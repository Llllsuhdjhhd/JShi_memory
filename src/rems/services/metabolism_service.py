from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

# 事件代谢：残影合并、边界检测、封存、未完成库维护、物理红线强制整理。
# 对照《REMS 记忆系统规范解析》4.1–4.2。
# 2026-05 升级：引入"评估 → 修复 → 保留 oversized UC"三段式处理，
# 取代旧的"越阈值就 force-seal"盲目兜底（盲目封存会让 enrichment 对
# 未闭环叙事产生噪声角色/摘要）。物理红线仍作为最后一道防线保留。

from ..config import REMSConfig
from ..models.event import Event, EventRoleEntry
from ..models.metabolism import Shadow, UnclosedEvent
from ..models.role import Role
from ..port import MemoryExperience
from ..services.event_service import EventService
from ..utils.text import segment_text_hits
from ..skills.boundary_detection import BoundaryDetectionSkill, BoundaryResult
from ..skills.shadow_compaction import ShadowCompactionSkill
from ..skills.boundary_split import (
    BoundaryForceThresholdEvaluator,
    BoundarySkillContext,
    OverlongUCSplitRemediator,
    OverlongUCSplitSkill,
)
from ..skills.evaluation import SkillEvaluator, SkillRemediator, run_skill_with_eval
from ..storage.repository import EventRepository, MetabolismRepository

logger = logging.getLogger(__name__)


def _objects_for_event(
    content: str | None, memory: MemoryExperience | None
) -> dict[str, str]:
    """整批合并时按事件内容归属对象：命中子段的对象并集，未命中回退全批对象池。"""
    if memory is None:
        return {}
    subs = getattr(memory, "sub_segments", None)
    if not content or not subs:
        return dict(memory.objects or {})
    hits: dict[str, str] = {}
    for sub in subs:
        if segment_text_hits(sub.text or "", content):
            hits.update(sub.objects)
    return hits or dict(memory.objects or {})


class MetabolismService:
    """Manages shadow buffer, unclosed-event library and event trigger logic.

    Lifecycle:
        1. Accept raw input.
        2. Merge with shadow.
        3. Run boundary detection.
        4. Evaluate boundary output; if oversized UC detected, remediate via
           ``OverlongUCSplitSkill`` (one pass only — evaluator does NOT re-run
           on remediation output, by design).
        5. Seal completed events (including split prefixes); update unclosed
           library; when a tail UC later closes, inherit its ``split_prefix_event_ids``
           chain to the sealed event and append the successor id to each prefix event.
        6. Enforce physical-redline compaction when needed (last-resort defence).

    管理残影（Shadow）、未完成事件库及封存触发逻辑。典型生命周期为：接收原始输入并与残影合并；
    调用边界检测技能划分已闭环片段与剩余残影；对已闭环内容调用 ``EventService.seal_event``；
    更新未完成库；当残影与未完成总长超过 ``config.physical_redline`` 时执行强制压缩与遗忘策略。
    """

    def __init__(
        self,
        config: REMSConfig,
        meta_repo: MetabolismRepository,
        boundary_skill: BoundaryDetectionSkill,
        event_service: EventService,
        *,
        event_repo: EventRepository | None = None,
        boundary_evaluators: list[SkillEvaluator] | None = None,
        boundary_remediator: SkillRemediator | None = None,
        shadow_compaction: ShadowCompactionSkill | None = None,
    ):
        self._config = config
        self._repo = meta_repo
        self._boundary = boundary_skill
        self._event_svc = event_service
        self._event_repo = event_repo
        self._shadow_compaction = shadow_compaction
        # 默认评估链：仅规则型评估器（零 LLM 成本，始终开）。LLM 二级评估默认关。
        self._boundary_evaluators: list[SkillEvaluator] = (
            boundary_evaluators
            if boundary_evaluators is not None
            else [BoundaryForceThresholdEvaluator(config)]
        )
        # 默认修复链：调用 OverlongUCSplitSkill，仅在 ``boundary_remediation_enabled`` 打开时装配。
        self._boundary_remediator: SkillRemediator | None = boundary_remediator

    # ------------------------------------------------------------------
    # Factory helpers (便于 pipeline / tests 复用默认装配)
    # ------------------------------------------------------------------

    @classmethod
    def with_default_boundary_repair(
        cls,
        config: REMSConfig,
        meta_repo: MetabolismRepository,
        boundary_skill: BoundaryDetectionSkill,
        event_service: EventService,
        *,
        event_repo: EventRepository | None = None,
        llm=None,
        shadow_compaction: ShadowCompactionSkill | None = None,
    ) -> "MetabolismService":
        """Construct a service wired with the default evaluator + split remediator.

        调用侧若希望拿"开箱即用"的 80/20 分裂修复链，用这个工厂方法即可；
        ``llm`` 若未显式传入，则从 ``boundary_skill`` 上读取（两者共享同一 provider）。
        ``boundary_remediation_enabled=False`` 时不装配 remediator，评估只做记录。
        """
        evaluators: list[SkillEvaluator] = [BoundaryForceThresholdEvaluator(config)]
        remediator: SkillRemediator | None = None
        llm_for_split = llm if llm is not None else getattr(boundary_skill, "_llm", None)
        if config.boundary_remediation_enabled:
            split_skill = OverlongUCSplitSkill(llm_for_split, config)
            remediator = OverlongUCSplitRemediator(split_skill)
        if shadow_compaction is None and llm_for_split is not None:
            shadow_compaction = ShadowCompactionSkill(llm_for_split, config)
        return cls(
            config,
            meta_repo,
            boundary_skill,
            event_service,
            event_repo=event_repo,
            boundary_evaluators=evaluators,
            boundary_remediator=remediator,
            shadow_compaction=shadow_compaction,
        )

    # ------------------------------------------------------------------
    # Main entry: process a raw input string
    # ------------------------------------------------------------------

    def process_input(
        self,
        raw_input: str,
        *,
        force_save: bool = False,
        input_id: str | None = None,
        known_roles_hint: list[Role] | None = None,
        pre_role_entries: list[EventRoleEntry] | None = None,
        ingest_session: Any | None = None,
        memory: MemoryExperience | None = None,
    ) -> list[Event]:
        """Ingest *raw_input*, return list of newly sealed events (may be empty).

        摄入字符串 *raw_input*，返回本轮新封存的基本事件列表（可能为空列表）。
        ``force_save=True`` 时跳过边界模型，立即合并残影与未完成项并封存（手动 /save 类触发）。

        ``pre_role_entries``：pipeline pre-recall 阶段在 ``shadow + raw_input`` 上抽到的
        「全局富信息池」（每个 EventRoleEntry 含 snapshot + 8 维情绪）。下放给
        ``EventService.seal_event``，让事件级 enrichment 走 names_only 分支并按 role_id
        从池里回填 snapshot/情感（详见 ``EventService.seal_event`` 的 ``pre_role_entries``）。

        白皮书 4.2 说明：``msg_len × 1.2`` 的兜底阈值作用于**未闭环事件的累积长度**，
        而非整体输入。边界检测不会因「残影 + 当前输入」过长被跳过；超长输入在边界剥离
        得到的单条未闭环片段越过红线时，会由 80/20 分裂修复器尝试切成前缀 + 尾部；
        修复失败时仅标记 ``oversized=True`` 并保留，物理红线层作为最后兜底。
        """
        if len(raw_input) > self._config.len_msg:
            logger.warning(
                "Input length %d exceeds len_msg %d; boundary detection will still run",
                len(raw_input),
                self._config.len_msg,
            )

        unclosed = self._repo.get_unclosed_events()
        now = self._ingest_time(memory)

        abandoned: list[Event] = []
        if not force_save:
            abandoned, unclosed = self._seal_abandoned_unclosed(
                unclosed,
                now=now,
                memory=memory,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
            )

        # 白皮书新定义：残影是未完成事件的直接拼接
        shadow_content = "\n".join(ue.merged_content for ue in unclosed)

        if force_save:
            sealed = self._force_save_all(
                shadow_content, raw_input, unclosed,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
                memory=memory,
            )
            self._raise_dormant_same_object(sealed)
            return sealed

        # 边界检测仅负责事件切分；摘要/角色等衍生字段由 EventEnrichment 在 seal 时生成。
        # 评估 → 修复 → 采用 三段式：评估器失败时，若配置了修复器则调用一次；
        # 修复后的输出**不再评估**（评估只做一层）。
        ctx = BoundarySkillContext(
            force_threshold=int(self._config.len_msg * self._config.unclosed_force_ratio),
            split_ratio_target=self._config.boundary_split_ratio_target,
            split_ratio_min=self._config.boundary_split_ratio_min,
            split_ratio_max=self._config.boundary_split_ratio_max,
            unclosed_events=tuple(unclosed),
        )

        def _run_boundary():
            if ingest_session is not None and getattr(ingest_session, "turn", 0) >= 1:
                return ingest_session.detect_boundary_followup(
                    shadow_content, raw_input, unclosed,
                )
            return self._boundary.detect(shadow_content, raw_input, unclosed)

        result, report = run_skill_with_eval(
            _run_boundary,
            evaluators=self._boundary_evaluators,
            remediator=self._boundary_remediator,
            skill_input=ctx,
        )
        if not report.ok:
            logger.info(
                "Boundary evaluator flagged %d issue(s): %s (remediator=%s)",
                len(report.issues),
                [i.code for i in report.issues],
                type(self._boundary_remediator).__name__ if self._boundary_remediator else None,
            )

        sealed = self._apply_boundary_result(
            result, unclosed,
            input_id=input_id,
            known_roles_hint=known_roles_hint,
            pre_role_entries=pre_role_entries,
            memory=memory,
        )
        sealed = abandoned + sealed
        self._raise_dormant_same_object(sealed)
        return sealed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_boundary_result(
        self,
        result: BoundaryResult,
        unclosed: list[UnclosedEvent],
        input_id: str | None = None,
        known_roles_hint: list[Role] | None = None,
        pre_role_entries: list[EventRoleEntry] | None = None,
        memory: MemoryExperience | None = None,
    ) -> list[Event]:
        """Apply LLM boundary result and refresh shadow / unclosed library.

        残影是未完成库的拼接。本轮模型把「残影 + 新输入」的序号划进事件、续写或新的未完成线。
        旧线被划走后删除，避免同一段文字存两份。没被任何序号领走的句子已由边界解析收成一条未完成线。
        本轮结束时：

            1. 已被 ``continuation_of`` 命中的旧 UC 在循环中删除；
            2. 删除其余旧 UC，避免与重新声明的内容叠成两份；
            3. 用 ``result.new_unclosed``（含未领走的句子）重建未完成库；
            4. 残影 = join(未完成线原文)。

        ``known_roles_hint`` 由 pipeline 在 pre-recall 阶段抽取得到，向下传给
        ``EventEnrichmentSkill``，仅作为代词消解的提示——不直接覆盖事件 role_list，
        因为单条 sealed event 的真实参与角色应由 enrichment 在该事件原文上重新精确判定。

        80/20 分裂链路处理（2026-05 新增）：
            - ``completed_events`` 中 ``is_split_prefix=True`` 的片段按常规封存，
              同时记录 ``split_id → event_id`` 映射；
            - ``new_unclosed`` 中若声明 ``split_id``，则落库的 UC 带上
              ``split_prefix_event_ids=[mapped_event_id]``（可跨多轮累积，故为 list）；
            - ``continuation_of`` 命中旧 UC 时，新封存事件从该 UC 继承
              ``split_prefix_event_ids``；并反向把新事件 id 追加到每个前缀事件的
              ``split_successor_event_ids``（供审计与未来的前缀链多跳展开）；
            - **不再**对越阈值的 ``new_unclosed`` 做 force-seal；评估失败且修复亦失败
              的 oversized 条目标记为 ``oversized=True`` 保留在未完成库，物理红线层兜底。
        """
        sealed: list[Event] = []
        consumed_uc_ids: set[str] = set()
        kept: list[UnclosedEvent] = []
        # split_id → prefix event_id，便于 tail UC 落库时回写 split_prefix_event_ids。
        split_id_to_prefix_event_id: dict[str, str] = {}
        minimum = int(self._config.event_min_chars or 0)

        for frag in result.completed_events:
            ue, content, inherited_prefix_chain = self._resolve_fragment(frag, unclosed)
            if ue:
                consumed_uc_ids.add(ue.id)
            hold_short = (
                not frag.is_split_prefix
                and minimum > 0
                and len(content) < minimum
            )
            if hold_short:
                same = ue is not None and content == ue.merged_content
                kept.append(self._window_line(
                    ue=ue,
                    content=content,
                    role="formed",
                    prefix=inherited_prefix_chain,
                    memory=memory,
                    last_hit=ue.last_hit_time if same and ue is not None else None,
                ))
                continue

            event = self._event_svc.seal_event(
                content,
                split_prefix_event_ids=inherited_prefix_chain or None,
                seal_reason="split" if frag.is_split_prefix else "closed",
                **self._seal_memory_kwargs(
                    memory,
                    content=content,
                    input_id=input_id,
                    known_roles_hint=known_roles_hint,
                    pre_role_entries=pre_role_entries,
                ),
            )
            sealed.append(event)

            # 如果该 completed 是一条分裂前缀，把它登记起来供 tail UC 关联。
            if frag.is_split_prefix and frag.split_id:
                split_id_to_prefix_event_id[frag.split_id] = event.event_id

            # 反向链：若本事件继承到前缀链，则把它作为 successor 登记回每个前缀事件。
            if inherited_prefix_chain and self._event_repo is not None:
                for prefix_id in inherited_prefix_chain:
                    try:
                        self._event_repo.append_split_successor(prefix_id, event.event_id)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "append_split_successor(%s, %s) failed; skipping",
                            prefix_id, event.event_id, exc_info=True,
                        )

        for text in result.no_form_texts:
            logger.info("Dust exit (no_form), not stored: %s", text[:200])

        # 已形成但本轮没被改写的短事件留在窗口里，不退回残影，也不删掉。
        for ue in unclosed:
            if ue.formation_role == "formed" and ue.id not in consumed_uc_ids:
                continue
            self._repo.delete_unclosed_event(ue.id)

        for item in kept:
            self._repo.save_unclosed_event(item)

        force_threshold = int(self._config.len_msg * self._config.unclosed_force_ratio)
        for nu in result.new_unclosed:
            is_oversized = len(nu.content) > force_threshold
            if is_oversized:
                logger.warning(
                    "Unclosed fragment length %d exceeds %.2f×len_msg (%d); "
                    "keeping as oversized UC (split remediation already attempted)",
                    len(nu.content),
                    self._config.unclosed_force_ratio,
                    force_threshold,
                )
            source = self._find_unclosed(unclosed, nu.continuation_of) if nu.continuation_of else None
            prefix_ids = list(source.split_prefix_event_ids or []) if source else []
            prefix_event_id = split_id_to_prefix_event_id.get(nu.split_id) if nu.split_id else None
            if prefix_event_id:
                prefix_ids.append(prefix_event_id)
            self._repo.save_unclosed_event(self._window_line(
                ue=None,
                content=nu.content,
                role="residual",
                prefix=prefix_ids,
                memory=memory,
                logical_gaps=nu.logical_gaps,
                oversized=is_oversized,
            ))

        for text in result.pending_texts:
            self._repo.save_unclosed_event(self._window_line(
                ue=None,
                content=text,
                role="rejudge",
                prefix=[],
                memory=memory,
            ))

        # 更新残影记录（为保持一致性，每次代谢后同步更新）
        final_unclosed = self._repo.get_unclosed_events()
        new_shadow_content = "\n".join(ue.merged_content for ue in final_unclosed)
        self._repo.update_shadow(Shadow(
            content=new_shadow_content,
            updated_at=datetime.now(),
            subject_id=memory.subject_id if memory is not None else "",
        ))

        self._check_physical_redline()

        return sealed

    # ------------------------------------------------------------------
    def _force_save_all(
        self,
        shadow_content: str,
        raw_input: str,
        unclosed: list[UnclosedEvent],
        *,
        is_suspicious: bool = False,
        input_id: str | None = None,
        known_roles_hint: list[Role] | None = None,
        pre_role_entries: list[EventRoleEntry] | None = None,
        memory: MemoryExperience | None = None,
    ) -> list[Event]:
        """Manual trigger (/save, /mem) or length-based fallback: seal everything immediately.

        手动触发或长度被迫兜底：将残影与当前输入合并后尽可能封存；遍历未完成库中已有内容的条目逐一封存并删除；
        最后清空残影。用于用户显式「保存记忆」或文本溢出边界强制回收。
        """
        sealed: list[Event] = []

        combined = (shadow_content + "\n" + raw_input).strip()
        if combined:
            event = self._event_svc.seal_event(
                combined,
                is_suspicious=is_suspicious,
                seal_reason="truncated",
                **self._seal_memory_kwargs(
                    memory,
                    content=combined,
                    input_id=input_id,
                    known_roles_hint=known_roles_hint,
                    pre_role_entries=pre_role_entries,
                ),
            )
            sealed.append(event)

        for ue in unclosed:
            if ue.total_length > 0:
                event = self._event_svc.seal_event(
                    ue.merged_content,
                    is_suspicious=is_suspicious,
                    split_prefix_event_ids=list(ue.split_prefix_event_ids or []) or None,
                    seal_reason="truncated",
                    **self._seal_memory_kwargs(
                        memory,
                        content=ue.merged_content,
                        input_id=input_id,
                        known_roles_hint=known_roles_hint,
                        pre_role_entries=pre_role_entries,
                    ),
                )
                # 反向链：force_save 也要登记 successor。
                if ue.split_prefix_event_ids and self._event_repo is not None:
                    for prefix_id in ue.split_prefix_event_ids:
                        try:
                            self._event_repo.append_split_successor(prefix_id, event.event_id)
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "append_split_successor(%s, %s) failed; skipping",
                                prefix_id, event.event_id, exc_info=True,
                            )
                sealed.append(event)
            self._repo.delete_unclosed_event(ue.id)

        self._repo.update_shadow(Shadow(content="", updated_at=datetime.now()))
        return sealed

    @staticmethod
    def _seal_memory_kwargs(
        memory: MemoryExperience | None,
        *,
        content: str | None = None,
        input_id: str | None,
        known_roles_hint: list[Role] | None,
        pre_role_entries: list[EventRoleEntry] | None,
    ) -> dict:
        """记忆路径（memory 非 None）传主体/对象/来源/时间/origin；旧路径传角色提示。

        整批合并时按 ``content`` 与各子段的重叠归属对象，避免整批对象并集串扰；
        未匹配到子段时回退到全批对象池（保持旧行为兼容）。
        """
        if memory is not None:
            objects = _objects_for_event(content, memory)
            return {
                "input_id": memory.segment_id,
                "subject_id": memory.subject_id,
                "objects": objects,
                "source_ids": memory.source_ids,
                "occurred_at": memory.occurred_at,
                "origin": memory.origin,
            }
        return {
            "input_id": input_id,
            "known_roles": known_roles_hint,
            "pre_role_entries": pre_role_entries,
        }

    # ------------------------------------------------------------------
    def _check_physical_redline(self) -> None:
        """总长越过物理红线时，把过长或已超时的整条未完成线按 truncated 封存。

        不改写残影，也不把残影字符串截尾。封存后按剩余未完成线重拼残影。
        """
        shadow = self._repo.get_shadow()
        unclosed = self._repo.get_unclosed_events()
        total = shadow.length + sum(ue.total_length for ue in unclosed)

        if total <= self._config.physical_redline:
            return

        logger.warning("Physical redline hit (%d > %d), sealing overlong lines", total, self._config.physical_redline)

        force_threshold = self._config.len_msg * self._config.unclosed_force_ratio
        sealed_ids: set[str] = set()
        for ue in unclosed:
            if ue.total_length < force_threshold:
                continue
            event = self._seal_unclosed_as_event(
                ue,
                memory=None,
                input_id=None,
                known_roles_hint=None,
                pre_role_entries=None,
                seal_reason="truncated",
            )
            sealed_ids.add(ue.id)
            logger.info("Force-sealed unclosed %s as event %s (physical-redline tier)", ue.id, event.event_id)

        self._forget_stale_unclosed([ue for ue in unclosed if ue.id not in sealed_ids])
        self._rebuild_shadow()

    # ------------------------------------------------------------------
    def _seal_abandoned_unclosed(
        self,
        unclosed: list[UnclosedEvent],
        *,
        now: datetime,
        memory: MemoryExperience | None,
        input_id: str | None,
        known_roles_hint: list[Role] | None,
        pre_role_entries: list[EventRoleEntry] | None,
    ) -> tuple[list[Event], list[UnclosedEvent]]:
        """按闲置规则封存未完成条目，原因记为 truncated。

        单条字数变长不在边界检测前封存，先交给边界和 80/20 切开。
        闲置启用时：字数 > 0.5×上限且闲置达到部分天数，或闲置达到硬天数。
        说话人更换不触发封存。
        """
        char_limit = int(self._config.unclosed_char_limit or 0)
        idle_on = bool(self._config.unclosed_idle_seal_enabled)
        if char_limit <= 0 and not idle_on:
            return [], list(unclosed)

        sealed: list[Event] = []
        remaining: list[UnclosedEvent] = []
        for ue in unclosed:
            if ue.total_length <= 0:
                self._repo.delete_unclosed_event(ue.id)
                continue
            if not self._is_abandoned_unclosed(ue, now=now):
                remaining.append(ue)
                continue
            if (ue.formation_role or "residual") == "formed":
                logger.info("Sealing held event %s after wait (closed)", ue.id)
                sealed.append(
                    self._seal_unclosed_as_event(
                        ue,
                        memory=memory,
                        input_id=input_id,
                        known_roles_hint=known_roles_hint,
                        pre_role_entries=pre_role_entries,
                        seal_reason="closed",
                    )
                )
                continue
            logger.info("Rejudging abandoned residual %s before seal", ue.id)
            sealed.extend(self._rejudge_abandoned(
                [ue],
                memory=memory,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
            ))

        if sealed:
            subject_id = memory.subject_id if memory is not None else ""
            self._repo.update_shadow(Shadow(
                content="\n".join(ue.merged_content for ue in remaining),
                updated_at=now,
                subject_id=subject_id,
            ))
        return sealed, remaining

    def _is_abandoned_unclosed(
        self,
        ue: UnclosedEvent,
        *,
        now: datetime,
    ) -> bool:
        """单条未完成事件是否应封存。"""
        length = int(ue.total_length)
        char_limit = int(self._config.unclosed_char_limit or 0)

        if not self._config.unclosed_idle_seal_enabled:
            return False

        hit = self._as_naive(ue.last_hit_time or ue.updated_at)
        if hit is None:
            return False
        idle = now - hit
        hard_days = float(self._config.unclosed_idle_hard_days or 0.0)
        partial_days = float(self._config.unclosed_idle_partial_days or 0.0)
        partial_ratio = float(self._config.unclosed_idle_partial_ratio or 0.5)

        # 1.3 闲置 ≥ 硬上限天数（默认 7 天）
        if hard_days > 0 and idle >= timedelta(days=hard_days):
            return True
        # 1.3 字数 > 0.5×限制 且闲置 ≥ 部分天数（默认 3 天）
        if (
            char_limit > 0
            and partial_days > 0
            and length > char_limit * partial_ratio
            and idle >= timedelta(days=partial_days)
        ):
            return True
        return False

    def _seal_unclosed_as_event(
        self,
        ue: UnclosedEvent,
        *,
        memory: MemoryExperience | None,
        input_id: str | None,
        known_roles_hint: list[Role] | None,
        pre_role_entries: list[EventRoleEntry] | None,
        seal_reason: str = "truncated",
    ) -> Event:
        event = self._event_svc.seal_event(
            ue.merged_content,
            split_prefix_event_ids=list(ue.split_prefix_event_ids or []) or None,
            seal_reason=seal_reason,
            **self._seal_memory_kwargs(
                memory,
                content=ue.merged_content,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
            ),
        )
        if ue.split_prefix_event_ids and self._event_repo is not None:
            for prefix_id in ue.split_prefix_event_ids:
                try:
                    self._event_repo.append_split_successor(prefix_id, event.event_id)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "append_split_successor(%s, %s) failed; skipping",
                        prefix_id, event.event_id, exc_info=True,
                    )
        self._repo.delete_unclosed_event(ue.id)
        self._raise_dormant_same_object([event])
        return event

    def _forget_stale_unclosed(self, unclosed: list[UnclosedEvent], stale_hours: int = 24) -> None:
        # 物理红线兜底：过久未续写的未完成改为封存，不再丢弃原文。
        now = datetime.now()
        cutoff = now - timedelta(hours=stale_hours)
        for ue in unclosed:
            if ue.total_length <= 0:
                continue
            hit = ue.last_hit_time or ue.updated_at
            if hit is None or hit >= cutoff:
                continue
            if self._repo.get_unclosed_event(ue.id) is None:
                continue
            logger.info("Sealing stale unclosed event %s (physical-redline idle)", ue.id)
            self._seal_unclosed_as_event(
                ue,
                memory=None,
                input_id=None,
                known_roles_hint=None,
                pre_role_entries=None,
            )

    def _rebuild_shadow(self) -> None:
        remaining = self._repo.get_unclosed_events()
        self._repo.update_shadow(Shadow(
            content="\n".join(ue.merged_content for ue in remaining),
            updated_at=datetime.now(),
        ))

    def _raise_dormant_same_object(self, events: list[Event]) -> None:
        """同一对象出现新事件时，把该对象仍低于静默阈值的旧事件抬到阈值。

        只改遗忘因子，不改原文，也不改召回管线。
        """
        if not events or self._event_repo is None:
            return
        threshold = float(self._config.forgetting_silence_threshold or 0.02)
        for event in events:
            object_ids = {
                entry.role_id
                for entry in (event.role_list or [])
                if not getattr(entry, "is_subject", False) and entry.role_id
            }
            if not object_ids:
                continue
            subject = event.subject_id or ""
            for other in self._event_repo.list_all(exclude_tombstoned=True):
                if other.event_id == event.event_id:
                    continue
                if subject and other.subject_id and other.subject_id != subject:
                    continue
                if float(other.forgetting_factor or 0) >= threshold:
                    continue
                other_ids = {
                    entry.role_id
                    for entry in (other.role_list or [])
                    if not getattr(entry, "is_subject", False) and entry.role_id
                }
                if object_ids & other_ids:
                    self._event_repo.update_forgetting_factor(other.event_id, threshold)

    @staticmethod
    def _ingest_time(memory: MemoryExperience | None) -> datetime:
        if memory is not None and getattr(memory, "occurred_at", None):
            return MetabolismService._as_naive(memory.occurred_at)
        return datetime.now()

    @staticmethod
    def _as_naive(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def _resolve_fragment(
        self,
        frag,
        unclosed: list[UnclosedEvent],
    ) -> tuple[UnclosedEvent | None, str, list[str]]:
        ue = self._find_unclosed(unclosed, frag.continuation_of) if frag.continuation_of else None
        if getattr(frag, "content_includes_source", False):
            content = frag.content_raw
        elif ue and frag.content_raw:
            content = ue.merged_content + "\n" + frag.content_raw
        elif ue:
            content = ue.merged_content
        else:
            content = frag.content_raw
        prefix = list(ue.split_prefix_event_ids or []) if ue else []
        return ue, content, prefix

    def _window_line(
        self,
        *,
        ue: UnclosedEvent | None,
        content: str,
        role: str,
        prefix: list[str],
        memory: MemoryExperience | None,
        last_hit: datetime | None = None,
        logical_gaps: str | None = None,
        oversized: bool = False,
    ) -> UnclosedEvent:
        now = self._ingest_time(memory)
        hit = last_hit or now
        subject = ""
        if ue is not None and ue.subject_id:
            subject = ue.subject_id
        elif memory is not None:
            subject = memory.subject_id
        interlocutor = ue.interlocutor if ue is not None else None
        if memory is not None and getattr(memory, "interlocutor", None):
            interlocutor = memory.interlocutor
        return UnclosedEvent(
            id=ue.id if ue is not None else f"UC-{secrets.token_hex(4)}",
            subject_id=subject,
            content_fragments=[content],
            logical_gaps=logical_gaps if logical_gaps is not None else (ue.logical_gaps if ue else None),
            split_prefix_event_ids=list(prefix or []),
            oversized=oversized,
            interlocutor=interlocutor,
            created_at=ue.created_at if ue is not None else now,
            updated_at=now,
            last_hit_time=hit,
            formation_role=role,
        )

    def _rejudge_abandoned(
        self,
        lines: list[UnclosedEvent],
        *,
        memory: MemoryExperience | None,
        input_id: str | None,
        known_roles_hint: list[Role] | None,
        pre_role_entries: list[EventRoleEntry] | None,
    ) -> list[Event]:
        """搁置的残影再划分一次。已有边界的封为 closed，仍未形成的封为 truncated。"""
        try:
            result = self._boundary.detect("", "", lines)
        except Exception:
            logger.exception("Formation rejudge failed; sealing residual as truncated")
            return [
                self._seal_unclosed_as_event(
                    ue,
                    memory=memory,
                    input_id=input_id,
                    known_roles_hint=known_roles_hint,
                    pre_role_entries=pre_role_entries,
                    seal_reason="truncated",
                )
                for ue in lines
            ]

        sealed: list[Event] = []
        handled: set[str] = set()
        no_form = set(result.no_form_texts)
        if result.completed_events:
            for frag in result.completed_events:
                ue, content, prefix = self._resolve_fragment(frag, lines)
                if ue:
                    handled.add(ue.id)
                event = self._event_svc.seal_event(
                    content,
                    split_prefix_event_ids=prefix or None,
                    seal_reason="closed",
                    **self._seal_memory_kwargs(
                        memory,
                        content=content,
                        input_id=input_id,
                        known_roles_hint=known_roles_hint,
                        pre_role_entries=pre_role_entries,
                    ),
                )
                sealed.append(event)
                if ue is not None:
                    self._repo.delete_unclosed_event(ue.id)
                    self._raise_dormant_same_object([event])
        for ue in lines:
            if ue.id in handled:
                continue
            if ue.merged_content and ue.merged_content in no_form:
                logger.info("Dust exit on rejudge, not stored: %s", ue.merged_content[:200])
                self._repo.delete_unclosed_event(ue.id)
                continue
            sealed.append(self._seal_unclosed_as_event(
                ue,
                memory=memory,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
                seal_reason="truncated",
            ))
        return sealed

    # ------------------------------------------------------------------
    @staticmethod
    def _find_unclosed(events: list[UnclosedEvent], ue_id: str) -> Optional[UnclosedEvent]:
        for ue in events:
            if ue.id == ue_id:
                return ue
        return None
