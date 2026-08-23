from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

# 事件代谢：残影合并、边界检测、封存、未完成库维护、物理红线强制整理。
# 对照《REMS 记忆系统规范解析》4.1–4.2。
# 2026-05 升级：引入"评估 → 修复 → 保留 oversized UC"三段式处理，
# 取代旧的"越阈值就 force-seal"盲目兜底（盲目封存会让 enrichment 对
# 未闭环叙事产生噪声角色/摘要）。物理红线仍作为最后一道防线保留。

from ..config import REMSConfig
from ..models.event import Event, EventRoleEntry
from ..models.metabolism import Shadow, UnclosedEvent
from ..models.role import Role
from ..port import MemoryInput
from ..services.event_service import EventService
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
        memory: MemoryInput | None = None,
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
        # 白皮书新定义：残影是未完成事件的直接拼接
        shadow_content = "\n".join(ue.merged_content for ue in unclosed)

        if force_save:
            return self._force_save_all(
                shadow_content, raw_input, unclosed,
                input_id=input_id,
                known_roles_hint=known_roles_hint,
                pre_role_entries=pre_role_entries,
                memory=memory,
            )

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

        return self._apply_boundary_result(
            result, unclosed,
            input_id=input_id,
            known_roles_hint=known_roles_hint,
            pre_role_entries=pre_role_entries,
            memory=memory,
        )

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
        memory: MemoryInput | None = None,
    ) -> list[Event]:
        """Apply LLM boundary result and refresh shadow / unclosed library.

        关键不变量（修自 P0-5 残影跨轮重复 bug）：**残影仅由当前未完成事件拼接而成**
        （白皮书 §4.1 视图定义）。本轮入口处 ``shadow = join(previous_unclosed)``，
        模型已经看到 shadow 全部内容；剩余仍未闭合的部分**必须**通过
        ``result.new_unclosed`` 重新声明，否则按白皮书 §4.2.2 机制 3「无主碎屑垃圾回收」
        被丢弃（trace decay）。所以本轮结束时：

            1. 已被 ``continuation_of`` 命中的旧 UC 在循环中删除；
            2. 把**所有**未被命中的旧 UC 一次性清空——避免与 new_unclosed 内容重复持有；
            3. 用 ``result.new_unclosed`` 重建未完成库；
            4. 残影 = join(new_unclosed.merged_content)。

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
        # split_id → prefix event_id，便于 tail UC 落库时回写 split_prefix_event_ids。
        split_id_to_prefix_event_id: dict[str, str] = {}

        for frag in result.completed_events:
            if frag.continuation_of:
                ue = self._find_unclosed(unclosed, frag.continuation_of)
                if ue:
                    content = ue.merged_content + "\n" + frag.content_raw
                    # 继承该 UC 持有的前缀链（若它本身是 tail UC）。
                    inherited_prefix_chain = list(ue.split_prefix_event_ids or [])
                    self._repo.delete_unclosed_event(ue.id)
                    consumed_uc_ids.add(ue.id)
                else:
                    content = frag.content_raw
                    inherited_prefix_chain = []
            else:
                content = frag.content_raw
                inherited_prefix_chain = []

            event = self._event_svc.seal_event(
                content,
                split_prefix_event_ids=inherited_prefix_chain or None,
                **self._seal_memory_kwargs(
                    memory,
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

        # 关键修复：清空所有未被 continuation_of 命中的旧 UC——它们的内容已经作为 shadow
        # 整段进入 boundary 模型；模型若仍认为它们未闭合，应通过 ``new_unclosed`` 重新声明，
        # 否则被认定为已彻底失去叙事价值的"无主碎屑"，按 trace decay 丢弃。
        # 不这样做就会"旧 UC + 新 UC 同时持有同一段文本"——长跑会单调膨胀。
        for ue in unclosed:
            if ue.id not in consumed_uc_ids:
                self._repo.delete_unclosed_event(ue.id)

        # 新建未完成事件：不再盲目 force-seal；越阈值条目仅标记 oversized=True。
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

            prefix_event_id = split_id_to_prefix_event_id.get(nu.split_id) if nu.split_id else None
            ue = UnclosedEvent(
                id=f"UC-{secrets.token_hex(4)}",
                subject_id=memory.subject_id if memory is not None else "",
                content_fragments=[nu.content],
                logical_gaps=nu.logical_gaps,
                split_prefix_event_ids=[prefix_event_id] if prefix_event_id else [],
                oversized=is_oversized,
            )
            self._repo.save_unclosed_event(ue)

        # 更新残影记录（为保持一致性，每次代谢后同步更新）
        final_unclosed = self._repo.get_unclosed_events()
        new_shadow_content = "\n".join(ue.merged_content for ue in final_unclosed)
        self._repo.update_shadow(Shadow(
            content=new_shadow_content,
            updated_at=datetime.now(),
            subject_id=memory.subject_id if memory is not None else "",
        ))

        self._maybe_shadow_compact(final_unclosed)
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
        memory: MemoryInput | None = None,
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
                **self._seal_memory_kwargs(
                    memory,
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
                    **self._seal_memory_kwargs(
                        memory,
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
        memory: MemoryInput | None,
        *,
        input_id: str | None,
        known_roles_hint: list[Role] | None,
        pre_role_entries: list[EventRoleEntry] | None,
    ) -> dict:
        """记忆路径（memory 非 None）传主体/对象/来源/时间/origin；旧路径传角色提示。"""
        if memory is not None:
            return {
                "input_id": memory.input_id,
                "subject_id": memory.subject_id,
                "objects": memory.objects,
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
    def _maybe_shadow_compact(self, unclosed: list[UnclosedEvent]) -> None:
        """Rewrite fragmented unclosed narratives when fragment count exceeds threshold (§4.2.2)."""
        skill = self._shadow_compaction
        if skill is None or not skill.should_compact(unclosed):
            return
        for ue in unclosed:
            compacted = skill.compact(ue.merged_content)
            if compacted != ue.merged_content:
                ue.merged_content = compacted
                self._repo.save_unclosed_event(ue)
                logger.info("Shadow-compacted unclosed %s", ue.id)

    def _check_physical_redline(self) -> None:
        """If shadow + unclosed exceed physical redline, force-compact.

        当残影长度与所有未完成事件片段长度之和超过 ``physical_redline`` 时触发：
        对超过 ``len_msg * unclosed_force_ratio`` 的未完成条目强制封存；按策略遗忘陈旧未完成项；
        若残影仍高于 ``safe_watermark`` 则截断保留尾部，防止缓冲区无限增长（白皮书 4.2 工程化防溢出）。

        注意：这是"最后一道物理防线"——即便认知层（评估 → 修复）失败，也不允许
        残影无上限膨胀。此处 force-seal 仍保留，并在封存时继承 ``split_prefix_event_ids``。
        """
        shadow = self._repo.get_shadow()
        unclosed = self._repo.get_unclosed_events()
        total = shadow.length + sum(ue.total_length for ue in unclosed)

        if total <= self._config.physical_redline:
            return

        logger.warning("Physical redline hit (%d > %d), forcing compaction", total, self._config.physical_redline)

        for ue in unclosed:
            force_threshold = self._config.len_msg * self._config.unclosed_force_ratio
            if ue.total_length >= force_threshold:
                event = self._event_svc.seal_event(
                    ue.merged_content,
                    split_prefix_event_ids=list(ue.split_prefix_event_ids or []) or None,
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
                logger.info("Force-sealed unclosed %s as event %s (physical-redline tier)", ue.id, event.event_id)

        self._forget_stale_unclosed(unclosed)

        remaining_shadow = self._repo.get_shadow()
        if remaining_shadow.length > self._config.safe_watermark:
            trimmed = remaining_shadow.content[-self._config.safe_watermark:]
            self._repo.update_shadow(Shadow(content=trimmed, updated_at=datetime.now()))

    # ------------------------------------------------------------------
    def _forget_stale_unclosed(self, unclosed: list[UnclosedEvent], stale_hours: int = 24) -> None:
        # 长时间未续写的未完成条目丢弃，避免库无限膨胀（工程策略）。
        cutoff = datetime.now() - timedelta(hours=stale_hours)
        for ue in unclosed:
            if ue.last_hit_time < cutoff:
                logger.info("Forgetting stale unclosed event %s", ue.id)
                self._repo.delete_unclosed_event(ue.id)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_unclosed(events: list[UnclosedEvent], ue_id: str) -> Optional[UnclosedEvent]:
        for ue in events:
            if ue.id == ue_id:
                return ue
        return None
