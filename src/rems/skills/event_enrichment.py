from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# 事件充实（Event Enrichment）技能（白皮书 1.1.3、2.1、2.3）：
#   单次 LLM 调用同时产出「L1..Ln 递归摘要 + 角色列表（含 8 维情绪）」。
#   替代原来分离的 SummaryGenerationSkill + RoleExtractionSkill 组合（仅基本事件流使用）。
#   三种工作模式（详见 EventEnrichmentSkill.enrich）：
#   - full         ：summaries + 完整角色（snapshot + 8 维情绪），一次拿全；
#   - names_only   ：summaries + 仅角色名 (+ 可对齐的 role_id)，配合 pre-recall 全局池回填；
#   - summary_only ：仅 summaries（外层若已传入 role_entries，可省 roles 解析）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import (
    ENRICHMENT_FULL_USER,
    ENRICHMENT_MEMORY_USER,
    ENRICHMENT_SUMMARY_AND_NAMES_USER,
    ENRICHMENT_SUMMARY_ONLY_USER,
    build_enrichment_system_message,
)
from ..models.event import BasicEmotionVector, EmotionalModel, RoleSnapshot
from ..skills.role_extraction import ExtractedRole

if TYPE_CHECKING:
    from ..models.event import CompressionBudget
    from ..models.role import Role
    from .role_extraction import RoleExtractionSkill

logger = logging.getLogger(__name__)


class EnrichmentResult(BaseModel):
    """Unified output of a single event enrichment LLM call.

    ``summaries``       — L1..Ln dict, fuse-stop rule honored by the model.
    ``summary_lengths`` — len(text) per level, computed locally for downstream audits.
    ``actual_max_level``— number of generated levels; mirrors ``SummaryResult.actual_max_level`` semantics.
    ``roles``           — extracted role list (snapshots + emotion), same shape as ``RoleExtractionResult.roles``.
    """

    summaries: dict[str, str] = Field(default_factory=dict)
    summary_lengths: dict[str, int] = Field(default_factory=dict)
    actual_max_level: int = 0
    roles: list[ExtractedRole] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    location: str | None = None
    # memory 模式：事件级主体情感；对象表现的 L1 文本（name → L1）；完整级数在 object_levels。
    emotion: EmotionalModel | None = None
    object_snapshots: dict[str, str] = Field(default_factory=dict)
    object_levels: dict[str, dict[str, str]] = Field(default_factory=dict)
    object_white_paintings: dict[str, RoleSnapshot] = Field(default_factory=dict)


class EventEnrichmentSkill:
    """Single-call skill producing summaries (+ optional roles) for a sealed basic event.

    把原来两次 LLM 调用（summary_generation + role_extraction）合并为一次。``enrich(...)`` 三种工作模式：
        - ``skip_roles=True``  ：仅 summaries；外部已经传入完整 ``role_entries`` 时使用。
        - ``names_only=True``  ：summaries + 仅角色名（外加可对齐的 role_id），**不要 snapshot/情感**；
                                 配合 pipeline pre-recall 的「全局富信息池」，事件级 enrich 不重做 snapshot。
        - 默认 ``full``        ：summaries + 完整角色（snapshot + 8 维情绪）。
    优先级：``skip_roles`` > ``names_only`` > full。
    仅当 full / names_only 解析失败 + 显式提供 ``role_fallback`` 时，才会作为兜底再调一次 RoleExtractionSkill。
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: REMSConfig,
        role_fallback: "RoleExtractionSkill | None" = None,
    ):
        self._llm = llm
        self._config = config
        self._role_fallback = role_fallback

    def enrich(
        self,
        content_raw: str,
        *,
        known_roles: "list[Role] | None" = None,
        budget: "CompressionBudget | None" = None,
        skip_roles: bool = False,
        names_only: bool = False,
        memory_objects: dict[str, str] | None = None,
        seal_reason: str = "closed",
    ) -> EnrichmentResult:
        memory_mode = memory_objects is not None
        if memory_mode:
            # memory 路径：不抽角色；对象与情感由专门 prompt 产出。
            skip_roles = True
            names_only = False

        # skip_roles 优先：完全不出 roles 字段。
        if skip_roles:
            names_only = False

        summary_budget_text = "按系统默认要求"
        if budget:
            summary_budget_text = "\n".join(
                f"- {lvl}: {b} 字以内" for lvl, b in budget.summary_level_budgets.items()
            )

        fuse_min = self._config.summary_fuse_min_chars

        known_desc = "无已知角色" if not known_roles else "\n".join(
            f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
            for r in (known_roles or [])
        )

        if memory_mode:
            object_list = "\n".join(
                f"- {name}（{oid}）" for name, oid in memory_objects.items()
            ) or "（无对象，主体记忆）"
            user_msg = ENRICHMENT_MEMORY_USER.format(
                content_raw=content_raw,
                object_list=object_list,
                summary_budget_table=summary_budget_text,
            )
            if seal_reason in ("split", "truncated"):
                user_msg += (
                    "\n\n## 封存口径\n"
                    "本段因切开或搁置而封存，不是自然结束。"
                    "摘要和对象事实只描述原文里已经出现的内容，"
                    "不要补写结局，不要写成这件事已经结束。"
                )
            mode = "memory"
        elif skip_roles:
            user_msg = ENRICHMENT_SUMMARY_ONLY_USER.format(
                content_raw=content_raw,
                summary_budget_table=summary_budget_text,
            )
            mode = "summary_only"
        elif names_only:
            user_msg = ENRICHMENT_SUMMARY_AND_NAMES_USER.format(
                content_raw=content_raw,
                known_roles=known_desc,
                summary_budget_table=summary_budget_text,
            )
            mode = "names_only"
        else:
            snapshot_budgets_text = "按系统默认要求"
            if budget:
                snapshot_budgets_text = "\n".join(
                    f"- {lvl}: {b} 字以内" for lvl, b in budget.snapshot_level_budgets.items()
                )
            user_msg = ENRICHMENT_FULL_USER.format(
                content_raw=content_raw,
                known_roles=known_desc,
                summary_budget_table=summary_budget_text,
                snapshot_budgets=snapshot_budgets_text,
            )
            mode = "full"

        system_msg = build_enrichment_system_message(
            self._config, fuse_min_chars=fuse_min, mode=mode,
        )

        data = self._llm.complete_json(
            "event_enrichment",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        summaries = self._parse_summaries(data.get("summaries"))
        summary_lengths = {k: len(v) for k, v in summaries.items()}
        actual_max_level = len(summaries)

        roles: list[ExtractedRole] = []
        if not skip_roles and not memory_mode:
            roles = self._parse_roles(data.get("roles"), names_only=names_only)
            # 兜底：解析失败但配置了 fallback 时，单独再调一次（保留旧行为以增强健壮性）。
            # names_only 模式下也允许兜底——但兜底拿回的是完整 ExtractedRole（含 snapshot/情感），
            # 上层 EventService 仍按"按名字回填池"的语义处理：池里有就用池里的，池外的角色才用兜底数据。
            if not roles and self._role_fallback is not None:
                try:
                    extraction = self._role_fallback.extract(content_raw, known_roles=known_roles)
                    roles = list(extraction.roles)
                except Exception as e:
                    logger.warning("Fallback role extraction failed: %s", e)

        emotion: EmotionalModel | None = None
        object_snapshots: dict[str, str] = {}
        object_levels: dict[str, dict[str, str]] = {}
        object_white_paintings: dict[str, RoleSnapshot] = {}
        if memory_mode:
            raw_emo = data.get("emotion") or {}
            if isinstance(raw_emo, dict):
                emotion_init: dict[str, float] = {}
                for k, v in raw_emo.items():
                    if k in BasicEmotionVector.model_fields:
                        try:
                            emotion_init[k] = float(v)
                        except (TypeError, ValueError):
                            emotion_init[k] = 0.0
                if emotion_init:
                    emotion = EmotionalModel.from_emotion(BasicEmotionVector(**emotion_init))
            raw_objs = data.get("objects")
            if isinstance(raw_objs, list):
                for od in raw_objs:
                    if not isinstance(od, dict):
                        continue
                    name = (od.get("name") or "").strip()
                    if not name:
                        continue
                    raw_fact = od.get("fact")
                    levels = self._parse_summaries(raw_fact if isinstance(raw_fact, dict) else od.get("levels"))
                    if levels:
                        ordered = sorted(levels, key=lambda key: int(key[1:]))
                        object_levels[name] = levels
                        object_snapshots[name] = levels[ordered[0]]
                        continue
                    if isinstance(raw_fact, str) and raw_fact.strip():
                        object_snapshots[name] = raw_fact.strip()
                        continue
                    summary = (od.get("summary") or "").strip()
                    if summary:
                        object_snapshots[name] = summary

        keywords: list[str] = []
        location: str | None = None
        if memory_mode or self._config.recall_enrichment_metadata_enabled:
            raw_kw = data.get("keywords")
            if isinstance(raw_kw, list):
                keywords = [str(x).strip() for x in raw_kw if str(x).strip()]
            loc = data.get("location")
            if isinstance(loc, str) and loc.strip():
                location = loc.strip()

        return EnrichmentResult(
            summaries=summaries,
            summary_lengths=summary_lengths,
            actual_max_level=actual_max_level,
            roles=roles,
            keywords=keywords,
            location=location,
            emotion=emotion,
            object_snapshots=object_snapshots,
            object_levels=object_levels,
            object_white_paintings=object_white_paintings,
        )

    @staticmethod
    def _parse_summaries(raw: object) -> dict[str, str]:
        out: dict[str, str] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            if not isinstance(k, str) or not k.startswith("L") or not k[1:].isdigit():
                continue
            if not isinstance(v, str):
                continue
            text = v.strip()
            if not text:
                continue
            out[k] = text
        return out

    @staticmethod
    def _parse_roles(raw: object, *, names_only: bool = False) -> list[ExtractedRole]:
        if not isinstance(raw, list):
            return []
        out: list[ExtractedRole] = []
        seen_names: set[str] = set()
        for rd in raw:
            if names_only:
                # names_only 路径契约：模型直接输出名字字符串数组 ["角色名1", "角色名2"]。
                # 兼容旧契约：若仍是 dict 形态，只取 name 字段，role_id 由代码后续用
                # known_roles 做字符串匹配自行解析（详见 EventService.seal_event）。
                if isinstance(rd, str):
                    name = rd.strip()
                elif isinstance(rd, dict):
                    name = (rd.get("name") or "").strip()
                else:
                    continue
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                out.append(ExtractedRole(name=name))
                continue
            if not isinstance(rd, dict):
                continue
            name = rd.get("name", "") or ""
            snap = rd.get("snapshot") or {}
            emo = rd.get("emotion") or {}
            emotion_init: dict[str, float] = {}
            if isinstance(emo, dict):
                for k, v in emo.items():
                    if k in BasicEmotionVector.model_fields:
                        try:
                            emotion_init[k] = float(v)
                        except (TypeError, ValueError):
                            emotion_init[k] = 0.0
            emotion = BasicEmotionVector(**emotion_init)
            out.append(ExtractedRole(
                role_id=rd.get("role_id"),
                name=name,
                entity_type=rd.get("entity_type", "person"),
                importance=rd.get("importance", "C"),
                snapshot=RoleSnapshot(
                    l1_mention=snap.get("l1_mention") if isinstance(snap, dict) else None,
                    l2_interaction=snap.get("l2_interaction") if isinstance(snap, dict) else None,
                    l3_decision=snap.get("l3_decision") if isinstance(snap, dict) else None,
                ),
                emotional_model=EmotionalModel.from_emotion(emotion),
            ))
        return out
