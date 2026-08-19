from __future__ import annotations

import logging
from dataclasses import dataclass

# 归纳演化：多条基本/低阶事件 → 合成一条抽象事件（``is_abstract=True``）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import (
    EVOLUTION_SYSTEM,
    EVOLUTION_USER,
    NARRATIVE_COHERENCE_SYSTEM,
    NARRATIVE_COHERENCE_USER,
)
from ..models.event import Event, EventStatus, generate_event_id

logger = logging.getLogger(__name__)

COGNITIVE_RELATIONS = frozenset({
    "CAUSALITY", "PROTOTYPE_INVARIANT", "TEMPORAL_CHRONO",
    "COGNITIVE_DIALECTIC", "MERONYMY", "NONE",
})


@dataclass(frozen=True)
class SynthesisOutcome:
    event: Event | None
    cognitive_relation: str
    shadow_lambda: float
    rejected_none: bool = False


@dataclass(frozen=True)
class NarrativeCoherenceOutcome:
    """LLM partitioning of mined subset events into one coherent narrative strand vs unrelated items."""

    coherent_event_ids: frozenset[str]
    excluded_event_ids: frozenset[str]
    rate_a: float
    passed_gate: bool  # coherent count >= configured min


class InductiveEvolutionSkill:
    """Synthesise an abstract event from a cluster of basic/lower-order events.

    由多条基本/低阶事件归纳生成抽象事件（``is_abstract=True``，并填充 ``source_events``、
    ``content_raw``、可选 ``insight`` / ``decoration``）。抽象事件 **无角色、无角色快照、无情感模型**；
    summaries 由外层 ``SummaryGenerationSkill`` 基于 ``content_raw`` 递归生成（白皮书 §3.1–§3.3）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def evaluate_narrative_coherence(self, events: list[Event]) -> NarrativeCoherenceOutcome:
        """Partition ``events`` into one coherent narrative strand vs unrelated.

        Judges **logical subset members**（挖矿条目）—not only flattened leaf basics.
        On JSON/validation failure → all events treated excluded, ``rate_a=0``.
        """

        cfg = self._config
        if not events:
            return NarrativeCoherenceOutcome(
                coherent_event_ids=frozenset(),
                excluded_event_ids=frozenset(),
                rate_a=0.0,
                passed_gate=False,
            )
        expected = frozenset(e.event_id for e in events)

        logical_text = self._build_logical_evidence_text(events)
        user_msg = NARRATIVE_COHERENCE_USER.format(
            event_contents=logical_text,
        )

        try:
            data = self._llm.complete_json(
                "narrative_coherence",
                [
                    {"role": "system", "content": NARRATIVE_COHERENCE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:
            logger.warning("narrative coherence LLM failure: %s", exc)
            return NarrativeCoherenceOutcome(
                coherent_event_ids=frozenset(),
                excluded_event_ids=expected,
                rate_a=0.0,
                passed_gate=False,
            )

        raw_coh = list(data.get("coherent_event_ids") or [])
        raw_exc = list(data.get("excluded_event_ids") or [])
        coh_set = frozenset(str(x) for x in raw_coh if x)
        exc_set = frozenset(str(x) for x in raw_exc if x)

        invalid = False
        if coh_set - expected or exc_set - expected:
            invalid = True
        if coh_set & exc_set:
            invalid = True
        if coh_set | exc_set != expected:
            invalid = True

        if invalid:
            logger.warning(
                "narrative coherence invalid partition "
                "(expected=%d coherent=%d excluded=%d)",
                len(expected),
                len(coh_set),
                len(exc_set),
            )
            return NarrativeCoherenceOutcome(
                coherent_event_ids=frozenset(),
                excluded_event_ids=expected,
                rate_a=0.0,
                passed_gate=False,
            )

        n = len(expected)
        rate_a = len(coh_set) / max(1, n)
        min_c = max(1, getattr(cfg, "abstract_narrative_coherence_min_count", 3))
        passed_gate = len(coh_set) >= min_c
        return NarrativeCoherenceOutcome(
            coherent_event_ids=coh_set,
            excluded_event_ids=exc_set,
            rate_a=rate_a,
            passed_gate=passed_gate,
        )

    def synthesize(
        self,
        events: list[Event],
        abstraction_level: int = 1,
        *,
        evidence_events: list[Event] | None = None,
    ) -> SynthesisOutcome:
        """Create an abstract event with semantic posterior in one LLM call (§3.2.3)."""
        evidence = evidence_events or events
        event_contents = self._build_content_raw_text(evidence)
        leaf_count, leaf_avg_len, target_content_len = self._leaf_evidence_stats(evidence)
        insight_enabled = self._config.enable_abstract_insight
        insight_instruction = (
            "- `insight`：开启。JSON 对象含 relation（认知关系常量）与 conclusion（规律结论）。"
            if insight_enabled
            else "- `insight`：关闭。不要输出 insight 字段。"
        )
        json_schema = (
            '{\n'
            '  "cognitive_relation": "CAUSALITY|PROTOTYPE_INVARIANT|TEMPORAL_CHRONO|COGNITIVE_DIALECTIC|MERONYMY|NONE",\n'
            '  "shadow_lambda": 0.0,\n'
            '  "content_raw": "压缩后的抽象事件主文本",\n'
            '  "insight": {"relation": "CAUSALITY", "conclusion": "认知结论"},\n'
            '  "decoration": null\n'
            '}'
            if insight_enabled
            else '{\n'
            '  "cognitive_relation": "CAUSALITY|...|NONE",\n'
            '  "shadow_lambda": 0.0,\n'
            '  "content_raw": "压缩后的抽象事件主文本",\n'
            '  "decoration": null\n'
            '}'
        )

        user_msg = EVOLUTION_USER.format(
            count=len(evidence),
            event_contents=event_contents,
            target_content_len=target_content_len,
            leaf_count=leaf_count,
            leaf_avg_len=leaf_avg_len,
            insight_instruction=insight_instruction,
            json_schema=json_schema,
        )

        system_msg = EVOLUTION_SYSTEM.format(
            target_content_len=target_content_len,
            leaf_count=leaf_count,
            leaf_avg_len=leaf_avg_len,
        )

        data = self._llm.complete_json(
            "abstraction",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        relation = str(data.get("cognitive_relation", "NONE")).upper()
        if relation not in COGNITIVE_RELATIONS:
            relation = "NONE"
        shadow_lambda = float(data.get("shadow_lambda", 0.5) or 0.5)
        shadow_lambda = min(max(shadow_lambda, 0.0), 1.0)

        if relation == "NONE":
            return SynthesisOutcome(None, relation, shadow_lambda, rejected_none=True)

        insight_val = None
        if insight_enabled:
            raw_ins = data.get("insight")
            if isinstance(raw_ins, dict):
                insight_val = f"[{raw_ins.get('relation', relation)}] {raw_ins.get('conclusion', '')}".strip()
            elif raw_ins:
                insight_val = f"[{relation}] {raw_ins}"

        event = Event(
            event_id=generate_event_id(),
            content_raw=data.get("content_raw", ""),
            is_abstract=True,
            status=EventStatus.ACTIVE,
            decoration=data.get("decoration"),
            insight=insight_val,
            abstraction_level=abstraction_level,
            source_events=[e.event_id for e in events],
            role_list=[],
        )
        return SynthesisOutcome(event, relation, shadow_lambda, rejected_none=False)

    # ------------------------------------------------------------------

    def _build_logical_evidence_text(self, events: list[Event]) -> str:
        """Prompt body for coherence gate: logical mining members (basic or abstract)."""

        lines: list[str] = []
        for e in events:
            role_context = self._build_role_context(e)
            label = "抽象事件" if getattr(e, "is_abstract", False) else "基本事件"
            lines.append(
                f"### {label} {e.event_id} (创建于 {e.create_time.isoformat()})\n"
                f"content_raw:\n{e.content_raw}\n"
                f"角色线索:\n{role_context}"
            )
        return "\n\n".join(lines)

    def _build_content_raw_text(self, events: list[Event]) -> str:
        lines: list[str] = []
        for e in events:
            role_context = self._build_role_context(e)
            lines.append(
                f"### 基本事件 {e.event_id} (创建于 {e.create_time.isoformat()})\n"
                f"content_raw:\n{e.content_raw}\n"
                f"角色线索:\n{role_context}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _leaf_evidence_stats(events: list[Event]) -> tuple[int, int, int]:
        """``(leaf_count, avg_content_raw_len_rounded, target_len)`` for prompts.

        ``target_len`` = ``max(1, int(mean(len(content_raw)) * 1.2))``，与原先
        ``_target_content_len`` 一致；``avg`` 取四舍五入整数便于模型读数。
        """
        if not events:
            return 0, 0, 0
        n = len(events)
        total = sum(len(e.content_raw) for e in events)
        avg_f = total / n
        target = max(1, int(avg_f * 1.2))
        avg_rounded = int(round(avg_f))
        return n, avg_rounded, target

    @staticmethod
    def _build_role_context(event: Event) -> str:
        if not event.role_list:
            return "无显式角色线索"
        lines: list[str] = []
        for entry in event.role_list:
            snap = entry.role_snapshot
            fragments = [
                snap.l1_mention,
                snap.l2_interaction,
                snap.l3_decision,
            ]
            summary = " / ".join(f for f in fragments if f)
            importance = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            lines.append(f"- {entry.role_id}（{importance}）：{summary or '无快照'}")
        return "\n".join(lines)
