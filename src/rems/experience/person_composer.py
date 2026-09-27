"""Turn accumulated white paintings into a revisable interaction portrait."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from ..llm.provider import LLMProvider
from ..models.portrait import PendingPortraitSummary
from .models import (
    ContextualAdjustment,
    PersonExperiencePortrait,
    PortraitRevisionKind,
    RevisedClaim,
)


def render_person_experience(
    baseline: str,
    adjustments: tuple[ContextualAdjustment, ...],
    uncertainties: tuple[str, ...],
) -> str:
    """Make the structured stance usable as a short piece of context."""
    parts = [baseline.strip().rstrip("。；")]
    parts.extend(
        f"在{item.context.strip()}时，{item.approach.strip()}".rstrip("。；")
        for item in adjustments
    )
    if uncertainties:
        parts.append("尚不确定：" + "；".join(item.strip() for item in uncertainties))
    return "；".join(item for item in parts if item) + "。"


@dataclass(frozen=True)
class InteractionPortraitDraft:
    baseline: str
    adjustments: tuple[ContextualAdjustment, ...]
    uncertainties: tuple[str, ...]
    brief: str
    revision_kind: PortraitRevisionKind
    revision_reason: str = ""
    revised_claims: tuple[RevisedClaim, ...] = ()

    @property
    def full(self) -> str:
        return render_person_experience(
            self.baseline, self.adjustments, self.uncertainties
        )


@dataclass(frozen=True)
class InsufficientEvidence:
    reason: str = ""


class PersonExperienceComposerPort(Protocol):
    def compose(
        self,
        *,
        object_id: str,
        previous: PersonExperiencePortrait | None,
        pending: tuple[PendingPortraitSummary, ...],
        max_chars: int,
        brief_chars: int,
    ) -> InteractionPortraitDraft | InsufficientEvidence | None: ...


_SYSTEM = """你在维护「匠石与某个人长期相处形成的经验肖像」。
输入包括带来源事件的新增白描和此前的结构化经验肖像。白描是本轮证据。
任务是维护「匠石当前如何理解并与这个人相处」的简短经验状态，
不是总结履历、给人物贴人格标签，也不是把白描润色成人物肖像。

所有成功判断都输出完整新快照：
- baseline：简短、保守、较稳定的关系与相处基调。
- adjustments：有具体适用条件的相处经验，每项含 context（情境）与 approach（相处方式）。
- uncertainties：证据不足、存在反例或适用边界尚不明确的认识。
- brief：几句话的极简长期背景，不复述事件履历，不重复 baseline 和 adjustments。

维护原则：
1. 白描是证据，不要简单改写为肖像内容。
2. 不得猜测无证据支持的动机、人格、情绪或偏好；不得把单次表现说成长期规律。
3. 只有跨事件证据、重复出现的表现，或明确且有长期意义的信息，才适合成为稳定经验。
4. 情境性认识优先写入 adjustments，不要扩大成 baseline。
5. 旧肖像中没有被新证据推翻、修正或证明失效的内容应默认保留；
   不得因为本轮白描没有再次提及就静默删除。
6. 新旧证据冲突时，结合证据数量、情境差异和反例判断；
   不要用最新一次经历直接覆盖长期认识。
7. 描述的是匠石形成的相处经验，不是对人物的绝对评价。

revision_kind 只能是：
- initial：此前没有肖像，本轮首次形成。
- confirm：新证据与旧认识一致，或不足以支持任何变化；返回与旧肖像完全相同的完整快照。
- extend：增加此前未覆盖的情境、边界或相处经验，保留未受挑战的旧认识。
- revise：有足够证据修改旧认识；在 revised_claims 列出修改。
- uncertain：旧认识遇到有意义的反例，但不足以形成新的稳定结论；
  在 revised_claims 标明受挑战的认识，并在 uncertainties 留下边界。
除 initial 外必须说明 revision_reason。
revised_claims 每项格式为
{"previous": "此前认识", "updated": "修正后或更保守的当前表述", "reason": "修改或降低确定性的原因"}。
revise 和 uncertain 必须提供非空 revised_claims；其他类型使用空数组。

仅当此前没有肖像且证据不足以首次形成时返回 {"insufficient_evidence": true}。
已有肖像但本轮材料不足以改变认识时应返回 confirm 和完整旧快照。
其他情况只输出合法 JSON，不输出解释文字，字段为：
{"insufficient_evidence": false, "revision_kind": "initial | confirm | extend | revise | uncertain",
 "revision_reason": "本轮判断原因", "revised_claims": [], "baseline": "关系基调",
 "adjustments": [{"context": "具体情境", "approach": "合适的相处方式"}],
 "uncertainties": [], "brief": "极简长期背景"}。
"""


class LLMPersonExperienceComposer:
    """Model boundary; invalid or unsupported decisions leave evidence pending."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def compose(
        self,
        *,
        object_id: str,
        previous: PersonExperiencePortrait | None,
        pending: tuple[PendingPortraitSummary, ...],
        max_chars: int,
        brief_chars: int,
    ) -> InteractionPortraitDraft | InsufficientEvidence | None:
        old_shape = None if previous is None else {
            "baseline": previous.baseline,
            "adjustments": [
                {"context": item.context, "approach": item.approach}
                for item in previous.adjustments
            ],
            "uncertainties": list(previous.uncertainties),
            "brief": previous.brief,
        }
        evidence = [
            {"event_id": item.source_event_id, "white_painting": item.text}
            for item in pending
        ]
        data = self._llm.complete_json("experience_portrait", [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": (
                f"object_id: {object_id}\n"
                f"已有肖像：{json.dumps(old_shape, ensure_ascii=False)}\n"
                f"新增白描：{json.dumps(evidence, ensure_ascii=False)}\n"
                f"完整渲染最多 {max_chars} 字；brief 最多 {brief_chars} 字。"
            )},
        ])
        if data.get("insufficient_evidence") is True:
            if previous is not None:
                return None
            return InsufficientEvidence(str(data.get("reason") or "").strip())
        kind = data.get("revision_kind")
        if kind not in {"initial", "confirm", "extend", "revise", "uncertain"}:
            return None
        reason = str(data.get("revision_reason") or "").strip()
        if previous is None and kind != "initial":
            return None
        if previous is not None and kind == "initial":
            return None
        if kind != "initial" and not reason:
            return None
        baseline = data.get("baseline")
        brief = data.get("brief")
        adjustments_raw = data.get("adjustments")
        uncertainties_raw = data.get("uncertainties")
        revised_raw = data.get("revised_claims")
        if (
            not isinstance(baseline, str) or not baseline.strip()
            or not isinstance(brief, str) or not brief.strip()
            or not isinstance(adjustments_raw, list)
            or not isinstance(uncertainties_raw, list)
            or not isinstance(revised_raw, list)
        ):
            return None
        adjustments: list[ContextualAdjustment] = []
        for item in adjustments_raw:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("context"), str)
                or not isinstance(item.get("approach"), str)
                or not item["context"].strip()
                or not item["approach"].strip()
            ):
                return None
            adjustments.append(ContextualAdjustment(
                context=item["context"].strip(),
                approach=item["approach"].strip(),
            ))
        if any(not isinstance(item, str) or not item.strip() for item in uncertainties_raw):
            return None
        revised_claims: list[RevisedClaim] = []
        for item in revised_raw:
            if (
                not isinstance(item, dict)
                or any(
                    not isinstance(item.get(key), str) or not item[key].strip()
                    for key in ("previous", "updated", "reason")
                )
            ):
                return None
            revised_claims.append(RevisedClaim(
                previous=item["previous"].strip(),
                updated=item["updated"].strip(),
                reason=item["reason"].strip(),
            ))
        draft = InteractionPortraitDraft(
            baseline=baseline.strip(),
            adjustments=tuple(adjustments),
            uncertainties=tuple(item.strip() for item in uncertainties_raw),
            brief=brief.strip(),
            revision_kind=kind,
            revision_reason=reason,
            revised_claims=tuple(revised_claims),
        )
        if (
            len(draft.full) > max_chars or len(draft.brief) > brief_chars
            or draft.full in {item.text.strip() for item in pending}
            or draft.brief in {item.text.strip() for item in pending}
        ):
            return None
        if kind in {"revise", "uncertain"} and not draft.revised_claims:
            return None
        if kind in {"initial", "confirm", "extend"} and draft.revised_claims:
            return None
        if kind == "confirm" and previous is not None and (
            draft.baseline != previous.baseline
            or draft.adjustments != previous.adjustments
            or draft.uncertainties != previous.uncertainties
            or draft.brief != previous.brief
        ):
            return None
        if kind == "uncertain" and not draft.uncertainties:
            return None
        return draft
