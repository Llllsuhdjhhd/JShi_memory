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
    for item in adjustments:
        context = item.context.strip().rstrip("。；，")
        approach = item.approach.strip().rstrip("。；")
        if not context or not approach:
            continue
        if context.startswith("在"):
            context = context[1:]
        condition = f"在{context}" if context.endswith(("时", "时候")) else f"在{context}时"
        parts.append(f"{condition}，{approach}")
    if uncertainties:
        parts.append(
            "尚不确定："
            + "；".join(item.strip().rstrip("。；") for item in uncertainties)
        )
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
8. baseline、adjustments、uncertainties 和 brief 统一用第三人称：称互动对象为「对方」或其名字，称主体为「匠石」。不要输出没有引号的「我」「你」，以免把对象的话误写成匠石的要求。
9. 如果对象明确限定某偏好的适用场景，严格保留这个条件；不同场景的做法分开写。明确给出的边界不要再写成「尚不确定」，但可以保留真正缺乏跨事件证据的部分。
10. 互斥场景必须拆成不同 adjustments，禁止把两种做法合进一项。例如「操作时要三条短步骤，倾诉时不要清单」应分别写成「值班操作→只给三条短步骤，不额外扩写」和「倾诉心情→先听，不列步骤或清单」。操作项不得包含「先讲心情」「不塞清单」；倾诉项不得包含操作步骤。

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
                f"完整渲染目标约 {max_chars} 字，brief 目标约 {brief_chars} 字；"
                "有依据的情境需要时可以超出，不要为压字数删去仍然有效的认识。"
            )},
        ])
        if data.get("insufficient_evidence") is True:
            if previous is None:
                return InsufficientEvidence(str(data.get("reason") or "").strip())
            # 已有肖像时「证据不足」就是本批不改：原样确认，白描算已吸收。
            return InteractionPortraitDraft(
                baseline=previous.baseline,
                adjustments=previous.adjustments,
                uncertainties=previous.uncertainties,
                brief=previous.brief,
                revision_kind="confirm",
                revision_reason=str(data.get("reason") or "").strip(),
            )
        # 类别由调用方按新旧内容的实际差异重定，这里只做格式校验。
        kind = data.get("revision_kind")
        if kind not in {"initial", "confirm", "extend", "revise", "uncertain"}:
            kind = "initial" if previous is None else "revise"
        reason = str(data.get("revision_reason") or "").strip()
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
        ):
            return None
        if not isinstance(revised_raw, list):
            revised_raw = []
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
                continue
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
            draft.full in {item.text.strip() for item in pending}
            or draft.brief in {item.text.strip() for item in pending}
        ):
            return None
        return draft
