"""80/20 forced-split: evaluator, dedicated splitter skill, remediator.

模块内三个角色：

1. ``BoundaryForceThresholdEvaluator`` —— 规则型评估器，零 LLM 成本。
   检查 ``BoundaryResult``：
     - ``new_unclosed`` 中是否有单条超过 ``len_msg × unclosed_force_ratio`` 的未完成片段（``oversized_uc``）；
     - 所有 ``split_id`` 在 ``completed_events`` 与 ``new_unclosed`` 两侧是否一一配对（``broken_split_pair``）；
     - 分裂前缀的相对长度是否落在可接受区间（``split_ratio_off``，告警级别）。

2. ``OverlongUCSplitSkill`` —— 只负责"把一段过长文本按逻辑闭环切成两段"的单点 LLM 调用。
   与 ``BoundaryDetectionSkill`` 分离，一是避免污染主 prompt；二是让 remediation 的输入/输出
   非常窄，便于做一层"最小修复"。

3. ``OverlongUCSplitRemediator`` —— 只修复 ``oversized_uc`` / ``broken_split_pair``：
   - 对每一条 oversized UC 独立调用一次 ``OverlongUCSplitSkill``；
   - 成功：把原条从 ``new_unclosed`` 替换为"前缀 completed + 尾部 UC"一对；
   - 失败：保留原条，但把 ``split_id`` 以及任何残留的"孤儿前缀"做一次清理；在 UC 侧
     由 ``MetabolismService`` 把条目落盘为 ``oversized=True``（审计），**不再** force-seal。
   **评估只做一层**：修复后的输出不再进入评估器，只要成功就直接采用。
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Optional

from ..config import REMSConfig
from ..llm.prompts import OVERLONG_UC_SPLIT_SYSTEM, OVERLONG_UC_SPLIT_USER
from ..llm.provider import LLMProvider
from ..models.metabolism import UnclosedEvent
from .boundary_detection import BoundaryResult, CompletedFragment, NewUnclosed
from .evaluation import EvalIssue, EvalReport, SkillEvaluator, SkillRemediator

logger = logging.getLogger(__name__)


# =====================================================================
# Input context passed to evaluator / remediator
# =====================================================================

@dataclass(frozen=True)
class BoundarySkillContext:
    """输入侧上下文：由 ``MetabolismService`` 在调用 skill 前构造。

    评估器与修复器借此拿到：
        - ``force_threshold`` 具体数值（由 config 计算出）；
        - 当前持有的 ``UnclosedEvent`` 列表（用于在修复时把某条 UC 的拆分结果回写到正确的原 UC）。
    """

    force_threshold: int
    split_ratio_target: float
    split_ratio_min: float
    split_ratio_max: float
    unclosed_events: tuple[UnclosedEvent, ...] = ()


# =====================================================================
# Evaluator
# =====================================================================

class BoundaryForceThresholdEvaluator(SkillEvaluator):
    """Rule-based evaluator: oversized_uc + broken_split_pair + split_ratio_off."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def evaluate(self, skill_input: Any, skill_output: Any) -> EvalReport:
        if not isinstance(skill_output, BoundaryResult):
            return EvalReport.success()

        ctx: Optional[BoundarySkillContext] = (
            skill_input if isinstance(skill_input, BoundarySkillContext) else None
        )
        if ctx is None:
            # 无上下文无法检查；保持保守通过。
            return EvalReport.success()

        issues: list[EvalIssue] = []

        # 1. oversized_uc：任一 new_unclosed 越过阈值 且**没有**声明 split_id。
        for idx, nu in enumerate(skill_output.new_unclosed):
            if len(nu.content) <= ctx.force_threshold:
                continue
            if nu.split_id:
                # 已配对的尾段允许略长，但尾段绝不能单独越过阈值（否则分裂无效）。
                issues.append(EvalIssue(
                    code="split_tail_still_oversized",
                    severity="error",
                    detail=(
                        f"tail UC #{idx} (split_id={nu.split_id}) length={len(nu.content)} "
                        f"exceeds force_threshold={ctx.force_threshold}"
                    ),
                    ctx={"new_unclosed_index": idx, "split_id": nu.split_id},
                ))
                continue
            issues.append(EvalIssue(
                code="oversized_uc",
                severity="error",
                detail=(
                    f"new_unclosed #{idx} length={len(nu.content)} > force_threshold="
                    f"{ctx.force_threshold} and was not split"
                ),
                ctx={"new_unclosed_index": idx},
            ))

        # 2. broken_split_pair：两侧 split_id 集合是否完全一致（顺序无关）。
        prefix_ids = {f.split_id for f in skill_output.completed_events if f.is_split_prefix and f.split_id}
        tail_ids = {nu.split_id for nu in skill_output.new_unclosed if nu.split_id}
        orphan_prefix = prefix_ids - tail_ids
        orphan_tail = tail_ids - prefix_ids
        for sid in orphan_prefix:
            issues.append(EvalIssue(
                code="broken_split_pair",
                severity="error",
                detail=f"split_id={sid} appears on a completed prefix but no tail UC references it",
                ctx={"split_id": sid, "side": "prefix_without_tail"},
            ))
        for sid in orphan_tail:
            issues.append(EvalIssue(
                code="broken_split_pair",
                severity="error",
                detail=f"split_id={sid} appears on a tail UC but no completed prefix references it",
                ctx={"split_id": sid, "side": "tail_without_prefix"},
            ))

        # 3. split_ratio_off：前缀 / (前缀 + 尾段) 落在区间外。告警级别（不触发修复）。
        for sid in prefix_ids & tail_ids:
            prefix = next(
                (f for f in skill_output.completed_events if f.is_split_prefix and f.split_id == sid),
                None,
            )
            tail = next(
                (nu for nu in skill_output.new_unclosed if nu.split_id == sid),
                None,
            )
            if prefix is None or tail is None:
                continue
            total = len(prefix.content_raw) + len(tail.content)
            if total <= 0:
                continue
            ratio = len(prefix.content_raw) / total
            if ratio < ctx.split_ratio_min or ratio > ctx.split_ratio_max:
                issues.append(EvalIssue(
                    code="split_ratio_off",
                    severity="warn",
                    detail=(
                        f"split_id={sid} prefix_ratio={ratio:.2f} outside "
                        f"[{ctx.split_ratio_min:.2f}, {ctx.split_ratio_max:.2f}]"
                    ),
                    ctx={"split_id": sid, "ratio": ratio},
                ))

        if not issues:
            return EvalReport.success()
        has_error = any(i.severity == "error" for i in issues)
        return EvalReport(ok=not has_error, issues=tuple(issues))


# =====================================================================
# OverlongUCSplitSkill —— 单点 LLM 分裂
# =====================================================================

@dataclass(frozen=True)
class SplitOutcome:
    prefix_text: str
    tail_text: str
    aborted: bool = False


class OverlongUCSplitSkill:
    """Split a single overlong piece of text at the nearest logical closure point."""

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def split(self, content: str) -> SplitOutcome:
        if not content:
            return SplitOutcome("", "", aborted=True)

        cfg = self._config
        user_msg = OVERLONG_UC_SPLIT_USER.format(
            content=content,
            target_ratio=cfg.boundary_split_ratio_target,
            ratio_lo=cfg.boundary_split_ratio_min,
            ratio_hi=cfg.boundary_split_ratio_max,
        )
        try:
            data = self._llm.complete_json(
                "overlong_uc_split",
                [
                    {"role": "system", "content": OVERLONG_UC_SPLIT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OverlongUCSplitSkill LLM call failed: %s", exc)
            return SplitOutcome("", "", aborted=True)

        if data.get("abort") is True:
            return SplitOutcome("", "", aborted=True)

        prefix = (data.get("prefix_text") or "").strip()
        tail = (data.get("tail_text") or "").strip()

        if not prefix or not tail:
            return SplitOutcome("", "", aborted=True)

        # 字符等价校验：允许首尾空白差异，但不允许内容增减（防止 LLM 改写）。
        if self._canonical(prefix + tail) != self._canonical(content):
            # 小概率：模型把分隔符吞了或重复了一下。做一次宽容处理：用 prefix 在原文里定位切点。
            joined = prefix + tail
            if self._canonical(joined).replace("\n", "") != self._canonical(content).replace("\n", ""):
                logger.warning(
                    "OverlongUCSplitSkill content mismatch; aborting remediation. "
                    "prefix_len=%d tail_len=%d orig_len=%d",
                    len(prefix), len(tail), len(content),
                )
                return SplitOutcome("", "", aborted=True)

        # 比例硬防线（区间外直接放弃；不再二次评估）。
        total = len(prefix) + len(tail)
        ratio = len(prefix) / total if total else 0.0
        if ratio < cfg.boundary_split_ratio_min - 0.05 or ratio > cfg.boundary_split_ratio_max + 0.05:
            logger.info(
                "OverlongUCSplitSkill returned off-ratio split (%.2f); aborting remediation",
                ratio,
            )
            return SplitOutcome("", "", aborted=True)

        return SplitOutcome(prefix_text=prefix, tail_text=tail, aborted=False)

    @staticmethod
    def _canonical(s: str) -> str:
        """Normalize whitespace for equivalence check."""
        return "".join(s.split())


# =====================================================================
# Remediator
# =====================================================================

class OverlongUCSplitRemediator(SkillRemediator):
    """Fix ``oversized_uc`` / ``broken_split_pair`` by calling ``OverlongUCSplitSkill`` once per offender.

    修复策略：
        - 评估报告里标出的每一条 ``oversized_uc``（按 ``new_unclosed_index`` 定位）：
          独立调用一次 ``OverlongUCSplitSkill``；
            * 成功 → 在 ``new_unclosed`` 中把原条**替换**为新 tail，同时在
              ``completed_events`` 中**追加**一条 ``is_split_prefix=True`` 的前缀；
              两侧共享一个新生成的 ``split_id``；
            * 失败 → 保留原条不动（调用方应把它落库为 ``oversized=True``）。
        - ``broken_split_pair`` 的单侧孤儿：
            * ``prefix_without_tail`` → 把该 completed 片段**去分裂化**（清空 split_id /
              is_split_prefix），它就是一条正常的 completed_event；
            * ``tail_without_prefix`` → 清空该 UC 的 ``split_id``；评估器仍会把它
              重新归类到 ``oversized_uc`` 路径（如果它仍然过长），再进入上面的分裂修复。
        - 评估只做一层，所以这里修复后的结果**不再评估**，直接返回。
    """

    def __init__(self, split_skill: OverlongUCSplitSkill):
        self._split = split_skill

    def remediate(
        self,
        skill_input: Any,
        skill_output: Any,
        report: EvalReport,
    ) -> Optional[Any]:
        if not isinstance(skill_output, BoundaryResult):
            return None

        # 深拷贝（Pydantic v2: model_copy）以便安全改写。
        result = skill_output.model_copy(deep=True)

        # 1. 清理孤儿前缀：改写对应 completed_events 让其变为普通 completed。
        orphan_prefix_ids = {
            i.ctx.get("split_id") for i in report.issues
            if i.code == "broken_split_pair" and i.ctx.get("side") == "prefix_without_tail"
        }
        if orphan_prefix_ids:
            for f in result.completed_events:
                if f.split_id in orphan_prefix_ids:
                    f.is_split_prefix = False
                    f.split_id = None

        # 2. 清理孤儿尾段：仅清空 split_id，内容保留。
        orphan_tail_ids = {
            i.ctx.get("split_id") for i in report.issues
            if i.code == "broken_split_pair" and i.ctx.get("side") == "tail_without_prefix"
        }
        if orphan_tail_ids:
            for nu in result.new_unclosed:
                if nu.split_id in orphan_tail_ids:
                    nu.split_id = None

        # 3. 修复 oversized_uc / split_tail_still_oversized —— 逐条分裂。
        oversize_indices: set[int] = set()
        for i in report.issues:
            if i.code in ("oversized_uc", "split_tail_still_oversized"):
                idx = i.ctx.get("new_unclosed_index")
                if isinstance(idx, int):
                    oversize_indices.add(idx)

        if not oversize_indices:
            return result

        # 注意：修复期间 new_unclosed 列表要"替换+追加 completed"，
        # 为了避免索引错乱，采用"按原序重建"策略。
        new_new_unclosed: list[NewUnclosed] = []
        for idx, nu in enumerate(result.new_unclosed):
            if idx not in oversize_indices:
                new_new_unclosed.append(nu)
                continue
            outcome = self._split.split(nu.content)
            if outcome.aborted or not outcome.prefix_text or not outcome.tail_text:
                logger.info(
                    "OverlongUCSplitRemediator gave up on new_unclosed #%d (len=%d); "
                    "keeping original oversized UC.",
                    idx, len(nu.content),
                )
                new_new_unclosed.append(nu)
                continue

            split_id = f"SP-{secrets.token_hex(3)}"
            result.completed_events.append(CompletedFragment(
                content_raw=outcome.prefix_text,
                continuation_of=None,
                is_split_prefix=True,
                split_id=split_id,
            ))
            new_new_unclosed.append(NewUnclosed(
                content=outcome.tail_text,
                logical_gaps=nu.logical_gaps,
                split_id=split_id,
            ))

        result.new_unclosed = new_new_unclosed
        return result
