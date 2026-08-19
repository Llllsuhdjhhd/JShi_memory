"""Generic skill-output evaluation & remediation protocol.

作用域：每一个 LLM 技能（BoundaryDetection / EventEnrichment / InductiveEvolution / ...）
在调用后都应经过 "评估 → 必要时修复" 一道；评估只做一层，**不递归评估修复后的结果**
（防止"评估器无穷嵌套"），这是本模块的硬性设计约束。

典型用法::

    raw_result = boundary_skill.detect(shadow, input, uc)
    final_result, report = run_skill_with_eval(
        lambda: raw_result,
        evaluators=[BoundaryForceThresholdEvaluator(config)],
        remediator=OverlongUCSplitRemediator(llm, config),
        skill_input={"shadow": shadow, "input": input, "uc": uc},
        config=config,
    )

评估 / 修复的角色分离：
    - ``SkillEvaluator``  —— 对 ``(skill_input, skill_output)`` 给出 ``EvalReport``；
      可以是规则检查（零 LLM 成本，始终开启），也可以是另一个 LLM 调用（默认关闭）。
    - ``SkillRemediator`` —— 仅在评估失败时触发；返回修复后的新 ``skill_output``
      （类型与原始输出同构），或 ``None`` 表示放弃修复、返回原输出。

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

TOut = TypeVar("TOut")


# =====================================================================
# Report objects
# =====================================================================

@dataclass(frozen=True)
class EvalIssue:
    """单条评估问题。``code`` 机器可读，``detail`` 人类可读；``severity`` 区分告警 / 致命。

    常见 ``code`` 取值（各 skill 自行扩展）：
        - ``oversized_uc``        未闭环片段超过 force_threshold
        - ``broken_split_pair``   LLM 返回的分裂对配对不完整
        - ``split_ratio_off``     分裂点显著偏离期望区间
        - ``empty_roles``         事件充实未能抽到任何角色
    """

    code: str
    severity: str  # "warn" | "error"
    detail: str
    ctx: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    """评估报告聚合对象。``ok=True`` 表示无需修复；否则 ``issues`` 非空。"""

    ok: bool
    issues: tuple[EvalIssue, ...] = ()

    @classmethod
    def success(cls) -> "EvalReport":
        return cls(ok=True, issues=())

    @classmethod
    def failure(cls, issues: list[EvalIssue]) -> "EvalReport":
        return cls(ok=False, issues=tuple(issues))

    def merge(self, other: "EvalReport") -> "EvalReport":
        merged_issues = tuple(self.issues) + tuple(other.issues)
        return EvalReport(
            ok=self.ok and other.ok,
            issues=merged_issues,
        )

    def has_code(self, code: str) -> bool:
        return any(i.code == code for i in self.issues)


# =====================================================================
# Protocols
# =====================================================================

@runtime_checkable
class SkillEvaluator(Protocol):
    """Evaluate a single skill output against validation rules."""

    def evaluate(self, skill_input: Any, skill_output: Any) -> EvalReport: ...


@runtime_checkable
class SkillRemediator(Protocol):
    """Fix a failed skill output. Returns a *new* output or None to give up."""

    def remediate(self, skill_input: Any, skill_output: Any, report: EvalReport) -> Optional[Any]: ...


# =====================================================================
# Runner
# =====================================================================

def run_skill_with_eval(
    skill_call: Callable[[], TOut],
    *,
    evaluators: list[SkillEvaluator] | None = None,
    remediator: Optional[SkillRemediator] = None,
    skill_input: Any = None,
) -> tuple[TOut, EvalReport]:
    """Run a skill, evaluate its output, optionally remediate once.

    语义约束：
        1. 评估器以 **严格合并** 聚合（任一失败 → 整体失败）。
        2. 修复只做一次；修复后的输出 **不再评估**（防递归）。
        3. 修复器返回 ``None`` 时回退原始输出，同时保留失败报告供调用方决策。
        4. 修复器未注册 / 报告为 ok 时，直接返回原始输出。

    调用方应检查返回的 ``EvalReport``：
        - ``ok=True``：正常链路；
        - ``ok=False`` 且输出已被修复：走 happy path，但可记 metric；
        - ``ok=False`` 且修复失败：走 "降级保留" 策略（例如在代谢层保留 oversized UC、
          不再强制封存），由调用方决定。
    """
    out = skill_call()

    if not evaluators:
        return out, EvalReport.success()

    report = EvalReport.success()
    for ev in evaluators:
        try:
            report = report.merge(ev.evaluate(skill_input, out))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluator %s raised: %s", type(ev).__name__, exc, exc_info=True)
            report = report.merge(EvalReport.failure([
                EvalIssue(
                    code="evaluator_error",
                    severity="warn",
                    detail=f"{type(ev).__name__}: {exc}",
                )
            ]))

    if report.ok or remediator is None:
        return out, report

    try:
        repaired = remediator.remediate(skill_input, out, report)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Remediator %s raised: %s", type(remediator).__name__, exc, exc_info=True)
        repaired = None

    if repaired is None:
        logger.info(
            "Skill remediation gave up; returning original output with %d issue(s): %s",
            len(report.issues),
            [i.code for i in report.issues],
        )
        return out, report

    logger.info(
        "Skill remediation applied; %d issue(s) were targeted: %s",
        len(report.issues),
        [i.code for i in report.issues],
    )
    return repaired, report
