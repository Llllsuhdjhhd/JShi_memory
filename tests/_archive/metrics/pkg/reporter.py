"""Markdown report builder for white-paper live metric suites."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rems.config import REMSConfig

from .probes import ProbeResult

_STATS_COMMENT_RE = re.compile(
    r"<!--\s*wp-live-stats:\s*(\{.*?\})\s*-->",
    re.DOTALL,
)


def report_dir_from_env() -> Path:
    import os

    raw = os.environ.get("REMS_WP_REPORT_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parents[1] / "reports"


def default_report_path(filename: str) -> Path:
    """Resolved path under ``REMS_WP_REPORT_DIR`` or ``tests/reports``."""
    return report_dir_from_env() / filename


def _escape_md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _fmt_actual(actual: Any) -> str:
    if isinstance(actual, (dict, list)):
        try:
            return json.dumps(actual, ensure_ascii=False)[:200]
        except TypeError:
            return str(actual)[:200]
    return str(actual)[:200]


@dataclass
class _SectionDraft:
    clause: str
    title: str
    parts: list[str] = field(default_factory=list)


class SectionCtx:
    """Per-section Markdown accumulator (used as context manager)."""

    def __init__(self, reporter: "WPReporter", clause: str, title: str) -> None:
        self._r = reporter
        self._draft = _SectionDraft(clause=clause, title=title)

    def quote(self, text: str) -> None:
        lines = ["> " + ln if ln.strip() else ">" for ln in text.strip().splitlines()]
        self._draft.parts.append("### 白皮书原文\n\n" + "\n".join(lines) + "\n")

    def method(self, lines: str | list[str]) -> None:
        if isinstance(lines, str):
            bullets = [lines]
        else:
            bullets = list(lines)
        body = "\n".join(f"- {b}" for b in bullets)
        self._draft.parts.append("### 检查方法\n\n" + body + "\n")

    def note(self, text: str) -> None:
        self._draft.parts.append("### 备注\n\n" + text.strip() + "\n")

    def json_block(self, label: str, obj: Any, *, max_chars: int = 4000) -> None:
        try:
            raw = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            raw = str(obj)
        if len(raw) > max_chars:
            raw = raw[: max_chars - 3] + "..."
        self._draft.parts.append(f"### {label}\n\n```json\n{raw}\n```\n")

    def probe(self, pr: ProbeResult) -> None:
        self._r.add_probe(self._draft.clause, pr)
        tag = _probe_status_tag(pr.ok)
        detail = ""
        if pr.detail is not None:
            try:
                djson = json.dumps(pr.detail, ensure_ascii=False, indent=2, default=str)[:1500]
            except TypeError:
                djson = str(pr.detail)[:1500]
            detail = f"\n\n```\n{djson}\n```\n"
        self._draft.parts.append(
            f"### 现场数据 · {pr.name} {tag}\n\n"
            f"- **期望**：{pr.expected}\n"
            f"- **实测**：`{_fmt_actual(pr.actual)}`\n"
            f"{detail}"
        )

    def raw_md(self, markdown: str) -> None:
        self._draft.parts.append(markdown.rstrip() + "\n")

    def __enter__(self) -> SectionCtx:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._r._flush_section(self._draft)


def _probe_status_tag(ok: Optional[bool]) -> str:
    if ok is True:
        return "**`PASS`**"
    if ok is False:
        return "**`FAIL`**"
    return "**`INFO`**（无硬判定）"


class WPReporter:
    """Accumulates probes + section bodies and writes a Markdown report + index."""

    def __init__(self, report_path: Path, title: str) -> None:
        self.report_path = Path(report_path).resolve()
        self.title = title
        self._cfg_snapshot: dict[str, Any] | None = None
        self._llm_env_lines: list[str] = []
        self._sections: list[_SectionDraft] = []
        self._probes: list[tuple[str, ProbeResult]] = []
        self._run_summary_lines: list[str] = []
        self._repro_lines: list[str] = []
        self._table_rows: list[tuple[str, str, str, str]] = []

    def set_config_snapshot(self, cfg: REMSConfig) -> None:
        """Serialize a compact, human-readable config view."""
        self._cfg_snapshot = {
            "context_window": cfg.context_window,
            "chars_per_token": cfg.chars_per_token,
            "len_msg": cfg.len_msg,
            "physical_redline": cfg.physical_redline,
            "safe_watermark": cfg.safe_watermark,
            "user_mode": cfg.user_mode.value,
            "core_user_role_id": cfg.core_user_role_id,
            "active_participants": list(cfg.active_participants),
            "recall_cluster_threshold": cfg.recall_cluster_threshold,
            "summary_fuse_min_chars": cfg.summary_fuse_min_chars,
            "ae_high_threshold": cfg.ae_high_threshold,
            "ae_score_weight": cfg.ae_score_weight,
            "activation_energy_gain": cfg.activation_energy_gain,
            "activation_energy_weight": cfg.activation_energy_weight,
            "ema_smoothing_alpha": cfg.ema_smoothing_alpha,
            "wp_primary_field": cfg.wp_primary_field,
            "wp_default_field": cfg.wp_default_field,
            "wp_half_life_days": cfg.wp_half_life_days,
            "ae_forgetting_multiplier": cfg.ae_forgetting_multiplier,
            "semantic_card_max_keys": cfg.semantic_card_max_keys,
            "hallucination_anchor_prob": cfg.hallucination_anchor_prob,
        }

    def set_llm_env_summary(self, lines: list[str]) -> None:
        self._llm_env_lines = lines

    def add_probe(self, clause: str, pr: ProbeResult) -> None:
        self._probes.append((clause, pr))
        status = "PASS" if pr.ok is True else "FAIL" if pr.ok is False else "INFO"
        self._table_rows.append(
            (pr.name, status, _escape_md_cell(_fmt_actual(pr.actual)), clause),
        )

    def section(self, clause: str, title: str) -> SectionCtx:
        return SectionCtx(self, clause, title)

    def add_run_summary(self, lines: str | list[str]) -> None:
        if isinstance(lines, str):
            self._run_summary_lines.append(lines)
        else:
            self._run_summary_lines.extend(lines)

    def set_repro_footer(
        self,
        *,
        pytest_cmd: str,
        extra_env_keys: list[str] | None = None,
    ) -> None:
        keys = ["REMS_RUN_LIVE_METABOLISM_TEST", "REMS_WP_REPORT_DIR", "REMS_LIVE_ALLOW_FAKE_EMBEDDING"]
        if extra_env_keys:
            keys.extend(extra_env_keys)
        lines = [
            f"- **生成时间**：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Python**：`{sys.executable}`",
            f"- **pytest 命令**：`{pytest_cmd}`",
            "- **相关环境变量键**（值不落盘）：" + "、".join(f"`{k}`" for k in sorted(set(keys))),
        ]
        self._repro_lines = lines

    def _flush_section(self, draft: _SectionDraft) -> None:
        self._sections.append(draft)

    def _render(self) -> str:
        parts: list[str] = [f"# WP-LIVE · {self.title}\n"]
        parts.append(f"- **报告文件**：`{self.report_path.name}`\n")

        if self._cfg_snapshot:
            parts.append("\n## 配置快照\n\n```json\n")
            parts.append(json.dumps(self._cfg_snapshot, ensure_ascii=False, indent=2))
            parts.append("\n```\n")

        if self._llm_env_lines:
            parts.append("\n## LLM / 嵌入环境\n\n")
            parts.extend(f"- {ln}\n" for ln in self._llm_env_lines)

        if self._table_rows:
            parts.append("\n## 指标核对表\n\n")
            parts.append("| 指标 | 状态 | 实测摘要 | 章节 |\n")
            parts.append("| --- | --- | --- | --- |\n")
            for name, st, act, cl in self._table_rows:
                parts.append(f"| {name} | **{st}** | {act} | {cl} |\n")

        for sec in self._sections:
            parts.append(f"\n## {sec.clause} {sec.title}\n\n")
            parts.extend(sec.parts)

        if self._run_summary_lines:
            parts.append("\n## 运行摘要\n\n")
            parts.extend(ln if ln.endswith("\n") else ln + "\n" for ln in self._run_summary_lines)

        parts.append("\n## 环境 & 复现命令\n\n")
        parts.extend(ln + "\n" for ln in self._repro_lines)

        p, f, n = self._count_pf()
        stats = {"report": self.report_path.name, "pass": p, "fail": f, "na": n}
        parts.append(f"\n<!-- wp-live-stats: {json.dumps(stats, ensure_ascii=False)} -->\n")
        return "".join(parts)

    def _count_pf(self) -> tuple[int, int, int]:
        p = f = n = 0
        for _, pr in self._probes:
            if pr.ok is True:
                p += 1
            elif pr.ok is False:
                f += 1
            else:
                n += 1
        return p, f, n

    def finalize(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(self._render(), encoding="utf-8")
        _rewrite_index(self.report_path.parent)


def _rewrite_index(report_dir: Path) -> None:
    report_dir = Path(report_dir).resolve()
    index_path = report_dir / "wp_live_INDEX.md"
    rows: list[tuple[str, int, int, int, str]] = []
    for path in sorted(report_dir.glob("wp_live_*.md")):
        if path.name == "wp_live_INDEX.md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        m = _STATS_COMMENT_RE.search(text)
        if m:
            try:
                stats = json.loads(m.group(1))
            except json.JSONDecodeError:
                stats = {}
        else:
            stats = {}
        p = int(stats.get("pass", 0))
        f = int(stats.get("fail", 0))
        n = int(stats.get("na", 0))
        overall = "PASS" if f == 0 else "FAIL"
        rows.append((path.name, p, f, n, overall))

    lines = [
        "# WP-LIVE 报告总入口\n",
        f"\n> 自动生成于 `{report_dir}`。每项为最近一次 `finalize()` 写入的统计（HTML 注释块）。\n\n",
        "| 报告 | PASS | FAIL | INFO | 总判 |\n",
        "| --- | ---: | ---: | ---: | --- |\n",
    ]
    for name, p, f, n, overall in rows:
        lines.append(f"| [`{name}`](./{name}) | {p} | {f} | {n} | **{overall}** |\n")
    if not rows:
        lines.append("| （尚无子报告） | — | — | — | — |\n")

    index_path.write_text("".join(lines), encoding="utf-8")
