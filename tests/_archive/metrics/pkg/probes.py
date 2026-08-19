"""Indicator probes mapped to specific clauses of the REMS white paper.

每个函数返回一个 ``ProbeResult`` 结构，由 :class:`WPReporter` 渲染为指标核对表与章节段落。
约定：
- ``name``: 指标简称（将作为表格首列）。
- ``expected``: 期望值或阈值的**人类可读**描述。
- ``actual``: 实测值（数字、字符串、短字典均可）。
- ``ok``: 是否通过。``None`` 表示"陈列指标但不做 pass/fail 判定"（如 §4.2 模糊截断实测比例）。
- ``detail``: 可选的补充信息（多行字符串或字典）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from rems.config import REMSConfig, UserMode
from rems.models.event import Event
from rems.models.metabolism import RecallBlock
from rems.models.role import WhitePaintingEntry


@dataclass
class ProbeResult:
    name: str
    expected: str
    actual: Any
    ok: Optional[bool] = None  # None == 陈列指标，不做判定
    detail: Any = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# §1 Basic Event indicators
# =====================================================================


def event_id_monotonic(events: Iterable[Event]) -> ProbeResult:
    """§1.1.1 — ``event_id`` 前缀 ``EVT-`` + 时间戳段应使 ID 字典序大致随时间递增。"""
    events = list(events)
    ids = [e.event_id for e in events if e.event_id.startswith("EVT-")]
    sorted_ids = sorted(ids)
    ok = ids == sorted_ids
    return ProbeResult(
        name="§1.1.1 event_id 字典序递增",
        expected="封存顺序下 event_id 列表整体呈升序",
        actual={"count": len(ids), "first": ids[:3], "last": ids[-3:]},
        ok=ok,
        detail=None if ok else {"observed": ids, "sorted": sorted_ids},
    )


def event_length_vs_budget(event: Event, cfg: REMSConfig) -> ProbeResult:
    """§1.1.7 — 单事件 ``event_length`` 必须 ≤ ``len_msg``。"""
    ok = event.event_length <= cfg.len_msg
    return ProbeResult(
        name="§1.1.7 event_length ≤ len_msg",
        expected=f"≤ {cfg.len_msg}",
        actual=event.event_length,
        ok=ok,
        detail={"event_id": event.event_id},
    )


def summary_fuse_state(event: Event, cfg: REMSConfig) -> ProbeResult:
    """§1.1.3 — 递归摘要应在字数不足 ``summary_fuse_min_chars`` 时熔断。"""
    lengths = dict(event.summary_lengths)
    levels = sorted(lengths.keys(), key=lambda k: int(k[1:]))
    tail_len = lengths[levels[-1]] if levels else 0
    beyond_tail = event.actual_max_level > len(levels) if levels else False

    # 合规：要么熔断生效（尾部长度 < 阈值或达到 5 层上限），要么仍未达任何摘要
    ok: Optional[bool]
    if not levels:
        ok = None
    elif tail_len < cfg.summary_fuse_min_chars or len(levels) >= 5:
        ok = True
    else:
        ok = False

    return ProbeResult(
        name="§1.1.3 摘要熔断",
        expected=f"末级摘要字数 < {cfg.summary_fuse_min_chars} 或 actual_max_level == 5",
        actual={"actual_max_level": event.actual_max_level, "summary_lengths": lengths},
        ok=ok,
        detail={"event_id": event.event_id, "tail_len": tail_len, "beyond_tail": beyond_tail},
    )


def anti_fragmentation(event: Event, keywords: list[str]) -> ProbeResult:
    """§1.1.7 — 防碎片化聚合：单事件 ``content_raw`` 内应同时包含多枚琐碎动作的关键字。"""
    hits = [kw for kw in keywords if kw in event.content_raw]
    ok = len(hits) >= 2
    return ProbeResult(
        name="§1.1.7 防碎片化聚合",
        expected=f"同一事件的 content_raw 同时包含 ≥2 枚关键字：{keywords}",
        actual={"hits": hits, "content_preview": event.content_raw[:100]},
        ok=ok,
        detail={"event_id": event.event_id},
    )


# =====================================================================
# §2 Role System indicators
# =====================================================================


def prompt_contains_user_mode_block(
    captured_system_prompt: str,
    mode: UserMode,
    core_user: Optional[str] = None,
    participants: Optional[list[str]] = None,
) -> ProbeResult:
    """§2.2 — 提示词注入引擎：system prompt 应包含对应模式的注入段。"""
    text = captured_system_prompt or ""
    if mode == UserMode.SINGLE:
        key_markers = ["单人隔离模式", "核心用户"]
        if core_user:
            key_markers.append(core_user)
    else:
        key_markers = ["多人交互模式", "活跃参与者"]
        if participants:
            key_markers.extend(participants[:2])  # 只校验前两个避免误伤
    present = {m: (m in text) for m in key_markers}
    ok = all(present.values())
    return ProbeResult(
        name=f"§2.2 {mode.value} 模式提示词注入",
        expected=f"system prompt 包含：{key_markers}",
        actual=present,
        ok=ok,
        detail={"system_prompt_head": text[:500]},
    )


def white_painting_tier_check(
    wp_entry: WhitePaintingEntry,
    source_event: Event,
    is_primary: bool,
    cfg: REMSConfig,
) -> ProbeResult:
    """§2.3 — 白描收集粒度：主角（S/A 或单人核心用户）取 ``l3_decision``，其他取 ``l2_interaction``。"""
    # 找到 source_event 对应这个角色的 snapshot
    snap = None
    for re in source_event.role_list:
        if re.role_id == wp_entry.role_id:
            snap = re.role_snapshot
            break
    if snap is None:
        return ProbeResult(
            name="§2.3 白描粒度",
            expected="wp_entry.role_id 在 source_event 中存在",
            actual="role not found in source event",
            ok=False,
            detail={"wp_event_id": wp_entry.event_id, "role_id": wp_entry.role_id},
        )

    target_field = cfg.wp_primary_field if is_primary else cfg.wp_default_field
    target_text = getattr(snap, target_field, None) or ""
    wp_text = wp_entry.role_summary or ""

    if not target_text:
        # 目标字段为空，允许回退到 L2/L1 —— 只要 wp 文本非空即通过
        ok = bool(wp_text)
        actual = "目标字段缺失，已回退"
    else:
        ok = (wp_text == target_text)
        actual = {"picked": wp_text[:80], "expected_field": target_field}

    return ProbeResult(
        name=f"§2.3 白描粒度（{'主角' if is_primary else '配角'}）",
        expected=f"wp.role_summary == snapshot.{target_field}",
        actual=actual,
        ok=ok,
        detail={"wp_event_id": wp_entry.event_id, "role_id": wp_entry.role_id},
    )


def dynamic_forgetting_retention(
    entries: list[WhitePaintingEntry],
    cfg: REMSConfig,
    *,
    now: Optional[datetime] = None,
) -> ProbeResult:
    """§2.3 — 动态遗忘：高 AE 条目的半衰乘数应让其有效留存分高于低 AE。"""
    now = now or datetime.now()
    high: list[float] = []
    low: list[float] = []
    for e in entries:
        age_days = max((now - e.create_time).total_seconds() / 86400, 0.0)
        multiplier = cfg.ae_forgetting_multiplier if e.memory_weight >= cfg.ae_high_threshold else 1.0
        half_life = cfg.wp_half_life_days * multiplier
        retention = math.exp(-0.693 * age_days / half_life)
        score = e.memory_weight * 0.4 + retention * 0.6
        (high if e.memory_weight >= cfg.ae_high_threshold else low).append(score)
    ok: Optional[bool]
    if not high or not low:
        ok = None  # 样本不足无法对比
    else:
        ok = (sum(high) / len(high)) > (sum(low) / len(low))
    return ProbeResult(
        name="§2.3 动态遗忘：高 AE > 低 AE 有效留存",
        expected="avg(high_ae_retention) > avg(low_ae_retention)",
        actual={
            "high_avg": round(sum(high) / len(high), 4) if high else None,
            "low_avg": round(sum(low) / len(low), 4) if low else None,
            "high_count": len(high),
            "low_count": len(low),
        },
        ok=ok,
    )


def semantic_card_key_cap(card_data: dict, cfg: REMSConfig) -> ProbeResult:
    """§2.4 — 语义卡片键数 ≤ ``semantic_card_max_keys``。"""
    n = len(card_data or {})
    ok = n <= cfg.semantic_card_max_keys
    return ProbeResult(
        name="§2.4 语义卡片键上限",
        expected=f"≤ {cfg.semantic_card_max_keys}",
        actual=n,
        ok=ok,
        detail={"keys": list((card_data or {}).keys())},
    )


def ema_smoothing_check(
    history_mean: float,
    current_snapshot: float,
    smoothed: float,
    alpha: float,
    *,
    tolerance: float = 1e-3,
) -> ProbeResult:
    """§2.5 — EMA 指数平滑：``smoothed ≈ α·current + (1-α)·history``。"""
    expected_val = alpha * current_snapshot + (1 - alpha) * history_mean
    diff = abs(smoothed - expected_val)
    ok = diff <= tolerance
    return ProbeResult(
        name="§2.5 EMA 指数平滑",
        expected=f"smoothed ≈ {alpha}*current + {1-alpha}*history（容差 {tolerance}）",
        actual={
            "history_mean": round(history_mean, 4),
            "current": round(current_snapshot, 4),
            "smoothed": round(smoothed, 4),
            "expected": round(expected_val, 4),
            "abs_diff": round(diff, 4),
        },
        ok=ok,
    )


def activation_energy_binding(event: Event, cfg: REMSConfig) -> ProbeResult:
    """§2.5 — 记忆初始值硬绑定：高 AE 事件 ``activation_energy`` 应被 1.5× 加成。"""
    ae = event.affective_energy
    act = event.activation_energy
    if ae < cfg.ae_high_threshold:
        # 非重大事件：仅校验 0 ≤ act ≤ 1 且 act ≤ ae * gain + ε
        ok = 0.0 <= act <= 1.0 and act <= ae * cfg.activation_energy_gain + 1e-6
        expected = f"0 ≤ act ≤ min(1, ae·gain)={min(1.0, ae*cfg.activation_energy_gain):.3f}"
    else:
        lower = min(1.0, ae * cfg.activation_energy_gain * 1.5) - 1e-3
        ok = act >= lower
        expected = f"≥ min(1, ae·gain·1.5) = {min(1.0, ae*cfg.activation_energy_gain*1.5):.3f}"
    return ProbeResult(
        name="§2.5 activation_energy 硬绑定",
        expected=expected,
        actual={"ae": round(ae, 4), "activation_energy": round(act, 4)},
        ok=ok,
        detail={"event_id": event.event_id, "gain": cfg.activation_energy_gain},
    )


# =====================================================================
# §3 Abstract Event indicators
# =====================================================================


def abstraction_cluster_trigger(
    recalled_count: int,
    cfg: REMSConfig,
    produced: bool,
) -> ProbeResult:
    """§3.2 — 召回聚集触发：数量达阈值 → 应生成抽象事件。"""
    should_trigger = recalled_count >= cfg.recall_cluster_threshold
    ok = (should_trigger and produced) or (not should_trigger and not produced)
    return ProbeResult(
        name="§3.2 召回聚集触发",
        expected=(
            f"recalled ≥ {cfg.recall_cluster_threshold} ⇒ produced=True；"
            "否则 produced=False"
        ),
        actual={"recalled": recalled_count, "produced": produced},
        ok=ok,
    )


def abstraction_level_chain(abstract_event: Event, cluster: list[Event]) -> ProbeResult:
    """§3.1/§1.1.8 — ``abstraction_level == max(cluster 内已有)+1`` 且 ``source_events`` 完整。"""
    max_existing = max((e.abstraction_level or 0) for e in cluster)
    expected_level = max_existing + 1
    source_ids = set(abstract_event.source_events or [])
    cluster_ids = {e.event_id for e in cluster}
    ok_level = abstract_event.abstraction_level == expected_level
    ok_sources = source_ids.issuperset(cluster_ids - {abstract_event.event_id})
    ok = ok_level and ok_sources
    return ProbeResult(
        name="§1.1.8/§3.1 抽象层级与证据链",
        expected=f"abstraction_level == {expected_level}, source_events ⊇ cluster - self",
        actual={
            "abstraction_level": abstract_event.abstraction_level,
            "source_events_count": len(source_ids),
            "cluster_size": len(cluster),
        },
        ok=ok,
        detail={
            "event_id": abstract_event.event_id,
            "missing_sources": list((cluster_ids - {abstract_event.event_id}) - source_ids),
        },
    )


def hallucination_anchor_effect(
    prompt_when_anchor_on: str,
    prompt_when_anchor_off: str,
) -> ProbeResult:
    """§3.3 — 锚定概率作用开关：强制锚定时 prompt 出现 L1 标签；关闭时使用中位摘要标签。"""
    on_ok = "L1（锚定事实层）" in prompt_when_anchor_on or "锚定" in prompt_when_anchor_on
    off_ok = "L1（锚定事实层）" not in prompt_when_anchor_off
    ok = on_ok and off_ok
    return ProbeResult(
        name="§3.3 防幻觉锚定开关",
        expected="on 时出现 L1（锚定事实层），off 时不出现",
        actual={"on_has_anchor": on_ok, "off_no_anchor": off_ok},
        ok=ok,
    )


# =====================================================================
# §4 Metabolism / Recall indicators
# =====================================================================


def recall_block_budget(block: RecallBlock, cfg: REMSConfig) -> ProbeResult:
    """§4.4 — 回忆块总字数应 ≤ ``physical_redline``。"""
    ok = block.total_length <= cfg.physical_redline
    return ProbeResult(
        name="§4.4 回忆块 ≤ physical_redline",
        expected=f"≤ {cfg.physical_redline} 字",
        actual=block.total_length,
        ok=ok,
        detail={
            "items": len(block.items),
            "levels": [it.summary_level for it in block.items],
        },
    )


def recall_has_ultra_fallback(block: RecallBlock) -> ProbeResult:
    """§4.4 — 懒索引降级：当总字数逼近上限时应出现 ``summary_level == "ultra"``。"""
    levels = [it.summary_level for it in block.items]
    has_ultra = any(lv == "ultra" for lv in levels)
    return ProbeResult(
        name="§4.4 ultra fallback 触发",
        expected="items 中至少一条 summary_level == 'ultra'",
        actual={"levels": levels, "has_ultra": has_ultra},
        ok=has_ultra,
    )


def tombstone_exclusion(block: RecallBlock, tombstoned_id: str) -> ProbeResult:
    """§4.3 — 墓碑事件不应出现在回忆块。"""
    ids = [it.event_id for it in block.items]
    excluded = tombstoned_id not in ids
    return ProbeResult(
        name="§4.3 墓碑回忆排除",
        expected=f"回忆块不含 {tombstoned_id}",
        actual={"recall_ids": ids, "excluded": excluded},
        ok=excluded,
    )


def physical_redline_compaction(
    shadow_length: int,
    unclosed_total: int,
    cfg: REMSConfig,
    triggered: bool,
) -> ProbeResult:
    """§4.2 — 物理红线触发：shadow + unclosed 超过上限应执行强制封存/压缩。"""
    total = shadow_length + unclosed_total
    should_trigger = total > cfg.physical_redline
    ok = (should_trigger and triggered) or (not should_trigger and not triggered)
    return ProbeResult(
        name="§4.2 物理红线触发",
        expected=f"sum > {cfg.physical_redline} ⇔ 触发强制整理",
        actual={"total": total, "triggered": triggered},
        ok=ok,
    )


def truncation_ratio(front_len: int, back_len: int) -> ProbeResult:
    """§4.2 — 模糊截断比例陈列（不做 pass/fail；白皮书 0.8:0.2 为建议值）。"""
    total = front_len + back_len
    ratio_front = front_len / total if total else 0.0
    return ProbeResult(
        name="§4.2 0.8:0.2 模糊截断比例",
        expected="建议前约 80%、后约 20%（LLM 执行，无硬判定）",
        actual={"front_ratio": round(ratio_front, 3), "back_ratio": round(1 - ratio_front, 3)},
        ok=None,
        detail={"front_len": front_len, "back_len": back_len},
    )


# =====================================================================
# §5 Output Mode indicators
# =====================================================================


def dialogue_package_shape(result) -> ProbeResult:
    """§5.1 — DIALOGUE 模式必须提供 Context Package。"""
    ok = result.context_package is not None
    return ProbeResult(
        name="§5.1 DIALOGUE context_package 非空",
        expected="context_package is not None",
        actual={"is_none": result.context_package is None},
        ok=ok,
    )


def passive_log_silence(result) -> ProbeResult:
    """§5.2 — PASSIVE_LOG 模式不应组装 Context Package。"""
    ok = result.context_package is None
    return ProbeResult(
        name="§5.2 PASSIVE_LOG 静默",
        expected="context_package is None",
        actual={"is_none": result.context_package is None, "sealed": len(result.sealed_events)},
        ok=ok,
    )


def npc_directives_shape(result) -> ProbeResult:
    """§5.3 — NPC_AGENT 模式：``npc_directives`` 为结构化动作列表，含 ``action`` 与 ``klesha_delta`` 键。"""
    ds = result.npc_directives or []
    ok = bool(ds) and all("action" in d and "klesha_delta" in d for d in ds)
    return ProbeResult(
        name="§5.3 NPC_AGENT 结构化指令",
        expected="npc_directives 非空且每条含 action + klesha_delta",
        actual=ds,
        ok=ok,
    )
