"""WP-LIVE §1 / §2 / §4.4 — single-user narrative through full ``ingest``.

运行：``REMS_RUN_LIVE_METABOLISM_TEST=1``；可选 ``REMS_WP_REPORT_DIR``、``REMS_LIVE_ALLOW_FAKE_EMBEDDING=1``、
``REMS_WP_LIVE_NARRATIVE_ROUNDS``（正整数时只跑剧本前 *N* 条，便于短程真模型试跑）。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from rems.config import UserMode
from rems.models.event import EmotionalModel, Importance
from rems.models.role import Role, WhitePaintingEntry
from rems.pipeline import ProcessingMode
from rems.services.emotion_service import EMAEvolver

from .metrics import (
    InstrumentationStore,
    ProbeResult,
    WPReporter,
    default_report_path,
    format_hybrid_line,
    install_llm_tracer,
    install_recall_scoring_tracer,
    install_vector_tracer,
)
from .metrics import probes as P
from .metrics.live_common import build_pipeline, live_llm_skip_if_disabled

CORE_ROLE_ID = "ROL-wp-live-core"

NARRATIVE_INPUTS: list[str] = [
    "早上七点我起床洗漱，心情还不错。",
    "我去跑了步，然后回家，接着打扫了卫生，又看了一小时小说。",
    "中午自己做了一顿简单的午饭。",
    "下午和朋友小李通了电话，约好周末爬山。",
    "傍晚我开始整理本周的工作邮件。",
    "晚上十点我准备休息，觉得今天挺充实。",
    "（次日）早上到公司，先和主管汇报了进度。",
    "上午一直在写设计文档，思路比较顺。",
    "中午和同事在食堂吃饭，聊了新项目。",
    "下午开会时有些紧张，但结果还可以。",
    "下班后我去超市买了水果。",
    "回家后练了二十分钟吉他。",
    "临睡前读了十几页书。",
    "（第三天）我得知项目提前验收，非常开心，几乎跳起来。",
    "我立刻给团队发了感谢消息。",
    "晚上奖励自己看了一部电影。",
    "（第四天）我平静地复盘了这几天的节奏。",
    "今天想聊聊之前那些经历对我意味着什么。",
]


def _narrative_inputs_this_run() -> list[str]:
    raw = os.environ.get("REMS_WP_LIVE_NARRATIVE_ROUNDS", "").strip()
    if not raw:
        return list(NARRATIVE_INPUTS)
    try:
        n = int(raw)
    except ValueError:
        return list(NARRATIVE_INPUTS)
    if n <= 0:
        return list(NARRATIVE_INPUTS)
    return NARRATIVE_INPUTS[:n]


@pytest.mark.live_llm
def test_wp_live_single_mode_narrative(tmp_path, monkeypatch) -> None:
    live_llm_skip_if_disabled()

    narrative_inputs = _narrative_inputs_this_run()

    pipeline, cfg, emb_notes = build_pipeline(
        tmp_path,
        user_mode=UserMode.SINGLE,
        core_user_role_id=CORE_ROLE_ID,
        context_window=4096,
        chars_per_token=1.0,
    )
    pipeline.role_repo.save(
        Role(role_id=CORE_ROLE_ID, name="核心用户", entity_type="person", aliases=["我"]),
    )

    store = InstrumentationStore()
    un_llm = install_llm_tracer(store, pipeline.llm)
    un_vec = install_vector_tracer(store, pipeline.vector_store)
    un_hyb = install_recall_scoring_tracer(store, pipeline.recall_service)

    ema_captures: list[tuple[float, float, float]] = []

    _orig_evolve = EMAEvolver.evolve_event

    def _evolve_with_log(self, event):
        for entry in event.role_list:
            if entry.role_id != CORE_ROLE_ID:
                continue
            recent = pipeline.role_repo.get_white_painting(
                entry.role_id,
                limit=cfg.ema_history_window,
            )
            if not recent:
                break
            baseline = EMAEvolver._average_emotion(e.emotional_model for e in recent)
            cur_joy = float(entry.emotional_model.vedana.joy)
            _orig_evolve(self, event)
            ema_captures.append(
                (float(baseline.vedana.joy), cur_joy, float(entry.emotional_model.vedana.joy)),
            )
            return
        _orig_evolve(self, event)

    monkeypatch.setattr(EMAEvolver, "evolve_event", _evolve_with_log)

    rep = WPReporter(default_report_path("wp_live_single_mode.md"), "单人模式全景叙述")
    rep.set_config_snapshot(cfg)
    rounds_note = os.environ.get("REMS_WP_LIVE_NARRATIVE_ROUNDS", "").strip()
    rep.set_llm_env_summary(
        emb_notes
        + (
            [f"**剧本截断**: `REMS_WP_LIVE_NARRATIVE_ROUNDS={rounds_note}` → 本轮 **{len(narrative_inputs)}** 条"]
            if rounds_note
            else []
        ),
    )
    rep.set_repro_footer(
        pytest_cmd=(
            "python -m pytest tests/integration/test_wp_live_single_mode_narrative.py -v -s"
            + (f"  # REMS_WP_LIVE_NARRATIVE_ROUNDS={rounds_note}" if rounds_note else "")
        ),
    )

    all_sealed: list = []
    card_snapshots: list[dict] = []
    high_ae_events: list = []
    last_recall_block = None
    frag_event = None

    for i, raw in enumerate(narrative_inputs, start=1):
        result = pipeline.ingest(raw, mode=ProcessingMode.DIALOGUE)
        all_sealed.extend(result.sealed_events)
        for ev in result.sealed_events:
            if ev.affective_energy >= cfg.ae_high_threshold:
                high_ae_events.append(ev)
            if frag_event is None and "跑" in ev.content_raw and "打扫" in ev.content_raw and "小说" in ev.content_raw:
                frag_event = ev
        if result.context_package and result.context_package.recall_block.items:
            last_recall_block = result.context_package.recall_block

        card = pipeline.role_service.get_semantic_card(CORE_ROLE_ID)
        if card and card.data:
            card_snapshots.append(dict(list(card.data.items())[:12]))
        if len(card_snapshots) > 15:
            card_snapshots.pop(0)

    un_llm()
    un_vec()
    un_hyb()

    basic_all = pipeline.event_repo.list_all(is_abstract=False, exclude_tombstoned=True)
    with rep.section("§1.1.1", "event_id 递增") as s:
        s.quote("生成字典序大致随时间递增、全局唯一的 `event_id`（`EVT-` 前缀）。")
        s.method(["按封存顺序收集基本事件 ID，检查非递减排序。"])
        s.probe(P.event_id_monotonic(basic_all))

    if all_sealed:
        last = all_sealed[-1]
        with rep.section("§1.1.3", "递归摘要熔断") as s:
            s.quote("递归摘要应在字数不足 `summary_fuse_min_chars` 时熔断。")
            s.method(["读取最后一条封存事件的 `actual_max_level` 与 `summary_lengths`。"])
            s.probe(P.summary_fuse_state(last, cfg))
        with rep.section("§1.1.7", "event_length 上限") as s:
            s.quote("单事件 `event_length` 不得超过 `len_msg`。")
            s.probe(P.event_length_vs_budget(last, cfg))

    with rep.section("§1.1.7", "防碎片化聚合") as s:
        s.quote(
            "对于描述同一连续时间段内、性质相近的琐碎日常动作……不得将其拆分为多个极短的独立事件。"
        )
        s.method(["使用含跑步→回家→打扫→看小说的单轮输入诱导合并。"])
        if frag_event:
            s.probe(P.anti_fragmentation(frag_event, ["跑", "打扫", "小说"]))
            s.json_block("合并事件摘录", {"event_id": frag_event.event_id, "l0": frag_event.content_raw[:500]})
        else:
            s.probe(
                ProbeResult(
                    name="§1.1.7 防碎片化",
                    expected="单条 content_raw 含多动作",
                    actual="未匹配到诱导句产生的单事件",
                    ok=None,
                )
            )

    sp = store.first_role_extraction_system or ""
    with rep.section("§2.2", "单人模式提示词注入") as s:
        s.quote("单人隔离模式下，system prompt 应注入核心用户与模式说明（白皮书 2.2）。")
        s.method(["拦截首次 `role_extraction` 的 system 消息。"])
        s.probe(P.prompt_contains_user_mode_block(sp, UserMode.SINGLE, core_user=CORE_ROLE_ID))
        s.json_block("system prompt 前 300 字", sp[:300])

    with rep.section("§2.3", "白描收集粒度") as s:
        s.quote("主角取 L3 决策白描，配角取 L2 互动白描。")
        s.method(["抽样含多角色的事件，对照 `WhitePaintingEntry` 与 `role_snapshot`。"])
        found_secondary = False
        for ev in reversed(all_sealed):
            if len(ev.role_list) < 2:
                continue
            for entry in ev.role_list:
                if entry.role_id == CORE_ROLE_ID:
                    continue
                wps = pipeline.role_repo.get_white_painting(entry.role_id, limit=8)
                for wp in wps:
                    if wp.event_id == ev.event_id:
                        s.probe(P.white_painting_tier_check(wp, ev, False, cfg))
                        found_secondary = True
                        break
                if found_secondary:
                    break
            if found_secondary:
                break
        found_primary = False
        for ev in reversed(all_sealed):
            for entry in ev.role_list:
                if entry.role_id != CORE_ROLE_ID:
                    continue
                wps = pipeline.role_repo.get_white_painting(entry.role_id, limit=8)
                for wp in wps:
                    if wp.event_id == ev.event_id:
                        s.probe(P.white_painting_tier_check(wp, ev, True, cfg))
                        found_primary = True
                        break
                break
            if found_primary:
                break
        if not found_secondary and not found_primary:
            s.note("_未找到多角色白描对齐样本，可能模型未拆分角色。_")

    pipeline.role_repo.add_white_painting_entry(
        CORE_ROLE_ID,
        WhitePaintingEntry(
            event_id="SYN-OLD-LOW",
            role_summary="（合成）低能量旧条目",
            importance=Importance.C,
            create_time=datetime.now() - timedelta(days=30),
            memory_weight=0.2,
            emotional_model=EmotionalModel(),
        ),
    )
    pipeline.role_repo.add_white_painting_entry(
        CORE_ROLE_ID,
        WhitePaintingEntry(
            event_id="SYN-OLD-HIGH",
            role_summary="（合成）高能量旧条目",
            importance=Importance.A,
            create_time=datetime.now() - timedelta(days=30),
            memory_weight=0.85,
            emotional_model=EmotionalModel(),
        ),
    )
    entries = pipeline.role_repo.get_white_painting(CORE_ROLE_ID, limit=80)
    with rep.section("§2.3", "动态遗忘（半衰）") as s:
        s.quote("高 AE 条目半衰期按 `ae_forgetting_multiplier` 拉长。")
        s.probe(P.dynamic_forgetting_retention(entries, cfg))

    card = pipeline.role_service.get_semantic_card(CORE_ROLE_ID)
    data = card.data if card else {}
    with rep.section("§2.4", "语义卡片键上限") as s:
        s.quote("语义卡片键数不得超过 `semantic_card_max_keys`。")
        s.probe(P.semantic_card_key_cap(data, cfg))
        s.json_block("最近卡片 data 快照", card_snapshots[-3:])

    with rep.section("§2.5", "EMA 指数平滑") as s:
        s.quote("smoothed = α·current + (1-α)·history（白皮书 2.5）。")
        s.method(["对核心用户记录历史均值 joy、当前 joy、平滑后 joy。"])
        if not ema_captures:
            s.probe(
                ProbeResult(
                    name="§2.5 EMA",
                    expected="至少一轮存在白描历史后的封存",
                    actual="无捕获（首轮或无历史）",
                    ok=None,
                )
            )
        else:
            for hist, cur, sm in ema_captures[-5:]:
                s.probe(P.ema_smoothing_check(hist, cur, sm, cfg.ema_smoothing_alpha))

    with rep.section("§2.5", "activation_energy 硬绑定") as s:
        s.quote("高 AE 事件的 `activation_energy` 应体现重大事件硬绑定（×1.5 增益上限）。")
        s.method(["筛选 `affective_energy ≥ ae_high_threshold` 的封存事件。"])
        if not high_ae_events:
            s.probe(
                ProbeResult(
                    name="§2.5 AE 硬绑定样本",
                    expected="至少一例高 AE 封存",
                    actual="本轮无高 AE 事件",
                    ok=None,
                )
            )
        else:
            for ev in high_ae_events[:5]:
                s.probe(P.activation_energy_binding(ev, cfg))

    with rep.section("§4.4", "回忆块预算与混合分") as s:
        s.quote("回忆块总长应受 `physical_redline` 约束；混合分为余弦/时间/角色/AE/Act 加权。")
        s.method(["最后一轮 `ContextPackage.recall_block` 与 `hybrid_scores` 轨迹。"])
        if last_recall_block:
            s.probe(P.recall_block_budget(last_recall_block, cfg))
            top_lines = "\n".join(
                format_hybrid_line(r)
                for r in store.hybrid_scores[-12:]
            )
            s.raw_md("**Top 混合分分量（近期轨迹）**\n\n```text\n" + top_lines + "\n```\n")
            topk_md = "\n".join(
                f"- `{it.event_id}` score={it.score:.4f} level={it.summary_level}"
                for it in last_recall_block.items[:8]
            )
            s.raw_md("**RecallBlock Top-K**\n\n" + topk_md + "\n")
        else:
            s.probe(
                ProbeResult(
                    name="§4.4 回忆块",
                    expected="最后一轮存在 recall_block",
                    actual="无（历史不足或检索为空）",
                    ok=None,
                )
            )

    # ultra：缩小上下文窗口重建管线仅用于回忆
    tiny = build_pipeline(
        tmp_path / "ultra_sub",
        user_mode=UserMode.SINGLE,
        core_user_role_id=CORE_ROLE_ID,
        context_window=1536,
        chars_per_token=1.0,
    )[0]
    tiny.role_repo.save(
        Role(role_id=CORE_ROLE_ID, name="核心用户", entity_type="person", aliases=["我"]),
    )
    for ev in basic_all[:15]:
        tiny.event_repo.save(ev)
        tiny.vector_store.add_event(
            ev.event_id,
            ev.summaries.get("L1", ev.content_raw),
            {"is_abstract": False},
        )
    shadow = tiny.meta_repo.get_shadow()
    block = tiny.recall_service.build_recall_block(
        "项目 会议 邮件 爬山 吉他",
        shadow,
        focus_role_ids={CORE_ROLE_ID},
    )
    with rep.section("§4.4", "ultra 降级") as s:
        s.quote("当总字数逼近上限时，启用 ultra 极简映射。")
        s.method(["使用窄 `physical_redline` 的临时配置强行组装回忆块。"])
        s.probe(P.recall_has_ultra_fallback(block))

    rep.add_run_summary(
        [
            f"- 剧本轮次：**{len(narrative_inputs)}**（全量剧本共 {len(NARRATIVE_INPUTS)} 条）",
            f"- 新封存事件数：**{len(all_sealed)}**",
            f"- LLM 调用次数：**{len(store.llm_calls)}**",
        ]
    )
    rep.set_llm_invocation_metrics(pipeline.llm.invocation_history())
    rep.finalize()

    assert rep.report_path.is_file()
