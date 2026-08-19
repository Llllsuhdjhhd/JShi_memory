"""WP-LIVE §3.2–3.3 / §4.3–4.4 — hand-seeded events, abstraction, tombstone, recall."""

from __future__ import annotations

import pytest

from rems.config import UserMode
from rems.models.event import (
    Event,
    EventRoleEntry,
    EventStatus,
    Importance,
    RoleSnapshot,
    generate_event_id,
)
from rems.models.role import Role
from .metrics import (
    InstrumentationStore,
    WPReporter,
    default_report_path,
    install_llm_tracer,
    install_recall_scoring_tracer,
    format_hybrid_line,
)
from .metrics import probes as P
from .metrics.live_common import build_pipeline, live_llm_skip_if_disabled

CORE = "ROL-abs-core"
THEME = "张三在会议室讨论项目进度"


def _seed_cluster_event(idx: int, suffix: str) -> Event:
    body = f"{THEME}：{suffix}（记录-{idx}）。"
    rid = generate_event_id()
    return Event(
        event_id=rid,
        content_raw=body,
        summaries={"L1": body, "L2": body},
        summary_lengths={"L1": len(body), "L2": len(body)},
        actual_max_level=2,
        role_list=[
            EventRoleEntry(
                role_id=CORE,
                importance=Importance.A,
                role_snapshot=RoleSnapshot(
                    l3_decision=f"张三关注进度-{idx}",
                    l2_interaction=body,
                ),
            )
        ],
        status=EventStatus.ACTIVE,
    )


@pytest.mark.live_llm
def test_wp_live_abstraction_and_recall(tmp_path) -> None:
    live_llm_skip_if_disabled()

    pipeline, cfg, emb_notes = build_pipeline(
        tmp_path,
        user_mode=UserMode.SINGLE,
        core_user_role_id=CORE,
        recall_cluster_threshold=3,
        context_window=8192,
        chars_per_token=1.0,
    )
    pipeline.role_repo.save(Role(role_id=CORE, name="张三", entity_type="person", aliases=[]))

    store = InstrumentationStore()
    un = install_llm_tracer(store, pipeline.llm)

    rep = WPReporter(default_report_path("wp_live_abstraction_and_recall.md"), "抽象与回忆专项")
    rep.set_config_snapshot(cfg)
    rep.set_llm_env_summary(emb_notes)
    rep.set_repro_footer(
        pytest_cmd="python -m pytest tests/integration/test_wp_live_abstraction_and_recall.py -v -s",
    )

    cluster: list[Event] = []
    for i, suf in enumerate(
        [
            "梳理风险清单",
            "确认里程碑日期",
            "协调接口人",
            "回顾上周延期原因",
            "对齐测试计划",
        ]
    ):
        ev = _seed_cluster_event(i, suf)
        cluster.append(ev)
        pipeline.event_repo.save(ev)
        pipeline.vector_store.add_event(ev.event_id, ev.summaries["L1"], {"is_abstract": False})

    anchor = cluster[0]
    recalled_n = len(cluster) - 1

    cfg.hallucination_anchor_prob = 1.0
    abs_on = pipeline.abstraction_service.check_and_abstract(anchor)

    with rep.section("§3.2", "召回聚集触发") as s:
        s.quote("近邻基本事件数达到阈值时应归纳抽象事件。")
        s.method(["手造同主题事件 + `check_and_abstract`。"])
        produced = abs_on is not None
        s.probe(P.abstraction_cluster_trigger(recalled_n, cfg, produced))
        if abs_on:
            s.json_block(
                "抽象事件摘录",
                {
                    "event_id": abs_on.event_id,
                    "abstraction_level": abs_on.abstraction_level,
                    "source_events": abs_on.source_events,
                    "insight": (abs_on.insight or "")[:500],
                },
            )
            src_models = [
                pipeline.event_repo.get(x)
                for x in (abs_on.source_events or [])
                if x != abs_on.event_id
            ]
            src_cluster = [e for e in src_models if e is not None]
            if src_cluster:
                s.probe(P.abstraction_level_chain(abs_on, src_cluster))

    msgs_anchor_on = list(store.abstraction_user_messages)
    cfg.hallucination_anchor_prob = 0.0
    # 第二簇：未吸收的基本事件
    cluster_b: list[Event] = []
    for i, suf in enumerate(
        [
            "补充预算评估",
            "讨论资源瓶颈",
            "确认外包范围",
            "记录决策待办",
        ]
    ):
        ev = _seed_cluster_event(i + 10, f"B-{suf}")
        cluster_b.append(ev)
        pipeline.event_repo.save(ev)
        pipeline.vector_store.add_event(ev.event_id, ev.summaries["L1"], {"is_abstract": False})

    cfg.hallucination_anchor_prob = 0.0
    _ = pipeline.abstraction_service.check_and_abstract(cluster_b[0])
    msgs_anchor_off = store.abstraction_user_messages[len(msgs_anchor_on) :]
    if not msgs_anchor_off:
        # 若向量近邻未触发第二次抽象，直接调用演化技能以捕获 prompt
        pipeline.abstraction_service._evolution.synthesize(cluster_b[:3], abstraction_level=2)
        msgs_anchor_off = store.abstraction_user_messages[len(msgs_anchor_on) :]

    with rep.section("§3.3", "防幻觉锚定开关") as s:
        s.quote("锚定概率强制为 1 时用户提示应出现 L1 锚定标签；为 0 时不出现。")
        s.method(["两次 `check_and_abstract`，比较 abstraction user 消息。"])
        on_txt = msgs_anchor_on[-1] if msgs_anchor_on else ""
        off_txt = msgs_anchor_off[-1] if msgs_anchor_off else ""
        s.probe(P.hallucination_anchor_effect(on_txt, off_txt))
        s.json_block("锚定 ON 消息头", on_txt[:1200])
        s.json_block("锚定 OFF 消息头", off_txt[:1200])

    lone = _seed_cluster_event(99, "独立审计条目（不触发抽象）")
    pipeline.event_repo.save(lone)
    pipeline.vector_store.add_event(lone.event_id, lone.summaries["L1"], {"is_abstract": False})
    tomb_id = lone.event_id
    pipeline.tombstone(tomb_id, reason="测试墓碑：逻辑覆写", replacement_id=None)

    shadow = pipeline.meta_repo.get_shadow()
    block_before = pipeline.recall_service.build_recall_block(
        "项目进度 张三 会议",
        shadow,
        focus_role_ids={CORE},
    )
    block_after = pipeline.recall_service.build_recall_block(
        "项目进度 张三 会议",
        shadow,
        focus_role_ids={CORE},
    )

    with rep.section("§4.3", "墓碑排除") as s:
        s.quote("墓碑事件不得出现在回忆检索结果中。")
        s.method(["`tombstone` 后 `build_recall_block`。"])
        s.probe(P.tombstone_exclusion(block_after, tomb_id))
        s.raw_md(
            "**墓碑前 ID 预览**\n\n"
            + "\n".join(f"- `{it.event_id}`" for it in block_before.items[:8])
            + "\n\n**墓碑后 ID 预览**\n\n"
            + "\n".join(f"- `{it.event_id}`" for it in block_after.items[:8])
            + "\n"
        )

    # §4.4 ultra + 长摘要：窄窗口管线
    tiny = build_pipeline(
        tmp_path / "recall_tiny",
        user_mode=UserMode.SINGLE,
        core_user_role_id=CORE,
        context_window=1280,
        chars_per_token=1.0,
    )[0]
    tiny.role_repo.save(Role(role_id=CORE, name="张三", entity_type="person", aliases=[]))
    long_tail = "X" * 400
    for ev in cluster_b[:4]:
        l1 = (ev.summaries.get("L1", "") + long_tail)[:2000]
        l2 = ev.summaries.get("L2", "") or ""
        ev2 = ev.model_copy(
            deep=True,
            update={
                "summaries": {"L1": l1, "L2": l2},
                "summary_lengths": {"L1": len(l1), "L2": len(l2)},
            },
        )
        tiny.event_repo.save(ev2)
        tiny.vector_store.add_event(
            ev2.event_id,
            ev2.summaries["L1"],
            {"is_abstract": False},
        )
    ultra_block = tiny.recall_service.build_recall_block(
        "项目进度 预算 外包",
        tiny.meta_repo.get_shadow(),
        focus_role_ids={CORE},
    )
    with rep.section("§4.4", "ultra 与物理红线") as s:
        s.quote("回忆块总长不得超过 `physical_redline`，必要时 ultra 极简降级。")
        s.method(["窄上下文 + 超长 L1 触发 `_ultra_concise_fallback`。"])
        s.probe(P.recall_block_budget(ultra_block, tiny.config))
        s.probe(P.recall_has_ultra_fallback(ultra_block))

    hyb_store = InstrumentationStore()
    un_h2 = install_recall_scoring_tracer(hyb_store, tiny.recall_service)
    _ = tiny.recall_service.build_recall_block(
        "张三 会议 进度",
        tiny.meta_repo.get_shadow(),
        focus_role_ids={CORE},
    )
    with rep.section("§4.4", "混合打分分量") as s:
        s.quote("混合分由余弦、时间衰减、角色 boost、AE、activation_energy 组成。")
        s.method(["`install_recall_scoring_tracer` 记录分项贡献。"])
        lines = "\n".join(format_hybrid_line(r) for r in hyb_store.hybrid_scores[:10])
        s.raw_md("```text\n" + lines + "\n```\n")
    un_h2()

    rep.add_run_summary(
        [
            f"- 抽象事件生成：**{abs_on.event_id if abs_on else '（未触发）'}**",
            f"- 墓碑 event_id：`{tomb_id}`",
        ]
    )
    rep.set_llm_invocation_metrics(pipeline.llm.invocation_history())
    rep.finalize()
    un()

    assert rep.report_path.is_file()
