"""WP-LIVE §2.2–2.3 / §5.1–5.3 — multi-user mode + output modes.

运行：``REMS_RUN_LIVE_METABOLISM_TEST=1``；可选 ``REMS_WP_REPORT_DIR``、
``REMS_WP_LIVE_MULTI_ROUNDS``（正整数时只跑多人剧本前 *N* 条 ingest）。
"""

from __future__ import annotations

import os

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
from rems.pipeline import ProcessingMode

from .metrics import InstrumentationStore, ProbeResult, WPReporter, default_report_path, install_llm_tracer
from .metrics import probes as P
from .metrics.live_common import build_pipeline, live_llm_skip_if_disabled

NPC_ID = "ROL-npc-guard"

MULTI_NARRATIVE_INPUTS: list[str] = [
    "她把杯子递给他，然后我们都笑了。Alice 点了点头，Bob 耸耸肩，Carol 在记笔记。",
    "Carol 提议周末去爬山，Bob 说可以带相机。",
    "Alice 和 Bob 争论预算，Carol 建议先写邮件问财务。",
    "下班路上 Bob 打电话给 Carol，Alice 自己先走了。",
    "三个人在群里约好周二当面复盘项目风险。",
]


def _multi_inputs_this_run() -> list[str]:
    raw = os.environ.get("REMS_WP_LIVE_MULTI_ROUNDS", "").strip()
    if not raw:
        return list(MULTI_NARRATIVE_INPUTS)
    try:
        n = int(raw)
    except ValueError:
        return list(MULTI_NARRATIVE_INPUTS)
    if n <= 0:
        return list(MULTI_NARRATIVE_INPUTS)
    return MULTI_NARRATIVE_INPUTS[:n]


@pytest.mark.live_llm
def test_wp_live_multi_and_modes(tmp_path) -> None:
    live_llm_skip_if_disabled()

    multi_inputs = _multi_inputs_this_run()

    pipeline, cfg, emb_notes = build_pipeline(
        tmp_path,
        user_mode=UserMode.MULTI,
        core_user_role_id=None,
        active_participants=["Alice", "Bob", "Carol"],
    )

    store = InstrumentationStore()
    un = install_llm_tracer(store, pipeline.llm)

    rep = WPReporter(default_report_path("wp_live_multi_and_modes.md"), "多人与多场景输出")
    rep.set_config_snapshot(cfg)
    rounds_note = os.environ.get("REMS_WP_LIVE_MULTI_ROUNDS", "").strip()
    rep.set_llm_env_summary(
        emb_notes
        + (
            [f"**多人剧本截断**: `REMS_WP_LIVE_MULTI_ROUNDS={rounds_note}` → 本轮 **{len(multi_inputs)}** 条"]
            if rounds_note
            else [f"**多人剧本**: 本轮 **{len(multi_inputs)}** 条（全量 {len(MULTI_NARRATIVE_INPUTS)}）"]
        ),
    )
    rep.set_repro_footer(
        pytest_cmd=(
            "python -m pytest tests/integration/test_wp_live_multi_and_modes.py -v -s"
            + (f"  # REMS_WP_LIVE_MULTI_ROUNDS={rounds_note}" if rounds_note else "")
        ),
    )

    r_dialogue: object | None = None
    for raw in multi_inputs:
        r_dialogue = pipeline.ingest(raw, mode=ProcessingMode.DIALOGUE)
    sp = store.first_role_extraction_system or ""

    with rep.section("§2.2", "多人模式提示词注入") as s:
        s.quote("多人交互模式应注入活跃参与者花名册，禁止默认归一核心用户。")
        s.method(["拦截 `role_extraction` system prompt，校验模板关键字。"])
        s.probe(
            P.prompt_contains_user_mode_block(
                sp,
                UserMode.MULTI,
                participants=["Alice", "Bob"],
            )
        )
        s.json_block("system prompt 摘要", sp[:500])

    sealed_multi = pipeline.event_repo.list_all(is_abstract=False, exclude_tombstoned=True)
    last_roles = sealed_multi[-1].role_list if sealed_multi else []
    with rep.section("§2.2", "多人抽取规模（陈列）") as s:
        s.quote("模型对多方代词的实际消解效力需人工审阅；此处仅统计抽取角色数。")
        s.method(["读取最近一条封存事件的 `role_list` 长度。"])
        s.probe(
            ProbeResult(
                name="§2.2 抽取角色数 ≥2",
                expected="同一输入下 roles ≥2",
                actual=len(last_roles),
                ok=len(last_roles) >= 2 if last_roles else None,
                detail=[r.role_id for r in last_roles],
            )
        )

    with rep.section("§2.3", "多人配角白描粒度") as s:
        s.quote("多人模式无核心用户时，配角白描应走 `l2_interaction` 档位（非 S/A）。")
        s.method(["对非 S/A 角色的白描条目执行 `white_painting_tier_check(..., is_primary=False)`。"])
        found = False
        for ev in reversed(sealed_multi):
            for entry in ev.role_list:
                if entry.importance in (Importance.S, Importance.A):
                    continue
                wps = pipeline.role_repo.get_white_painting(entry.role_id, limit=6)
                for wp in wps:
                    if wp.event_id == ev.event_id:
                        s.probe(P.white_painting_tier_check(wp, ev, False, cfg))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            s.note("_未找到合适的非 S/A 白描样本。_")

    assert r_dialogue is not None
    with rep.section("§5.1", "DIALOGUE 上下文包") as s:
        s.quote("标准交互模式应组装 Context Package 供上游生成回复。")
        s.method(["`context_package is not None`，且 `recall_block.total_length ≤ physical_redline`。"])
        s.probe(P.dialogue_package_shape(r_dialogue))
        if r_dialogue.context_package:
            s.probe(P.recall_block_budget(r_dialogue.context_package.recall_block, cfg))

    r_passive = pipeline.ingest(
        "会议室里只有键盘声，没有人说话。",
        mode=ProcessingMode.PASSIVE_LOG,
        force_save=True,
    )
    with rep.section("§5.2", "PASSIVE_LOG 静默") as s:
        s.quote("被动日志模式只记不说，不组装 Context Package。")
        s.method(["`context_package is None`；仍尝试 `force_save` 推动封存。"])
        s.probe(P.passive_log_silence(r_passive))

    # --- NPC：预置多条含 NPC id 的索引文本，抬高 threat_score ---
    pipeline.role_repo.save(Role(role_id=NPC_ID, name="守卫", entity_type="npc", aliases=[]))

    def _npc_event(i: int) -> Event:
        rid = generate_event_id()
        l1 = f"玩家在第{i}次冲突中攻击了 {NPC_ID}，守卫后退并警告。"
        return Event(
            event_id=rid,
            content_raw=l1,
            summaries={"L1": l1, "L2": l1},
            summary_lengths={"L1": len(l1), "L2": len(l1)},
            actual_max_level=2,
            role_list=[
                EventRoleEntry(
                    role_id=NPC_ID,
                    importance=Importance.B,
                    role_snapshot=RoleSnapshot(l2_interaction=l1),
                )
            ],
            status=EventStatus.ACTIVE,
        )

    for i in range(3):
        ev = _npc_event(i)
        pipeline.event_repo.save(ev)
        pipeline.vector_store.add_event(ev.event_id, ev.summaries["L1"], {"is_abstract": False})

    r_npc = pipeline.ingest(
        f"玩家再次逼近 {NPC_ID}，气氛紧张。",
        mode=ProcessingMode.NPC_AGENT,
        npc_role_id=NPC_ID,
    )
    with rep.section("§5.3", "NPC_AGENT 指令") as s:
        s.quote("NPC 模式输出结构化 `action` 与 `klesha_delta`（白皮书 5.3）。")
        s.method(["预置多条回忆文本包含 npc_role_id 以抬高威胁启发分。"])
        s.probe(P.npc_directives_shape(r_npc))
        if r_npc.npc_directives:
            action = r_npc.npc_directives[0].get("action")
            s.probe(
                ProbeResult(
                    name="§5.3 NPC action ∈ watch/flee",
                    expected="action 为 watch 或 flee（威胁阈值示例）",
                    actual=action,
                    ok=action in ("watch", "flee"),
                    detail=r_npc.npc_directives[:3],
                )
            )
            s.json_block("npc_directives", r_npc.npc_directives)
        else:
            s.probe(
                ProbeResult(
                    name="§5.3 NPC directives 非空",
                    expected="threat_score 足够触发非 idle",
                    actual=[],
                    ok=False,
                )
            )

    rep.add_run_summary(
        [
            f"- 多人 DIALOGUE 剧本：**{len(multi_inputs)}** 轮",
            f"- 剧本结束后基本事件数：**{len(sealed_multi)}**",
            f"- 全程 LLM 调用：**{len(store.llm_calls)}**",
            f"- NPC 指令条数：**{len(r_npc.npc_directives)}**",
        ]
    )
    rep.set_llm_invocation_metrics(pipeline.llm.invocation_history())
    rep.finalize()
    un()

    assert rep.report_path.is_file()
