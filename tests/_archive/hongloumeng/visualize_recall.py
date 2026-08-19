import sys
import json
import time
from pathlib import Path

# Ensure we can import rems and local scenario helpers
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode

from recall_trace_util import (
    install_recall_hooks,
    recall_block_to_json,
    write_recall_trace_json,
)

def create_visual_report(chunk_idx, input_text, shadow_text, roles_info, stream_a, stream_b, merged_items):
    report_file = Path("tests/scenarios/hongloumeng/outputs/continuous_run/recall_visual_explanation.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# REMS 记忆召回过程深度解析 (Chunk {chunk_idx})\n\n")
        f.write("REMS 使用**双流混合召回 (Dual-Stream Hybrid Recall)** 架构。这就像人类记忆：一方面通过当前情景产生“联想”（语义流），另一方面通过当前在场的人物回忆起“与他有关的事”（角色流）。\n\n")
        
        f.write("## 1. 输入上下文 (Input Context)\n")
        f.write("> 这是系统当前“看到”的内容，它是所有记忆检索的起点。\n\n")
        f.write(f"**残影 (Shadow):**\n```text\n{shadow_text or '(空)'}\n```\n\n")
        f.write(f"**当前输入 (Current Input):**\n```text\n{input_text}\n```\n\n")
        
        f.write("## 2. 检索触发词 (Retrieval Triggers)\n")
        f.write("系统首先提取当前场景中的关键人物及其瞬时状态，作为检索记忆的“锚点”。\n\n")
        f.write("| 人物 (Role) | 瞬时摘要 (Snapshot/Trigger) |\n")
        f.write("| :--- | :--- |\n")
        for r in roles_info:
            f.write(f"| {r['name']} | {r['snapshot']} |\n")
        f.write("\n")
        
        f.write("## 3. 双流并行检索 (Dual-Stream Retrieval)\n\n")
        
        f.write("### A 流：语义关联流 (Stream A: Semantic)\n")
        f.write("> **原理**：直接用当前文字在向量库中寻找语义最接近的“历史事件”。\n")
        f.write("> **侧重**：发生过类似的事情，或者话题相关的记忆。\n\n")
        f.write("| 排名 | 事件 ID | 语义内容 (Summary) | 相关度距离 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for i, h in enumerate(stream_a[:10]):
            dist = h.get('distance', 0)
            doc = h.get('document') or h.get('document_preview') or ''
            f.write(f"| {i+1} | `{h['event_id'][:8]}...` | {doc[:50]}... | {dist:.4f} |\n")
        f.write("\n")
        
        f.write("### B 流：角色白描流 (Stream B: Role-Aware)\n")
        f.write("> **原理**：利用提取到的人物状态，在“角色白描库”中搜索。它回答的是：“这些人在过去有过类似的表现吗？”\n")
        f.write("> **侧重**：人物关系的延续性，特定人物的性格一致性记忆。\n\n")
        f.write("| 排名 | 角色 | 相关事件 ID | 白描片段 (White-Painting) | 相关度距离 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for i, h in enumerate(stream_b[:10]):
            dist = h.get('distance', 0)
            doc = h.get('document') or h.get('document_preview') or ''
            parts = h['wp_id'].split("::")
            if len(parts) == 2:
                role_id, event_id = parts
                f.write(f"| {i+1} | {role_id} | `{event_id[:8]}...` | {doc[:50]}... | {dist:.4f} |\n")
        f.write("\n")
        
        f.write("## 4. 记忆融合与竞争 (Merge & Competition)\n")
        f.write("系统将两个流的结果进行合并。**同时出现在两个流中的记忆（交集）**被认为是最可靠、最相关的，优先进入大脑。\n\n")
        
        ids_a = {h['event_id'] for h in stream_a}
        ids_b = {h['wp_id'].split("::")[1] for h in stream_b if "::" in h['wp_id']}
        intersection = ids_a.intersection(ids_b)
        
        f.write(f"**核心交集记忆 ({len(intersection)} 个):**\n")
        if intersection:
            for eid in list(intersection)[:5]:
                f.write(f"- `{eid}` (同时符合当前语义和人物状态)\n")
        else:
            f.write("- (无交集)\n")
        f.write("\n")
        
        f.write("## 5. 最终进入上下文的记忆 (Final Recall Block)\n")
        f.write("经过排序、去重和**动态长度压缩**后，最终呈献给 LLM 的记忆如下：\n\n")
        f.write("| 序号 | 事件 ID | 摘要等级 | 最终呈现内容预览 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for i, it in enumerate(merged_items[:10]):
            f.write(f"| {i+1} | `{it.event_id[:8]}...` | {it.summary_level} | {it.content[:100]}... |\n")
        
        f.write("\n---\n*本报告由 REMS 视觉化诊断工具自动生成。*")

def run_visual_diagnostic():
    base_dir = Path("tests/scenarios/hongloumeng/outputs/continuous_run")
    state_file = base_dir / "simulation_state.json"
    
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        last_idx = state["last_chunk_idx"]
    
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{base_dir}/rems_sim.db",
            qdrant_path=str(base_dir / "qdrant_sim")
        ),
        embedding={"provider": "hash"},
        user_mode=UserMode.MULTI
    )
    pipeline = REMSPipeline.from_config(config)
    
    dataset_path = Path("data/hongloumeng_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    current_chunk_idx = last_idx - 1
    input_text = chunks[current_chunk_idx]['content']
    shadow = pipeline.meta_repo.get_shadow()
    
    print(f"Diagnosing Recall for Chunk {current_chunk_idx + 1}...")
    
    # 1. Capture Roles
    extraction_result = pipeline.role_skill.extract(shadow.content + "\n" + input_text)
    roles_info = []
    focus_role_ids = set()
    focus_role_entries = []
    
    # Correctly call resolve_and_register with the list
    mapping = pipeline.role_service.resolve_and_register(extraction_result.roles)
    
    for er in extraction_result.roles:
        snap = er.snapshot.l2_interaction or er.snapshot.l1_mention or "No description"
        roles_info.append({"name": er.name, "snapshot": snap})
        
        key = er.role_id or er.name
        rid = mapping[key]
        focus_role_ids.add(rid)
        focus_role_entries.append(pipeline.role_skill.to_event_role_entry(er, rid))

    # 2. B-Query reconstruction (for JSON + parity with RecallService)
    search_text = shadow.content + "\n" + input_text
    b_query = search_text
    snapshot_texts = [r["snapshot"] for r in roles_info]
    if snapshot_texts:
        b_query += "\n" + "\n".join(snapshot_texts)

    # 3. Single build_recall_block with hooks (Stream A/B + RRF captured once)
    trace, uninstall = install_recall_hooks(pipeline.recall_service)
    try:
        recall_block = pipeline.recall_service.build_recall_block(
            input_text, shadow, focus_role_ids=focus_role_ids, focus_role_entries=focus_role_entries
        )
    finally:
        uninstall()

    stream_a = trace["stream_a"][-1]["hits"]
    stream_b = trace["stream_b"][-1]["hits"]
    for h in stream_a:
        if "document" not in h and "document_preview" in h:
            h["document"] = h["document_preview"]
    for h in stream_b:
        if "document" not in h and "document_preview" in h:
            h["document"] = h["document_preview"]

    # 4. Generate Report + machine-readable trace
    create_visual_report(
        current_chunk_idx + 1,
        input_text,
        shadow.content,
        roles_info,
        stream_a,
        stream_b,
        recall_block.items,
    )

    trace_path = base_dir / "recall_trace.json"
    write_recall_trace_json(
        trace_path,
        {
            "kind": "visualize_recall",
            "chunk_idx": current_chunk_idx + 1,
            "input_excerpt": input_text[:800],
            "shadow_excerpt": (shadow.content or "")[:800],
            "b_query_excerpt": b_query[:1200],
            "stream_a": trace["stream_a"],
            "stream_b": trace["stream_b"],
            "rrf": trace["rrf"],
            "recall_block": recall_block_to_json(recall_block),
        },
    )

    print(f"Visual report generated at: tests/scenarios/hongloumeng/outputs/continuous_run/recall_visual_explanation.md")
    print(f"Recall trace JSON at: {trace_path}")

if __name__ == "__main__":
    run_visual_diagnostic()
