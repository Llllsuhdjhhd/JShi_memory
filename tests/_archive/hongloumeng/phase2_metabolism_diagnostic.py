import sys
import json
import time
from pathlib import Path
from dataclasses import asdict

# Force unbuffered output
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode
from rems.llm.provider import LLMProvider
from rems.models.metabolism import Shadow

def report(msg):
    """极致透明报告打印"""
    print(msg, flush=True)
    report_path = Path("tests/scenarios/hongloumeng/outputs/continuous_run/phase2_diagnostic.md")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_phase2_diagnostic(chunk_idx: int):
    base_dir = Path("tests/scenarios/hongloumeng/outputs/continuous_run")
    report_path = base_dir / "phase2_diagnostic.md"
    
    if not report_path.exists():
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# REMS Phase 2 代谢与存续深度诊断报告\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

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
    
    if chunk_idx >= len(chunks):
        print(f"Error: Chunk {chunk_idx} out of range.")
        return

    content = chunks[chunk_idx]['content']
    
    report("\n" + "="*100)
    report(f"## 【Phase 2 诊断】CHUNK {chunk_idx + 1} 开始执行")
    report("="*100)

    # 1. 获取初始残影 (Shadow = Concatenated Unclosed Events)
    initial_unclosed = pipeline.meta_repo.get_unclosed_events()
    initial_shadow = pipeline.meta_repo.get_shadow().content
    
    report("\n### [A] 初始代谢状态 (Initial State)")
    report("-" * 60)
    report(f"**当前残影 (Shadow Content - Verbatim):**\n```text\n{initial_shadow}\n```")
    report(f"**未完成库条数:** {len(initial_unclosed)}")
    for i, ue in enumerate(initial_unclosed):
        report(f"- 未完成项 #{i+1} ID: {ue.id} (长度: {ue.total_length})")
    report("-" * 60)

    # 2. 注入 Hooks 捕获中间态
    captured_boundary = None
    orig_detect = pipeline.metabolism_service._boundary.detect
    def hooked_detect(shadow, raw, unclosed=None):
        nonlocal captured_boundary
        res = orig_detect(shadow, raw, unclosed)
        captured_boundary = res
        return res
    pipeline.metabolism_service._boundary.detect = hooked_detect

    captured_seals = []
    orig_seal = pipeline.event_service.seal_event
    def hooked_seal(content, **kwargs):
        event = orig_seal(content, **kwargs)
        captured_seals.append((content, event))
        return event
    pipeline.event_service.seal_event = hooked_seal

    # 3. 执行 Ingest
    report("\n### [B] 正在执行 Ingest (Recall -> Metabolism -> Evolution)")
    report(f"**输入文本 (Verbatim):**\n```text\n{content}\n```")
    
    # 捕获召回路径
    captured_recall = []
    orig_recall = pipeline.recall_service.build_recall_block
    def hooked_recall(query, shadow=None, **kwargs):
        block = orig_recall(query, shadow=shadow, **kwargs)
        captured_recall.append((query, shadow.content if shadow else "", block))
        return block
    pipeline.recall_service.build_recall_block = hooked_recall

    start_time = time.time()
    result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
    duration = time.time() - start_time
    
    # 4. 全链路溯源展示
    report("\n### [C] 回忆全链路溯源 (Full Traceability)")
    report("-" * 60)
    if captured_recall:
        query, shad, block = captured_recall[0]
        report(f"**1. 向量检索词 (Search Text):**\n```text\n{shad}\n[SEP]\n{query}\n```")
        report(f"\n**2. 召回项 (Top 5 Matches):**")
        for i, item in enumerate(block.items[:5]):
            report(f"- **MATCH #{i+1}** [{item.event_id}] (Score: {item.score:.4f})")
            report(f"  摘要原文: {item.content[:200]}...")
    report("-" * 60)

    # 5. 代谢流转观测
    report("\n### [D] 代谢流转观测 (Metabolism Lifecycle)")
    report("-" * 60)
    if captured_boundary:
        report("**1. 边界探测结果 (Boundary Detection Result):**")
        report(f"- **已完成片段 (Completed):** {len(captured_boundary.completed_events)} 条")
        for i, frag in enumerate(captured_boundary.completed_events):
            report(f"  - 片段 #{i+1} 原文: {frag.content_raw[:100]}...")
            report(f"    续写自: {frag.continuation_of or 'None'}")
            
        report(f"\n- **新未完成片段 (New Unclosed):** {len(captured_boundary.new_unclosed)} 条")
        for i, nu in enumerate(captured_boundary.new_unclosed):
            report(f"  - 片段 #{i+1} 原文: {nu.content[:100]}...")

    report(f"\n**2. 封存执行 (Sealing):** {len(captured_seals)} 个事件入库")
    for content_raw, event in captured_seals:
        report(f"- **EVENT [{event.event_id}]** (角色数: {len(event.role_list)})")
        report(f"  摘要: {event.summaries.get('L1', 'N/A')}")

    # 6. 最终状态
    final_unclosed = pipeline.meta_repo.get_unclosed_events()
    final_shadow = pipeline.meta_repo.get_shadow().content
    
    report(f"\n**3. 最终残影状态 (Final Shadow - Verbatim):**\n```text\n{final_shadow}\n```")
    report(f"**最终未完成库条数:** {len(final_unclosed)}")
    report("-" * 60)
    
    # 更新状态
    state_file = base_dir / "simulation_state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump({"last_chunk_idx": chunk_idx + 1}, f)

    report(f"\n[DONE] Chunk {chunk_idx + 1} 诊断完成。耗时: {duration:.2f}s")

if __name__ == "__main__":
    # 从状态文件读取下一个 index
    base_dir = Path("tests/scenarios/hongloumeng/outputs/continuous_run")
    state_file = base_dir / "simulation_state.json"
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        next_idx = state["last_chunk_idx"]
    
    run_phase2_diagnostic(next_idx)
