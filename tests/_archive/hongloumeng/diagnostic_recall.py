import sys
import json
import time
from pathlib import Path

# Force unbuffered output
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recall_trace_util import (
    install_recall_hooks,
    latest_recall_log_row,
    recall_block_to_json,
    write_recall_trace_json,
)
from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode

def log(msg):
    # 同时打印到终端（供我观察）和报告文件（供你阅读）
    print(msg, file=sys.stderr, flush=True)
    report_file = Path("tests/scenarios/hongloumeng/outputs/continuous_run/diagnostic_report.md")
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_diagnostic_recall():
    base_dir = Path("tests/scenarios/hongloumeng/outputs/continuous_run")
    report_file = base_dir / "diagnostic_report.md"
    
    # 每次运行前清空报告，或者你也可以改为追加
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# REMS 记忆系统深度诊断报告\n\n")
    log_dir = base_dir / "llm_calls"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file = base_dir / "simulation_state.json"
    
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        last_idx = state["last_chunk_idx"]
    
    # Setup LLM logging
    orig_complete = LLMProvider.complete
    def complete_with_logging(self, task_type, messages, **kwargs):
        model = self._get_model(task_type)
        log(f"  [LLM] Calling {task_type} ({model})...")
        from dataclasses import asdict
        content = orig_complete(self, task_type, messages, **kwargs)
        metrics = self._invocations[-1]
        log_file = log_dir / f"diag_chunk_{last_idx + 1}_{task_type}_{time.time()}.json"
        log_data = {
            "task": task_type,
            "model": model,
            "messages": messages,
            "response": content,
            "metrics": asdict(metrics)
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return content
    LLMProvider.complete = complete_with_logging
    
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
    
    current_chunk_idx = last_idx
    content = chunks[current_chunk_idx]['content']
    
    log("\n" + "="*80)
    log(f" 【极致透明诊断：CHUNK {current_chunk_idx + 1}】 ")
    log("="*80)

    # 1. 原始输入与残影
    shadow_content = pipeline.meta_repo.get_shadow().content
    log("\n### 第一步：原始输入 (Raw Input & Shadow)")
    log("-" * 40)
    log(f"【残影 (Shadow)】:\n{shadow_content}")
    log(f"\n【当前输入 (Input)】:\n{content}")
    log("-" * 40)

    # 2. 提取人物摘要
    extraction_result = pipeline.role_skill.extract(shadow_content + "\n" + content)
    current_roles = extraction_result.roles
    log("\n### 第二步：提取的人物瞬时摘要 (Verbatim Snapshots)")
    log("-" * 40)
    for cr in current_roles:
        snap = cr.snapshot.l2_interaction or cr.snapshot.l1_mention or "无描述"
        log(f"人物: {cr.name}")
        log(f"摘要原文: {snap}")
        log("")
    log("-" * 40)

    # 3. Recall trace hooks (Stream A / B / RRF) — single ingest, then export JSON
    trace, uninstall_recall = install_recall_hooks(pipeline.recall_service)
    try:
        result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
    finally:
        uninstall_recall()

    trace_path = base_dir / "recall_trace.json"
    rb_model = result.context_package.recall_block if result.context_package else None
    write_recall_trace_json(
        trace_path,
        {
            "kind": "diagnostic_recall",
            "chunk_idx": current_chunk_idx + 1,
            "stream_a": trace["stream_a"],
            "stream_b": trace["stream_b"],
            "rrf": trace["rrf"],
            "recall_block": recall_block_to_json(rb_model) if rb_model else [],
            "recall_log_last": latest_recall_log_row(pipeline.recall_log_repo),
        },
    )
    log(f"\n[Recall trace JSON] {trace_path}")

    if result.context_package:
        rb = result.context_package.recall_block
        log(f"\n[Observation] Memory List (Top 10 Items)")
        log("="*60)
        for i, item in enumerate(rb.items[:10]):
            log(f"\nITEM #{i+1}: {item.event_id}")
            log(f"  > PUBLIC SUMMARY [{item.summary_level}]:")
            log(f"    {item.content[:200]}...")
            
            # 获取该事件下的所有角色白描
            event = pipeline.event_repo.get(item.event_id)
            if event and event.role_list:
                log(f"  > PRIVATE WHITE-PAINTINGS ({len(event.role_list)} roles):")
                for er in event.role_list:
                    role = pipeline.role_repo.get(er.role_id)
                    role_name = role.name if role else er.role_id
                    wp = pipeline.role_repo.get_white_painting_by_event(er.role_id, item.event_id)
                    if wp:
                        log(f"    - [{role_name}] Importance: {er.importance}")
                        log(f"      Role Summary: {wp.role_summary[:200]}...")
        log("="*60)
    
    state["last_chunk_idx"] = current_chunk_idx + 1
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    log("\n" + "="*80)
    log(f" DIAGNOSTIC COMPLETE. State updated to Chunk {state['last_chunk_idx']} ")
    log("="*80)

if __name__ == "__main__":
    run_diagnostic_recall()
