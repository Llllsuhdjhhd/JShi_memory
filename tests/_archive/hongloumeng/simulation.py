import os
import json
import shutil
import time
import sqlite3
import sys
from pathlib import Path

# Ensure all prints are flushed immediately for Windows debugging
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

print("DEBUG: Simulation script started.")

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode
from rems.llm.provider import LLMProvider

print("DEBUG: Imports completed.")

def run_simulation(max_chunks=5):
    """
    Simulation runner with:
    1. Automatic verification after Chunk 1.
    2. Detailed LLM request/response logging (literal Chinese).
    """
    timestamp = int(time.time())
    test_dir = Path(__file__).parent / "outputs" / "sim_runs" / f"sim_run_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for LLM logs
    log_dir = test_dir / "llm_calls"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = test_dir / "rems_sim.db"
    chroma_path = test_dir / "chroma_sim"
    
    print(f"\n{'='*60}")
    print(f" REMS HONGLOUMENG SIMULATION ({max_chunks} chunks) ")
    print(f"{'='*60}")
    print(f"Workspace: {test_dir}")
    print(f"LLM Logs:  {log_dir}")
    print(f"Goal: Process {max_chunks} chunks with full transparency.")

    # 1. Live Progress & Detail Logger
    orig_complete = LLMProvider.complete
    call_counter = 0

    def complete_with_logging(self, task_type, messages, **kwargs):
        nonlocal call_counter
        call_counter += 1
        current_call_id = call_counter
        
        model = self._get_model(task_type)
        print(f"  [LLM] #{current_call_id:03} {task_type:.<18} | {model:.<20}", end="", flush=True)
        
        t_start = time.perf_counter()
        try:
            content = orig_complete(self, task_type, messages, **kwargs)
            duration_ms = (time.perf_counter() - t_start) * 1000
            metrics = self._invocations[-1]
            
            # Save the raw request and response (ensure_ascii=False for Chinese)
            log_file = log_dir / f"{current_call_id:03}_{task_type}_{model}.json"
            log_data = {
                "call_id": current_call_id,
                "task_type": task_type,
                "model": model,
                "messages": messages,
                "response": content,
                "metrics": {
                    "latency_ms": duration_ms,
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens
                }
            }
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f" DONE ({duration_ms:5.0f}ms | Tokens: {metrics.total_tokens:4})")
            return content
        except Exception as e:
            print(f" FAILED: {str(e)[:50]}")
            raise
    
    LLMProvider.complete = complete_with_logging

    # 2. Initialization
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            qdrant_path=str(chroma_path)
        ),
        embedding={"provider": "hash"},
        user_mode=UserMode.MULTI
    )
    pipeline = REMSPipeline.from_config(config)
    
    # 修正数据路径，从 scenarios/hongloumeng/ 回溯到根目录的 data/
    dataset_path = Path(__file__).parents[3] / "data" / "hongloumeng_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    total_start_time = time.perf_counter()
    chunk_times = []

    # 3. Processing Loop
    for i in range(min(max_chunks, len(chunks))):
        chunk = chunks[i]
        content = chunk['content']
        print(f"\n--- CHUNK {i+1} START (Chars: {len(content)}) ---")
        
        chunk_start = time.perf_counter()
        try:
            import traceback
            result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
        except Exception as e:
            print("FATAL ERROR IN INGEST:")
            traceback.print_exc()
            sys.exit(1)
        chunk_duration = time.perf_counter() - chunk_start
        chunk_times.append(chunk_duration)
        
        print(f"  [CHUNK {i+1} OK] {chunk_duration:.1f}s | Events: {len(result.sealed_events)}")
        
        # --- VERIFICATION STEP AFTER CHUNK 1 ---
        if i == 0:
            print(f"\n[VERIFICATION: CHUNK 1 DATABASE STORAGE]")
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT summaries FROM events LIMIT 1")
                row = cursor.fetchone()
                if row:
                    raw_data = row[0]
                    print(f"  Raw SQL Data (First Event Summary Snapshot):")
                    print(f"  {raw_data[:200]}...") 
                    if "\\u" not in raw_data:
                        print("  >> SUCCESS: literal Chinese detected in DB.")
                    else:
                        print("  >> WARNING: Unicode escape sequences found in DB.")
                conn.close()
            except Exception as ev:
                print(f"  Verification failed: {ev}")
            print("\nProceeding to remaining chunks...")

    total_duration = time.perf_counter() - total_start_time

    # 4. Final Diagnostic Report
    print(f"\n{'='*60}")
    print(f" FINAL PERFORMANCE REPORT ({max_chunks} CHUNKS) ")
    print(f"{'='*60}")
    print(f"Total Wall Time: {total_duration:.2f}s")
    print(f"Workspace: {test_dir}")
    print(f"Total LLM Calls Logged: {call_counter}")

if __name__ == "__main__":
    run_simulation(1)
