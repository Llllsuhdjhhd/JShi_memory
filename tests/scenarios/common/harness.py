from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.observability.ingest_trace import IngestTrace
from rems.llm.provider import LLMProvider
from rems.pipeline import REMSPipeline


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in [here, *here.parents]:
        if (d / "data" / "hongloumeng_dataset.json").is_file():
            return d
        if (d / "pyproject.toml").is_file() and (d / "src" / "rems").is_dir():
            return d
    raise FileNotFoundError("Cannot locate REMS repo root (need data/hongloumeng_dataset.json or pyproject.toml)")


@dataclass
class ScenarioWorkspace:
    """Persistent run directory for chunk-by-chunk scenario ingest."""

    base_dir: Path
    run_name: str = "continuous_run"

    @classmethod
    def create(cls, base_dir: Path, *, run_name: str = "continuous_run") -> ScenarioWorkspace:
        ws = cls(base_dir=base_dir, run_name=run_name)
        ws.base_dir.mkdir(parents=True, exist_ok=True)
        return ws

    @classmethod
    def default_continuous(cls, repo_root: Path | None = None) -> ScenarioWorkspace:
        root = repo_root or find_repo_root()
        return cls.create(
            root / "tests" / "scenarios" / "hongloumeng" / "outputs" / "continuous_run",
        )

    @property
    def state_file(self) -> Path:
        return self.base_dir / "simulation_state.json"

    @property
    def db_path(self) -> Path:
        return self.base_dir / "rems_sim.db"

    @property
    def qdrant_path(self) -> Path:
        return self.base_dir / "qdrant_sim"

    @property
    def debug_root(self) -> Path:
        return self.base_dir / "debug"

    def chunk_debug_dir(self, chunk_id: int, *, create: bool = False) -> Path:
        """Per-chunk debug artifacts: report / traces / llm summaries."""
        d = self.debug_root / f"chunk_{chunk_id}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def load_last_chunk_idx(self) -> int:
        if not self.state_file.exists():
            return 0
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return int(data.get("last_chunk_idx", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0

    def save_last_chunk_idx(self, chunk_idx: int, *, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "last_chunk_idx": chunk_idx,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> None:
        import shutil
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build_config(self, **overrides: Any) -> REMSConfig:
        cfg = REMSConfig(
            storage=StorageConfig(
                database_url=f"sqlite:///{self.db_path.as_posix()}",
                qdrant_path=str(self.qdrant_path),
            ),
            embedding={"provider": "local"},
            user_mode=UserMode.MULTI,
        )
        for key, value in overrides.items():
            if key == "embedding" and isinstance(value, dict):
                for ek, ev in value.items():
                    setattr(cfg.embedding, ek, ev)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class ChunkIngestSnapshot:
    chunk_id: int
    chunk_chars: int
    events_total: int
    events_created: int
    roles_total: int
    abstracts_total: int
    tier1_pool_size: int
    qdrant_points: int
    recall_log_rows: int
    unclosed_count: int
    shadow_chars: int
    recall_block_items: int
    intent_weights: tuple[float, float, float] | None = None
    tri_band_hits: int = 0
    bypass_triggered: bool = False
    elapsed_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def install_llm_file_logger(
    log_dir: Path,
    chunk_id: int,
    *,
    ingest_trace: IngestTrace | None = None,
    write_files: bool = True,
    full_payload: bool = False,
) -> tuple[Callable[[], None], list[Path]]:
    """Patch LLMProvider.complete; optionally persist compact per-call summaries."""
    orig = LLMProvider.complete
    written: list[Path] = []
    counter = {"n": 0}
    if write_files:
        log_dir.mkdir(parents=True, exist_ok=True)

    def complete_with_logging(self, task_type, messages, **kwargs):
        counter["n"] += 1
        n = counter["n"]
        model = self._get_model(task_type)
        t0 = time.perf_counter()
        content = orig(self, task_type, messages, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        metrics = self._invocations[-1] if self._invocations else None
        log_name = f"{n:03d}_{task_type}.json"
        if write_files:
            payload: dict[str, Any] = {
                "chunk_id": chunk_id,
                "task_type": task_type,
                "model": model,
                "latency_ms": ms,
                "metrics": asdict(metrics) if metrics else {},
            }
            if full_payload:
                payload["messages"] = messages
                payload["response"] = content
            else:
                payload["message_count"] = len(messages or [])
                payload["response_chars"] = len(content or "")
            log_path = log_dir / log_name
            log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(log_path)
        if ingest_trace is not None:
            ingest_trace.record_llm(
                task_type=task_type,
                model=model,
                latency_ms=ms,
                prompt_tokens=getattr(metrics, "prompt_tokens", None) if metrics else None,
                completion_tokens=getattr(metrics, "completion_tokens", None) if metrics else None,
                total_tokens=getattr(metrics, "total_tokens", None) if metrics else None,
                log_file=log_name if write_files else None,
            )
        elif write_files:
            print(f"  [LLM] #{n:03d} {task_type} ({ms:.0f}ms) -> {log_name}", flush=True)
        return content

    LLMProvider.complete = complete_with_logging  # type: ignore[method-assign]

    def uninstall():
        LLMProvider.complete = orig  # type: ignore[method-assign]

    return uninstall, written


def write_chunk_debug_bundle(
    workspace: ScenarioWorkspace,
    chunk_id: int,
    *,
    snapshot: ChunkIngestSnapshot,
    pipeline_steps: list[dict[str, Any]],
    recall_trace: dict[str, Any] | None = None,
) -> Path:
    """Write all per-chunk debug JSON once at ingest end (minimal I/O)."""
    debug_dir = workspace.chunk_debug_dir(chunk_id, create=True)
    llm_dir = debug_dir / "llm"
    rel = debug_dir.relative_to(workspace.base_dir).as_posix()

    snap = asdict(snapshot)
    snap["extra"] = {**snap.get("extra", {}), "debug_dir": rel}

    (debug_dir / "report.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "pipeline_trace.json").write_text(
        json.dumps({"chunk_id": chunk_id, "steps": pipeline_steps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if recall_trace is not None:
        (debug_dir / "recall_trace.json").write_text(
            json.dumps(recall_trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if llm_dir.is_dir() and not any(llm_dir.iterdir()):
        llm_dir.rmdir()
    return debug_dir


def build_pipeline(workspace: ScenarioWorkspace, **config_overrides: Any) -> REMSPipeline:
    cfg = workspace.build_config(**config_overrides)
    return REMSPipeline.from_config(cfg)
