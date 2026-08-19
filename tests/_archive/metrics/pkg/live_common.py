"""Shared gates and pipeline bootstrap for ``test_wp_live_*.py``."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rems.config import REMSConfig, StorageConfig
from rems.pipeline import REMSPipeline

from ..conftest import FakeEmbeddingFunction


def live_metabolism_enabled() -> bool:
    return os.environ.get("REMS_RUN_LIVE_METABOLISM_TEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def fake_embedding_escape_enabled() -> bool:
    return os.environ.get("REMS_LIVE_ALLOW_FAKE_EMBEDDING", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def probe_torch() -> tuple[bool, str]:
    exe = sys.executable
    try:
        import torch

        _ = torch.zeros(1)
        return True, f"`torch` {torch.__version__} · 解释器: `{exe}`"
    except OSError as e:
        return False, f"解释器: `{exe}`\n`OSError`: {e!r}"
    except ImportError as e:
        return False, f"解释器: `{exe}`\n`ImportError`: {e!r}"


def isolated_config(tmp_path: Path, **updates) -> REMSConfig:
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    base = REMSConfig()
    db_path = root / "wp_live.sqlite3"
    chroma_path = root / "chroma_wp_live"
    storage = StorageConfig(
        database_url=f"sqlite:///{db_path.as_posix()}",
        qdrant_url=":memory:",
    )
    return base.model_copy(update={"storage": storage, **updates})


def require_live_llm_api_key(cfg: REMSConfig) -> None:
    if not (cfg.llm.api_key or "").strip():
        pytest.skip("REMS_LLM__API_KEY is empty — configure .env before running this test.")


def build_pipeline(
    tmp_path: Path,
    **cfg_updates,
) -> tuple[REMSPipeline, REMSConfig, list[str]]:
    """Return (pipeline, cfg, embedding_note_lines). Applies fake embedding patch when env set."""
    cfg = isolated_config(tmp_path, **cfg_updates)
    require_live_llm_api_key(cfg)
    notes: list[str] = []
    if fake_embedding_escape_enabled():
        cfg.embedding.provider = "hash"
        notes.append(
            "**Qdrant 嵌入**: hash 向量（`REMS_LIVE_ALLOW_FAKE_EMBEDDING=1`），语义检索质量不保证。"
        )
    else:
        torch_ok, torch_msg = probe_torch()
        if not torch_ok:
            pytest.skip(
                "当前解释器无法加载 PyTorch。\n"
                f"{torch_msg}\n"
                "请改用可用 torch 的环境，或设置 `REMS_LIVE_ALLOW_FAKE_EMBEDDING=1`。"
            )
        notes.append(f"**Qdrant 嵌入**: `{cfg.embedding.model_name}`（{torch_msg}）")
    pipeline = REMSPipeline.from_config(cfg)
    return pipeline, cfg, notes


def live_llm_skip_if_disabled() -> None:
    if not live_metabolism_enabled():
        pytest.skip(
            "Set REMS_RUN_LIVE_METABOLISM_TEST=1 to run this live LLM test.",
        )
