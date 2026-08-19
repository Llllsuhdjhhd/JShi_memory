"""Live-LLM helper utilities shared by ``test_wp_live_*`` integration tests.

行为：
- ``live_llm_skip_if_disabled()``：在 ``REMS_RUN_LIVE_METABOLISM_TEST`` 未设为真值时
  立刻 ``pytest.skip``，让默认 ``pytest`` 收集这些测试时既能完成 collection 又不真正执行。
- ``build_pipeline(tmp_path, **cfg_kwargs)``：薄包装 ``REMSPipeline.create_default``，
  把数据库 / Chroma 路径定向到 ``tmp_path``，把 ``cfg_kwargs`` 直接覆盖到 ``REMSConfig``，
  返回 ``(pipeline, cfg, emb_notes)``——其中 ``emb_notes`` 是注释字符串列表，记录嵌入
  Provider 的选择，用于报告里说明。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from rems.config import REMSConfig
from rems.pipeline import REMSPipeline


def live_llm_skip_if_disabled() -> None:
    """Skip current test unless ``REMS_RUN_LIVE_METABOLISM_TEST`` is truthy."""
    if not _truthy_env("REMS_RUN_LIVE_METABOLISM_TEST"):
        pytest.skip(
            "wp-live test skipped: set REMS_RUN_LIVE_METABOLISM_TEST=1 to enable live LLM run."
        )


def _truthy_env(var: str) -> bool:
    raw = os.environ.get(var, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def build_pipeline(
    tmp_path: Path,
    **config_overrides: Any,
) -> tuple[REMSPipeline, REMSConfig, list[str]]:
    """Construct a REMSPipeline rooted at *tmp_path* with config overrides applied."""
    cfg = REMSConfig()
    cfg.storage.database_url = f"sqlite:///{tmp_path}/wp_live.db"
    cfg.storage.qdrant_url = ":memory:"

    if _truthy_env("REMS_LIVE_ALLOW_FAKE_EMBEDDING"):
        cfg.embedding.provider = "hash"

    for key, value in config_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    pipeline = REMSPipeline.from_config(cfg)
    emb_notes = [f"embedding.provider={cfg.embedding.provider}"]
    return pipeline, cfg, emb_notes
