"""Run one Hongloumeng chunk via the 2026.06 runner (replaces subprocess to continuous_simulation)."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig

from tests.scenarios.common.harness import ScenarioWorkspace, find_repo_root
from tests.scenarios.conftest import live_llm_skip_if_disabled
from tests.scenarios.hongloumeng.runner import run_chunks


@pytest.mark.live_llm
def test_hongloumeng_one_chunk_via_runner() -> None:
    live_llm_skip_if_disabled()

    cfg = REMSConfig()
    if not (cfg.llm.api_key or "").strip():
        pytest.skip("REMS_LLM__API_KEY is empty — configure .env before running this test.")

    repo = find_repo_root()
    dataset = repo / "data" / "hongloumeng_dataset.json"
    assert dataset.is_file(), f"红楼梦数据集缺失: {dataset}"

    ws = ScenarioWorkspace.default_continuous(repo)
    results = run_chunks(workspace=ws, count=1, reset=False, offline=False)
    assert len(results) == 1
    assert results[0].chunk_id >= 1
