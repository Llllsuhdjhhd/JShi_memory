"""Shared pytest fixtures for scenario tests (hongloumeng, etc.)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.scenarios.common.harness import ScenarioWorkspace, find_repo_root


def live_metabolism_enabled() -> bool:
    return os.environ.get("REMS_RUN_LIVE_METABOLISM_TEST", "").strip().lower() in (
        "1", "true", "yes",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return find_repo_root()


@pytest.fixture(scope="session")
def hongloumeng_dataset_path(repo_root: Path) -> Path:
    path = repo_root / "data" / "hongloumeng_dataset.json"
    if not path.is_file():
        pytest.skip(f"红楼梦数据集缺失: {path}")
    return path


@pytest.fixture
def scenario_workspace(tmp_path, request) -> ScenarioWorkspace:
    """Isolated workspace per test (under pytest tmp_path)."""
    name = request.node.name.replace("[", "_").replace("]", "_")[:80]
    return ScenarioWorkspace.create(tmp_path / "scenario_runs" / name)


def live_llm_skip_if_disabled() -> None:
    if not live_metabolism_enabled():
        pytest.skip(
            "Set REMS_RUN_LIVE_METABOLISM_TEST=1 to run live LLM scenario tests."
        )
