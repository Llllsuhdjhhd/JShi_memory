"""Hongloumeng chunk runner tests (offline + optional live LLM)."""

from __future__ import annotations

import json

import pytest

from rems.config import REMSConfig
from tests.scenarios.common.harness import ScenarioWorkspace
from tests.scenarios.hongloumeng.dataset import get_chunk, load_dataset
from tests.scenarios.hongloumeng.runner import run_chunks
from tests.scenarios.conftest import live_llm_skip_if_disabled


def test_dataset_has_many_chunks(hongloumeng_dataset_path):
    data = json.loads(hongloumeng_dataset_path.read_text(encoding="utf-8"))
    assert len(data) >= 100


def test_offline_ingest_one_chunk(scenario_workspace: ScenarioWorkspace, monkeypatch):
    """Hash embed + isolated workspace; patches LLM to avoid network (minimal smoke)."""
    from tests.conftest import FakeLLM
    import rems.llm.provider as llm_mod

    fake = FakeLLM()
    # Boundary: one completed event covering full chunk
    fake.push_response({
        "completed_events": [{"content_raw_indices": [1], "continuation_of": None}],
        "new_unclosed_indices": [],
    })
    fake.push_response({"summaries": {"L1": "甄士隐梦境"}, "roles": []})

    class _FakeProvider:
        def __init__(self, config):
            self.config = config

        def complete(self, task_type, messages, **kwargs):
            return fake.complete(task_type, messages, **kwargs)

        def complete_json(self, task_type, messages, **kwargs):
            return fake.complete_json(task_type, messages, **kwargs)

        def _get_model(self, task_type):
            return "fake"

    monkeypatch.setattr(llm_mod, "LLMProvider", _FakeProvider)

    chunk = get_chunk(1)
    results = run_chunks(
        workspace=scenario_workspace,
        count=1,
        chunk_id=chunk.id,
        reset=True,
        offline=True,
    )
    assert len(results) == 1
    assert results[0].chunk_id == 1
    assert results[0].events_total >= 1
    assert (scenario_workspace.chunk_debug_dir(chunk.id) / "report.json").is_file()


@pytest.mark.live_llm
def test_live_ingest_one_chunk(scenario_workspace: ScenarioWorkspace):
    live_llm_skip_if_disabled()
    cfg = REMSConfig()
    if not (cfg.llm.api_key or "").strip():
        pytest.skip("REMS_LLM__API_KEY is empty")

    last = scenario_workspace.load_last_chunk_idx()
    chunks = load_dataset()
    next_id = last + 1 if last > 0 else 1
    if not any(c.id == next_id for c in chunks):
        pytest.skip(f"Chunk {next_id} not in dataset")

    results = run_chunks(
        workspace=scenario_workspace,
        count=1,
        chunk_id=next_id,
        reset=False,
        offline=False,
    )
    assert len(results) == 1
    assert results[0].events_created >= 0
