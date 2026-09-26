"""Offline contract checks against the adjacent JShi project's real MemoryPort."""

from __future__ import annotations

import inspect
import sys
from datetime import datetime
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
JSHI_SRC = REPO_ROOT.parent / "JShi" / "src"
if not JSHI_SRC.is_dir():
    pytest.skip("requires sibling JShi/src checkout", allow_module_level=True)
sys.path.insert(0, str(JSHI_SRC))

from jshi.memory.port import MemoryPort as JshiMemoryPort  # noqa: E402
from jshi.memory.port import RecalledFragment as JshiRecalledFragment  # noqa: E402
from jshi.memory.coordinator import RecallCoordinator  # noqa: E402
from jshi.models import RecallRequest  # noqa: E402
from jshi.memory.rems3 import from_recalled_fragments  # noqa: E402
from rems.pipeline import REMSPipeline  # noqa: E402
from rems.port import RecalledFragment as RemsRecalledFragment  # noqa: E402


def test_rems_pipeline_accepts_all_current_jshi_recall_arguments():
    jshi_parameters = set(inspect.signature(JshiMemoryPort.recall).parameters) - {"self"}
    rems_parameters = set(inspect.signature(REMSPipeline.recall).parameters)

    assert jshi_parameters <= rems_parameters


def test_jshi_coordinator_sends_its_actual_recall_request_shape():
    class MemorySpy:
        def __init__(self):
            self.call = None

        def recall(self, subject_id, query, **kwargs):
            self.call = (subject_id, query, kwargs)
            return ()

    memory = MemorySpy()
    coordinator = RecallCoordinator(repository=object(), memory=memory)
    request = RecallRequest(
        query="去年小米的维修金额",
        budget=4,
        object_ids=("OBJ-MI", "OBJ-OTHER"),
        anchor_event_ids=("EVT-ANCHOR",),
        level=6,
    )

    coordinator.execute_round(
        subject_id="stone",
        activity_id="ACT-1",
        perception_id="PER-1",
        round_number=1,
        requests=(request,),
        known_ids=set(),
        interlocutor_object_id="OBJ-MI",
    )

    assert memory.call == (
        "stone",
        "去年小米的维修金额",
        {
            "limit": 4,
            "object_ids": ("OBJ-MI", "OBJ-OTHER"),
            "interlocutor_object_id": "OBJ-MI",
            "level": 6,
            "anchor_event_ids": ("EVT-ANCHOR",),
        },
    )


def test_rems_pipeline_uses_first_jshi_object_as_interlocutor_when_not_explicit():
    class RecallSpy:
        def __init__(self):
            self.kwargs = None

        def recall(self, subject_id, query, **kwargs):
            self.kwargs = kwargs
            return ()

    pipeline = object.__new__(REMSPipeline)
    spy = RecallSpy()
    pipeline.recall_pipeline = spy
    pipeline._record_recall_trace = lambda *args, **kwargs: None

    pipeline.recall(
        "stone",
        "回忆",
        object_ids=("OBJ-SPEAKER", "OBJ-OTHER"),
    )

    assert spy.kwargs["interlocutor_object_id"] == "OBJ-SPEAKER"
    assert spy.kwargs["object_ids"] == ("OBJ-SPEAKER", "OBJ-OTHER")


def test_jshi_adapter_maps_rems_fragment_to_its_public_fragment_shape():
    occurred_at = datetime(2025, 6, 1, 12, 30)
    raw = RemsRecalledFragment(
        event_id="EVT-1",
        text="stone：与甲确认维修金额680元",
        content="与甲确认维修金额680元",
        type="object_event_fact",
        kind="fact",
        object_id="OBJ-A",
        source_ids=["source-1"],
        score=0.92,
        summary_level="L2",
        representation_level="L2",
        signals={"semantic": 0.92, "lexical": 1.0, "object": 1.0},
        occurred_at=occurred_at,
    )

    (mapped,) = from_recalled_fragments((raw,))

    assert isinstance(mapped, JshiRecalledFragment)
    assert mapped.event_id == "EVT-1"
    assert mapped.event_type == "memory"
    assert mapped.text == raw.text
    assert mapped.content == raw.content
    assert mapped.kind == "fact"
    assert mapped.object_id == "OBJ-A"
    assert mapped.source_ids == ("source-1",)
    assert mapped.score == pytest.approx(0.92)
    assert mapped.summary_level == "L2"
    assert mapped.interlocutor is None
