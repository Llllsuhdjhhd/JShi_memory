"""Acceptance tests for the JShi-facing recall design (design/1010)."""

from __future__ import annotations

from datetime import datetime, timedelta
from inspect import signature
from types import SimpleNamespace

import pytest

from rems.models.event import Event, EventRoleEntry, Importance
from rems.models.role import Role, WhitePaintingEntry
from rems.pipeline import REMSPipeline
from rems.port import MemoryBackendPort
from rems.recall import HashEmbedding, QdrantRecallVectorStore, RecallPipeline
from rems.storage.repository import EventRepository, ObjectTimelineRepository, RoleRepository


SUBJECT_ID = "stone"


@pytest.fixture()
def recall_case(db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    pipeline = RecallPipeline(
        HashEmbedding(),
        QdrantRecallVectorStore("recall-contract", url=":memory:"),
        event_repo,
        ObjectTimelineRepository(db),
        role_repo=role_repo,
        include_unclosed=False,
        default_budget_chars=10_000,
    )
    return pipeline, event_repo, role_repo


def _save_event(
    recall_case,
    *,
    text: str,
    summary: str,
    object_ids: tuple[str, ...] = (),
    occurred_at: datetime | None = None,
) -> Event:
    pipeline, event_repo, _role_repo = recall_case
    roles = [
        EventRoleEntry(role_id=SUBJECT_ID, is_subject=True, importance=Importance.S)
    ]
    roles.extend(
        EventRoleEntry(role_id=object_id, importance=Importance.C)
        for object_id in object_ids
    )
    event = Event(
        subject_id=SUBJECT_ID,
        content_raw=text,
        summaries={"L1": summary},
        role_list=roles,
        occurred_at=occurred_at,
    )
    event_repo.save(event)
    pipeline.index_event(event)
    return event


def _save_object_fact(
    role_repo: RoleRepository,
    *,
    object_id: str,
    event_id: str,
    fact: str,
) -> None:
    if role_repo.get(object_id) is None:
        role_repo.save(Role(role_id=object_id, name=object_id))
    role_repo.add_white_painting_entry(
        object_id,
        WhitePaintingEntry(
            subject_id=SUBJECT_ID,
            event_id=event_id,
            role_summary=fact,
            l1_mention=fact,
        ),
    )


def test_explicit_object_and_time_filters_intersect(recall_case):
    pipeline, _event_repo, role_repo = recall_case
    start = datetime(2025, 6, 1)
    end = datetime(2025, 6, 30)
    at_start = _save_event(
        recall_case,
        text="stone 和甲讨论维修金额680元",
        summary="维修金额是680元",
        object_ids=("OBJ-A",),
        occurred_at=start,
    )
    at_end = _save_event(
        recall_case,
        text="stone 和甲确认维修金额680元",
        summary="维修金额仍是680元",
        object_ids=("OBJ-A",),
        occurred_at=end,
    )
    wrong_object = _save_event(
        recall_case,
        text="stone 和乙讨论维修金额680元",
        summary="维修金额是680元",
        object_ids=("OBJ-B",),
        occurred_at=start + timedelta(days=10),
    )
    wrong_time = _save_event(
        recall_case,
        text="stone 和甲后来提到维修金额680元",
        summary="维修金额是680元",
        object_ids=("OBJ-A",),
        occurred_at=end + timedelta(seconds=1),
    )
    for event in (at_start, at_end, wrong_object, wrong_time):
        object_id = event.role_list[1].role_id
        _save_object_fact(
            role_repo,
            object_id=object_id,
            event_id=event.event_id,
            fact=f"{object_id} 的维修金额是680元",
        )

    fragments = pipeline.recall(
        SUBJECT_ID,
        "680元",
        object_id="OBJ-A",
        time_range=(start, end),
        channels=("lexical",),
        reinforce=False,
    )

    expected_events = {at_start.event_id, at_end.event_id}
    assert {fragment.event_id for fragment in fragments} == expected_events
    assert {fragment.type for fragment in fragments} == {"event", "object_event_fact"}
    assert all(
        fragment.type != "object_event_fact" or fragment.object_id == "OBJ-A"
        for fragment in fragments
    )
    assert all(fragment.signals["object"] == 1.0 for fragment in fragments)
    assert all(fragment.signals["temporal"] == 1.0 for fragment in fragments)


def test_query_mentions_do_not_become_hard_object_filters(recall_case):
    pipeline, _event_repo, _role_repo = recall_case
    a = _save_event(
        recall_case,
        text="stone 和小明确认维修费680元",
        summary="维修费是680元",
        object_ids=("OBJ-A",),
    )
    b = _save_event(
        recall_case,
        text="stone 和小红也确认维修费680元",
        summary="维修费也是680元",
        object_ids=("OBJ-B",),
    )

    fragments = pipeline.recall(
        SUBJECT_ID,
        "680元",
        channels=("lexical",),
        reinforce=False,
    )

    event_fragments = [fragment for fragment in fragments if fragment.type == "event"]
    assert {fragment.event_id for fragment in event_fragments} == {a.event_id, b.event_id}
    assert "object" not in event_fragments[0].signals or event_fragments[0].signals["object"] == 0.0


def test_event_lexical_recall_uses_sealed_summaries_not_raw_text(recall_case):
    pipeline, _event_repo, _role_repo = recall_case
    _save_event(
        recall_case,
        text="原文中提到只有原始记录才有的秘密口令",
        summary="记录了周末的天气",
    )

    fragments = pipeline.recall(
        SUBJECT_ID,
        "秘密口令",
        channels=("lexical",),
        reinforce=False,
    )

    assert fragments == ()


def test_query_is_embedded_verbatim(db):
    class RecordingEmbedding(HashEmbedding):
        def __init__(self):
            super().__init__()
            self.queries: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return super().embed_query(text)

    embedding = RecordingEmbedding()
    pipeline = RecallPipeline(
        embedding,
        QdrantRecallVectorStore("verbatim-query", url=":memory:"),
        EventRepository(db),
        include_unclosed=False,
    )
    query = "去年，小米维修时最后花了多少？"

    pipeline.recall(SUBJECT_ID, query, channels=("semantic",), reinforce=False)

    assert embedding.queries == [query]


def test_matching_object_fact_and_event_reinforce_the_memory_once(recall_case):
    pipeline, event_repo, role_repo = recall_case
    event = _save_event(
        recall_case,
        text="stone 与甲确认维修金额680元",
        summary="维修金额是680元",
        object_ids=("OBJ-A",),
    )
    _save_object_fact(
        role_repo,
        object_id="OBJ-A",
        event_id=event.event_id,
        fact="OBJ-A 的维修金额是680元",
    )

    fragments = pipeline.recall(
        SUBJECT_ID,
        "680元",
        object_id="OBJ-A",
        channels=("lexical",),
        reinforce=True,
    )

    assert {fragment.type for fragment in fragments} == {"event", "object_event_fact"}
    assert event_repo.get(event.event_id).forgetting_factor == pytest.approx(1.5)


def test_backend_recall_keeps_the_current_jshi_call_shape():
    jshi_keywords = {"object_id", "level", "limit", "anchor_event_ids"}
    pipeline_parameters = set(signature(REMSPipeline.recall).parameters)
    port_parameters = set(signature(MemoryBackendPort.recall).parameters)

    assert {"subject_id", "query", *jshi_keywords} <= pipeline_parameters
    assert {"self", "subject_id", "query", *jshi_keywords} <= port_parameters


def test_pipeline_forwards_the_current_jshi_recall_arguments():
    class RecallSpy:
        def __init__(self):
            self.call = None

        def recall(self, subject_id, query, **kwargs):
            self.call = (subject_id, query, kwargs)
            return ()

    spy = RecallSpy()
    pipeline = object.__new__(REMSPipeline)
    pipeline.recall_pipeline = spy
    pipeline.config = SimpleNamespace(recall_trace_enabled=False)
    pipeline.recall_trace_repo = None

    result = pipeline.recall(
        SUBJECT_ID,
        "原样线索",
        object_id="OBJ-A",
        level=7,
        limit=5,
        anchor_event_ids=("EVT-1",),
    )

    assert result == ()
    assert spy.call == (
        SUBJECT_ID,
        "原样线索",
        {
            "object_id": "OBJ-A",
            "object_ids": None,
            "interlocutor_object_id": None,
            "time_range": None,
            "budget_chars": None,
            "expand_raw": False,
            "level": 7,
            "limit": 5,
            "anchor_event_ids": ("EVT-1",),
        },
    )
