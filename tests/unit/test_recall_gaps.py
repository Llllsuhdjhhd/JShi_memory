"""P0 recall-gap fixes: arousal->forgetting_factor, object route rebind, backfill."""

from __future__ import annotations

import pytest

from rems.models.event import Event, EventRoleEntry, Importance
from rems.recall import HashEmbedding, NullReranker, QdrantRecallVectorStore, RecallPipeline
from rems.recall.intent import RecallIntent, RuleIntentClassifier
from rems.services.event_service import initial_forgetting_factor
from rems.storage.backfill import backfill_event_forgetting_factors
from rems.storage.database import EventRecord
from rems.storage.repository import EventRepository, ObjectTimelineRepository


def test_initial_forgetting_factor_mapping():
    assert initial_forgetting_factor(0.5, gain=2.0) == pytest.approx(1.0)
    assert initial_forgetting_factor(1.0, gain=2.0) == pytest.approx(2.0)
    assert initial_forgetting_factor(0.0, gain=2.0) == pytest.approx(0.5)
    # clamped: out-of-range inputs collapse to [0, 1]
    assert initial_forgetting_factor(1.2, gain=2.0) == pytest.approx(2.0)
    assert initial_forgetting_factor(-0.2, gain=2.0) == pytest.approx(0.5)


@pytest.fixture()
def recall_pipeline(config, db):
    event_repo = EventRepository(db)
    obj_repo = ObjectTimelineRepository(db)
    embedding = HashEmbedding()
    vector_store = QdrantRecallVectorStore("events", url=":memory:")
    pipe = RecallPipeline(
        embedding, vector_store, event_repo, obj_repo, reranker=NullReranker()
    )
    return pipe, event_repo, obj_repo


def _mk_role_list(object_id: str) -> list[EventRoleEntry]:
    return [
        EventRoleEntry(role_id="jshi-1", is_subject=True, importance=Importance.S),
        EventRoleEntry(role_id=object_id, importance=Importance.C),
    ]


def test_object_route_uses_role_list_without_timeline(recall_pipeline):
    pipe, event_repo, _ = recall_pipeline
    a = Event(
        content_raw="和deepseek讨论测试流程",
        subject_id="jshi-1",
        summaries={"L1": "讨论测试流程"},
        role_list=_mk_role_list("OBJ-DEEPSEEK"),
    )
    b = Event(
        content_raw="独自整理代码",
        subject_id="jshi-1",
        summaries={"L1": "整理代码"},
    )
    event_repo.save(a)
    event_repo.save(b)
    pipe.index_event(a)
    pipe.index_event(b)

    frags = pipe.recall("jshi-1", "讨论", object_id="OBJ-DEEPSEEK")
    assert {f.event_id for f in frags} == {a.event_id}


def test_backfill_idempotent_and_skips_reinforced(config, db):
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="EVT-legacy-a",
                subject_id="jshi-1",
                create_time=__import__("datetime").datetime.now(),
                content_raw="旧事件高唤醒",
                summaries={"L1": "旧事件"},
                role_list=[],
                activation_energy=0.9,
                forgetting_factor=1.0,
            )
        )
        session.add(
            EventRecord(
                event_id="EVT-legacy-b",
                subject_id="jshi-1",
                create_time=__import__("datetime").datetime.now(),
                content_raw="旧事件已强化",
                summaries={"L1": "已强化"},
                role_list=[],
                activation_energy=0.9,
                forgetting_factor=1.5,
            )
        )
        session.commit()

    n = backfill_event_forgetting_factors(db, gain=2.0)
    assert n == 1

    with db.session() as session:
        a = session.get(EventRecord, "EVT-legacy-a")
        b = session.get(EventRecord, "EVT-legacy-b")
        assert a.forgetting_factor == pytest.approx(2.0 ** (2.0 * (0.9 - 0.5)))
        assert b.forgetting_factor == pytest.approx(1.5)

    # second run is a no-op
    assert backfill_event_forgetting_factors(db, gain=2.0) == 0


def test_intent_classifier_object_emotion_recency():
    clf = RuleIntentClassifier(
        known_objects={"deepseek": "OBJ-DEEPSEEK"},
        fact_lexicon=["为什么", "怎么"],
    )
    intent = clf.classify("deepseek最近为什么生气")
    assert intent.object_id == "OBJ-DEEPSEEK"
    assert intent.recency is True
    assert intent.emotion == "negative"
    assert intent.fact is True


def test_intent_classifier_empty_query():
    clf = RuleIntentClassifier()
    assert clf.classify("") == RecallIntent()


def test_recency_route_orders_by_create_time(recall_pipeline):
    from datetime import datetime, timedelta

    pipe, event_repo, _ = recall_pipeline
    older = Event(
        content_raw="旧事", subject_id="jshi-1", summaries={"L1": "旧事"},
        create_time=datetime.now() - timedelta(days=2),
    )
    newer = Event(
        content_raw="新事", subject_id="jshi-1", summaries={"L1": "新事"},
        create_time=datetime.now() - timedelta(days=1),
    )
    event_repo.save(older)
    event_repo.save(newer)
    ids = list(pipe._recency_route([older, newer]))
    assert ids == [newer.event_id, older.event_id]


def test_channels_ablation_lexical_only(recall_pipeline):
    pipe, event_repo, _ = recall_pipeline
    a = Event(content_raw="今天讨论回忆方案", subject_id="jshi-1", summaries={"L1": "讨论回忆方案"})
    b = Event(content_raw="散步买咖啡", subject_id="jshi-1", summaries={"L1": "散步"})
    event_repo.save(a)
    event_repo.save(b)
    pipe.index_event(a)
    pipe.index_event(b)

    # 只开词法路：bigram 命中 a；不开语义路也不影响
    frags = pipe.recall("jshi-1", "回忆方案", channels=("lexical",), reinforce=False)
    assert {f.event_id for f in frags} == {a.event_id}


def test_reinforce_false_keeps_forgetting_factor(recall_pipeline):
    pipe, event_repo, _ = recall_pipeline
    ev = Event(content_raw="重要的事", subject_id="jshi-1", summaries={"L1": "重要"})
    event_repo.save(ev)
    pipe.index_event(ev)
    assert event_repo.get(ev.event_id).forgetting_factor == pytest.approx(1.0)

    pipe.recall("jshi-1", "重要", reinforce=False)
    assert event_repo.get(ev.event_id).forgetting_factor == pytest.approx(1.0)
