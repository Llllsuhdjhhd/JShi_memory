"""回忆第一版：对象范围、可及性不压过强线索、词面命中、预算选级。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rems.models.event import Event, EventRoleEntry, Importance
from rems.models.interlocutor import InterlocutorAttribution
from rems.models.role import Role, WhitePaintingEntry
from rems.recall import HashEmbedding, NullReranker, QdrantRecallVectorStore, RecallPipeline
from rems.recall.pipeline import _Candidate, material_source_hash, order_by_relevance
from rems.storage.repository import EventRepository, ObjectTimelineRepository, RoleRepository


def _cand(
    event_id: str,
    semantic: float,
    lexical: float,
    accessibility: float,
    object_role_weight: float = 1.0,
) -> _Candidate:
    return _Candidate(
        material_type="event",
        event_id=event_id,
        object_id=None,
        semantic=semantic,
        lexical=lexical,
        object_hit=0.0,
        temporal=None,
        accessibility=accessibility,
        levels={"L1": event_id},
        raw=event_id,
        kind="external",
        object_role_weight=object_role_weight,
    )


def test_accessibility_does_not_reverse_distinct_relevance():
    strong = _cand("strong", semantic=0.95, lexical=0.0, accessibility=0.01)
    weak = _cand("weak", semantic=0.72, lexical=0.0, accessibility=100.0)
    ordered = order_by_relevance([weak, strong], close=0.08)
    assert [c.event_id for c in ordered] == ["strong", "weak"]


def test_accessibility_reorders_only_when_relevance_is_close():
    higher = _cand("higher", semantic=0.90, lexical=0.0, accessibility=0.2)
    lower = _cand("lower", semantic=0.86, lexical=0.0, accessibility=5.0)
    ordered = order_by_relevance([higher, lower], close=0.08)
    assert [c.event_id for c in ordered] == ["lower", "higher"]


def test_interlocutor_70_percent_weight_can_edge_other_object_30_percent():
    speaker = _cand(
        "speaker", semantic=0.40, lexical=0.0, accessibility=1.0,
        object_role_weight=0.7,
    )
    involved = _cand(
        "involved", semantic=0.90, lexical=0.0, accessibility=1.0,
        object_role_weight=0.3,
    )

    ordered = order_by_relevance([involved, speaker], close=0.0)

    assert [item.event_id for item in ordered] == ["speaker", "involved"]
    assert speaker.relevance == pytest.approx(0.40)
    assert speaker.rank_score == pytest.approx(0.28)
    assert involved.rank_score == pytest.approx(0.27)


@pytest.fixture()
def recall_pipeline(db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    pipe = RecallPipeline(
        HashEmbedding(),
        QdrantRecallVectorStore("events", url=":memory:"),
        event_repo,
        ObjectTimelineRepository(db),
        reranker=NullReranker(),
        role_repo=role_repo,
        include_unclosed=False,
    )
    return pipe, event_repo, role_repo


def _roles(*object_ids: str) -> list[EventRoleEntry]:
    rows = [EventRoleEntry(role_id="jshi-1", is_subject=True, importance=Importance.S)]
    rows.extend(EventRoleEntry(role_id=oid, importance=Importance.C) for oid in object_ids)
    return rows


def _save_fact(role_repo: RoleRepository, role_id: str, name: str, event_id: str, levels: dict[str, str]) -> None:
    role_repo.save(Role(role_id=role_id, name=name))
    role_repo.add_white_painting_entry(role_id, WhitePaintingEntry(
        subject_id="jshi-1",
        event_id=event_id,
        role_summary=levels["L1"],
        l1_mention=levels["L1"],
        l2_interaction=levels.get("L2"),
        l3_decision=levels.get("L3"),
    ))


class TestRecallV1:
    def test_recall_applies_speaker_and_involved_object_weights(self, recall_pipeline):
        pipe, _event_repo, role_repo = recall_pipeline
        speaker_event = Event(
            content_raw="说话人提到维修金额680元",
            subject_id="jshi-1",
            summaries={"L1": "维修金额680元"},
            role_list=_roles("OBJ-SPEAKER"),
            interlocutor_attributions=[
                InterlocutorAttribution(segment_id="seg-speaker", object_id="OBJ-SPEAKER"),
            ],
        )
        involved_event = Event(
            content_raw="涉及对象也提到维修金额680元",
            subject_id="jshi-1",
            summaries={"L1": "维修金额680元"},
            role_list=_roles("OBJ-OTHER"),
        )
        for event in (speaker_event, involved_event):
            pipe._event_repo.save(event)
            _save_fact(
                role_repo,
                event.role_list[1].role_id,
                event.role_list[1].role_id,
                event.event_id,
                {"L1": "维修金额680元"},
            )
            pipe.index_event(event)

        fragments = pipe.recall(
            "jshi-1",
            "680元",
            object_ids=("OBJ-SPEAKER", "OBJ-OTHER"),
            interlocutor_object_id="OBJ-SPEAKER",
            channels=("lexical",),
            reinforce=False,
        )
        facts = [item for item in fragments if item.type == "object_event_fact"]

        assert [item.object_id for item in facts] == ["OBJ-SPEAKER", "OBJ-OTHER"]
        assert [item.score for item in facts] == pytest.approx([0.7, 0.3])
        assert [item.signals["lexical"] for item in facts] == [1.0, 1.0]
        assert [item.signals["object_role_weight"] for item in facts] == [0.7, 0.3]

    def test_object_scope_does_not_expand_to_other_facts(self, recall_pipeline):
        pipe, event_repo, role_repo = recall_pipeline
        ev = Event(
            content_raw="发布会上谈到新手机",
            subject_id="jshi-1",
            summaries={"L1": "发布会上谈到新手机"},
            role_list=_roles("OBJ-MI", "OBJ-HUAN"),
        )
        event_repo.save(ev)
        _save_fact(role_repo, "OBJ-MI", "小米", ev.event_id, {"L1": "小米想买新手机"})
        _save_fact(role_repo, "OBJ-HUAN", "小欢", ev.event_id, {"L1": "小欢觉得新手机太贵"})
        pipe.index_event(ev)
        hits = pipe._vector_store.search(
            pipe._embedding.embed_query("新手机"),
            top_k=10,
            payload_filter={"subject_id": "jshi-1"},
        )
        indexed = {(h["payload"].get("type"), h["payload"].get("object_id")) for h in hits}
        assert ("object_event_fact", "OBJ-MI") in indexed
        assert ("object_event_fact", "OBJ-HUAN") in indexed
        assert any(h["payload"].get("type") == "event" for h in hits)

        frags = pipe.recall("jshi-1", "新手机", object_ids=("OBJ-MI",), reinforce=False)
        facts = [f for f in frags if f.type == "object_event_fact"]
        events = [f for f in frags if f.type == "event"]
        assert [f.object_id for f in facts] == ["OBJ-MI"]
        assert facts[0].content == "小米想买新手机"
        assert events and events[0].event_id == ev.event_id
        assert all(f.object_id != "OBJ-HUAN" or f.type != "object_event_fact" for f in frags)
        assert frags[0].type == "object_event_fact"
        assert "temporal" not in frags[0].signals
        assert frags[0].signals["object"] == 1.0

    def test_exact_number_hit_survives_low_accessibility(self, recall_pipeline):
        pipe, event_repo, _role = recall_pipeline
        cold = Event(
            content_raw="水温保持在42℃",
            subject_id="jshi-1",
            summaries={"L1": "水温保持在42℃"},
            forgetting_factor=0.01,
            create_time=datetime.now() - timedelta(days=400),
        )
        warm = Event(
            content_raw="大家心情都不错",
            subject_id="jshi-1",
            summaries={"L1": "大家心情都不错"},
            forgetting_factor=100.0,
            create_time=datetime.now(),
        )
        event_repo.save(cold)
        event_repo.save(warm)
        pipe.index_event(cold)
        pipe.index_event(warm)

        frags = pipe.recall("jshi-1", "42℃", reinforce=False)
        ids = [f.event_id for f in frags if f.type == "event"]
        assert cold.event_id in ids
        if warm.event_id in ids:
            assert ids.index(cold.event_id) < ids.index(warm.event_id)
        hit = next(f for f in frags if f.event_id == cold.event_id and f.type == "event")
        assert hit.signals["lexical"] == 1.0

    def test_budget_picks_longest_level_that_fits(self, recall_pipeline):
        pipe, event_repo, _role = recall_pipeline
        l1 = "甲" * 890 + "42℃"
        l2 = "乙" * 430
        l3 = "丙" * 210
        raw = "原文" + "丁" * 1200
        ev = Event(
            content_raw=raw,
            subject_id="jshi-1",
            summaries={"L1": l1, "L2": l2, "L3": l3},
        )
        event_repo.save(ev)
        pipe.index_event(ev)

        def chosen(budget: int, *, expand_raw: bool = False) -> str | None:
            frags = pipe.recall(
                "jshi-1", "42℃", budget_chars=budget, expand_raw=expand_raw, reinforce=False,
            )
            events = [f for f in frags if f.type == "event" and f.event_id == ev.event_id]
            if not events:
                return None
            return events[0].content

        assert chosen(250) == l3
        assert chosen(500) == l2
        assert chosen(1000) == l1
        assert chosen(50) is None
        assert chosen(2000, expand_raw=True) == raw

    def test_short_budget_does_not_return_truncated_raw(self, recall_pipeline):
        pipe, event_repo, _role = recall_pipeline
        raw = "水温保持在42℃，后面还有一整段没有被截开的原文。"
        ev = Event(
            content_raw=raw,
            subject_id="jshi-1",
            summaries={"L1": "水温保持在42℃"},
        )
        event_repo.save(ev)
        pipe.index_event(ev)

        frags = pipe.recall("jshi-1", "42℃", budget_chars=4, reinforce=False)
        assert frags == ()


class _CountingEmbedding(HashEmbedding):
    def __init__(self):
        super().__init__()
        self.docs = 0

    def embed_documents(self, texts):
        self.docs += 1
        return super().embed_documents(texts)


def test_index_is_idempotent_until_l1_changes(db):
    embedding = _CountingEmbedding()
    store = QdrantRecallVectorStore("events", url=":memory:")
    event_repo = EventRepository(db)
    pipe = RecallPipeline(embedding, store, event_repo, include_unclosed=False)
    ev = Event(
        content_raw="原文里还有对象编号 OBJ-MI，但不该进入向量",
        subject_id="jshi-1",
        summaries={"L1": "讨论机器人记忆"},
        location="会议室",
        role_list=_roles("OBJ-MI"),
    )
    event_repo.save(ev)
    pipe.index_event(ev)
    pipe.index_event(ev)
    assert embedding.docs == 1

    stored = store.get(ev.event_id)
    assert stored["representation_level"] == "L1"
    assert stored["source_hash"] == material_source_hash("讨论机器人记忆")
    assert stored["embedding_version"] == 1
    assert "OBJ-MI" not in stored["source_hash"]

    ev.summaries = {"L1": "改成另一句 L1"}
    event_repo.save(ev)
    pipe.index_event(ev)
    assert embedding.docs == 2
    assert store.get(ev.event_id)["source_hash"] == material_source_hash("改成另一句 L1")


def test_missing_l1_does_not_embed_raw(db):
    embedding = _CountingEmbedding()
    store = QdrantRecallVectorStore("events", url=":memory:")
    event_repo = EventRepository(db)
    pipe = RecallPipeline(embedding, store, event_repo, include_unclosed=False)
    ev = Event(content_raw="只有原文，没有 L1", subject_id="jshi-1", summaries={})
    event_repo.save(ev)
    pipe.index_event(ev)
    assert embedding.docs == 0
    assert store.get(ev.event_id) is None


def test_vector_search_respects_object_and_time(recall_pipeline):
    pipe, event_repo, role_repo = recall_pipeline
    now = datetime.now()
    recent = Event(
        content_raw="最近的记录",
        subject_id="jshi-1",
        summaries={"L1": "阿尔法装置记录"},
        occurred_at=now - timedelta(days=1),
        role_list=_roles("OBJ-MI", "OBJ-HUAN"),
    )
    older = Event(
        content_raw="更早的记录",
        subject_id="jshi-1",
        summaries={"L1": "阿尔法装置记录"},
        occurred_at=now - timedelta(days=40),
        role_list=_roles("OBJ-MI"),
    )
    event_repo.save(recent)
    event_repo.save(older)
    _save_fact(role_repo, "OBJ-MI", "小米", recent.event_id, {"L1": "小米看到阿尔法装置"})
    _save_fact(role_repo, "OBJ-HUAN", "小欢", recent.event_id, {"L1": "小欢看到阿尔法装置"})
    pipe.index_event(recent)
    pipe.index_event(older)

    start = (now - timedelta(days=3)).timestamp()
    end = now.timestamp()
    hits = pipe._vector_store.search(
        pipe._embedding.embed_query("阿尔法装置"),
        top_k=10,
        payload_filter={
            "subject_id": "jshi-1",
            "embedding_model": "hash",
            "embedding_version": 1,
        },
        any_of={"object_ids": ["OBJ-MI"]},
        ranges={"occurred_at": (start, end)},
    )
    assert hits
    assert all(h["payload"].get("object_id") != "OBJ-HUAN" for h in hits)
    assert older.event_id not in {h["payload"].get("event_id") for h in hits}
    assert any(h["payload"].get("object_id") == "OBJ-MI" for h in hits)


def test_dimension_change_does_not_drop_the_collection():
    store = QdrantRecallVectorStore("events", url=":memory:")
    store.upsert("keep", [0.2] * 4, {"event_id": "keep", "source_hash": "a"})
    with pytest.raises(RuntimeError):
        store.upsert("other", [0.2] * 8, {"event_id": "other"})
    assert store.get("keep")["source_hash"] == "a"
