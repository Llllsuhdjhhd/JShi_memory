"""回忆管线（design/1010）单元测试：多路召回、对象过滤、档位、记忆恢复。"""

from __future__ import annotations

import pytest

from rems.models.event import Event
from rems.models.metabolism import UnclosedEvent
from rems.models.object_entry import ObjectMemoryEntry
from rems.port import RecalledFragment
from rems.recall import HashEmbedding, NullReranker, QdrantRecallVectorStore, RecallPipeline
from rems.storage.repository import (
    EventRepository,
    MetabolismRepository,
    ObjectTimelineRepository,
)


@pytest.fixture()
def recall_pipeline(config, db):
    event_repo = EventRepository(db)
    obj_repo = ObjectTimelineRepository(db)
    meta_repo = MetabolismRepository(db)
    embedding = HashEmbedding()
    vector_store = QdrantRecallVectorStore("events", url=":memory:")
    pipe = RecallPipeline(
        embedding, vector_store, event_repo, obj_repo,
        reranker=NullReranker(),
        meta_repo=meta_repo,
    )
    return pipe, event_repo, obj_repo, meta_repo


def _mk(event_repo, text, summaries, subject_id="jshi-1", **kw):
    ev = Event(content_raw=text, subject_id=subject_id, summaries=summaries, **kw)
    event_repo.save(ev)
    return ev


class TestRecall:
    def test_semantic_recall_returns_fragment(self, recall_pipeline):
        pipe, event_repo, _obj, _meta = recall_pipeline
        ev = _mk(event_repo, "今天看到美丽的落日", {"L1": "看到落日"})
        pipe.index_event(ev)

        frags = pipe.recall("jshi-1", "落日")
        assert len(frags) == 1
        f = frags[0]
        assert isinstance(f, RecalledFragment)
        assert f.event_id == ev.event_id
        assert f.text == "今天看到美丽的落日"
        assert f.content == "看到落日"
        assert f.summary_level == "L1"
        assert f.kind == "external"

    def test_subject_scoping_excludes_other_subject(self, recall_pipeline):
        pipe, event_repo, _obj, _meta = recall_pipeline
        mine = _mk(event_repo, "我的私事", {"L1": "私事"})
        other = _mk(event_repo, "别人的私事", {"L1": "别人的"}, subject_id="other-1")
        pipe.index_event(mine)
        pipe.index_event(other)

        frags = pipe.recall("jshi-1", "私事")
        assert {f.event_id for f in frags} == {mine.event_id}

    def test_object_filter_via_timeline(self, recall_pipeline):
        pipe, event_repo, obj_repo, _meta = recall_pipeline
        a = _mk(event_repo, "和阿明一起爬山", {"L1": "爬山"})
        pipe.index_event(a)
        obj_repo.append(ObjectMemoryEntry(
            subject_id="jshi-1", object_id="OBJ-AMING", name="阿明",
            event_id=a.event_id, summary="一起爬山",
        ))
        b = _mk(event_repo, "独自看电影", {"L1": "看电影"})
        pipe.index_event(b)

        frags_am = pipe.recall("jshi-1", "爬山", object_id="OBJ-AMING")
        assert {f.event_id for f in frags_am} == {a.event_id}

        frags_all = pipe.recall("jshi-1", "爬山 电影")
        ids = {f.event_id for f in frags_all}
        assert a.event_id in ids and b.event_id in ids

    def test_reinforcement_updates_forgetting_factor(self, recall_pipeline):
        pipe, event_repo, _obj, _meta = recall_pipeline
        ev = _mk(event_repo, "重要的事", {"L1": "重要"})
        pipe.index_event(ev)
        assert event_repo.get(ev.event_id).forgetting_factor == pytest.approx(1.0)

        pipe.recall("jshi-1", "重要")
        assert event_repo.get(ev.event_id).forgetting_factor == pytest.approx(1.5)

    def test_empty_query_returns_empty(self, recall_pipeline):
        pipe, *_ = recall_pipeline
        assert pipe.recall("jshi-1", "") == ()

    def test_same_object_unclosed_is_recalled(self, recall_pipeline):
        pipe, event_repo, _obj, meta_repo = recall_pipeline
        ev = _mk(event_repo, "以前和甲去过公园", {"L1": "去过公园"})
        pipe.index_event(ev)
        meta_repo.save_unclosed_event(UnclosedEvent(
            id="UC-plan",
            subject_id="jshi-1",
            content_fragments=["甲说周末出门，还没定哪天"],
            interlocutor="OBJ-A",
        ))
        meta_repo.save_unclosed_event(UnclosedEvent(
            id="UC-other",
            subject_id="jshi-1",
            content_fragments=["乙在讲另一件事"],
            interlocutor="OBJ-B",
        ))

        frags = pipe.recall("jshi-1", "周末怎么安排", object_id="OBJ-A")
        ids = {f.event_id for f in frags}
        assert "UC-plan" in ids
        assert "UC-other" not in ids
        uc = next(f for f in frags if f.event_id == "UC-plan")
        assert uc.kind == "unclosed"
        assert uc.interlocutor == "OBJ-A"
        assert "还没定哪天" in uc.content

    def test_unclosed_only_still_returns(self, recall_pipeline):
        pipe, _event_repo, _obj, meta_repo = recall_pipeline
        meta_repo.save_unclosed_event(UnclosedEvent(
            id="UC-only",
            subject_id="jshi-1",
            content_fragments=["我们商量出门但还没结果"],
            interlocutor="OBJ-A",
        ))
        frags = pipe.recall("jshi-1", "出门", object_id="OBJ-A")
        assert [f.event_id for f in frags] == ["UC-only"]

    def test_query_overlap_unclosed_without_object_id(self, recall_pipeline):
        pipe, event_repo, _obj, meta_repo = recall_pipeline
        ev = _mk(event_repo, "昨天吃了面", {"L1": "吃面"})
        pipe.index_event(ev)
        meta_repo.save_unclosed_event(UnclosedEvent(
            id="UC-trip",
            subject_id="jshi-1",
            content_fragments=["周末出门的事还没定"],
            interlocutor="OBJ-A",
        ))
        frags = pipe.recall("jshi-1", "周末出门定了没")
        assert "UC-trip" in {f.event_id for f in frags}
