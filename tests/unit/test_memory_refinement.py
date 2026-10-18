from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rems.config import MemoryRefinementConfig, RefinementStage
from rems.models.event import Event
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience
from rems.services.refinement_service import RefinementService
from rems.storage.database import WhitePaintingRecord


@pytest.fixture()
def case(config, fake_llm):
    config.memory_refinement.enabled = True
    config.event_min_chars = 900  # Production legacy minimum must not hold small events.
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    service = RefinementService(config, pipe.event_repo, pipe.event_service,
                                pipe.recall_pipeline, fake_llm)
    yield pipe, service, fake_llm
    pipe.recall_pipeline._vector_store._client.close()
    pipe.db.engine.dispose()


def small(pipe, text, date, *, subject="stone", identity=None):
    event = pipe.event_service.seal_event(text, objects={"Lee": "lee"},
        subject_id=subject, source_ids=(identity or text,), occurred_at=date)
    pipe.recall_pipeline.index_event(event)
    return event


def responses(llm, count=1, summary="Music."):
    llm.push_response({"events": [{"indices": list(range(1, count + 1))}], "residual": []})
    llm.push_response({"summaries": {"L1": summary}, "objects": [{"name": "Lee", "fact": {"L1": summary}}],
                       "emotion": {"joy": 0.8}})


def test_partial_ack_replay_keeps_each_source_fragment_once(case):
    pipe, _, llm = case
    date = datetime(2026, 1, 1)
    first = MemoryExperience(subject_id="stone", text="Lee plays Bach。Lee prepares a concert。",
        source_ids=("D1:1",), segment_id="turn-1", occurred_at=date)
    llm.push_response({"events": [{"indices": [1]}], "residual": [{"indices": [2]}]})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(first,)))
    assert not result.errors
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    second = MemoryExperience(subject_id="stone", text="The concert is on Friday。",
        source_ids=("D1:2",), segment_id="turn-2", occurred_at=date)
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(first, second)))
    assert not result.errors
    roots = pipe.event_repo.list_all()
    text = "\n".join(event.content_raw for event in roots)
    assert text.count("Lee plays Bach") == text.count("Lee prepares a concert") == 1
    assert "Friday" in text and not pipe.meta_repo.get_unclosed_events()
    identities = [identity for event in roots for identity in event.source_fragment_ids]
    assert len(identities) == len(set(identities)) == 3
    # Reloaded persistent ownership also protects the force-save path.
    result = pipe.metabolism_service.process_input(first.text, memory=first, force_save=True)
    assert not result and len(pipe.event_repo.list_all()) == len(roots)


def test_source_fragment_identity_preserves_legitimate_repetition(case):
    pipe, _, _ = case
    for source, text in (("D1:1", "Same。Same。"), ("D2:1", "Same。")):
        memory = MemoryExperience(subject_id="stone", text=text, source_ids=(source,),
                segment_id=source, occurred_at=datetime(2026, 1, 1))
        result = pipe.metabolism_service.process_input(text, memory=memory, force_save=True)
        assert result
    roots = pipe.event_repo.list_all()
    assert sum(event.content_raw.count("Same") for event in roots) == 3
    assert len({identity for event in roots for identity in event.source_fragment_ids}) == 3


def test_replayed_oversized_sentence_chunks_are_stable(case):
    pipe, _, _ = case
    pipe.config.memory_refinement.small_event_max_chars = 30
    memory = MemoryExperience(subject_id="stone", text="a" * 83, source_ids=("long",), segment_id="long")
    first = pipe.metabolism_service.process_input(memory.text, memory=memory, force_save=True)
    assert len(first) == 3
    second = pipe.metabolism_service.process_input(memory.text, memory=memory, force_save=True)
    assert not second and len(pipe.event_repo.list_all()) == 3


def test_ingest_closes_short_event_without_enrichment_or_emotion(case):
    pipe, _, llm = case
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text="Lee likes Bach。Lee plays Mozart。",
            objects={"Lee": "lee"}, source_ids=("D1:1",), segment_id="turn-1",
            occurred_at=datetime(2026, 1, 1)),)))
    assert not result.errors and len(result.sealed_event_ids) == 1
    event = pipe.event_repo.get(result.sealed_event_ids[0])
    assert event.memory_stage == 0 and event.summaries == {} and event.emotion is None
    assert not event.role_list and event.refinement_objects == {"Lee": "lee"}
    assert [call["task_type"] for call in llm.calls] == ["boundary_detection"]
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "没有最低字数要求" in prompt and "ev_len ≤" not in prompt
    found = pipe.recall_pipeline.recall("stone", "Bach", object_id="lee",
                                       channels=("lexical",), reinforce=False)
    assert found and "Bach" in found[0].content and found[0].summary_level == "raw"


def test_backward_window_keeps_whole_events_in_chronological_order(case):
    pipe, service, _ = case
    cfg = pipe.config.memory_refinement
    cfg.candidate_chars, cfg.batch_chars = 40, 24
    now = datetime(2026, 1, 1)
    originals = [small(pipe, str(i) * 10, now - timedelta(days=100 - i)) for i in range(3)]
    small(pipe, "future", now - timedelta(days=89))
    small(pipe, "other", now - timedelta(days=120), subject="other")
    selected = service.select("stone", 1, now)
    assert [e.event_id for e in selected] == [e.event_id for e in originals[1:]]
    cfg.direction = "forward"
    assert [e.event_id for e in service.select("stone", 1, now)] == [e.event_id for e in originals[:2]]


def test_first_refinement_keeps_verbatim_l0_emotion_and_atomic_lineage(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    b = small(pipe, "Lee plays Mozart.", now - timedelta(days=99))
    responses(llm, 2)
    result = service.refine_once("stone", 1, now)
    assert result["status"] == "refined"
    parent = pipe.event_repo.get(result["parent_ids"][0])
    assert parent.content_raw == a.content_raw + "\n\n" + b.content_raw
    assert parent.memory_stage == 1 and parent.emotion is not None
    assert parent.event_time_start == a.event_time_start
    assert parent.event_time_end == b.event_time_end
    assert parent.refined_at == now and parent.source_events == [a.event_id, b.event_id]
    for child in (a, b):
        stored = pipe.event_repo.get(child.event_id)
        assert stored.content_raw == child.content_raw and stored.refinement_hidden_by == [parent.event_id]
        assert pipe.recall_pipeline._vector_store.get(child.event_id) is None
    with pipe.db.session() as session:
        assert session.query(WhitePaintingRecord).filter_by(event_id=parent.event_id).count() == 1
    assert service.refine_once("stone", 1, now)["status"] == "idle"


def test_all_three_stages_use_previous_l1_and_original_age(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach and Mozart.", datetime(2022, 1, 1))
    for _ in range(3):
        responses(llm)
    result = service.run("stone", now)
    assert not result["errors"]
    events = pipe.event_repo.list_all()
    assert sorted(e.memory_stage for e in events) == [0, 1, 2, 3]
    parents = {e.memory_stage: e for e in events}
    assert parents[1].content_raw == leaf.content_raw
    assert parents[2].content_raw == parents[1].summaries["L1"]
    assert parents[3].content_raw == parents[2].summaries["L1"]
    assert all(e.event_time_end.date() == leaf.event_time_end.date() for e in events)
    assert all(e.refined_at.replace(tzinfo=None) == now for e in events if e.memory_stage)
    assert not parents[3].refinement_hidden_by
    assert service.run("stone", now)["batches"][0]["status"] == "idle"
    assert len(llm.calls) == 6
    found = pipe.recall_pipeline.recall("stone", "Music", channels=("lexical",), reinforce=False)
    assert found and {f.event_id for f in found} == {parents[3].event_id}
    assert not pipe.recall_pipeline.recall("stone", "Bach", channels=("lexical",), reinforce=False)
    assert pipe.event_repo.get(leaf.event_id).content_raw == leaf.content_raw


def test_age_uses_latest_source_not_publication_and_handles_timezones(case):
    pipe, service, _ = case
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = small(pipe, "A complete old event.", now - timedelta(days=200))
    event.event_time_end = now - timedelta(days=89)
    pipe.event_repo.save(event)
    assert not service.select("stone", 1, now)
    event.event_time_end = now - timedelta(days=90)
    event.create_time = now
    pipe.event_repo.save(event)
    assert service.select("stone", 1, now)[0].event_id == event.event_id


def test_grouping_leaves_unconsumed_sources_visible(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    a = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    b = small(pipe, "Lee likes Mozart.", now - timedelta(days=99))
    llm.push_response({"events": [{"indices": [1]}], "residual": [{"indices": [2]}]})
    llm.push_response({"summaries": {"L1": "Bach."}, "objects": []})
    result = service.refine_once("stone", 1, now)
    assert result["consumed_ids"] == [a.event_id]
    assert not pipe.event_repo.get(b.event_id).refinement_hidden_by


def test_model_failure_does_not_publish_or_hide(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    for _ in range(3):
        llm.push_response({"summaries": {"L1": "X" * 1000}, "objects": []})
    with pytest.raises(ValueError):
        service.refine_once("stone", 1, now)
    assert len(pipe.event_repo.list_all()) == 1
    assert not pipe.event_repo.get(leaf.event_id).refinement_hidden_by
    assert pipe.recall_pipeline._vector_store.get(leaf.event_id)


def test_transaction_failure_cleans_staged_vectors_and_keeps_children(case, monkeypatch):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    responses(llm)
    def fail(*args, **kwargs):
        raise RuntimeError("publication failure")
    monkeypatch.setattr(pipe.event_repo, "commit_refinement", fail)
    with pytest.raises(RuntimeError, match="publication failure"):
        service.refine_once("stone", 1, now)
    assert not pipe.event_repo.get(leaf.event_id).refinement_hidden_by
    assert pipe.recall_pipeline._vector_store.get(leaf.event_id)
    points, _ = pipe.recall_pipeline._vector_store._client.scroll(
        pipe.recall_pipeline._vector_store._collection, limit=100)
    assert len(points) == 1


def test_conflicting_second_commit_cannot_publish_duplicate_parent(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    responses(llm)
    result = service.refine_once("stone", 1, now)
    parent = pipe.event_repo.get(result["parent_ids"][0])
    duplicate = parent.model_copy(update={"event_id": "duplicate"})
    with pytest.raises(ValueError, match="already consumed"):
        pipe.event_repo.commit_refinement([leaf], [duplicate])
    assert pipe.event_repo.get("duplicate") is None


def test_later_stage_missing_l1_never_falls_back_to_original(case):
    pipe, service, _ = case
    event = Event(subject_id="stone", content_raw="Raw must not silently replace missing L1.",
                  memory_stage=1, occurred_at=datetime(2020, 1, 1))
    pipe.event_repo.save(event)
    with pytest.raises(ValueError, match="Missing prior L1"):
        service.select("stone", 2, datetime(2026, 1, 1))
    assert not pipe.event_repo.get(event.event_id).refinement_hidden_by


def test_policy_validation_and_default_ages():
    assert [s.age_days for s in MemoryRefinementConfig().stages] == [90, 365, 1095]
    with pytest.raises(ValidationError):
        MemoryRefinementConfig(stages=[RefinementStage(age_days=365), RefinementStage(age_days=90)])
    with pytest.raises(ValidationError):
        MemoryRefinementConfig(candidate_chars=100, batch_chars=200)


def test_legacy_events_are_not_automatically_recompressed(case):
    pipe, service, llm = case
    pipe.event_repo.save(Event(subject_id="stone", content_raw="Legacy event.",
                              occurred_at=datetime(2020, 1, 1), summaries={"L1": "Old."}))
    result = service.run("stone", datetime(2026, 1, 1))
    assert not result["errors"] and all(b["status"] == "idle" for b in result["batches"])
    assert not llm.calls


def test_sql_failure_after_child_updates_rolls_back_visibility(case, monkeypatch):
    pipe, _, _ = case
    now = datetime(2026, 1, 1)
    a = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    b = small(pipe, "Lee likes Mozart.", now - timedelta(days=100))
    parent = Event(event_id="will-fail", subject_id="stone", content_raw="Grouped.",
                   memory_stage=1, source_events=[a.event_id, b.event_id])
    original = pipe.event_repo.save
    def fail_parent(event, **kwargs):
        if event.event_id == parent.event_id:
            raise RuntimeError("SQL write failed")
        return original(event, **kwargs)
    monkeypatch.setattr(pipe.event_repo, "save", fail_parent)
    with pytest.raises(RuntimeError, match="SQL write failed"):
        pipe.event_repo.commit_refinement([a, b], [parent])
    assert not pipe.event_repo.get(a.event_id).refinement_hidden_by
    assert not pipe.event_repo.get(b.event_id).refinement_hidden_by
    assert not pipe.event_repo.get(a.event_id).is_abstracted
    assert pipe.event_repo.get(parent.event_id) is None


def test_index_failure_does_not_hide_source(case, monkeypatch):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    responses(llm)
    def fail(*args, **kwargs):
        raise RuntimeError("vector write failed")
    monkeypatch.setattr(pipe.recall_pipeline, "index_event", fail)
    with pytest.raises(RuntimeError, match="vector write failed"):
        service.refine_once("stone", 1, now)
    assert len(pipe.event_repo.list_all()) == 1
    assert not pipe.event_repo.get(leaf.event_id).refinement_hidden_by


def test_preserved_source_ranges_reach_enrichment_without_changing_l0(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    a = small(pipe, "Lee signed up yesterday.", datetime(2025, 7, 3))
    b = small(pipe, "Lee finished today.", datetime(2025, 8, 4))
    responses(llm, 2)
    result = service.refine_once("stone", 1, now)
    parent = pipe.event_repo.get(result["parent_ids"][0])
    units = parent.refinement_source_units
    assert [u["event_id"] for u in units] == [a.event_id, b.event_id]
    for unit, child in zip(units, [a, b]):
        assert parent.content_raw[unit["start_char"]:unit["end_char"]] == child.content_raw
    prompt = llm.calls[-1]["messages"][1]["content"]
    assert "2025-07-03" in prompt and "2025-08-04" in prompt
    assert "never the refinement date" in prompt


def test_hidden_vector_cleanup_failure_is_filtered_and_backfill_recovers(case, monkeypatch):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    responses(llm)
    store = pipe.recall_pipeline._vector_store
    original = store.delete
    def fail_child(key):
        if key == leaf.event_id:
            raise RuntimeError("cleanup unavailable")
        return original(key)
    monkeypatch.setattr(store, "delete", fail_child)
    result = service.refine_once("stone", 1, now)
    assert result["cleanup_pending"] == [leaf.event_id]
    assert store.get(leaf.event_id) is not None
    assert not pipe.recall_pipeline.recall("stone", "Bach", channels=("lexical",), reinforce=False)
    monkeypatch.setattr(store, "delete", original)
    pipe.recall_pipeline.backfill_index()
    assert store.get(leaf.event_id) is None


def test_refinement_time_range_matches_source_span_not_publication_time(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    small(pipe, "Lee likes music.", datetime(2025, 5, 1))
    small(pipe, "Lee plays music.", datetime(2025, 6, 1))
    responses(llm, 2)
    result = service.refine_once("stone", 1, now)
    found = pipe.recall_pipeline.recall("stone", "Music", reinforce=False,
        time_range=(datetime(2025, 5, 15), datetime(2025, 5, 16)), channels=("semantic",))
    assert found and found[0].event_id in result["parent_ids"]


def test_list_group_schema_variant_is_not_silently_deferred(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    a = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    b = small(pipe, "Lee likes Mozart.", now - timedelta(days=100))
    llm.push_response({"events": [[1, 2]], "residual": []})
    llm.push_response({"summaries": {"L1": "Music."}, "objects": []})
    result = service.refine_once("stone", 1, now)
    assert result["status"] == "refined"
    assert set(result["consumed_ids"]) == {a.event_id, b.event_id}


@pytest.mark.parametrize("data", [
    {"events": "invalid"},
    {"events": [{"indices": [1]}, {"indices": [1]}]},
    {"events": [{"indices": [99]}]},
    {"events": [{"indices": [1]}], "residual": [1]},
])
def test_invalid_schema_or_claims_fail_without_hiding(case, data):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    llm.push_response(data)
    with pytest.raises(ValueError):
        service.refine_once("stone", 1, now)
    assert not pipe.event_repo.get(leaf.event_id).refinement_hidden_by
    assert len(llm.calls) == 1


def test_background_entry_runs_one_bounded_pass_and_closes_resources(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    from rems.config import REMSConfig
    from rems.jobs import refine_memory
    calls, closed = [], []
    def run(subject, *, now):
        calls.append((subject, now))
        return {"subject_id": subject, "batches": [], "errors": []}
    pipe = SimpleNamespace(refine_memory=run,
        recall_pipeline=SimpleNamespace(_vector_store=SimpleNamespace(
            _client=SimpleNamespace(close=lambda: closed.append("index")))),
        db=SimpleNamespace(engine=SimpleNamespace(dispose=lambda: closed.append("db"))))
    monkeypatch.setattr(refine_memory, "REMSConfig", lambda: REMSConfig(_env_file=None))
    monkeypatch.setattr(refine_memory.REMSPipeline, "from_config", lambda cfg: pipe)
    monkeypatch.setattr(sys, "argv", ["refine-memory", "--subject", "stone", "--now", "2026-01-01"])
    refine_memory.main()
    assert calls == [("stone", datetime(2026, 1, 1))]
    assert closed == ["index", "db"] and '"errors": []' in capsys.readouterr().out


def test_losing_worker_cannot_delete_winning_workers_vectors(case, monkeypatch):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    responses(llm)
    responses(llm)
    original = pipe.event_repo.commit_refinement
    nested, winner = [], []
    def race(children, parents):
        if not nested:
            nested.append(True)
            winner.append(service.refine_once("stone", 1, now))
        original(children, parents)
    monkeypatch.setattr(pipe.event_repo, "commit_refinement", race)
    with pytest.raises(ValueError, match="already consumed"):
        service.refine_once("stone", 1, now)
    parent_id = winner[0]["parent_ids"][0]
    assert pipe.event_repo.get(leaf.event_id).refinement_hidden_by == [parent_id]
    assert pipe.recall_pipeline._vector_store.get(parent_id) is not None
