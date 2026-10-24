from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rems.config import MemoryRefinementConfig, RefinementStage
from rems.models.event import Event
from rems.models.metabolism import BufferSentence, UnclosedEvent
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience, RecalledFragment, clip_fragment_content
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
    bach = "Lee plays Bach" + (" and practices steadily" * 4) + "。"
    concert = "Lee prepares a concert。"
    friday = "The concert is on Friday" + (" with the same audience" * 4) + "。"
    first = MemoryExperience(subject_id="stone", text=bach + concert,
        source_ids=("D1:1",), segment_id="turn-1", occurred_at=date)
    llm.push_response({"events": [{"indices": [1]}], "residual": [{"indices": [2]}]})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(first,)))
    assert not result.errors
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    second = MemoryExperience(subject_id="stone", text=friday,
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
    text = ("Lee likes Bach and practices every morning before breakfast。"
            "Lee plays Mozart in the small room after dinner with friends。")
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=text,
            objects={"Lee": "lee"}, source_ids=("D1:1",), segment_id="turn-1",
            occurred_at=datetime(2026, 1, 1)),)))
    assert not result.errors and len(result.sealed_event_ids) == 1
    event = pipe.event_repo.get(result.sealed_event_ids[0])
    assert event.memory_stage == 0 and event.summaries == {} and event.emotion is None
    assert not event.role_list and event.refinement_objects == {"Lee": "lee"}
    assert [call["task_type"] for call in llm.calls] == ["boundary_detection"]
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "100" in prompt and "300" in prompt and "500" in prompt
    assert "没有最低字数" not in prompt and "ev_len ≤" not in prompt
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


def test_both_stages_use_previous_l1_and_original_age(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "Lee likes Bach and Mozart.", datetime(2022, 1, 1))
    for _ in range(2):
        responses(llm)
    result = service.run("stone", now)
    assert not result["errors"]
    events = pipe.event_repo.list_all()
    assert sorted(e.memory_stage for e in events) == [0, 1, 2]
    parents = {e.memory_stage: e for e in events}
    assert parents[1].content_raw == leaf.content_raw
    assert parents[2].content_raw == parents[1].summaries["L1"]
    assert all(e.event_time_end.date() == leaf.event_time_end.date() for e in events)
    assert all(e.refined_at.replace(tzinfo=None) == now for e in events if e.memory_stage)
    assert not parents[2].refinement_hidden_by
    assert parents[1].memory_sequence_start == leaf.memory_sequence == 1
    assert parents[2].memory_sequence_end == 1
    assert service.run("stone", now)["batches"][0]["status"] == "idle"
    assert len(llm.calls) == 4
    found = pipe.recall_pipeline.recall("stone", "Music", channels=("lexical",), reinforce=False)
    assert found and {f.event_id for f in found} == {parents[2].event_id}
    assert not found[0].detail_event_ids
    bach = pipe.recall_pipeline.recall("stone", "Bach", channels=("lexical",), reinforce=False)
    assert [item.event_id for item in bach if item.type == "event"] == [parents[2].event_id]
    assert not bach[0].detail_event_ids
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
    policy = MemoryRefinementConfig()
    assert [s.age_days for s in policy.stages] == [90, 365]
    assert (policy.small_event_min_chars, policy.small_event_target_chars,
            policy.small_event_max_chars) == (100, 300, 500)
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
    parent_id = result["parent_ids"][0]
    found = pipe.recall_pipeline.recall("stone", "Bach", channels=("lexical",), reinforce=False)
    assert [item.event_id for item in found if item.type == "event"] == [parent_id]
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


def test_sequences_follow_seal_order_and_parent_records_the_span(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    first = small(pipe, "Lee likes Bach.", now - timedelta(days=100))
    second = small(pipe, "Lee plays Mozart.", now - timedelta(days=99))
    assert (first.memory_sequence, second.memory_sequence) == (1, 2)
    responses(llm, 2)
    parent = pipe.event_repo.get(service.refine_once("stone", 1, now)["parent_ids"][0])
    assert (parent.memory_sequence_start, parent.memory_sequence_end) == (1, 2)
    assert [unit["memory_sequence"] for unit in parent.refinement_source_units] == [1, 2]
    assert pipe.event_repo.get(first.event_id).memory_sequence == 1


def test_interleaved_sentences_stay_verbatim_in_two_events(case):
    pipe, _, llm = case
    llm.push_response({"events": [{"indices": [1, 3]}, {"indices": [2, 4]}], "residual": []})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=(
            "甲喜欢苹果" + "果" * 90 + "。乙去公园" + "园" * 90
            + "。甲买了水果" + "果" * 90 + "。乙看见湖" + "湖" * 90 + "。"
        ),
            objects={"甲": "jia"}, source_ids=("D1",), segment_id="turn",
            occurred_at=datetime(2026, 1, 1)),)))
    assert not result.errors and len(result.sealed_event_ids) == 2
    events = [pipe.event_repo.get(event_id) for event_id in result.sealed_event_ids]
    events.sort(key=lambda event: event.memory_sequence)
    assert events[0].content_raw == "甲喜欢苹果" + "果" * 90 + "。\n甲买了水果" + "果" * 90 + "。"
    assert events[1].content_raw == "乙去公园" + "园" * 90 + "。\n乙看见湖" + "湖" * 90 + "。"
    assert events[0].summaries == {} and events[1].summaries == {}


def test_first_layer_targets_300_and_splits_only_above_500(case):
    pipe, _, llm = case
    long_sentence = "甲" * 399 + "。"
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    kept = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=long_sentence, objects={"甲": "jia"},
            source_ids=("long",), segment_id="long", occurred_at=datetime(2026, 1, 1)),)))
    assert len(kept.sealed_event_ids) == 1
    assert pipe.event_repo.get(kept.sealed_event_ids[0]).content_raw == long_sentence.strip()
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "100" in prompt and "300" in prompt and "500" in prompt
    parts = [f"句{i}" + ("乙" * 170) + "。" for i in range(4)]
    llm.push_response({"events": [{"indices": [1, 2, 3, 4]}], "residual": []})
    llm.push_response({"prefix_text": "\n".join(parts[:2]), "tail_text": "\n".join(parts[2:])})
    split = pipe.ingest_batch(MemoryBatch(subject_id="other", experiences=(
        MemoryExperience(subject_id="other", text="".join(parts), objects={"乙": "yi"},
            source_ids=("split",), segment_id="split", occurred_at=datetime(2026, 1, 2)),)))
    assert len(split.sealed_event_ids) == 1
    stored = pipe.event_repo.get(split.sealed_event_ids[0])
    assert stored.content_raw == "\n".join(parts[:2]) and len(stored.content_raw) <= 500
    residual = pipe.meta_repo.get_unclosed_events()
    assert len(residual) == 1 and parts[2] in residual[0].merged_content and parts[3] in residual[0].merged_content


def test_summary_keeps_a_query_term_without_attaching_raw(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    small(pipe, "The mentor is Dana.", now - timedelta(days=100))
    responses(llm, summary="Dana.")
    parent_id = service.refine_once("stone", 1, now)["parent_ids"][0]
    # 把摘要改成已经包含查询词，用来检查“摘要够用就不再附原文”。
    parent = pipe.event_repo.get(parent_id)
    parent.summaries = {"L1": "The mentor is Dana."}
    pipe.event_repo.save(parent)
    pipe.recall_pipeline.index_event(parent)
    found = [item for item in pipe.recall_pipeline.recall(
        "stone", "mentor Dana", channels=("lexical",), reinforce=True) if item.type == "event"]
    assert len(found) == 1 and found[0].event_id == parent_id and not found[0].detail_event_ids
    assert "原文" not in found[0].content
    hidden = next(event for event in pipe.event_repo.list_all() if event.memory_stage == 0)
    assert hidden.forgetting_factor == 1.0
    assert pipe.event_repo.get(parent_id).forgetting_factor > 1.0


def test_raw_term_missing_from_summary_is_attached_inside_the_same_result(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "The mentor is Dana.", now - timedelta(days=100))
    responses(llm, summary="Dana.")
    parent_id = service.refine_once("stone", 1, now)["parent_ids"][0]
    pipe.recall_pipeline._detail_min_similarity = 0
    found = [item for item in pipe.recall_pipeline.recall(
        "stone", "mentor Dana", reinforce=False, level=9) if item.type == "event"]
    assert [item.event_id for item in found] == [parent_id]
    assert found[0].detail_event_ids == [leaf.event_id]
    assert found[0].detail_levels == ["raw"]
    assert "原文 #1" in found[0].content and "The mentor is Dana." in found[0].content
    assert pipe.event_repo.get(leaf.event_id).forgetting_factor == 1.0


def test_detail_that_does_not_fit_is_left_out_whole(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    leaf = small(pipe, "The mentor is Dana.", now - timedelta(days=100))
    responses(llm, summary="Dana.")
    parent_id = service.refine_once("stone", 1, now)["parent_ids"][0]
    leaf.content_raw = "The mentor is Dana. " + ("x" * 800)
    pipe.event_repo.save(leaf)
    pipe.recall_pipeline._detail_min_similarity = 0
    found = [item for item in pipe.recall_pipeline.recall(
        "stone", "mentor Dana", reinforce=False, budget_chars=160)
        if item.type == "event"]
    assert [item.event_id for item in found] == [parent_id]
    assert not found[0].detail_event_ids
    assert found[0].detail_content_start is None
    assert "xxxx" not in found[0].content


def test_second_stage_raw_vector_finds_parent_without_becoming_a_candidate(case):
    pipe, _, _ = case
    when = datetime(2024, 6, 1)
    leaf = Event(event_id="leaf", subject_id="stone", content_raw="甲：公园里有一座恐龙模型。",
                 memory_stage=0, memory_sequence=4, occurred_at=when, source_dates=["2024-06-01"],
                 refinement_hidden_by=["mid"])
    mid = Event(event_id="mid", subject_id="stone", content_raw="孩子喜欢出门。", memory_stage=1,
                memory_sequence_start=4, memory_sequence_end=4, occurred_at=when,
                summaries={"L1": "孩子喜欢出门。"}, source_events=["leaf"],
                refinement_hidden_by=["top"])
    top = Event(event_id="top", subject_id="stone", content_raw="孩子喜欢出门。",
                memory_stage=2, memory_sequence_start=4, memory_sequence_end=4, occurred_at=when,
                summaries={"L1": "孩子喜欢出门。"}, source_events=["mid"], source_dates=["2024-06-01"])
    for event in (leaf, mid, top):
        pipe.event_repo.save(event)

    class Scripted:
        def _vec(self, kind):
            if kind == "raw":
                return [1.0, 0.0] + [0.0] * 62
            if kind == "query":
                return [1.0, 0.0] + [0.0] * 62
            return [0.0, 1.0] + [0.0] * 62

        def embed_documents(self, texts):
            return [self._vec("raw" if "恐龙" in text else "summary") for text in texts]

        def embed_query(self, text):
            return self._vec("query")

    pipe.recall_pipeline._embedding = Scripted()
    pipe.recall_pipeline.index_event(top)
    pipe.recall_pipeline._detail_min_similarity = 0.35
    sink = []
    found = pipe.recall_pipeline.recall(
        "stone", "他们上次看的是什么模型", channels=("semantic",), reinforce=False,
        candidate_sink=sink)
    assert [item.event_id for item in found] == ["top"]
    assert found[0].detail_event_ids == ["leaf"]
    assert "甲：公园里有一座恐龙模型。" in found[0].content
    assert "来源日期：2024-06-01" in found[0].content
    assert any(item.event_id == "leaf" and item.reason == "detail_attached" and item.similarity for item in sink)
    point = pipe.recall_pipeline._vector_store.get("top#detail#leaf")
    assert point["type"] == "event_detail" and point["source_event_id"] == "leaf"

    pipe.recall_pipeline._embedding.embed_documents = lambda texts: [
        [1.0, 0.0] + [0.0] * 62 if "恐龙" in text else [0.0, 1.0] + [0.0] * 62
        for text in texts
    ]
    pipe.recall_pipeline._embedding.embed_query = lambda text: [0.0, 1.0] + [0.0] * 62
    pipe.recall_pipeline.index_event(top)
    sink = []
    blocked = pipe.recall_pipeline.recall(
        "stone", "他们上次看的是什么模型", channels=("semantic",), reinforce=False,
        candidate_sink=sink)
    assert [item.event_id for item in blocked] == ["top"]
    assert not blocked[0].detail_event_ids
    logged = [item for item in sink if item.event_id == "leaf"]
    assert logged and logged[0].reason == "detail_below_threshold" and logged[0].similarity is not None


def test_level_does_not_increase_the_detail_count(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    small(pipe, "The mentor is Dana.", now - timedelta(days=100))
    small(pipe, "The gift was a watch.", now - timedelta(days=99))
    responses(llm, 2, summary="Lee had a day.")
    parent_id = service.refine_once("stone", 1, now)["parent_ids"][0]
    pipe.recall_pipeline._detail_min_similarity = 0
    found = [item for item in pipe.recall_pipeline.recall(
        "stone", "mentor watch day", reinforce=False, level=9)
        if item.type == "event"]
    assert [item.event_id for item in found] == [parent_id]
    assert len(found[0].detail_event_ids) == 1


def test_under_minimum_waits_in_residual(case):
    pipe, _, llm = case
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    text = "我们去湖边散步，聊得很开心。"
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=text, objects={"甲": "jia"},
            source_ids=("walk",), segment_id="walk", occurred_at=datetime(2026, 1, 1)),)))
    assert not result.errors and not result.sealed_event_ids
    leftover = pipe.meta_repo.get_unclosed_events()
    assert len(leftover) == 1 and text in leftover[0].merged_content


def test_split_keeps_speaker_prefixes_on_both_sides(case):
    pipe, _, llm = case
    first = "甲：" + "啊" * 280 + "。"
    second = "乙：" + "哦" * 280 + "。"
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    llm.push_response({"prefix_text": first, "tail_text": second})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=first + second, objects={"甲": "jia", "乙": "yi"},
            source_ids=("talk",), segment_id="talk", occurred_at=datetime(2026, 1, 1)),)))
    assert len(result.sealed_event_ids) == 1
    stored = pipe.event_repo.get(result.sealed_event_ids[0])
    assert stored.content_raw == first and stored.content_raw.startswith("甲：")
    residual = pipe.meta_repo.get_unclosed_events()
    assert len(residual) == 1 and residual[0].merged_content.startswith("乙：")


def test_unsplittable_event_is_sealed_whole(case):
    pipe, _, llm = case
    first = "甲：" + "啊" * 250 + "。"
    second = "乙：" + "哦" * 250 + "。"
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    llm.push_response({"abort": True})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=first + second, objects={"甲": "jia", "乙": "yi"},
            source_ids=("whole",), segment_id="whole", occurred_at=datetime(2026, 1, 1)),)))
    assert len(result.sealed_event_ids) == 1
    stored = pipe.event_repo.get(result.sealed_event_ids[0])
    assert stored.seal_reason == "closed"
    assert stored.content_raw.startswith("甲：") and "乙：" in stored.content_raw
    assert not pipe.meta_repo.get_unclosed_events()


def test_cut_inside_one_turn_is_rejected(case):
    pipe, _, llm = case
    first = "甲：" + "啊" * 250 + "。"
    second = "乙：" + "哦" * 250 + "。"
    prefix = first[:120]
    tail = first[120:] + "\n" + second
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    llm.push_response({"prefix_text": prefix, "tail_text": tail})
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(
        MemoryExperience(subject_id="stone", text=first + second, objects={"甲": "jia", "乙": "yi"},
            source_ids=("mid",), segment_id="mid", occurred_at=datetime(2026, 1, 1)),)))
    assert len(result.sealed_event_ids) == 1
    stored = pipe.event_repo.get(result.sealed_event_ids[0])
    assert stored.content_raw.startswith("甲：") and "乙：" in stored.content_raw
    assert not pipe.meta_repo.get_unclosed_events()


def _seed_residual(pipe, text, *, when, identity):
    pipe.meta_repo.save_unclosed_event(UnclosedEvent(
        id=identity, subject_id="stone", content_fragments=[text],
        buffer_items=[BufferSentence(text=text, residual_id=identity)],
        created_at=when, updated_at=when, last_hit_time=when,
    ))


def test_short_residual_waits_three_days_and_seals_at_seven(case):
    pipe, _, llm = case
    now = datetime(2026, 6, 1, 12, 0, 0)
    short = "我们去湖边散步，聊得很开心。"
    _seed_residual(pipe, short, when=now - timedelta(days=4), identity="UC-short")
    llm.push_response({"events": [{"indices": [1]}], "residual": [{"indices": [2]}]})
    memory = MemoryExperience(subject_id="stone", text="然后呢。", objects={"甲": "jia"},
        source_ids=("next",), segment_id="next", occurred_at=now)
    waiting = pipe.metabolism_service.process_input("然后呢。", memory=memory)
    assert waiting == []
    held = "\n".join(item.merged_content for item in pipe.meta_repo.get_unclosed_events())
    assert short in held and "然后呢" in held

    pipe.meta_repo.delete_unclosed_event("UC-short")
    for item in pipe.meta_repo.get_unclosed_events():
        pipe.meta_repo.delete_unclosed_event(item.id)
    _seed_residual(pipe, short, when=now - timedelta(days=8), identity="UC-week")
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    llm.push_response({"events": [], "residual": [{"indices": [1]}]})
    sealed = pipe.metabolism_service.process_input("接着说。", memory=memory.model_copy(
        update={"text": "接着说。", "occurred_at": now}))
    assert [event.content_raw for event in sealed] == [short]
    assert sealed[0].seal_reason == "truncated"
    leftover = "\n".join(item.merged_content for item in pipe.meta_repo.get_unclosed_events())
    assert short not in leftover and "接着说" in leftover


def test_unsplittable_residual_over_check_line_seals_after_three_idle_days(case):
    pipe, _, llm = case
    now = datetime(2026, 6, 1, 12, 0, 0)
    long = "丙：" + "嗯" * 600 + "。"
    _seed_residual(pipe, long, when=now - timedelta(days=4), identity="UC-long")
    llm.push_response({"events": [], "residual": [{"indices": [1]}]})
    memory = MemoryExperience(subject_id="stone", text="然后呢。", objects={"丙": "bing"},
        source_ids=("later",), segment_id="later", occurred_at=now)
    sealed = pipe.metabolism_service.process_input("然后呢。", memory=memory)
    assert len(sealed) == 1 and sealed[0].content_raw == long
    assert sealed[0].seal_reason == "truncated" and sealed[0].content_raw.startswith("丙：")
    leftover = pipe.meta_repo.get_unclosed_events()
    assert len(leftover) == 1 and leftover[0].merged_content == "然后呢。"


def test_resave_reads_sequence_from_storage(case):
    pipe, _, _ = case
    event = small(pipe, "甲说完了这件事。", datetime(2026, 1, 1))
    assert event.memory_sequence == 1
    event.memory_sequence = None
    pipe.event_repo.save(event)
    assert pipe.event_repo.get(event.event_id).memory_sequence == 1
    other = small(pipe, "乙又说了另一件事。", datetime(2026, 1, 2))
    assert other.memory_sequence == 2


def test_sequence_counter_initializes_from_existing_maximum(case):
    pipe, _, _ = case
    preset = Event(subject_id="stone", content_raw="库里已有的事。", memory_stage=0,
                   memory_sequence=7, occurred_at=datetime(2026, 1, 1))
    pipe.event_repo.save(preset)
    assert pipe.event_repo.get(preset.event_id).memory_sequence == 7
    preset.memory_sequence = 1
    pipe.event_repo.save(preset)
    assert pipe.event_repo.get(preset.event_id).memory_sequence == 7
    nxt = Event(subject_id="stone", content_raw="后写入的事。", memory_stage=0,
                occurred_at=datetime(2026, 1, 2))
    pipe.event_repo.save(nxt)
    assert nxt.memory_sequence == 8


def test_representative_marks_every_child_including_one_without_names(case):
    pipe, service, llm = case
    now = datetime(2026, 1, 1)
    walk = pipe.event_service.seal_event(
        "我们去湖边散步，聊得很开心。", objects={}, subject_id="stone",
        source_ids=("walk",), occurred_at=now - timedelta(days=100))
    long = "我们走了" + "很" * 70 + "远，一共12公里。"
    distance = pipe.event_service.seal_event(
        long, objects={}, subject_id="stone", source_ids=("distance",),
        occurred_at=now - timedelta(days=99))
    responses(llm, 2, summary="出去了。")
    parent = pipe.event_repo.get(service.refine_once("stone", 1, now)["parent_ids"][0])
    summary = parent.summaries["L1"]
    assert f"〔#{walk.memory_sequence}〕" in summary
    assert f"〔#{distance.memory_sequence}〕" in summary
    assert "湖边" in summary and "出去了。" in summary
    units = {unit["event_id"]: unit for unit in parent.refinement_source_units}
    assert units[walk.event_id]["representative"] == "我们去湖边散步，聊得很开心。"
    assert len(units[distance.event_id]["representative"]) <= 60
    assert "12" in units[distance.event_id]["representative"]
    assert parent.refinement_policy["representative_overflow"] is True


def test_budget_clip_keeps_or_drops_attached_raw_as_a_whole():
    body = "摘要。"
    raw = "甲：" + "啊" * 40
    content = body + "\n\n原文 #1" + raw
    start = len(body) + 2
    fragment = RecalledFragment(
        event_id="parent", text=body, content=content, detail_content_start=start,
        detail_event_ids=["leaf"], detail_levels=["raw"],
    )
    whole, ids, levels = clip_fragment_content(fragment, len(content))
    assert whole == content and ids == ["leaf"] and levels == ["raw"]
    dropped, ids, levels = clip_fragment_content(fragment, start + 4)
    assert dropped == body and ids == [] and levels == []
    assert "啊" not in dropped
    earlier, ids, levels = clip_fragment_content(fragment, len(body))
    assert earlier == body and ids == [] and levels == []


def test_date_anchor_locale_is_consistent():
    from rems.recall.pipeline import source_date_anchor
    from rems.config import RecallDetailConfig
    assert RecallDetailConfig().date_anchor_locale == "zh"
    assert RecallDetailConfig().min_similarity is None
    assert "来源日期：2024-06-01" in source_date_anchor(["2024-06-01"], "zh")
    assert "Source statement date: 2024-06-01" in source_date_anchor(["2024-06-01"], "en")
    assert "不能折算" in source_date_anchor(["2024-06-01", "2024-06-03"], "zh")
    assert "ambiguous" in source_date_anchor(["2024-06-01", "2024-06-03"], "en")
