from datetime import datetime

import pytest

from rems.models.event import Event
from rems.models.metabolism import BufferSentence, Shadow, UnclosedEvent
from rems.port import MemoryExperience, SubSegment
from rems.services.metabolism_service import _new_buffer_items
from rems.skills.boundary_detection import BoundaryResult, CompletedFragment, NewUnclosed
from rems.skills.event_enrichment import EventEnrichmentSkill
from .test_metabolism_service import metabolism_service


def test_rejudge_seals_source_once(metabolism_service, fake_llm, monkeypatch):
    service = metabolism_service
    text = "Melanie went running."
    line = UnclosedEvent(id="UC-one", content_fragments=[text])
    service._repo.save_unclosed_event(line)
    fake_llm.push_response({"events": [{"indices": [1]}], "residual": []})
    seen = []
    monkeypatch.setattr(service._event_svc, "seal_event",
        lambda content, **kwargs: seen.append(content) or Event(content_raw=content))
    result = service._rejudge_abandoned([line], memory=None, input_id=None,
        known_roles_hint=None, pre_role_entries=None)
    assert len(result) == 1 and seen == [text]
    assert service._repo.get_unclosed_events() == []


def test_force_save_splits_losslessly_without_duplicate_aggregate(metabolism_service, monkeypatch):
    service = metabolism_service
    service._config.event_min_chars = 900
    text = "x" * 4100
    line = UnclosedEvent(id="UC-long", content_fragments=[text])
    service._repo.save_unclosed_event(line)
    seen = []
    monkeypatch.setattr(service._event_svc, "seal_event",
        lambda content, **kwargs: seen.append(content) or Event(content_raw=content))
    result = service._force_save_all(text, "", [line], memory=None, input_id=None,
        known_roles_hint=None, pre_role_entries=None)
    assert len(result) == 3
    assert "".join(seen) == text
    assert all(len(part) < 1800 for part in seen)
    assert service._repo.get_unclosed_events() == []


def test_long_residual_and_overlapping_groups_keep_unique_source_ownership(metabolism_service):
    service = metabolism_service
    service._config.event_min_chars = 900
    buffer = [BufferSentence(text=str(i) * 600) for i in range(1, 7)]
    result = service._bound_groups(BoundaryResult(disentangle=True,
        completed_events=[CompletedFragment(content_raw="", source_indices=[1, 2]),
                          CompletedFragment(content_raw="", source_indices=[2, 3])],
        new_unclosed=[NewUnclosed(content="", source_indices=[4, 5, 6])]), buffer)
    indices = [i for group in [*result.completed_events, *result.new_unclosed] for i in group.source_indices]
    assert sorted(indices) == list(range(1, 7))
    assert all(len(group.content_raw) < 1800 for group in result.completed_events)


def test_source_timestamp_survives_buffer_and_sealing(metabolism_service):
    date = datetime(2023, 7, 3)
    memory = MemoryExperience(subject_id="test", text="Lee: I signed up yesterday.",
        sub_segments=[SubSegment(segment_id="s1", text="Lee: I signed up yesterday.",
            occurred_at=date, source_ids=("D5:4",))])
    buffer = _new_buffer_items(memory.text, memory)
    assert buffer[0].occurred_at == date
    kwargs = metabolism_service._seal_memory_kwargs(memory, input_id=None,
        known_roles_hint=None, pre_role_entries=None, buffer_items=buffer)
    assert kwargs["occurred_at"] == date
    assert kwargs["source_dates"] == [date.isoformat()]
    assert kwargs["source_ids"] == ("D5:4",)


def test_source_dates_persist_in_event_metadata(metabolism_service, fake_llm):
    service = metabolism_service._event_svc
    fake_llm.push_response({"summaries": {"L1": "Lee signed up yesterday."}, "objects": []})
    event = service.seal_event("Lee signed up for pottery yesterday.", subject_id="test",
        objects={}, source_ids=("D5:4",), occurred_at=datetime(2023, 7, 3),
        source_dates=["2023-07-03T00:00:00"])
    stored = service._event_repo.get(event.event_id)
    assert stored.source_dates == ["2023-07-03T00:00:00"]
    assert stored.occurred_at == datetime(2023, 7, 3)


def test_buffer_dates_roundtrip_through_both_persistent_buffers(metabolism_service):
    date = datetime(2023, 7, 3, 12, 30)
    item = BufferSentence(text="Lee signed up yesterday.", occurred_at=date,
        source_ids=("D5:4",), segment_id="s1")
    repo = metabolism_service._repo
    repo.update_shadow(Shadow(content=item.text, buffer_items=[item]))
    assert repo.get_shadow().buffer_items[0].occurred_at == date
    repo.save_unclosed_event(UnclosedEvent(id="UC-date", content_fragments=[item.text], buffer_items=[item]))
    restored = repo.get_unclosed_event("UC-date").buffer_items[0]
    assert restored.occurred_at == date and restored.source_ids == ("D5:4",)


def test_summary_repairs_oversize_l1_once(config, fake_llm, metabolism_service):
    budget = metabolism_service._event_svc._compute_budget(1000)
    fake_llm.push_response({"summaries": {"L1": "A" * 1000}, "objects": []})
    fake_llm.push_response({"summaries": {"L1": "Lee runs daily."}, "objects": []})
    result = EventEnrichmentSkill(fake_llm, config).enrich("Lee runs daily." * 80,
        budget=budget, memory_objects={})
    assert result.summaries == {"L1": "Lee runs daily."}
    assert len(fake_llm.calls) == 2


def test_summary_rejects_failed_repair(config, fake_llm, metabolism_service):
    budget = metabolism_service._event_svc._compute_budget(1000)
    for _ in range(2):
        fake_llm.push_response({"summaries": {"L1": "A" * 1000}, "objects": []})
    with pytest.raises(ValueError, match="after bounded repairs"):
        EventEnrichmentSkill(fake_llm, config).enrich("A" * 1000, budget=budget, memory_objects={})


def test_summary_packs_complete_specific_facts_without_clipping(config, fake_llm, metabolism_service):
    budget = metabolism_service._event_svc._compute_budget(1000)
    limit = budget.summary_level_budgets["L1"]
    facts = ["A" * (limit + 1), "Lee likes Bach and Mozart.", "Lee plays clarinet."]
    fake_llm.push_response({"summaries": {"L1": "A" * 1000}, "objects": []})
    fake_llm.push_response({"event_facts": facts, "objects": []})
    result = EventEnrichmentSkill(fake_llm, config).enrich("A" * 1000, budget=budget, memory_objects={})
    assert result.summaries["L1"] == "Lee likes Bach and Mozart. Lee plays clarinet."
    assert len(result.summaries["L1"]) <= limit


def test_short_caption_gets_bounded_measured_feedback(config, fake_llm, metabolism_service):
    budget = metabolism_service._event_svc._compute_budget(70)
    fake_llm.push_response({"summaries": {"L1": "A" * 100}, "objects": []})
    fake_llm.push_response({"event_facts": ["A" * 100], "objects": []})
    fake_llm.push_response({"event_facts": ["Cross-and-heart necklace."], "objects": []})
    result = EventEnrichmentSkill(fake_llm, config).enrich("A" * 70, budget=budget, memory_objects={})
    assert result.summaries["L1"] == "Cross-and-heart necklace."
    assert len(fake_llm.calls) == 3


def test_source_identical_retry_reuses_event_but_new_source_does_not(metabolism_service, fake_llm):
    service = metabolism_service._event_svc
    args = dict(subject_id="test", objects={}, source_ids=("D1:1",))
    fake_llm.push_response({"summaries": {"L1": "Lee runs."}, "objects": []})
    first = service.seal_event("Lee runs each morning.", **args)
    second = service.seal_event("Lee runs each morning.", **args)
    assert first.event_id == second.event_id
    fake_llm.push_response({"summaries": {"L1": "Lee runs."}, "objects": []})
    third = service.seal_event("Lee runs each morning.", **{**args, "source_ids": ("D2:1",)})
    assert third.event_id != first.event_id


def test_default_jshi_triggers_use_history_clock_and_all_pending():
    from pathlib import Path
    from rems.eval.runtime import load_jshi
    sibling = Path(__file__).resolve().parents[3] / "JShi"
    if not sibling.is_dir():
        pytest.skip("Adjacent JShi checkout required")
    load_jshi(sibling)
    from jshi.experienceledger import InProcessExperienceLedger
    from jshi.memorycontrol import InProcessMemoryControl
    ledger = InProcessExperienceLedger()
    date = datetime(2023, 7, 3, 12)
    control = InProcessMemoryControl(ledger, None, now=lambda: date)
    ledger.append_external("test", actor_object_id="lee", text_raw="short",
        objects={"Lee": "lee"}, source_ids=("1",), occurred_at=date)
    assert control.evaluate("test").should_trigger is False
    for i in range(19):
        ledger.append_external("test", actor_object_id="lee", text_raw="short",
            objects={"Lee": "lee"}, source_ids=(str(i+2),), occurred_at=date)
    assert control.evaluate("test").reason == "flush_max_segments"
    assert len(control.build_batch("test").experiences) == 20
