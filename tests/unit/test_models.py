"""Unit tests for Pydantic data models."""

from rems.models import (
    BasicEmotionVector,
    ContextPackage,
    EmotionalModel,
    Event,
    EventRoleEntry,
    EventStatus,
    Importance,
    RecallBlock,
    RecallItem,
    Role,
    Shadow,
    UnclosedEvent,
    WhitePaintingEntry,
)


class TestEvent:
    def test_defaults(self):
        e = Event(content_raw="Hello world")
        assert e.event_id.startswith("EVT-")
        assert e.event_length == 11
        assert e.is_abstract is False
        assert e.status == EventStatus.ACTIVE

    def test_summaries_mid_key(self):
        e = Event(
            content_raw="x",
            summaries={"L1": "a", "L2": "b", "L3": "c"},
        )
        assert e.mid_summary_key == "L2"

    def test_role_entry(self):
        entry = EventRoleEntry(role_id="ROL-test", importance=Importance.S)
        assert entry.importance == Importance.S
        assert entry.emotional_model.emotion.joy == 0.0

    def test_serialization_roundtrip(self):
        e = Event(content_raw="data", summaries={"L1": "d"})
        data = e.model_dump(mode="json")
        e2 = Event(**data)
        assert e2.event_id == e.event_id
        assert e2.summaries == {"L1": "d"}


class TestRole:
    def test_defaults(self):
        r = Role(name="Alice")
        assert r.role_id.startswith("ROL-")
        assert r.entity_type == "person"

    def test_white_painting(self):
        wp = WhitePaintingEntry(event_id="EVT-1", role_summary="did something")
        r = Role(name="Bob", white_painting=[wp])
        assert len(r.white_painting) == 1


class TestMetabolism:
    def test_shadow_length(self):
        s = Shadow(content="abc")
        assert s.length == 3

    def test_unclosed_merged(self):
        ue = UnclosedEvent(id="UC-1", content_fragments=["hello", "world"])
        assert ue.merged_content == "hello\nworld"
        assert ue.total_length == 10

    def test_recall_block(self):
        rb = RecallBlock(items=[
            RecallItem(event_id="E1", content="short"),
            RecallItem(event_id="E2", content="longer text"),
        ])
        rb.recompute_length()
        assert rb.total_length == len("short") + len("longer text")

    def test_context_package_assemble(self):
        cp = ContextPackage(
            shadow=Shadow(content="shadow"),
            current_input="input",
        )
        text = cp.assemble()
        assert "shadow" in text
        assert "input" in text


class TestEmotionalModel:
    def test_defaults_zero(self):
        em = EmotionalModel()
        assert em.emotion.joy == 0.0
        assert em.arousal == 0.0
        assert em.valence == 0.0

    def test_custom_values(self):
        em = EmotionalModel.from_emotion(BasicEmotionVector(joy=0.8, anger=0.5))
        assert em.emotion.joy == 0.8
        assert em.emotion.anger == 0.5
        assert em.arousal == 0.8
        assert em.valence > 0.0
