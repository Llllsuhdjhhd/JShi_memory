"""LoCoMo 适配层纯函数。"""

from datetime import datetime, timedelta

from rems.eval import locomo as L

SAMPLE = {
    "sample_id": "conv-x",
    "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hi!", "blip_caption": "a dog"},
        ],
        "session_2_date_time": "7:55 pm on 9 June, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Back."}],
    },
    "qa": [],
}


def test_parse_datetime():
    assert L.parse_locomo_datetime("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)
    assert L.parse_locomo_datetime("12:09 am on 13 September, 2023") == datetime(2023, 9, 13, 0, 9)


def test_session_batch_shape():
    s1 = L.iter_sessions(SAMPLE)[0]
    b = L.session_batch(SAMPLE, s1)
    (e,) = b.experiences
    assert b.subject_id == e.subject_id == "conv-x"
    assert e.text == "Caroline: Hey Mel!\nMelanie: Hi! [shares an image: a dog]"
    assert e.objects == {"Melanie": "conv-x:Melanie"}  # speaker_a = 主体，不进对象表
    assert e.source_ids == ("D1:1", "D1:2")
    assert e.segment_id == "conv-x_session_1"
    assert e.occurred_at == datetime(2023, 5, 8, 13, 56)


def test_time_reverse_lookup_and_hits():
    ss = L.iter_sessions(SAMPLE)
    m = L.map_event_to_session(ss, "E1", datetime(2023, 5, 8, 13, 58), ["D1:2", "D2:1"])
    assert m.matched and m.session == 1 and m.dia_ids == ["D1:2"]
    miss = L.map_event_to_session(ss, "E2", datetime(2023, 5, 8, 14, 30), ["D1:1"],
                                  tolerance=timedelta(minutes=5))
    assert not miss.matched and miss.session is None
    h = L.evidence_hits([m, miss], ["D1:1"])
    assert h["session_hit"] is True and h["dia_hit"] is False
    assert L.evidence_hits([m], ["D1:2; D2:1"])["dia_recall"] == 0.5


def test_scoring():
    assert L.score_answer(4, "7 May 2023", "7 May 2023") == 1.0
    assert L.score_answer(5, "Not mentioned in the conversation", "x") == 1.0
    assert L.score_answer(1, "hiking, painting", "painting, hiking") == 1.0
    assert 0 < L.token_f1("the big dog", "dog") < 1


def test_soft_trigger_chunks_on_turn_boundary_and_stays_in_session():
    turns = [{"speaker": "A", "dia_id": f"D1:{i}", "text": "x" * 100 + "."} for i in range(1, 31)]
    s = L.Session("conv-x", 1, datetime(2023, 1, 1), "", turns)
    batches = L.session_batches({"sample_id": "conv-x", "conversation": {"speaker_a": "A", "speaker_b": "B"}},
                                s, subject_id="A", soft_chars=1000)
    assert len(batches) > 1
    ids = [d for b in batches for d in b.experiences[0].source_ids]
    assert ids == [t["dia_id"] for t in turns]  # 不丢不重、不跨 session
    for b in batches:
        e = b.experiences[0]
        assert e.subject_id == "A" and e.segment_id.startswith("conv-x_session_1")
        assert all(line.endswith(".") for line in e.text.split("\n"))  # 整轮，不断句
    assert L.sample_of_segment(batches[1].experiences[0].segment_id) == "conv-x"


def test_cap_context_backs_off_to_sentence():
    text = "One sentence here. Two sentence here. Three sentence here."
    out = L.cap_context(text, 45)
    assert out == "One sentence here. Two sentence here."
    assert L.cap_context("short", 3000) == "short"


def test_lookup_restricted_to_source_sample():
    ss = L.iter_sessions(SAMPLE)
    m = L.map_event_to_session(ss, "E", datetime(2023, 5, 8, 13, 56), ["D1:1"], sample_id="other")
    assert not m.matched


def test_locate_turns_line_then_sentence_then_source_ids():
    s1 = L.iter_sessions(SAMPLE)[0]
    raw = "Caroline: Hey Mel!\nMelanie: Hi! [shares an image: a dog]"
    assert L.locate_turns(s1, "Melanie: hi!  [shares an image: a dog]", ["D1:1", "D1:2"]) == (["D1:2"], "line")
    turns = [{"speaker": "A", "dia_id": "D1:1", "text": "I went to the support group yesterday. It was great."},
             {"speaker": "B", "dia_id": "D1:2", "text": "Nice."}]
    s = L.Session("c", 1, datetime(2023, 1, 1), "", turns)
    assert L.locate_turns(s, "A: I went to the support group yesterday.", ["D1:1", "D1:2"]) == (["D1:1"], "sentence")
    assert L.locate_turns(s, "summary only", ["D1:2"]) == (["D1:2"], "source_ids")
    assert raw  # silence unused


def test_three_tier_hits():
    ss = L.iter_sessions(SAMPLE)
    m = L.map_event_to_session(ss, "E", datetime(2023, 5, 8, 13, 56), ["D1:1", "D1:2"],
                               content_raw="Caroline: Hey Mel!")
    assert m.turn_dia_ids == ["D1:1"] and m.turn_method == "line"
    h = L.evidence_hits([m], ["D1:2"])
    assert h["session_hit"] and h["chunk_hit"] and not h["turn_hit"]
    s = L.summarize([{"category": 4, "f1": 1.0, "evidence_hits": h, "recalled": [m.__dict__]}])
    assert s["evidence"]["turn_hit_rate"] == 0.0 and s["evidence"]["chunk_hit_rate"] == 1.0


def test_answer_prompt_styles():
    kw = dict(a="A", b="B", memories="[1] x", question="Q?")
    strict = L.build_answer_prompt("strict", **kw)
    lenient = L.build_answer_prompt("english_lenient", **kw)
    assert "reply exactly: Not mentioned" in strict and "English" not in strict
    assert "Always answer in English" in lenient and "truly no evidence" in lenient
    assert lenient.endswith("Question: Q?\nShort answer:")
    import pytest
    with pytest.raises(ValueError):
        L.build_answer_prompt("bogus", **kw)
