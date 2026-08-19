from __future__ import annotations

import json

import pytest

from rems.config import REMSConfig
from rems.skills.recall_query_compress import RecallQueryCompressSkill, _format_writing_guide
from tests.conftest import FakeLLM


@pytest.fixture()
def compress_skill(config: REMSConfig, fake_llm: FakeLLM):
    return RecallQueryCompressSkill(fake_llm, config)


def test_compress_returns_act_query(compress_skill, fake_llm):
    fake_llm.push_response({"act_query": "诊" + "脉" * 199})
    out = compress_skill.compress("很长的原文" * 50)
    assert "诊" in out
    assert len(out) <= compress_skill._config.tri_band.query_act_compress_max_chars
    assert len(out) >= 200


def test_compress_clamps_overlong_llm_output(compress_skill, fake_llm):
    cap = compress_skill._config.tri_band.query_act_compress_max_chars
    fake_llm.push_response({"act_query": "x" * (cap + 100)})
    out = compress_skill.compress("input" * 100)
    assert len(out) == cap


def test_compress_user_includes_writing_guide(compress_skill):
    guide = _format_writing_guide(compress_skill._config, 875)
    assert "875" in guide
    assert "200" in guide
    assert "密度对照" in guide
    assert "并列叙事线" in guide


def test_compress_retry_note_template(compress_skill):
    from rems.llm.prompts import RECALL_QUERY_COMPRESS_RETRY_NOTE

    note = RECALL_QUERY_COMPRESS_RETRY_NOTE.format(
        prev_chars=79, min_chars=200, target_chars=200,
    )
    assert "79" in note
    assert "续写扩展" in note


def test_compress_raises_on_empty_response(compress_skill, fake_llm):
    fake_llm.push_response({"act_query": ""})
    with pytest.raises(ValueError):
        compress_skill.compress("input" * 100)


def test_compress_retries_when_too_short(compress_skill, fake_llm):
    fake_llm.push_response({"act_query": "太短了"})
    fake_llm.push_response({"act_query": "扩" * 200})
    out = compress_skill.compress("很长的原文" * 50)
    assert len(out) >= 200
    assert len(fake_llm.calls) == 2
