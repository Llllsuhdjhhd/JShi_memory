import pytest
from pydantic import ValidationError

from rems.config import REMSConfig
from rems.llm.prompts import build_enrichment_system_message


def test_default_first_person_unchanged():
    msg = build_enrichment_system_message(REMSConfig(), fuse_min_chars=20, mode="memory")
    assert "摘要和对象事实用「我」的第一人称" in msg
    assert "视角（中立）" not in msg


def test_neutral_perspective_switch():
    cfg = REMSConfig(enrichment_perspective="neutral")
    msg = build_enrichment_system_message(cfg, fuse_min_chars=20, mode="memory")
    assert "摘要和对象事实用「我」的第一人称" not in msg
    assert "用「我」的视角写这个对象" not in msg
    assert "点名说话人" in msg and "视角（中立）" in msg


def test_enrichment_perspective_enum_validation():
    assert REMSConfig().enrichment_perspective == "first_person"
    assert REMSConfig(enrichment_perspective="neutral").enrichment_perspective == "neutral"
    with pytest.raises(ValidationError):
        REMSConfig(enrichment_perspective="third_person")
