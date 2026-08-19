from __future__ import annotations

import pytest

from rems.config import REMSConfig, TriBandConfig
from rems.embedding.query_act_processor import (
    LLMQueryActProcessor,
    TruncateFallbackQueryActProcessor,
    build_query_act_processor,
)
from rems.skills.recall_query_compress import RecallQueryCompressSkill


class _FakeCompressSkill:
    def __init__(self, response: str = "压缩摘要"):
        self._response = response
        self.calls: list[str] = []

    def compress(self, text: str) -> str:
        self.calls.append(text)
        return self._response


def test_truncate_fallback_when_compress_disabled():
    cfg = REMSConfig()
    cfg.recall_query_compress_enabled = False
    proc = build_query_act_processor(cfg)
    long_text = "a" * 300
    result = proc.resolve_act_text(long_text)
    assert result.source == "truncate_fallback"
    assert len(result.act_text) == cfg.tri_band.query_act_max_chars


def test_llm_processor_passthrough_short_text():
    cfg = REMSConfig()
    cfg.recall_query_compress_min_chars = 150
    skill = _FakeCompressSkill()
    proc = LLMQueryActProcessor(skill, cfg)  # type: ignore[arg-type]
    result = proc.resolve_act_text("短文本")
    assert result.source == "passthrough"
    assert skill.calls == []


def test_llm_processor_compresses_long_text():
    cfg = REMSConfig()
    cfg.recall_query_compress_min_chars = 50
    skill = _FakeCompressSkill("张友士为秦氏诊脉；贾敬寿辰筹备")
    proc = LLMQueryActProcessor(skill, cfg)  # type: ignore[arg-type]
    result = proc.resolve_act_text("x" * 200)
    assert result.source == "llm_compress"
    assert "诊脉" in result.act_text
    assert len(skill.calls) == 1


def test_llm_processor_fallback_on_skill_error():
    cfg = REMSConfig()
    cfg.recall_query_compress_min_chars = 10

    class _FailSkill:
        def compress(self, text: str) -> str:
            raise RuntimeError("boom")

    proc = LLMQueryActProcessor(_FailSkill(), cfg)  # type: ignore[arg-type]
    long_text = "z" * 300
    result = proc.resolve_act_text(long_text)
    assert result.source == "truncate_fallback"


def test_build_without_skill_uses_truncate():
    cfg = REMSConfig()
    cfg.recall_query_compress_enabled = True
    proc = build_query_act_processor(cfg, skill=None)
    assert isinstance(proc, TruncateFallbackQueryActProcessor)


def test_chunk58_dual_theme_compress():
    cfg = REMSConfig()
    cfg.recall_query_compress_min_chars = 50
    skill = _FakeCompressSkill("张友士为秦氏诊脉开方；贾敬寿辰贾琏贾蔷看座")
    proc = LLMQueryActProcessor(skill, cfg)  # type: ignore[arg-type]
    long_text = ("诊脉段落" * 100) + ("寿辰段落" * 100)
    result = proc.resolve_act_text(long_text)
    assert result.source == "llm_compress"
    assert "张友士" in result.act_text or "诊" in result.act_text
    assert "贾敬" in result.act_text or "寿" in result.act_text
