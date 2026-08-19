"""Tests for Chinese normalization used in focus-role matching."""

from __future__ import annotations

from rems.utils.zh_normalize import normalize_for_substring_match, to_simplified


class TestToSimplified:
    def test_hongloumeng_traditional_names(self):
        assert "宝玉" in to_simplified("寶玉")
        assert "凤姐" in to_simplified("鳳姐")
        assert "秦钟" in to_simplified("秦鐘")

    def test_normalize_for_match_lowercases_latin(self):
        assert normalize_for_substring_match("Hello") == "hello"

    def test_empty(self):
        assert to_simplified("") == ""
