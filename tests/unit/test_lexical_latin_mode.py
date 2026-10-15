"""lexical_score 英文规则：默认去停用词词重叠率；中文不变；旧行为可开关。"""

import pytest

from rems.recall.pipeline import lexical_score


def test_overlap_default_not_always_one():
    # 旧规则下 "did"/"the" 命中即 1.0
    assert lexical_score("What did the dog do", "the cat did sit") == 0.0
    assert lexical_score("When did Caroline paint a sunset", "Caroline went hiking") == pytest.approx(1 / 3)
    assert lexical_score("Caroline paint sunset", "caroline painted a sunset") == pytest.approx(2 / 3)


def test_token_mode_keeps_legacy_behaviour():
    assert lexical_score("What did the dog do", "the cat sat", latin_mode="token") == 1.0


def test_off_mode_ignores_latin():
    assert lexical_score("Caroline paint", "Caroline went", latin_mode="off") < 1.0


def test_numbers_and_full_phrase_still_hit():
    assert lexical_score("in 2023 what", "since 2023") == 1.0
    assert lexical_score("support group", "the lgbtq support group met") == 1.0


@pytest.mark.parametrize("mode", ["overlap", "token", "off"])
def test_chinese_unchanged(mode):
    q, t = "小明喝水的温度", "我把水调到四十二度，小明说刚好"
    assert lexical_score(q, t, latin_mode=mode) == lexical_score(q, t, latin_mode="off")
    assert lexical_score("落日", "今天看到美丽的落日", latin_mode=mode) == 1.0


def test_empty():
    assert lexical_score("", "x") == 0.0
