import pytest

from rems.recall.pipeline import lexical_score


def test_long_english_text_cannot_win_from_letter_fragments():
    query = "Which classical musicians does Melanie enjoy listening to?"
    unrelated = ("Caroline and Melanie talk about adoption, creating a loving family, "
                 "self acceptance, helping children, and supporting friends. ") * 20
    relevant = "Melanie enjoys classical music such as Bach and Mozart."
    bad = lexical_score(query, unrelated, english_words=True)
    good = lexical_score(query, relevant, english_words=True)
    assert bad == pytest.approx(1/5)
    assert good > bad


def test_function_words_and_partial_word_collisions_do_not_match():
    assert lexical_score("What did the dog do?", "The cat did sit.", english_words=True) == 0
    assert lexical_score("art", "a party", english_words=True) == 0
    assert lexical_score("books", "book", english_words=True) == 1


def test_possessive_name_is_not_full_topic_match():
    assert lexical_score("What do Melanie's kids like?", "Melanie likes pottery.", english_words=True) < 1


@pytest.mark.parametrize("query,text", [("去年小明的维修费", "去年小明花了680元维修"),
    ("lux的维修费", "lux谈到了自己的朋友"), ("2023", "发生于2023年")])
def test_chinese_mixed_and_number_queries_keep_existing_behavior(query, text):
    assert lexical_score(query, text, english_words=True) == lexical_score(query, text)
