import pytest

from voicekit.matching import MatchingAlgo, to_phonetic_skeleton


def test_skeleton_collapses_vowels_and_classes_consonants():
    assert to_phonetic_skeleton("list") == "423"
    assert to_phonetic_skeleton("show") == "2"


def test_skeleton_matches_near_homophones():
    assert to_phonetic_skeleton("list") == to_phonetic_skeleton("least")
    assert to_phonetic_skeleton("show") == to_phonetic_skeleton("go")


def test_levenshtein_identical_strings_score_100():
    matcher = MatchingAlgo("levenshtein")
    assert matcher.score("show", "show") == 100.0


def test_levenshtein_unrelated_strings_score_low():
    matcher = MatchingAlgo("levenshtein")
    assert matcher.score("show", "zzzzz") == 0.0


def test_phonetic_skeleton_algorithm_scores_homophones_highly():
    matcher = MatchingAlgo("phonetic_skeleton")
    assert matcher.score("list", "least") == 100.0


def test_fuzzy_best_match_finds_close_candidate():
    matcher = MatchingAlgo("fuzzy")
    assert matcher.best_match("eko", ["echo", "print", "list"], cutoff=50) == "echo"


def test_fuzzy_best_match_respects_cutoff():
    matcher = MatchingAlgo("fuzzy")
    assert matcher.best_match("xyzxyz", ["echo", "print", "list"], cutoff=50) is None


def test_best_match_empty_candidates_returns_none():
    matcher = MatchingAlgo("levenshtein")
    assert matcher.best_match("echo", [], cutoff=0) is None


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        MatchingAlgo("not-a-real-algo")


def test_word_error_rate_identical_sentences_score_100():
    matcher = MatchingAlgo("word_error_rate")
    assert matcher.score("hello world", "hello world") == 100.0


def test_word_error_rate_single_word_substitution():
    matcher = MatchingAlgo("word_error_rate")
    assert matcher.score("the quick brown fox jumps", "the quick brown fox jump") == 80.0


def test_word_error_rate_scores_lower_than_levenshtein_for_word_level_errors():
    # A single word substitution is a big jump at word-level but a small one
    # character-level (only the last letter differs).
    wer = MatchingAlgo("word_error_rate")
    lev = MatchingAlgo("levenshtein")
    sentence_a = "the quick brown fox jumps"
    sentence_b = "the quick brown fox jump"
    assert wer.score(sentence_a, sentence_b) < lev.score(sentence_a, sentence_b)


def test_word_error_rate_best_match():
    matcher = MatchingAlgo("word_error_rate")
    candidates = ["the quick brown fox jump", "a totally different sentence"]
    assert matcher.best_match("the quick brown fox jumps", candidates, cutoff=50) == candidates[0]
    assert matcher.best_match("completely unrelated words here", candidates, cutoff=50) is None
