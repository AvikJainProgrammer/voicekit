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
