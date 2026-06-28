"""String similarity / matching algorithms.

Unifies the three scoring approaches duplicated across the prototypes:
rapidfuzz Levenshtein (durga_names.py, all_voice2.py), difflib fuzzy
matching (voice_control.py, voice_shell_v2/v3.py), and the hand-rolled
consonant-skeleton trick (voice_shell_v3.py's get_phonetic_skeleton).
"""

import re
from difflib import get_close_matches

from rapidfuzz.distance import Levenshtein

ALGORITHMS = ("levenshtein", "phonetic_skeleton", "fuzzy", "word_error_rate")


def to_phonetic_skeleton(text: str) -> str:
    """Collapse a phrase into a string of phonetic sound-class digits.

    Strips vowels/glides and groups acoustically similar consonants into
    the same digit class, so homophone-ish mishearings still match.
    """
    text = text.lower().strip()
    text = re.sub(r"[aeiouwyh]", "", text)
    text = re.sub(r"[bfpv]", "1", text)
    text = re.sub(r"[cgjkqsxz]", "2", text)
    text = re.sub(r"[dt]", "3", text)
    text = re.sub(r"[l]", "4", text)
    text = re.sub(r"[mn]", "5", text)
    text = re.sub(r"[r]", "6", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text


class MatchingAlgo:
    """Scores similarity between two strings using a configurable algorithm."""

    def __init__(self, algorithm: str = "levenshtein"):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {ALGORITHMS}.")
        self.algorithm = algorithm

    def score(self, a: str, b: str) -> float:
        """Return a 0-100 normalized similarity score between two strings.

        For "word_error_rate", this is the word-level complement of WER
        (100 = no word errors), rather than character-level similarity.
        """
        a, b = a.lower().strip(), b.lower().strip()
        if self.algorithm == "phonetic_skeleton":
            a, b = to_phonetic_skeleton(a), to_phonetic_skeleton(b)
        elif self.algorithm == "word_error_rate":
            a, b = a.split(), b.split()
        return Levenshtein.normalized_similarity(a, b) * 100

    def best_match(self, text: str, candidates: list[str], cutoff: float = 0.0) -> str | None:
        """Return the candidate that best matches `text`, or None if below cutoff."""
        if not candidates:
            return None

        if self.algorithm == "fuzzy":
            matches = get_close_matches(text.lower(), candidates, n=1, cutoff=cutoff / 100)
            return matches[0] if matches else None

        best_candidate, best_score = None, -1.0
        for candidate in candidates:
            s = self.score(text, candidate)
            if s > best_score:
                best_candidate, best_score = candidate, s

        return best_candidate if best_score >= cutoff else None
