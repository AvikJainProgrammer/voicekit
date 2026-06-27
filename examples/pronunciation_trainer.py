"""Pronunciation trainer example built on voicekit.

Run with: python examples/pronunciation_trainer.py
Speak the target phrase; the script records, transcribes the words,
converts to IPA phonetics, and scores your pronunciation against the
target IPA.
"""

from voicekit import LanguageTranslator, MatchingAlgo, PhoneticTranslator, VoiceRecorder

FLASHCARDS = [
    {"phrase": "Guten Tag", "meaning": "Good day / Hello", "target_ipa": "ɡuːtntaːk"},
    {"phrase": "Danke", "meaning": "Thank you", "target_ipa": "daŋkə"},
    {"phrase": "Ja", "meaning": "Yes", "target_ipa": "jaː"},
    {"phrase": "Auf Wiedersehen", "meaning": "Goodbye", "target_ipa": "aʊfviːdɐzeːən"},
]


def main():
    print("Loading models (first run downloads them)...")
    recorder = VoiceRecorder()
    translator = LanguageTranslator(lang="german")
    phonetics = PhoneticTranslator()
    matcher = MatchingAlgo(algorithm="levenshtein")
    print("Ready.\n")

    for card in FLASHCARDS:
        print("=" * 50)
        print(f"Say: {card['phrase']}  ({card['meaning']})")
        print(f"Target IPA: {card['target_ipa']}")
        print(">>> Speak now...")

        audio = recorder.record()
        if audio.size == 0:
            print("(nothing heard, skipping)\n")
            continue

        words = translator.convert(audio)
        ipa = phonetics.convert(audio)
        score = matcher.score(ipa, card["target_ipa"])

        print(f"Heard words:         {words}")
        print(f"Heard phonetics:     {ipa}")
        print(f"Pronunciation match: {score:.1f}%")
        print("Correct!" if score >= 55.0 else "Try again.")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
