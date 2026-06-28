"""German pronunciation flashcards, built on voicekit.

Port of the original all_voice2.py prototype: speak each phrase until
your pronunciation scores high enough, then move to the next card.

Run with: python examples/flashcards_german.py
"""

from voicekit import LanguageTranslator, MatchingAlgo, PhoneticTranslator, VoiceRecorder

FLASHCARDS = [
    {"phrase": "Guten Tag", "meaning": "Good day / Hello", "target_ipa": "ɡuːtntaːk"},
    {"phrase": "Danke", "meaning": "Thank you", "target_ipa": "daŋkə"},
    {"phrase": "Ja", "meaning": "Yes", "target_ipa": "jaː"},
    {"phrase": "Auf Wiedersehen", "meaning": "Goodbye", "target_ipa": "aʊfviːdɐzeːən"},
]

MATCH_THRESHOLD = 55.0


def main():
    print("Loading models (first run downloads them)...")
    recorder = VoiceRecorder(silence_timeout=0.8)
    translator = LanguageTranslator(lang="german")
    phonetics = PhoneticTranslator()
    matcher = MatchingAlgo(algorithm="levenshtein")
    print("Ready.\n")

    for i, card in enumerate(FLASHCARDS, 1):
        print("=" * 50)
        print(f"FLASHCARD {i}/{len(FLASHCARDS)}: {card['phrase']} ({card['meaning']})")
        print(f"Target IPA: {card['target_ipa']}")

        while True:
            print(">>> Speak now...")
            audio = recorder.record()
            if audio.size == 0:
                print("(nothing heard, try again)\n")
                continue

            words = translator.convert(audio)
            ipa = phonetics.convert(audio)
            score = matcher.score(ipa, card["target_ipa"])

            print(f"Heard words:         {words}")
            print(f"Heard phonetics:     {ipa}")
            print(f"Pronunciation match: {score:.1f}%")

            if score >= MATCH_THRESHOLD:
                print("Correct! Brilliant.\n")
                break
            print("Try again! Make sure to enunciate clearly.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting flashcard app. Tschüss!")
