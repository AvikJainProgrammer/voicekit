"""Minimal live phonetics demo, built on voicekit.

The simplest possible use of PhoneticTranslator on its own: continuously
listens and prints the IPA phonetic transcription of each utterance you
speak. No word transcription, no matching/scoring.

Run with: python examples/live_phonetics_demo.py
"""

from voicekit import PhoneticTranslator, VoiceRecorder


def main():
    print("Loading models (first run downloads them)...")
    recorder = VoiceRecorder()
    phonetics = PhoneticTranslator()
    print("Ready. Speak now! (Ctrl+C to stop)\n")

    for audio in recorder.listen():
        ipa = phonetics.convert(audio)
        if ipa:
            print(f"Phonetics: {ipa}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
