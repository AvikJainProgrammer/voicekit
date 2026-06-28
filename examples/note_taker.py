"""Multilingual note taker, built on voicekit.

Continuously listens, transcribes each utterance in the selected
language, and saves timestamped notes to notes_output/ on exit.

Usage:
    python examples/note_taker.py                 # defaults to English
    python examples/note_taker.py --lang german
    python examples/note_taker.py --lang auto      # auto-detect language per utterance

Controls:
    Ctrl+C  -> stop and save notes
"""

import argparse
import datetime
from pathlib import Path

from voicekit import LANGUAGES, LanguageTranslator, VoiceRecorder

OUTPUT_DIR = Path(__file__).parent / "notes_output"


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual note taker (voicekit example)")
    parser.add_argument(
        "--lang", "-l", default="english", metavar="LANGUAGE",
        help=(
            "Language to transcribe. Options: "
            f"{', '.join(sorted(set(v[1] for v in LANGUAGES.values())))}. "
            "Default: english"
        ),
    )
    return parser.parse_args()


def save_notes(notes: list, lang_code: str | None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / f"{lang_code or 'auto'}.txt"
    session_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n-- Session: {session_ts} --\n")
        for ts, result in notes:
            lang_tag = f"[{result.language}] " if lang_code is None else ""
            f.write(f"[{ts}] {lang_tag}{result.text}\n")
    print(f"\nSaved {len(notes)} notes -> {out_file}")


def main():
    args = parse_args()
    print("Loading models (first run downloads them)...")
    recorder = VoiceRecorder()
    translator = LanguageTranslator(lang=args.lang)
    print(f"Ready. Language: {translator.lang_display}. Speak freely, Ctrl+C to stop and save.\n")

    notes = []
    try:
        for audio in recorder.listen():
            result = translator.convert(audio)
            if not result.text:
                continue
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            notes.append((ts, result))
            print(f"[{ts}] {result}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if notes:
            save_notes(notes, translator.lang_code)
        else:
            print("No notes captured.")


if __name__ == "__main__":
    main()
