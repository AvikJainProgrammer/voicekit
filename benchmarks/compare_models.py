"""Manual model comparison harness for voicekit.

Speak each built-in test phrase once; that single recording is fanned out
to every configured phonetic model and every configured language model so
you can judge quality side by side, with take-to-take variance removed.

Models are loaded and evaluated ONE AT A TIME, never held in memory
together: all phrases are recorded up front (only the lightweight VAD
model needed for that), then each model is loaded, run against every
recording, and freed before the next model loads. Peak memory is the
size of the single largest model, not the sum of all of them.

Language-model accuracy for `lang`s that render in a non-Latin script
(e.g. Hindi -> Devanagari) is scored against `target_text_alt` when one
is provided in TEST_PHRASES, instead of being structurally floored at 0%
against a Latin-script `text` reference it could never match.

Usage:
    python benchmarks/compare_models.py
    python benchmarks/compare_models.py --repeats 3
    python benchmarks/compare_models.py --language-models tiny,large-v3
    python benchmarks/compare_models.py --phonetic-models facebook/wav2vec2-lv-60-espeak-cv-ft

Note: the default model sets download whatever isn't already cached
locally (a few GB combined on first run). Pass a smaller --language-models
list for a quick first pass.
"""

import argparse
import gc
import statistics
import sys
import time

import torch

from voicekit import LanguageTranslator, MatchingAlgo, PhoneticTranslator, VoiceRecorder

# IPA output (e.g. the "long" mark 'ː') isn't representable in the legacy
# codepages some Windows consoles default to — replace rather than crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_PHONETIC_MODELS = [
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
]
DEFAULT_LANGUAGE_MODELS = ["tiny", "small", "medium", "large-v3", "distil-large-v3"]

# target_text_alt: Whisper transcribes Hindi-coded audio in Devanagari script,
# which can never match a Latin-script `text` reference at the word level —
# that's a scoring artifact, not a model failure. Where we know the expected
# native-script rendering, it's scored too (character-level) and the better
# of the two is kept.
TEST_PHRASES = [
    {"lang": "english", "text": "the quick brown fox jumps over the lazy dog", "target_ipa": None, "target_text_alt": None},
    {"lang": "english", "text": "she sells seashells by the seashore", "target_ipa": None, "target_text_alt": None},
    {"lang": "german", "text": "Guten Tag", "target_ipa": "ɡuːtntaːk", "target_text_alt": None},
    {"lang": "german", "text": "Auf Wiedersehen", "target_ipa": "aʊfviːdɐzeːən", "target_text_alt": None},
    {"lang": "hindi", "text": "Durgā", "target_ipa": "dʊrɡaː", "target_text_alt": "दुर्गा"},
    {"lang": "hindi", "text": "Durgamā", "target_ipa": "dʊrɡəmaː", "target_text_alt": "दुर्गमा"},
]


def cuda_mem_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1e6


def release_gpu_memory():
    """Force Python's GC and CUDA's allocator to actually reclaim memory.

    Must be called AFTER the caller has dereferenced its model (e.g. `del
    model` or `model = None`) — a helper that takes `model` as a parameter
    and does `del model` inside its own frame only drops *that* frame's
    reference, not the caller's, so the object would stay alive.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record_all_phrases(recorder, repeats):
    """Record every test phrase (x repeats) before any comparison model is loaded."""
    recordings = []
    try:
        for run in range(repeats):
            if repeats > 1:
                print(f"\n########## Recording run {run + 1}/{repeats} ##########")
            for phrase in TEST_PHRASES:
                ipa_hint = f"  (target IPA: {phrase['target_ipa']})" if phrase["target_ipa"] else ""
                alt_hint = f"  (native script: {phrase['target_text_alt']})" if phrase.get("target_text_alt") else ""
                print(f"\n>>> Say: \"{phrase['text']}\"{ipa_hint}{alt_hint}")
                audio = recorder.record()
                if audio.size == 0:
                    print("(nothing heard, skipping this phrase)")
                    continue
                recordings.append({
                    "phrase": phrase,
                    "audio": audio,
                    "audio_duration": len(audio) / recorder.sample_rate,
                    "results": {"phonetic": [], "language": []},
                })
    except KeyboardInterrupt:
        print("\nStopped recording early — evaluating what was captured so far.")
    return recordings


def run_phonetic_models(model_ids, recordings, matcher_ipa, load_stats):
    """Load each phonetic model in turn, score it against every recording, then free it."""
    for model_id in model_ids:
        print(f"\nLoading phonetic model: {model_id} ...")
        model = None
        try:
            before = cuda_mem_mb()
            t0 = time.perf_counter()
            model = PhoneticTranslator(model_id=model_id)
            load_time = time.perf_counter() - t0
            after = cuda_mem_mb()
            load_stats.append((model_id, "phonetic", load_time, (after - before) if before is not None else None))

            for rec in recordings:
                phrase, audio = rec["phrase"], rec["audio"]
                t0 = time.perf_counter()
                ipa = model.convert(audio)
                latency = time.perf_counter() - t0
                target = phrase["target_ipa"]
                accuracy = matcher_ipa.score(ipa, target) if target else None
                completeness = (len(ipa) / len(target) * 100) if target else None
                rec["results"]["phonetic"].append({
                    "model": model_id, "output": ipa, "latency": latency,
                    "accuracy": accuracy, "completeness": completeness,
                })
        except Exception as e:
            print(f"  Skipping {model_id}: {e}")
        finally:
            if model is not None:
                del model
                release_gpu_memory()
                print(f"  Unloaded {model_id}.")


def run_language_models(model_names, recordings, matcher_wer, matcher_char, load_stats):
    """Load each language model in turn, score it against every recording, then free it."""
    for name in model_names:
        print(f"\nLoading language model: {name} ...")
        model = None
        try:
            before = cuda_mem_mb()
            t0 = time.perf_counter()
            model = LanguageTranslator(model=name)
            load_time = time.perf_counter() - t0
            after = cuda_mem_mb()
            load_stats.append((name, "language", load_time, (after - before) if before is not None else None))

            for rec in recordings:
                phrase, audio = rec["phrase"], rec["audio"]
                t0 = time.perf_counter()
                result = model.convert(audio, lang=phrase["lang"])
                latency = time.perf_counter() - t0
                rtf = latency / rec["audio_duration"] if rec["audio_duration"] > 0 else None

                accuracy = matcher_wer.score(result.text, phrase["text"])
                if len(phrase["text"].split()) == 1:
                    # Word-level WER is binary (0% or 100%) when the
                    # reference is a single word — there's no second token
                    # to share partial credit with. Fall back to
                    # character-level similarity so near-misses (e.g.
                    # "Surgama." vs "Durgamā") aren't floored at 0%.
                    accuracy = max(accuracy, matcher_char.score(result.text, phrase["text"]))
                alt_text = phrase.get("target_text_alt")
                if alt_text:
                    # Same reasoning, against the native-script reference.
                    accuracy = max(accuracy, matcher_char.score(result.text, alt_text))

                ref_words = phrase["text"].split()
                out_words = result.text.split()
                completeness = (len(out_words) / len(ref_words) * 100) if ref_words else None
                rec["results"]["language"].append({
                    "model": name, "output": result.text, "latency": latency, "rtf": rtf,
                    "accuracy": accuracy, "completeness": completeness, "confidence": result.confidence,
                })
        except Exception as e:
            print(f"  Skipping {name}: {e}")
        finally:
            if model is not None:
                del model
                release_gpu_memory()
                print(f"  Unloaded {name}.")


def print_loading_table(stats):
    print("\n=== Model Loading ===")
    header = f"{'Model':<45} {'Type':<10} {'Load (s)':>10} {'VRAM (MB)':>10}"
    print(header)
    print("-" * len(header))
    for name, kind, load_time, vram_delta in stats:
        vram_str = f"{vram_delta:.0f}" if vram_delta is not None else "n/a"
        print(f"{name:<45} {kind:<10} {load_time:>10.1f} {vram_str:>10}")


def print_phonetic_table(rows):
    if not rows:
        return
    print("\n-- Phonetic models --")
    header = f"{'Model':<42} {'Output':<30} {'Acc %':>7} {'Cov %':>7} {'Lat (s)':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc = f"{r['accuracy']:.1f}" if r["accuracy"] is not None else "n/a"
        cov = f"{r['completeness']:.0f}" if r["completeness"] is not None else "n/a"
        print(f"{r['model']:<42} {r['output'][:30]:<30} {acc:>7} {cov:>7} {r['latency']:>8.2f}")


def print_language_table(rows):
    if not rows:
        return
    print("\n-- Language models --")
    header = f"{'Model':<16} {'Output':<35} {'WER %':>7} {'Cov %':>7} {'Lat (s)':>8} {'RTF':>6} {'Conf':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc = f"{r['accuracy']:.1f}" if r["accuracy"] is not None else "n/a"
        cov = f"{r['completeness']:.0f}" if r["completeness"] is not None else "n/a"
        rtf = f"{r['rtf']:.2f}" if r["rtf"] is not None else "n/a"
        print(
            f"{r['model']:<16} {r['output'][:35]:<35} {acc:>7} {cov:>7} "
            f"{r['latency']:>8.2f} {rtf:>6} {r['confidence']:>7.2f}"
        )


def _fmt_stats(values):
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.1f}+/-{stdev:.1f}"


def aggregate_and_print(all_results):
    results = [r for r in all_results if r is not None]
    if not results:
        print("\nNo phrases were captured; nothing to aggregate.")
        return

    phonetic_agg, language_agg = {}, {}
    for result in results:
        for r in result["phonetic"]:
            agg = phonetic_agg.setdefault(r["model"], {"accuracy": [], "latency": []})
            if r["accuracy"] is not None:
                agg["accuracy"].append(r["accuracy"])
            agg["latency"].append(r["latency"])
        for r in result["language"]:
            agg = language_agg.setdefault(r["model"], {"accuracy": [], "latency": [], "rtf": []})
            agg["accuracy"].append(r["accuracy"])
            agg["latency"].append(r["latency"])
            if r["rtf"] is not None:
                agg["rtf"].append(r["rtf"])

    print(f"\n=== Aggregate across {len(results)} phrase(s) ===")
    print("\n-- Phonetic models (accuracy % | latency s) --")
    for model, agg in phonetic_agg.items():
        print(f"  {model:<42} accuracy {_fmt_stats(agg['accuracy']):>14}   latency {_fmt_stats(agg['latency']):>12}")

    print("\n-- Language models (WER-accuracy % | latency s | RTF) --")
    for model, agg in language_agg.items():
        print(
            f"  {model:<16} accuracy {_fmt_stats(agg['accuracy']):>14}   "
            f"latency {_fmt_stats(agg['latency']):>12}   RTF {_fmt_stats(agg['rtf']):>10}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple phonetic/language models on voicekit")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="How many times to repeat the full phrase set (default 1)",
    )
    parser.add_argument(
        "--phonetic-models", type=str, default=None,
        help="Comma-separated wav2vec2 model IDs to compare",
    )
    parser.add_argument(
        "--language-models", type=str, default=None,
        help="Comma-separated faster-whisper model sizes to compare",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phonetic_model_ids = (
        [m.strip() for m in args.phonetic_models.split(",")]
        if args.phonetic_models else DEFAULT_PHONETIC_MODELS
    )
    language_model_names = (
        [m.strip() for m in args.language_models.split(",")]
        if args.language_models else DEFAULT_LANGUAGE_MODELS
    )

    print("Recording phrases first (only the lightweight VAD model is loaded for this step)...")
    recorder = VoiceRecorder()
    recordings = record_all_phrases(recorder, args.repeats)
    del recorder
    release_gpu_memory()

    if not recordings:
        print("Nothing was recorded; exiting.")
        return

    matcher_ipa = MatchingAlgo(algorithm="levenshtein")
    matcher_wer = MatchingAlgo(algorithm="word_error_rate")
    load_stats = []

    print(
        f"\nEvaluating {len(recordings)} recording(s) against "
        f"{len(phonetic_model_ids)} phonetic model(s) and {len(language_model_names)} "
        "language model(s) — one model at a time, to keep memory usage low.\n"
    )
    run_phonetic_models(phonetic_model_ids, recordings, matcher_ipa, load_stats)
    run_language_models(language_model_names, recordings, matcher_wer, matcher_ipa, load_stats)

    print_loading_table(load_stats)
    for rec in recordings:
        print(f"\nPhrase: \"{rec['phrase']['text']}\"")
        print_phonetic_table(rec["results"]["phonetic"])
        print_language_table(rec["results"]["language"])

    aggregate_and_print([rec["results"] for rec in recordings])


if __name__ == "__main__":
    main()
