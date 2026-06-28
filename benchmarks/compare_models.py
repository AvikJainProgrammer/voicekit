"""Manual model comparison harness for voicekit.

Speak each built-in test phrase once; that single recording is fanned out
to every configured phonetic model and every configured language model so
you can judge quality side by side, with take-to-take variance removed.

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
import statistics
import time

import torch

from voicekit import LanguageTranslator, MatchingAlgo, PhoneticTranslator, VoiceRecorder

DEFAULT_PHONETIC_MODELS = [
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
]
DEFAULT_LANGUAGE_MODELS = ["tiny", "small", "medium", "large-v3", "distil-large-v3"]

TEST_PHRASES = [
    {"lang": "english", "text": "the quick brown fox jumps over the lazy dog", "target_ipa": None},
    {"lang": "english", "text": "she sells seashells by the seashore", "target_ipa": None},
    {"lang": "german", "text": "Guten Tag", "target_ipa": "ɡuːtntaːk"},
    {"lang": "german", "text": "Auf Wiedersehen", "target_ipa": "aʊfviːdɐzeːən"},
    {"lang": "hindi", "text": "Durgā", "target_ipa": "dʊrɡaː"},
    {"lang": "hindi", "text": "Durgamā", "target_ipa": "dʊrɡəmaː"},
]


def cuda_mem_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1e6


def load_phonetic_models(model_ids):
    models, stats = {}, []
    for model_id in model_ids:
        before = cuda_mem_mb()
        t0 = time.perf_counter()
        models[model_id] = PhoneticTranslator(model_id=model_id)
        load_time = time.perf_counter() - t0
        after = cuda_mem_mb()
        delta = (after - before) if before is not None else None
        stats.append((model_id, "phonetic", load_time, delta))
    return models, stats


def load_language_models(model_names):
    models, stats = {}, []
    for name in model_names:
        before = cuda_mem_mb()
        t0 = time.perf_counter()
        models[name] = LanguageTranslator(model=name)
        load_time = time.perf_counter() - t0
        after = cuda_mem_mb()
        delta = (after - before) if before is not None else None
        stats.append((name, "language", load_time, delta))
    return models, stats


def print_loading_table(stats):
    print("\n=== Model Loading ===")
    header = f"{'Model':<45} {'Type':<10} {'Load (s)':>10} {'VRAM (MB)':>10}"
    print(header)
    print("-" * len(header))
    for name, kind, load_time, vram_delta in stats:
        vram_str = f"{vram_delta:.0f}" if vram_delta is not None else "n/a"
        print(f"{name:<45} {kind:<10} {load_time:>10.1f} {vram_str:>10}")


def evaluate_phrase(phrase, phonetic_models, language_models, matcher_ipa, matcher_wer, recorder):
    ipa_hint = f"  (target IPA: {phrase['target_ipa']})" if phrase["target_ipa"] else ""
    print(f"\n>>> Say: \"{phrase['text']}\"{ipa_hint}")
    audio = recorder.record()
    if audio.size == 0:
        print("(nothing heard, skipping this phrase)")
        return None
    audio_duration = len(audio) / recorder.sample_rate

    phonetic_rows = []
    for model_id, model in phonetic_models.items():
        t0 = time.perf_counter()
        ipa = model.convert(audio)
        latency = time.perf_counter() - t0
        target = phrase["target_ipa"]
        accuracy = matcher_ipa.score(ipa, target) if target else None
        completeness = (len(ipa) / len(target) * 100) if target else None
        phonetic_rows.append({
            "model": model_id, "output": ipa, "latency": latency,
            "accuracy": accuracy, "completeness": completeness,
        })

    language_rows = []
    for name, model in language_models.items():
        t0 = time.perf_counter()
        result = model.convert(audio, lang=phrase["lang"])
        latency = time.perf_counter() - t0
        rtf = latency / audio_duration if audio_duration > 0 else None
        accuracy = matcher_wer.score(result.text, phrase["text"])
        ref_words = phrase["text"].split()
        out_words = result.text.split()
        completeness = (len(out_words) / len(ref_words) * 100) if ref_words else None
        language_rows.append({
            "model": name, "output": result.text, "latency": latency, "rtf": rtf,
            "accuracy": accuracy, "completeness": completeness, "confidence": result.confidence,
        })

    return {"phonetic": phonetic_rows, "language": language_rows, "audio_duration": audio_duration}


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

    print("Loading models (first run downloads anything not already cached)...")
    recorder = VoiceRecorder()
    phonetic_models, phonetic_load_stats = load_phonetic_models(phonetic_model_ids)
    language_models, language_load_stats = load_language_models(language_model_names)
    print_loading_table(phonetic_load_stats + language_load_stats)

    matcher_ipa = MatchingAlgo(algorithm="levenshtein")
    matcher_wer = MatchingAlgo(algorithm="word_error_rate")

    all_results = []
    try:
        for run in range(args.repeats):
            if args.repeats > 1:
                print(f"\n########## Run {run + 1}/{args.repeats} ##########")
            for phrase in TEST_PHRASES:
                result = evaluate_phrase(
                    phrase, phonetic_models, language_models, matcher_ipa, matcher_wer, recorder
                )
                if result:
                    print_phonetic_table(result["phonetic"])
                    print_language_table(result["language"])
                all_results.append(result)
    except KeyboardInterrupt:
        print("\nStopped early.")

    aggregate_and_print(all_results)


if __name__ == "__main__":
    main()
