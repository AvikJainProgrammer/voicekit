# voicekit

Reusable voice recording, transcription, phonetic matching, and fuzzy
matching library — extracted from a set of standalone speech-processing
prototypes (pronunciation trainers, multilingual note takers, voice-controlled
shells) into four composable classes.

## Install

```
pip install -e ./project
```

## Quick start

```python
from voicekit import VoiceRecorder, LanguageTranslator, PhoneticTranslator, MatchingAlgo

recorder   = VoiceRecorder()
translator = LanguageTranslator(lang="german")
phonetics  = PhoneticTranslator()
matcher    = MatchingAlgo(algorithm="levenshtein")

audio = recorder.record()             # blocks until one utterance is captured
words = translator.convert(audio)     # TranscriptionResult; str()'s to the text
ipa   = phonetics.convert(audio)
print(words)
print(matcher.score(ipa, "ɡuːtntaːk"))
```

## Structure

- `src/voicekit/` — the library: `VoiceRecorder` (mic + VAD), `LanguageTranslator`
  (faster-whisper), `PhoneticTranslator` (wav2vec2 IPA), `MatchingAlgo`
  (`levenshtein` / `phonetic_skeleton` / `fuzzy` / `word_error_rate`).
- `examples/` — runnable demos: multilingual note taker, German flashcards,
  32 Names of Durga pronunciation trainer, minimal live-phonetics demo.
- `benchmarks/compare_models.py` — records each test phrase once and fans it
  out to multiple phonetic/language models so you can compare quality, speed,
  and accuracy side by side, with take-to-take variance removed.
- `tests/` — unit tests for the matching algorithms (no mic/GPU required).

## Benchmark findings

Measured with `benchmarks/compare_models.py --repeats 5` (30 test instances)
against the default model sets (`wav2vec2-lv-60-espeak-cv-ft` /
`wav2vec2-xlsr-53-espeak-cv-ft` for phonetics; `tiny` / `small` / `medium` /
`large-v3` / `distil-large-v3` for words) on English, German, and Sanskrit
(Durga names) test phrases. These numbers were revised once already as more
data came in (see the note at the end) — treat them as the current best
read, not a final verdict.

- **Phonetic models: `xlsr-53-espeak-cv-ft` beats `lv-60-espeak-cv-ft`, but
  modestly, not decisively.** Aggregate accuracy 39.0% vs 32.8%. Head-to-head
  across 20 scored instances: `xlsr-53` wins 9, `lv-60` wins 5, ties 6 —
  `lv-60` specifically wins on "Guten Tag" in most takes, suggesting a
  phrase-dependent effect rather than uniform superiority. Still the better
  aggregate choice, so it's `PhoneticTranslator`'s default.
- **`large-v3` has a small but consistent edge over `medium`** — 10
  head-to-head wins vs `medium`'s 3 (17 ties, mostly identical output),
  51.5% vs 47.8% aggregate. Both clearly ahead of `tiny` / `small`. This is
  `LanguageTranslator`'s default. (An earlier, smaller sample of 18
  instances showed these as statistically tied, which is why the default was
  briefly switched to `medium` for efficiency — the larger sample changed
  that conclusion.)
- **`distil-large-v3` is competitive on English but unreliable on German** —
  competitive on the English test phrases but hallucinates content unrelated
  to the audio on German in 9 of 10 attempts across 5 repeats (e.g. "Guten
  Tag" transcribed as "Putin talk." or "Brent talk."). Not recommended for
  non-English use.
- **Speed is not a meaningful differentiator at this scale.** Every tested
  Whisper size ran comfortably faster than real-time (RTF 0.03-0.36) on the
  test hardware — accuracy, not latency, should drive model choice here.
- **Methodology:** each phrase is recorded once and that exact recording is
  fed to every model under comparison, removing take-to-take pronunciation
  variance as a confound. `--repeats N` re-records the whole phrase set N
  times for additional statistical confidence (mean ± stdev). The `medium`
  vs `large-v3` revision above is a reminder that even N=18 wasn't enough to
  call a result reliably — `--repeats 5` or more is recommended before
  trusting a close result.

### Issues found and fixed during benchmarking

- **Devanagari/Latin script mismatch.** Whisper transcribes Hindi-coded audio
  in Devanagari script, which can never match a Latin-script reference at
  the word level — every Sanskrit-name attempt was scoring a structurally
  floored 0% regardless of actual quality. `compare_models.py` now also
  scores against a native-script reference (`target_text_alt`) when one is
  available, keeping whichever score is higher.
- **Binary word-level WER on single-word phrases.** Word-error-rate scoring
  on a one-word reference (e.g. "Durgamā") can only be 0% or 100% — there's
  no second token to share partial credit with, so a near-miss like
  "Surgama." scored a flat 0% despite being one letter off. `compare_models.py`
  now also falls back to character-level similarity for any single-word
  reference (not just the Devanagari one above), so near-misses score
  proportionally instead of being floored.
- **VAD onset clipping.** Short words (e.g. "Guten Tag") occasionally lost
  their first syllable in the captured audio — Silero VAD takes a window or
  two to ramp up confidence after speech actually starts, and that brief
  window was being discarded. `VoiceRecorder` now keeps a small rolling
  pre-roll buffer and prepends it the moment speech is detected, recovering
  the clipped onset.
- **Memory: load one model at a time.** Early versions of
  `compare_models.py` held every configured model in memory simultaneously,
  which reliably caused an out-of-memory crash. It now records all phrases
  up front, then loads, evaluates, and frees one model at a time — peak
  memory is bounded by the single largest model, not the sum of all of them.
