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

Measured with `benchmarks/compare_models.py --repeats 3` against the default
model sets (`wav2vec2-lv-60-espeak-cv-ft` / `wav2vec2-xlsr-53-espeak-cv-ft`
for phonetics; `tiny` / `small` / `medium` / `large-v3` / `distil-large-v3`
for words) on English, German, and Sanskrit (Durga names) test phrases:

- **Phonetic models: `xlsr-53-espeak-cv-ft` consistently beats
  `lv-60-espeak-cv-ft`.** It won or tied in all 12 head-to-head comparisons
  (10 strict wins), with higher aggregate accuracy (54.9% vs 36.6%) and a
  tighter spread (±18.6 vs ±12.9). Recommended over the current library
  default.
- **`medium` and `large-v3` are statistically tied** on word accuracy and
  both clearly ahead of `tiny` / `small`, as expected from model size.
- **`distil-large-v3` is competitive on English but unreliable on German** —
  it matched `large-v3` on the English test phrases but scored 0% on German
  in 5 of 6 instances, hallucinating unrelated phrases (e.g. "Guten Tag"
  audio transcribed as "Also, again." or "off with the end"). Not
  recommended for non-English use.
- **Speed is not a meaningful differentiator at this scale.** Every tested
  Whisper size ran comfortably faster than real-time (RTF 0.03-0.30) on the
  test hardware — accuracy, not latency, should drive model choice here.
- **Methodology:** each phrase is recorded once and that exact recording is
  fed to every model under comparison, removing take-to-take pronunciation
  variance as a confound. `--repeats N` re-records the whole phrase set N
  times for additional statistical confidence (mean ± stdev).

### Issues found and fixed during benchmarking

- **Devanagari/Latin script mismatch.** Whisper transcribes Hindi-coded audio
  in Devanagari script, which can never match a Latin-script reference at
  the word level — every Sanskrit-name attempt was scoring a structurally
  floored 0% regardless of actual quality. `compare_models.py` now also
  scores against a native-script reference (`target_text_alt`) when one is
  available, keeping whichever score is higher.
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
