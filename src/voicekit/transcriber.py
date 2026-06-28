"""Word-level speech transcription via faster-whisper.

Ports note_taker_ml.py's transcribe_segment + LANGUAGES priming and
voice_shell_v2.py's avg_logprob confidence scoring into a reusable class.
"""

from dataclasses import dataclass

import numpy as np

from ._device import resolve_device
from .languages import resolve_language


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float  # average log-probability reported by Whisper

    def __str__(self) -> str:
        return self.text


class LanguageTranslator:
    """Transcribes spoken audio into words for a given target language."""

    def __init__(
        self,
        lang: str = "english",
        model: str = "large-v3",
        device: str | None = None,
        beam_size: int = 5,
        initial_prompt: str | None = None,
    ):
        self.lang_code, self.lang_display, default_prompt = resolve_language(lang)
        self._explicit_prompt = initial_prompt
        self.prompt = initial_prompt if initial_prompt is not None else default_prompt
        self.beam_size = beam_size
        self.device = resolve_device(device)

        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            model,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
        )

    def convert(
        self, audio: np.ndarray, sample_rate: int = 16000, lang: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a float32 mono numpy array into a TranscriptionResult.

        `lang` overrides the instance's default language for this call only —
        lets one loaded model be reused across languages instead of needing a
        separate LanguageTranslator (and a separate model load) per language.
        """
        if lang is not None:
            lang_code, _, default_prompt = resolve_language(lang)
            prompt = self._explicit_prompt if self._explicit_prompt is not None else default_prompt
        else:
            lang_code, prompt = self.lang_code, self.prompt

        segments, info = self._model.transcribe(
            audio,
            language=lang_code,
            beam_size=self.beam_size,
            initial_prompt=prompt or None,
            vad_filter=False,
            without_timestamps=True,
            temperature=0.0,
        )
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list).strip()
        avg_logprob = (
            sum(s.avg_logprob for s in seg_list) / len(seg_list) if seg_list else -1.0
        )
        detected = info.language if info else (lang_code or "?")
        return TranscriptionResult(text=text, language=detected, confidence=avg_logprob)
