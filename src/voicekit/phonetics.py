"""Phonetic (IPA) transcription via Wav2Vec2.

Ports the model loading + decode logic from all_voice.py / durga_names.py.
"""

import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from ._device import resolve_device


class PhoneticTranslator:
    """Transcribes spoken audio directly into IPA phonetic symbols."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: str | None = None,
    ):
        self.device = resolve_device(device)
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self._model.eval()

    def convert(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a float32 mono numpy array into an IPA phonetic string."""
        input_values = self._feature_extractor(
            audio, return_tensors="pt", sampling_rate=sample_rate
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self._tokenizer.batch_decode(predicted_ids)[0].strip()
