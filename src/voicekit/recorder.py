"""Microphone capture with VAD-based utterance segmentation.

Ports the mic-callback + Silero-VAD + silence-timeout pipeline duplicated
across note_taker_ml.py and voice_shell_v3.py into a single reusable,
blocking primitive.
"""

import queue
import time

import numpy as np
import sounddevice as sd
import torch


class VoiceRecorder:
    """Captures one spoken utterance at a time from the default microphone.

    Speech boundaries are detected with Silero VAD: recording starts once
    voice is detected and ends after `silence_timeout` seconds of silence.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        vad_threshold: float = 0.45,
        silence_timeout: float = 1.3,
        max_duration: float = 30.0,
        device: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.max_duration = max_duration
        self.device = device

        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._vad_model.eval()
        self._stop_requested = False

    def stop(self) -> None:
        """Request that an in-progress record()/listen() stop as soon as possible."""
        self._stop_requested = True

    def record(self) -> np.ndarray:
        """Block until one utterance is captured and return it as float32 mono audio."""
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def mic_callback(indata, frames, time_info, status):
            audio_queue.put(indata[:, 0].copy())

        buffer: list[np.ndarray] = []
        vad_buf: list[np.ndarray] = []
        in_speech = False
        last_speech = time.time()
        start_time = time.time()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.block_size,
            dtype="float32",
            callback=mic_callback,
        ):
            while True:
                if self._stop_requested or (time.time() - start_time) > self.max_duration:
                    break

                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if in_speech and buffer and (time.time() - last_speech) > self.silence_timeout:
                        break
                    continue

                vad_buf.append(chunk)
                if sum(len(c) for c in vad_buf) < 512:
                    continue

                window = np.concatenate(vad_buf)
                vad_buf.clear()

                with torch.no_grad():
                    prob = self._vad_model(torch.from_numpy(window).float(), self.sample_rate).item()

                if prob >= self.vad_threshold:
                    buffer.append(window)
                    last_speech = time.time()
                    in_speech = True
                elif in_speech:
                    buffer.append(window)
                    if (time.time() - last_speech) > self.silence_timeout:
                        break

        if not buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(buffer).astype(np.float32)

    def listen(self):
        """Yield one utterance array at a time, forever, until stop() is called."""
        while not self._stop_requested:
            audio = self.record()
            if audio.size:
                yield audio
        self._stop_requested = False
