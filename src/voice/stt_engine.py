"""Local streaming speech-to-text (STT) engine powered by faster-whisper."""

import logging
import threading
from typing import Optional, List, Generator
import numpy as np

logger = logging.getLogger(__name__)


class StreamingWhisperSTT:
    """Local STT engine using faster-whisper with int8 quantization for ultra-low latency."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        force_mock: bool = False,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.force_mock = force_mock

        self._model = None
        self._lock = threading.Lock()
        if not self.force_mock:
            self._init_model()

    def _init_model(self) -> None:
        """Initializes the local faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            logger.info("Initializing faster-whisper (model=%s, device=%s, compute=%s)...", self.model_size, self.device, self.compute_type)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model '%s' loaded successfully.", self.model_size)
        except Exception as exc:
            logger.warning("Could not load faster-whisper (%s). Operating with mock fallback.", exc)
            self._model = None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribes a 16kHz float32 audio buffer into text."""
        if audio is None or len(audio) == 0:
            return ""

        if self._model is None:
            # Fallback mock for testing
            return self._mock_transcribe(audio)

        with self._lock:
            try:
                segments, info = self._model.transcribe(
                    audio,
                    beam_size=1,
                    language=self.language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400),
                )
                text_parts = [segment.text.strip() for segment in segments]
                full_text = " ".join(text_parts).strip()
                return full_text
            except Exception as exc:
                logger.error("Whisper transcription error: %s", exc)
                return ""

    def _mock_transcribe(self, audio: np.ndarray) -> str:
        """Deterministic mock transcriber for headless unit testing."""
        return getattr(self, "_mock_text", "Giriş yap")

    def set_mock_text(self, text: str) -> None:
        """Helper to inject simulated speech text in tests."""
        self._mock_text = text
