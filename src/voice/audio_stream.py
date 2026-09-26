"""Audio capture stream with ring-buffer, Push-to-Talk hotkey, and VAD energy gating."""

import time
import math
import queue
import logging
import threading
from typing import Optional, Generator, Callable, List
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # 16 kHz standard for Whisper models
BLOCK_SIZE = 1600    # 100ms per audio frame (1600 samples)
SILENCE_THRESHOLD_RMS = 0.015  # Energy threshold for speech detection
DEFAULT_PTT_KEY = "alt_r"      # Right Alt key for Push-to-Talk


class AudioCaptureStream:
    """Manages real-time microphone capture with Push-to-Talk and VAD continuous modes."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        push_to_talk: bool = True,
        ptt_key_name: str = DEFAULT_PTT_KEY,
        energy_threshold: float = SILENCE_THRESHOLD_RMS,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.push_to_talk = push_to_talk
        self.ptt_key_name = ptt_key_name
        self.energy_threshold = energy_threshold
        self.device_index = device_index

        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._is_active = False
        self._is_speaking = False
        self._ptt_pressed = False
        self._stream = None
        self._keyboard_listener = None

        if self.push_to_talk:
            self._init_ptt_listener()

    def _init_ptt_listener(self) -> None:
        """Initializes global keyboard hook for push-to-talk key states."""
        try:
            from pynput import keyboard

            def _is_match(key) -> bool:
                key_str = str(key).lower().replace("key.", "")
                target = self.ptt_key_name.lower()
                if target in ("alt_r", "alt_gr", "altgr"):
                    return any(k in key_str for k in ("alt_r", "alt_gr", "altgr", "menu"))
                return target in key_str or key_str == target

            def on_press(key):
                if _is_match(key):
                    if not self._ptt_pressed:
                        self._ptt_pressed = True
                        logger.info("[PUSH-TO-TALK] Activated (Key pressed). Listening...")

            def on_release(key):
                if _is_match(key):
                    if self._ptt_pressed:
                        self._ptt_pressed = False
                        logger.info("[PUSH-TO-TALK] Released. Processing speech utterance.")

            self._keyboard_listener = keyboard.Listener(
                on_press=on_press, on_release=on_release
            )
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            logger.info("Global Push-to-Talk listener armed on '%s' (supports Alt_R & Alt_Gr).", self.ptt_key_name)
        except Exception as exc:
            logger.warning("Could not initialize global keyboard listener (%s). Operating in VAD mode.", exc)
            self.push_to_talk = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback invoked by PortAudio / sounddevice for each chunk of incoming audio."""
        if status:
            logger.debug("SoundDevice stream status: %s", status)

        # Convert float32 or int16 to float32 1D numpy array
        audio_chunk = indata[:, 0].copy()
        rms_energy = np.sqrt(np.mean(audio_chunk**2))

        # Check gating condition
        should_record = False
        if self.push_to_talk:
            should_record = self._ptt_pressed
        else:
            # Continuous VAD threshold
            should_record = rms_energy >= self.energy_threshold

        if should_record:
            self._is_speaking = True
            try:
                self._audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(audio_chunk)
                except Exception:
                    pass
        else:
            self._is_speaking = False

    def start(self) -> None:
        """Starts the audio capture stream."""
        if self._is_active:
            return
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_active = True
            logger.info("Audio capture started. Rate: %dHz, PTT: %s", self.sample_rate, self.push_to_talk)
        except Exception as exc:
            logger.warning("Could not start real sounddevice input stream (%s). Audio running in mock/buffer mode.", exc)
            self._is_active = True

    def stop(self) -> None:
        """Stops the audio capture stream."""
        self._is_active = False
        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Retrieves the next audio chunk from the buffer."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_recording(self) -> bool:
        """Returns True if the user is currently speaking or holding the PTT key."""
        if self.push_to_talk:
            return self._ptt_pressed
        return self._is_speaking

    def set_manual_recording(self, recording: bool) -> None:
        """Manually sets recording state (e.g. from UI mic toggle button)."""
        self._ptt_pressed = recording
        self._is_speaking = recording

    def simulate_audio_input(self, audio_data: np.ndarray) -> None:
        """Helper for unit tests: injects simulated audio chunk into buffer."""
        self._audio_queue.put(audio_data)
