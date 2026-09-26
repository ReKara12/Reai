"""Minimal Black Dynamic Island & Overlay for Reflex-Agent.

Inspired by Andy Gao's viral macOS demo and Apple's Dynamic Island:
- Ultra-minimal pitch-black pill floating at top center of desktop
- Real-time speech transcription & sub-35ms speculative action display
- Expandable Settings drawer (⚙) with live microphone selector, mode toggle, and manual input
- Non-blocking asynchronous model loading (<200ms GUI startup)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Pre-import ctranslate2 before PyQt5 to prevent Windows OpenMP CRT DLL conflict
try:
    import ctranslate2
except ImportError:
    pass

import sys
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QComboBox, QCheckBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QColor, QFont, QPainter, QBrush, QPen

from ..orchestrator.models import AgentState, UIElement, ReflexDecision
from ..orchestrator.state_machine import ReflexStateMachine
from ..perception.accessibility import get_platform_driver
from ..perception.state_pruner import UIStatePruner
from ..reflex.laya_engine import LayaReflexEngine
from ..reflex.guardrail import SafetyGuardrail
from ..reflex.verifier import StateVerifier
from ..actuator.mouse_keyboard import get_actuator
from ..generative.llm_fallback import get_generative_provider
from ..voice.audio_stream import AudioCaptureStream
from ..voice.stt_engine import StreamingWhisperSTT

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/voice_settings.json")


def query_input_microphones() -> List[Tuple[int, str]]:
    """Retrieves list of active audio input devices (idx, name)."""
    devices = []
    try:
        import sounddevice as sd
        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", f"Cihaz {idx}").strip()
                devices.append((idx, f"[{idx}] {name}"))
    except Exception as e:
        logger.warning("Could not query audio devices: %s", e)
    return devices


def get_best_default_mic() -> Optional[int]:
    """Finds the best active microphone, avoiding 'Stereo Mix' speaker loopback."""
    mics = query_input_microphones()
    if not mics:
        return None

    # Priority 1: NVIDIA Broadcast, WO Mic, or dedicated mics
    for idx, name in mics:
        nl = name.lower()
        if "stereo" in nl:
            continue
        if "nvidia" in nl or "wo mic" in nl or "broadcast" in nl:
            return idx

    # Priority 2: Any real microphone array
    for idx, name in mics:
        nl = name.lower()
        if "stereo" in nl:
            continue
        if "mikrofon" in nl or "microphone" in nl or "dizisi" in nl:
            return idx

    # Priority 3: First device without stereo mix
    for idx, name in mics:
        if "stereo" not in name.lower():
            return idx

    return mics[0][0]


def load_saved_mic_index() -> Optional[int]:
    """Loads previously selected microphone device index from config."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                val = data.get("input_device_index")
                if val is not None:
                    return int(val)
    except Exception:
        pass
    return get_best_default_mic()


def save_mic_index(index: int):
    """Saves selected microphone device index to config."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"input_device_index": index}, f, indent=2)
    except Exception as e:
        logger.warning("Could not save mic index: %s", e)


DYNAMIC_ISLAND_STYLE = """
/* Minimal Dynamic Island */
QFrame#IslandPill {
    background-color: #06070a;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 22px;
}

QLabel#TranscriptLabel {
    color: #f8fafc;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
}

QPushButton#MicToggleBtn {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 14px;
    border-radius: 12px;
    padding: 2px 5px;
}
QPushButton#MicToggleBtn:hover {
    color: #00e5ff;
    background-color: rgba(255, 255, 255, 0.14);
}

QPushButton#SettingsGearBtn {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 15px;
    border-radius: 12px;
    padding: 2px 6px;
}
QPushButton#SettingsGearBtn:hover {
    color: #ffffff;
    background-color: rgba(255, 255, 255, 0.14);
}

/* Settings Drawer */
QFrame#SettingsCard {
    background-color: #0a0c12;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 12px;
}

QLabel.SettingHeader {
    color: #38bdf8;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QComboBox {
    background-color: #141824;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 8px;
    padding: 6px 10px;
    color: #ffffff;
    font-size: 12px;
}
QComboBox:hover {
    border: 1px solid #00e5ff;
}
QComboBox QAbstractItemView {
    background-color: #141824;
    color: #ffffff;
    selection-background-color: #0284c7;
    border: 1px solid rgba(255, 255, 255, 0.16);
}

QLineEdit#TestInput {
    background-color: #141824;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 8px;
    padding: 6px 10px;
    color: #ffffff;
    font-size: 12px;
}
QLineEdit#TestInput:focus {
    border: 1px solid #00e5ff;
}

QPushButton.ActionButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #3b82f6);
    border: none;
    border-radius: 8px;
    color: #040812;
    font-weight: bold;
    font-size: 11px;
    padding: 6px 14px;
}
QPushButton.ActionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #60a5fa);
}

QPushButton.SecondaryButton {
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    color: #cbd5e1;
    font-size: 11px;
    padding: 6px 12px;
}
QPushButton.SecondaryButton:hover {
    background-color: rgba(255, 255, 255, 0.16);
    color: #ffffff;
}

QCheckBox {
    color: #cbd5e1;
    font-size: 11px;
}
"""


class PulseIndicator(QWidget):
    """Pulsing indicator dot inside the Dynamic Island."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.state_color = QColor(16, 185, 129)  # Green = Ready
        self.glow_radius = 4.0
        self.expanding = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_pulse)
        self.timer.start(50)

    def set_color(self, hex_color: str):
        self.state_color = QColor(hex_color)
        self.update()

    def _animate_pulse(self):
        if self.expanding:
            self.glow_radius += 0.4
            if self.glow_radius >= 6.5:
                self.expanding = False
        else:
            self.glow_radius -= 0.4
            if self.glow_radius <= 3.2:
                self.expanding = True
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        glow_color = QColor(self.state_color)
        glow_color.setAlpha(85)
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.NoPen)
        center = self.rect().center()
        painter.drawEllipse(center, self.glow_radius, self.glow_radius)

        painter.setBrush(QBrush(self.state_color))
        painter.drawEllipse(center, 3.8, 3.8)


class VoiceThread(QThread):
    """Non-blocking background thread for real-time speech capture and transcription."""
    sig_partial = pyqtSignal(str)
    sig_final = pyqtSignal(str)
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str)

    def __init__(self, mic_index: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.mic_index = mic_index
        self._running = False
        self.audio_stream: Optional[AudioCaptureStream] = None
        self.stt: Optional[StreamingWhisperSTT] = None

    def start_listening(self):
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_listening(self):
        self._running = False

    def update_mic_index(self, index: int):
        self.mic_index = index
        if self.isRunning():
            self.stop_listening()
            self.wait(500)
            self.start_listening()

    def toggle_manual_recording(self) -> bool:
        """Toggles manual recording state on/off, returning new state."""
        if self.audio_stream:
            new_state = not self.audio_stream.is_recording()
            self.audio_stream.set_manual_recording(new_state)
            return new_state
        return False

    def run(self):
        try:
            self.stt = StreamingWhisperSTT(model_size="base", force_mock=False)
            self.audio_stream = AudioCaptureStream(
                push_to_talk=True,
                ptt_key_name="alt_r",
                device_index=self.mic_index
            )
            self.audio_stream.start()
            self.sig_status.emit("Konuşmak için Alt_R basılı tutun...")
        except Exception as exc:
            self.sig_error.emit(f"Ses donanımı başlatılamadı: {exc}")
            return

        accumulated_audio = []
        last_speech_time = time.time()

        try:
            while self._running:
                try:
                    chunk = self.audio_stream.get_chunk(timeout=0.1)
                    is_speaking = self.audio_stream.is_recording()

                    if chunk is not None:
                        accumulated_audio.append(chunk)
                        last_speech_time = time.time()

                        total_samples = sum(len(c) for c in accumulated_audio)
                        if total_samples >= 8000:  # 0.5 sec of speech
                            audio_np = np.concatenate(accumulated_audio)
                            partial_transcript = self.stt.transcribe(audio_np)
                            if partial_transcript and partial_transcript.strip():
                                self.sig_partial.emit(partial_transcript.strip())

                    if accumulated_audio and not is_speaking:
                        if time.time() - last_speech_time > 0.35:
                            full_audio = np.concatenate(accumulated_audio)
                            accumulated_audio.clear()
                            final_transcript = self.stt.transcribe(full_audio)
                            if final_transcript and final_transcript.strip():
                                self.sig_final.emit(final_transcript.strip())

                except Exception as e:
                    logger.debug("Voice read error: %s", e)
                self.msleep(20)
        finally:
            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                except Exception:
                    pass
                self.audio_stream = None


class AgentExecutionWorker(QObject):
    """Background worker for multi-step actuation on desktop."""
    sig_step = pyqtSignal(int, str, str)
    sig_anticipation = pyqtSignal(str, str, float)
    sig_finished = pyqtSignal(str, int)
    sig_ready = pyqtSignal()

    def __init__(self, mock_mode: bool = False):
        super().__init__()
        self.mock_mode = mock_mode
        self.state_machine: Optional[ReflexStateMachine] = None
        self._init_thread = threading.Thread(target=self._init_backend, daemon=True)
        self._init_thread.start()

    def _init_backend(self):
        try:
            driver = get_platform_driver(mock=self.mock_mode)
            actuator = get_actuator(mock=self.mock_mode)
            pruner = UIStatePruner(max_elements=25)
            reflex_engine = LayaReflexEngine(force_mock=self.mock_mode)
            guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
            verifier = StateVerifier(max_retries=3)
            llm = get_generative_provider("auto")

            self.state_machine = ReflexStateMachine(
                driver=driver,
                pruner=pruner,
                reflex_engine=reflex_engine,
                guardrail=guardrail,
                verifier=verifier,
                actuator=actuator,
                generative_provider=llm,
            )
            self.sig_ready.emit()
        except Exception as e:
            logger.error("Failed to initialize backend in worker: %s", e)

    def set_mode(self, mock_mode: bool):
        self.mock_mode = mock_mode
        if self.state_machine:
            self.state_machine.driver = get_platform_driver(mock=self.mock_mode)
            self.state_machine.actuator = get_actuator(mock=self.mock_mode)

    def execute_goal(self, goal: str, max_steps: int = 15, allow_destructive: bool = False):
        if not self.state_machine:
            time.sleep(1.0)
            if not self.state_machine:
                self.sig_finished.emit("Backend hazırlanıyor...", 0)
                return

        sm = self.state_machine
        try:
            # 1. Speculative pre-dispatch
            t0 = time.perf_counter()
            state, _ = sm.sense_and_prune(goal)
            spec = sm.reflex_engine.predict(state)
            lat = (time.perf_counter() - t0) * 1000.0
            target = spec.text_to_type or spec.selected_element_id or "Desktop"
            self.sig_anticipation.emit(spec.action_type, str(target), lat)
        except Exception:
            pass

        sm.verifier.reset()
        sm.history.clear()
        sm.step_count = 0

        for step_i in range(1, max_steps + 1):
            try:
                state, pre_hash = sm.sense_and_prune(goal)
                decision = sm.reflex_engine.predict(state)

                target_elem = None
                if decision.selected_element_id:
                    for el in state.available_elements:
                        if el.id == decision.selected_element_id:
                            target_elem = el
                            break

                decision = sm.guardrail.evaluate_and_annotate(
                    decision, element=target_elem, context_text=goal
                )
                if decision.is_destructive and not allow_destructive:
                    self.sig_step.emit(step_i, "GUARDRAIL_BLOCKED", f"Yıkıcı eylem: {decision.selected_element_id}")
                    self.sig_finished.emit("GUARDRAIL_BLOCKED", sm.step_count)
                    return

                if decision.is_task_completed and (decision.action_type == "WAIT" or not decision.selected_element_id):
                    self.sig_finished.emit("SUCCESS", sm.step_count)
                    return

                if decision.action_type == "CALL_LLM":
                    text = sm.generative_provider.generate(goal, context=f"Active Window: {state.active_window}")
                    if target_elem:
                        sm.actuator.execute_decision(
                            ReflexDecision(action_type="TYPE", selected_element_id=target_elem.id, text_to_type=text),
                            element=target_elem
                        )
                    else:
                        sm.actuator.paste_text(text)
                    sm.step_count += 1
                    self.sig_step.emit(step_i, "CALL_LLM", text[:25] + "...")
                    self.sig_finished.emit("SUCCESS", sm.step_count)
                    return
                else:
                    sm.actuator.execute_decision(decision, element=target_elem)

                sm.step_count += 1
                time.sleep(sm.step_delay)
                act_target = decision.text_to_type or decision.selected_element_id or "UI"
                self.sig_step.emit(step_i, decision.action_type, str(act_target))

            except Exception as e:
                self.sig_finished.emit(f"Hata: {e}", sm.step_count)
                return

        self.sig_finished.emit("MAX_STEPS", sm.step_count)


class FloatingHUD(QMainWindow):
    """Clean, pure black Dynamic Island with expandable Settings drawer."""

    def __init__(self, mock_mode: bool = False, enable_voice: bool = True, parent=None):
        super().__init__(parent)
        self.mock_mode = mock_mode
        self.enable_voice = enable_voice
        self.drag_position = QPoint()

        # Frameless, Always on Top, Persistent Window
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedWidth(460)

        # Position at top center of primary monitor
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = 20
        self.move(x, y)

        # Non-blocking async backend worker
        self.exec_worker = AgentExecutionWorker(mock_mode=self.mock_mode)
        self.exec_worker.sig_anticipation.connect(self._on_anticipation, Qt.QueuedConnection)
        self.exec_worker.sig_step.connect(self._on_step, Qt.QueuedConnection)
        self.exec_worker.sig_finished.connect(self._on_finished, Qt.QueuedConnection)
        self.exec_worker.sig_ready.connect(self._on_backend_ready, Qt.QueuedConnection)

        # Saved mic preference
        self.selected_mic_index = load_saved_mic_index()

        # Voice listener
        self.voice_worker = VoiceThread(mic_index=self.selected_mic_index)
        self.voice_worker.sig_partial.connect(self._on_voice_partial, Qt.QueuedConnection)
        self.voice_worker.sig_final.connect(self._on_voice_final, Qt.QueuedConnection)
        self.voice_worker.sig_status.connect(self._on_voice_status, Qt.QueuedConnection)
        self.voice_worker.sig_error.connect(self._on_voice_error, Qt.QueuedConnection)

        self._init_ui()

        if self.enable_voice:
            QTimer.singleShot(200, self.voice_worker.start_listening)

    def _init_ui(self):
        self.setStyleSheet(DYNAMIC_ISLAND_STYLE)

        self.root_widget = QWidget(self)
        self.root_widget.setStyleSheet("background: transparent;")
        self.setCentralWidget(self.root_widget)

        self.main_layout = QVBoxLayout(self.root_widget)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(8)

        # -------------------------------------------------------------
        # 1. THE DYNAMIC ISLAND (Small, Sleek, Pure Black Pill)
        # -------------------------------------------------------------
        self.island_pill = QFrame()
        self.island_pill.setObjectName("IslandPill")
        self.island_pill.setFixedHeight(44)

        pill_layout = QHBoxLayout(self.island_pill)
        pill_layout.setContentsMargins(14, 4, 12, 4)
        pill_layout.setSpacing(10)

        # Pulsing LED indicator
        self.pulse = PulseIndicator()
        pill_layout.addWidget(self.pulse)

        # Live speech transcription & action text
        self.lbl_transcript = QLabel("⚡ Model hazırlanıyor...")
        self.lbl_transcript.setObjectName("TranscriptLabel")
        self.lbl_transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pill_layout.addWidget(self.lbl_transcript)

        # Microphone toggle button
        self.btn_mic = QPushButton("🎙️")
        self.btn_mic.setObjectName("MicToggleBtn")
        self.btn_mic.setToolTip("Konuşmak için tıklayın veya 'Alt_R' / 'Alt Gr' basılı tutun")
        self.btn_mic.clicked.connect(self._toggle_mic_click)
        pill_layout.addWidget(self.btn_mic)

        # Settings gear icon
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setObjectName("SettingsGearBtn")
        self.btn_settings.setToolTip("Ayarlar (Mikrofon, Mod)")
        self.btn_settings.clicked.connect(self._toggle_settings)
        pill_layout.addWidget(self.btn_settings)

        self.main_layout.addWidget(self.island_pill)

        # -------------------------------------------------------------
        # 2. EXPANDABLE SETTINGS CARD (Hidden by default)
        # -------------------------------------------------------------
        self.settings_card = QFrame()
        self.settings_card.setObjectName("SettingsCard")
        self.settings_card.setVisible(False)

        card_layout = QVBoxLayout(self.settings_card)
        card_layout.setContentsMargins(12, 10, 12, 12)
        card_layout.setSpacing(10)

        # Header: Title + Close
        hdr_layout = QHBoxLayout()
        lbl_hdr = QLabel("Ayarlar & Mikrofon")
        lbl_hdr.setProperty("class", "SettingHeader")
        hdr_layout.addWidget(lbl_hdr)
        hdr_layout.addStretch()

        btn_close_settings = QPushButton("✕")
        btn_close_settings.setFixedSize(20, 20)
        btn_close_settings.setStyleSheet("background: transparent; color: #94a3b8; border: none; font-size: 11px;")
        btn_close_settings.clicked.connect(lambda: self.settings_card.setVisible(False))
        hdr_layout.addWidget(btn_close_settings)
        card_layout.addLayout(hdr_layout)

        # 1. Microphone Selector Dropdown
        mic_lbl = QLabel("🎙️ Mikrofon Girişi:")
        mic_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        card_layout.addWidget(mic_lbl)

        self.combo_mics = QComboBox()
        self.mics_list = query_input_microphones()
        selected_row = 0
        for row_i, (idx, name) in enumerate(self.mics_list):
            self.combo_mics.addItem(name, idx)
            if self.selected_mic_index is not None and idx == self.selected_mic_index:
                selected_row = row_i

        if self.mics_list:
            self.combo_mics.setCurrentIndex(selected_row)
        self.combo_mics.currentIndexChanged.connect(self._on_mic_changed)
        card_layout.addWidget(self.combo_mics)

        # 2. Mode Toggle (Physical vs Mock)
        mode_layout = QHBoxLayout()
        lbl_mode = QLabel("Çalışma Modu:")
        lbl_mode.setStyleSheet("color: #94a3b8; font-size: 11px;")
        mode_layout.addWidget(lbl_mode)
        mode_layout.addStretch()

        self.btn_mode_toggle = QPushButton("Simülasyon (Mock)" if self.mock_mode else "Gerçek Masaüstü (Physical)")
        self.btn_mode_toggle.setProperty("class", "SecondaryButton")
        self.btn_mode_toggle.clicked.connect(self._toggle_mode)
        mode_layout.addWidget(self.btn_mode_toggle)
        card_layout.addLayout(mode_layout)

        # 3. Manual Text Test Input
        card_layout.addWidget(QLabel("⌨️ Klavye ile Komut Testi:"))
        input_test_layout = QHBoxLayout()
        self.test_input = QLineEdit()
        self.test_input.setObjectName("TestInput")
        self.test_input.setPlaceholderText("Komut yazın (örn: open firefox)...")
        self.test_input.returnPressed.connect(self._on_manual_run)
        input_test_layout.addWidget(self.test_input)

        btn_test_run = QPushButton("⚡ Çalıştır")
        btn_test_run.setProperty("class", "ActionButton")
        btn_test_run.clicked.connect(self._on_manual_run)
        input_test_layout.addWidget(btn_test_run)
        card_layout.addLayout(input_test_layout)

        # 4. Guardrail Destructive Toggle
        self.chk_destructive = QCheckBox("Yıkıcı eylemlere (silme, kaldırma vb.) izin ver")
        card_layout.addWidget(self.chk_destructive)

        self.main_layout.addWidget(self.settings_card)
        self.adjustSize()

    def _on_backend_ready(self):
        self.pulse.set_color("#10b981")
        self.lbl_transcript.setText("Konuşmak için Alt_R / Alt Gr basılı tutun...")

    def _toggle_mic_click(self):
        """Toggles voice recording directly from mouse click on the mic button."""
        is_rec = self.voice_worker.toggle_manual_recording()
        if is_rec:
            self.pulse.set_color("#8b5cf6")
            self.lbl_transcript.setText("🎙️ Dinleniyor... (Bitince tekrar tıklayın)")
            self.btn_mic.setStyleSheet("color: #ef4444; background: rgba(239, 68, 68, 0.2);")
        else:
            self.pulse.set_color("#10b981")
            self.lbl_transcript.setText("İşleniyor...")
            self.btn_mic.setStyleSheet("")

    def _toggle_settings(self):
        is_visible = not self.settings_card.isVisible()
        self.settings_card.setVisible(is_visible)
        self.adjustSize()

    def _on_mic_changed(self, row: int):
        if row < 0 or row >= len(self.mics_list):
            return
        mic_idx, mic_name = self.mics_list[row]
        self.selected_mic_index = mic_idx
        save_mic_index(mic_idx)
        self.voice_worker.update_mic_index(mic_idx)
        self.lbl_transcript.setText(f"Mikrofon seçildi: {mic_name[:25]}...")

    def _toggle_mode(self):
        self.mock_mode = not self.mock_mode
        self.btn_mode_toggle.setText("Simülasyon (Mock)" if self.mock_mode else "Gerçek Masaüstü (Physical)")
        self.exec_worker.set_mode(self.mock_mode)
        self.lbl_transcript.setText(f"Mod: {'Mock' if self.mock_mode else 'Physical'}")

    def _on_manual_run(self):
        goal = self.test_input.text().strip()
        if not goal:
            return
        self.settings_card.setVisible(False)
        self.adjustSize()
        self.run_goal(goal)

    def run_goal(self, goal: str):
        self.pulse.set_color("#00e5ff")
        self.lbl_transcript.setText(f"Hedef: '{goal}'")

        allow_dest = self.chk_destructive.isChecked()
        threading.Thread(
            target=self.exec_worker.execute_goal,
            args=(goal, 15, allow_dest),
            daemon=True
        ).start()

    # Voice Callbacks
    def _on_voice_status(self, status: str):
        self.pulse.set_color("#10b981")
        self.lbl_transcript.setText(status)

    def _on_voice_partial(self, partial_text: str):
        self.pulse.set_color("#8b5cf6")
        self.lbl_transcript.setText(f"🎙️ \"{partial_text}\"")

    def _on_voice_final(self, final_text: str):
        self.lbl_transcript.setText(f"\"{final_text}\"")
        self.run_goal(final_text)

    def _on_voice_error(self, err: str):
        self.pulse.set_color("#ef4444")
        self.lbl_transcript.setText(err[:38])

    # Agent Execution Callbacks
    def _on_anticipation(self, action: str, target: str, latency: float):
        self.pulse.set_color("#00e5ff")
        self.lbl_transcript.setText(f"⚡ [{action}] ➔ {target} ({latency:.0f}ms)")

    def _on_step(self, step_num: int, action: str, target: str):
        self.pulse.set_color("#38bdf8")
        self.lbl_transcript.setText(f"Adım {step_num}: {action} ➔ {target}")

    def _on_finished(self, status: str, step_count: int):
        if status == "SUCCESS":
            self.pulse.set_color("#10b981")
            self.lbl_transcript.setText(f"✓ Tamamlandı ({step_count} adım)")
        elif status == "GUARDRAIL_BLOCKED":
            self.pulse.set_color("#ef4444")
            self.lbl_transcript.setText("🛡️ Güvenlik Engeli: Yıkıcı Eylem")
        else:
            self.pulse.set_color("#f59e0b")
            self.lbl_transcript.setText(f"Bitti: {status}")

        QTimer.singleShot(3500, self._reset_idle)

    def _reset_idle(self):
        self.pulse.set_color("#10b981")
        self.lbl_transcript.setText("Konuşmak için Alt_R / Alt Gr basılı tutun...")
        if hasattr(self, "btn_mic"):
            self.btn_mic.setStyleSheet("")

    def closeEvent(self, event):
        self.voice_worker.stop_listening()
        self.voice_worker.wait(500)
        event.accept()

    # Window Dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()


def launch_hud(mock: bool = False, enable_voice: bool = True):
    """Launches the minimal black Dynamic Island overlay."""
    import signal
    app = QApplication.instance() or QApplication(sys.argv)
    hud = FloatingHUD(mock_mode=mock, enable_voice=enable_voice)
    hud.show()
    hud.raise_()
    hud.activateWindow()

    # Periodic timer so Python interpreter catches SIGINT / Ctrl+C
    sig_timer = QTimer()
    sig_timer.start(250)
    sig_timer.timeout.connect(lambda: None)

    def _sigint_handler(sig, frame):
        print("\nDynamic Island kapatılıyor...", flush=True)
        hud.close()
        app.quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    print("\n" + "=" * 60)
    print("  ✨ ReAI Dynamic Island Overlay Aktif!")
    print("  📍 Konum: Ekranın en üst-orta kısmında süzülüyor")
    print("  🎙️ Sesli Komut: 'Alt_R' / 'Alt Gr' basılı tutun veya 🎙️ tıklayın")
    print("  ⚙️ Ayarlar & Mikrofon: Hapın sağındaki dişli simgesi")
    print("  ❌ Çıkış: Terminalde Ctrl+C")
    print("=" * 60 + "\n", flush=True)

    try:
        ret = app.exec_()
    except KeyboardInterrupt:
        ret = 0
    return ret


if __name__ == "__main__":
    mock_flag = "--mock" in sys.argv
    no_voice = "--no-voice" in sys.argv
    sys.exit(launch_hud(mock=mock_flag, enable_voice=not no_voice))
