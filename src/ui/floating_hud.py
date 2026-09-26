"""Modern Floating HUD and Glassmorphic Overlay for Reflex-Agent.

Inspired by Andy Gao's viral macOS Dynamic Island / Siri demo, built for Windows
with 100% local Laya ModernBERT (<35ms) speculative pre-dispatching and multi-step OS actuation.
"""

import sys
import os
import time
import threading
from typing import Optional, List, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea,
    QGraphicsDropShadowEffect, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QLinearGradient, QBrush, QPen

from ..orchestrator.models import AgentState, UIElement, ReflexDecision
from ..orchestrator.state_machine import ReflexStateMachine
from ..perception.accessibility import get_platform_driver, MockAccessibilityDriver
from ..perception.state_pruner import UIStatePruner
from ..reflex.laya_engine import LayaReflexEngine
from ..reflex.guardrail import SafetyGuardrail
from ..reflex.verifier import StateVerifier
from ..actuator.mouse_keyboard import get_actuator, MockActuator
from ..generative.llm_fallback import get_generative_provider
from ..voice.audio_stream import AudioCaptureStream
from ..voice.stt_engine import StreamingWhisperSTT
from ..voice.anticipator import SpeechToIntentAnticipator


HUD_STYLESHEET = """
QWidget#HUDContainer {
    background-color: rgba(14, 17, 27, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
}

QLabel {
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

QLineEdit#Omnibar {
    background-color: rgba(26, 32, 53, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 13px;
    padding: 10px 14px;
    color: #ffffff;
    font-size: 13px;
    selection-background-color: #00e5ff;
}
QLineEdit#Omnibar:focus {
    border: 1px solid #00e5ff;
    background-color: rgba(26, 32, 53, 0.95);
}

QPushButton.ActionButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #3b82f6);
    border: none;
    border-radius: 11px;
    color: #040812;
    font-weight: bold;
    font-size: 12px;
    padding: 8px 16px;
}
QPushButton.ActionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #60a5fa);
}
QPushButton.ActionButton:pressed {
    background: #0284c7;
}

QPushButton.PillButton {
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    color: #cbd5e1;
    font-size: 11px;
    padding: 6px 12px;
}
QPushButton.PillButton:hover {
    background-color: rgba(255, 255, 255, 0.16);
    color: #ffffff;
    border: 1px solid #00e5ff;
}

QPushButton.ActiveVoiceButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
    border: 1px solid #34d399;
    border-radius: 10px;
    color: #ffffff;
    font-size: 11px;
    font-weight: bold;
    padding: 6px 12px;
}

QFrame.CardFrame {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 8px;
}

QFrame#AnticipationBox {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 229, 255, 0.16), stop:1 rgba(59, 130, 246, 0.10));
    border: 1px solid rgba(0, 229, 255, 0.45);
    border-radius: 13px;
    padding: 10px;
}

QFrame#GuardrailBox {
    background-color: rgba(239, 68, 68, 0.18);
    border: 1px solid #ef4444;
    border-radius: 13px;
    padding: 10px;
}
"""


class AgentWorker(QObject):
    """Background worker for non-blocking agent decision and actuation execution."""
    sig_transcript = pyqtSignal(str)
    sig_anticipation = pyqtSignal(str, str, float)
    sig_step = pyqtSignal(int, str, str, bool)
    sig_finished = pyqtSignal(str, int)
    sig_guardrail = pyqtSignal(str, str)

    def __init__(self, state_machine: ReflexStateMachine):
        super().__init__()
        self.sm = state_machine
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run_goal(self, goal: str, max_steps: int = 15, allow_destructive: bool = False):
        self._is_cancelled = False
        self.sig_transcript.emit(f"Hedef: '{goal}'")

        # 1. Speculative Pre-Dispatch (<35ms System 1)
        try:
            t0 = time.perf_counter()
            state, _ = self.sm.sense_and_prune(goal)
            spec_decision = self.sm.reflex_engine.predict(state)
            latency = (time.perf_counter() - t0) * 1000.0

            target = spec_decision.text_to_type or spec_decision.selected_element_id or "Desktop"
            self.sig_anticipation.emit(spec_decision.action_type, str(target), latency)
        except Exception:
            pass

        # 2. Sequential Multi-Turn Actuation Loop
        self.sm.verifier.reset()
        self.sm.history.clear()
        self.sm.step_count = 0

        for step_i in range(1, max_steps + 1):
            if self._is_cancelled:
                self.sig_finished.emit("CANCELLED", self.sm.step_count)
                return

            try:
                state, pre_hash = self.sm.sense_and_prune(goal)
                decision = self.sm.reflex_engine.predict(state)

                target_elem = None
                if decision.selected_element_id:
                    for el in state.available_elements:
                        if el.id == decision.selected_element_id:
                            target_elem = el
                            break

                decision = self.sm.guardrail.evaluate_and_annotate(
                    decision, element=target_elem, context_text=goal
                )
                if decision.is_destructive and not allow_destructive:
                    self.sig_guardrail.emit(decision.action_type, f"Yıkıcı eylem '{decision.selected_element_id}' kullanıcı teyidi gerektirir!")
                    self.sig_finished.emit("GUARDRAIL_BLOCKED", self.sm.step_count)
                    return

                if decision.is_task_completed and (decision.action_type == "WAIT" or not decision.selected_element_id):
                    self.sig_finished.emit("SUCCESS", self.sm.step_count)
                    return

                if decision.action_type == "CALL_LLM":
                    text = self.sm.generative_provider.generate(goal, context=f"Active Window: {state.active_window}")
                    if target_elem:
                        self.sm.actuator.execute_decision(
                            ReflexDecision(action_type="TYPE", selected_element_id=target_elem.id, text_to_type=text),
                            element=target_elem
                        )
                    else:
                        self.sm.actuator.paste_text(text)
                    self.sm.step_count += 1
                    self.sig_step.emit(self.sm.step_count, "CALL_LLM", text[:35] + "...", True)
                    self.sig_finished.emit("SUCCESS", self.sm.step_count)
                    return
                else:
                    self.sm.actuator.execute_decision(decision, element=target_elem)

                self.sm.step_count += 1
                time.sleep(self.sm.step_delay)
                post_raw = self.sm.driver.get_ui_elements()
                post_pruned = self.sm.pruner.prune(post_raw)
                post_title = self.sm.driver.get_active_window_title()
                post_hash = self.sm.verifier.compute_state_hash(post_pruned, post_title)
                mutated = self.sm.verifier.verify_mutation(pre_hash, post_hash)

                act_target = decision.text_to_type or decision.selected_element_id or "UI"
                self.sig_step.emit(self.sm.step_count, decision.action_type, str(act_target), mutated)

            except Exception as exc:
                self.sig_finished.emit(f"HATA: {exc}", self.sm.step_count)
                return

        self.sig_finished.emit("MAX_STEPS", self.sm.step_count)


class PulseIndicator(QWidget):
    """Pulsing glow dot indicating listening or idle reflex state."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.state_color = QColor(16, 185, 129)  # Emerald green (ready)
        self.glow_radius = 4.0
        self.expanding = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_pulse)
        self.timer.start(45)

    def set_color(self, hex_color: str):
        self.state_color = QColor(hex_color)
        self.update()

    def _animate_pulse(self):
        if self.expanding:
            self.glow_radius += 0.4
            if self.glow_radius >= 7.0:
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
        painter.drawEllipse(center, 4.0, 4.0)


class FloatingHUD(QMainWindow):
    """Sleek Frameless Glassmorphic Overlay for Reflex-Agent OS Assistant."""

    sig_voice_partial = pyqtSignal(str)
    sig_voice_final = pyqtSignal(str)

    def __init__(self, mock_mode: bool = False, enable_voice: bool = True, parent=None):
        super().__init__(parent)
        self.mock_mode = mock_mode
        self.enable_voice = enable_voice
        self.drag_position = QPoint()

        # Frameless, Always on Top, Translucent
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(560, 500)

        # Center horizontally at top of screen (Dynamic Island / Andy Gao style)
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = 35
        self.move(x, y)

        # Initialize Backend
        self.driver = get_platform_driver(mock=self.mock_mode)
        self.actuator = get_actuator(mock=self.mock_mode)
        self.pruner = UIStatePruner(max_elements=25)
        self.reflex_engine = LayaReflexEngine(force_mock=self.mock_mode)
        self.guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
        self.verifier = StateVerifier(max_retries=3)
        self.llm = get_generative_provider("auto")

        self.state_machine = ReflexStateMachine(
            driver=self.driver,
            pruner=self.pruner,
            reflex_engine=self.reflex_engine,
            guardrail=self.guardrail,
            verifier=self.verifier,
            actuator=self.actuator,
            generative_provider=self.llm,
        )

        self.worker = AgentWorker(self.state_machine)
        self.worker.sig_transcript.connect(self._on_transcript)
        self.worker.sig_anticipation.connect(self._on_anticipation)
        self.worker.sig_step.connect(self._on_step)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.sig_guardrail.connect(self._on_guardrail)

        self.sig_voice_partial.connect(self._on_voice_partial)
        self.sig_voice_final.connect(self._on_voice_final)

        self.worker_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None
        self._is_voice_running = False

        self._init_ui()

        if self.enable_voice:
            self._start_voice_listener()

    def _init_ui(self):
        self.setStyleSheet(HUD_STYLESHEET)

        central = QWidget(self)
        central.setObjectName("HUDContainer")
        self.setCentralWidget(central)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 190))
        shadow.setOffset(0, 12)
        central.setGraphicsEffect(shadow)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(12)

        # 1. Header Bar: Pill Status & Controls
        header = QHBoxLayout()
        header.setSpacing(10)

        self.pulse = PulseIndicator()
        header.addWidget(self.pulse)

        self.lbl_title = QLabel("Reflex-Agent OS")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #f8fafc;")
        header.addWidget(self.lbl_title)

        self.lbl_model_badge = QLabel("⚡ Laya ModernBERT (<35ms)")
        self.lbl_model_badge.setStyleSheet(
            "background-color: rgba(0, 229, 255, 0.12); color: #00e5ff; "
            "border: 1px solid rgba(0, 229, 255, 0.35); border-radius: 9px; "
            "padding: 3px 8px; font-size: 11px; font-weight: 600;"
        )
        header.addWidget(self.lbl_model_badge)

        header.addStretch()

        self.btn_voice_indicator = QPushButton("🎙️ Ses (Alt_R)")
        self.btn_voice_indicator.setProperty("class", "ActiveVoiceButton" if self.enable_voice else "PillButton")
        self.btn_voice_indicator.clicked.connect(self._toggle_voice)
        header.addWidget(self.btn_voice_indicator)

        self.btn_mode = QPushButton("Mode: Mock" if self.mock_mode else "Mode: Physical")
        self.btn_mode.setProperty("class", "PillButton")
        self.btn_mode.clicked.connect(self._toggle_mode)
        header.addWidget(self.btn_mode)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(26, 26)
        self.btn_close.setStyleSheet(
            "background: rgba(255,255,255,0.06); border: none; border-radius: 13px; color: #94a3b8; font-size: 12px;"
        )
        self.btn_close.clicked.connect(self.close)
        header.addWidget(self.btn_close)

        layout.addLayout(header)

        # 2. Omnibar Input Deck
        input_deck = QHBoxLayout()
        input_deck.setSpacing(8)

        self.omnibar = QLineEdit()
        self.omnibar.setObjectName("Omnibar")
        self.omnibar.setPlaceholderText("Komut yazın veya Alt_R basılı tutun (örn: 'open firefox and go to youtube')...")
        self.omnibar.textChanged.connect(self._on_omnibar_typed)
        self.omnibar.returnPressed.connect(self.run_input_goal)
        input_deck.addWidget(self.omnibar)

        self.btn_run = QPushButton("⚡ Çalıştır")
        self.btn_run.setProperty("class", "ActionButton")
        self.btn_run.clicked.connect(self.run_input_goal)
        input_deck.addWidget(self.btn_run)

        layout.addLayout(input_deck)

        # 3. Quick Scenario Preset Pills (1-Click Test Runs)
        presets_layout = QHBoxLayout()
        presets_layout.setSpacing(6)

        p1 = QPushButton("🦊 Firefox & YouTube")
        p1.setProperty("class", "PillButton")
        p1.clicked.connect(lambda: self._set_and_run("open firefox and open a new tab and go to youtube in that new tab"))
        presets_layout.addWidget(p1)

        p2 = QPushButton("🚀 Antigravity Project")
        p2.setProperty("class", "PillButton")
        p2.clicked.connect(lambda: self._set_and_run("open antigravity and start a quick project"))
        presets_layout.addWidget(p2)

        p3 = QPushButton("📝 Notepad Notes")
        p3.setProperty("class", "PillButton")
        p3.clicked.connect(lambda: self._set_and_run("open notepad and type meeting notes"))
        presets_layout.addWidget(p3)

        p4 = QPushButton("💻 VSCode Terminal")
        p4.setProperty("class", "PillButton")
        p4.clicked.connect(lambda: self._set_and_run("focus vscode and toggle terminal"))
        presets_layout.addWidget(p4)

        presets_layout.addStretch()
        layout.addLayout(presets_layout)

        # 4. Andy Gao Anticipation Card (Speculative Pre-Dispatch Glow)
        self.anticipation_box = QFrame()
        self.anticipation_box.setObjectName("AnticipationBox")
        box_layout = QVBoxLayout(self.anticipation_box)
        box_layout.setContentsMargins(12, 10, 12, 10)
        box_layout.setSpacing(4)

        anticipate_header = QHBoxLayout()
        self.lbl_anticipate_title = QLabel("⚡ JEV / LAYA REFLEX PRE-DISPATCH")
        self.lbl_anticipate_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #00e5ff; letter-spacing: 0.6px;")
        anticipate_header.addWidget(self.lbl_anticipate_title)

        anticipate_header.addStretch()

        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_latency.setStyleSheet("font-size: 11px; color: #38bdf8; font-family: monospace; font-weight: bold;")
        anticipate_header.addWidget(self.lbl_latency)

        box_layout.addLayout(anticipate_header)

        self.lbl_anticipate_action = QLabel("Hazır: Kullanıcı komutunu veya ses akışını bekliyor...")
        self.lbl_anticipate_action.setStyleSheet("font-size: 12px; color: #ffffff; font-weight: 500;")
        box_layout.addWidget(self.lbl_anticipate_action)

        layout.addWidget(self.anticipation_box)

        # 5. Live Step Execution Feed (Scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; } "
            "QScrollBar:vertical { width: 4px; background: transparent; } "
            "QScrollBar::handle:vertical { background: rgba(255,255,255,0.2); border-radius: 2px; }"
        )
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background: transparent;")
        self.step_feed_layout = QVBoxLayout(self.scroll_widget)
        self.step_feed_layout.setContentsMargins(0, 0, 0, 0)
        self.step_feed_layout.setSpacing(6)
        self.step_feed_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)

        layout.addWidget(self.scroll_area, 1)

        # 6. Bottom Status Bar & Controls
        bottom_bar = QHBoxLayout()
        self.lbl_status = QLabel("Durum: Sistem 1 Karar Motoru Hazır (Alt_R tuşuna basılı tutarak konuşabilirsiniz)")
        self.lbl_status.setStyleSheet("font-size: 11px; color: #94a3b8;")
        bottom_bar.addWidget(self.lbl_status)

        bottom_bar.addStretch()

        self.btn_clear = QPushButton("Temizle")
        self.btn_clear.setProperty("class", "PillButton")
        self.btn_clear.clicked.connect(self._clear_feed)
        bottom_bar.addWidget(self.btn_clear)

        layout.addLayout(bottom_bar)

    def _on_omnibar_typed(self, text: str):
        """Speculative pre-dispatching in real time as user types (<35ms Laya inference)."""
        goal = text.strip()
        if len(goal) >= 4:
            try:
                t0 = time.perf_counter()
                state, _ = self.state_machine.sense_and_prune(goal)
                spec = self.reflex_engine.predict(state)
                lat = (time.perf_counter() - t0) * 1000.0

                target = spec.text_to_type or spec.selected_element_id or "Desktop"
                self.lbl_latency.setText(f"Latency: {lat:.1f} ms")
                self.lbl_anticipate_action.setText(f"⚡ Speculative Pre-Dispatch: [{spec.action_type}] ➔ {target}")
                self.pulse.set_color("#00e5ff")
            except Exception:
                pass

    def _set_and_run(self, goal: str):
        self.omnibar.setText(goal)
        self.run_input_goal()

    def run_input_goal(self):
        goal = self.omnibar.text().strip()
        if not goal:
            return

        self.pulse.set_color("#00e5ff")
        self.lbl_status.setText(f"İşleniyor: '{goal}'")

        self.worker_thread = threading.Thread(
            target=self.worker.run_goal,
            args=(goal,),
            daemon=True
        )
        self.worker_thread.start()

    def _toggle_mode(self):
        self.mock_mode = not self.mock_mode
        self.btn_mode.setText("Mode: Mock" if self.mock_mode else "Mode: Physical")
        self.driver = get_platform_driver(mock=self.mock_mode)
        self.actuator = get_actuator(mock=self.mock_mode)
        self.state_machine.driver = self.driver
        self.state_machine.actuator = self.actuator
        self.lbl_status.setText(f"Çalışma modu: {'Mock (Simülasyon)' if self.mock_mode else 'Physical (Gerçek Masaüstü)'}")

    def _toggle_voice(self):
        self.enable_voice = not self.enable_voice
        self.btn_voice_indicator.setProperty("class", "ActiveVoiceButton" if self.enable_voice else "PillButton")
        self.btn_voice_indicator.style().unpolish(self.btn_voice_indicator)
        self.btn_voice_indicator.style().polish(self.btn_voice_indicator)
        if self.enable_voice:
            self._start_voice_listener()
            self.lbl_status.setText("Sesli asistan aktif. Alt_R tuşuna basılı tutarak konuşun.")
        else:
            self._is_voice_running = False
            self.lbl_status.setText("Sesli asistan devre dışı bırakıldı.")

    def _start_voice_listener(self):
        if self._is_voice_running:
            return
        self._is_voice_running = True
        self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self._voice_thread.start()

    def _voice_loop(self):
        try:
            audio_stream = AudioCaptureStream(push_to_talk=True, ptt_key_name="alt_r")
            stt = StreamingWhisperSTT(model_size="base", force_mock=False)
            audio_stream.start()

            while self._is_voice_running:
                frames = audio_stream.get_audio_chunk(timeout=0.15)
                if frames is not None and len(frames) > 0:
                    text, is_final = stt.transcribe_chunk(frames)
                    if text:
                        if is_final:
                            self.sig_voice_final.emit(text)
                        else:
                            self.sig_voice_partial.emit(text)
                time.sleep(0.05)
        except Exception:
            pass

    def _on_voice_partial(self, partial_text: str):
        self.omnibar.setText(partial_text)
        self.pulse.set_color("#8b5cf6")  # Purple voice stream
        self.lbl_status.setText(f"🎙️ Dinleniyor: '{partial_text}'")

    def _on_voice_final(self, final_text: str):
        self.omnibar.setText(final_text)
        self.run_input_goal()

    def _clear_feed(self):
        while self.step_feed_layout.count() > 1:
            item = self.step_feed_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.lbl_anticipate_action.setText("Hazır: Yeni bir komut bekliyor...")
        self.lbl_latency.setText("Latency: -- ms")
        self.pulse.set_color("#10b981")

    # Worker Callbacks
    def _on_transcript(self, text: str):
        self.lbl_status.setText(text)

    def _on_anticipation(self, action: str, target: str, latency: float):
        self.lbl_latency.setText(f"Latency: {latency:.1f} ms")
        self.lbl_anticipate_action.setText(f"⚡ Speculative Pre-Dispatch: [{action}] ➔ {target}")
        self.pulse.set_color("#00e5ff")

    def _on_step(self, step_num: int, action: str, target: str, mutated: bool):
        card = QFrame()
        card.setProperty("class", "CardFrame")
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(10, 6, 10, 6)

        num_lbl = QLabel(f"Adım {step_num}")
        num_lbl.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 11px;")
        card_layout.addWidget(num_lbl)

        action_lbl = QLabel(f"{action}  ➔  '{target}'")
        action_lbl.setStyleSheet("color: #f1f5f9; font-size: 12px;")
        card_layout.addWidget(action_lbl)

        card_layout.addStretch()

        status_lbl = QLabel("✓ Değişti" if mutated else "✓ Yürütüldü")
        status_lbl.setStyleSheet("color: #10b981; font-size: 11px; font-weight: bold;")
        card_layout.addWidget(status_lbl)

        idx = max(0, self.step_feed_layout.count() - 1)
        self.step_feed_layout.insertWidget(idx, card)

    def _on_finished(self, status: str, step_count: int):
        if status == "SUCCESS":
            self.pulse.set_color("#10b981")
            self.lbl_status.setText(f"✓ Görev {step_count} adımda başarıyla tamamlandı.")
        elif status == "GUARDRAIL_BLOCKED":
            self.pulse.set_color("#ef4444")
            self.lbl_status.setText("⚠ Güvenlik Koruması (Guardrail) eylemi durdurdu.")
        else:
            self.pulse.set_color("#f59e0b")
            self.lbl_status.setText(f"Görev sonlandı: {status}")

    def _on_guardrail(self, action: str, reason: str):
        alert_card = QFrame()
        alert_card.setObjectName("GuardrailBox")
        al_layout = QVBoxLayout(alert_card)
        al_layout.setContentsMargins(10, 8, 10, 8)

        title = QLabel(f"🛡️ GÜVENLİK KORUMASI: {action} ENGELLENDİ")
        title.setStyleSheet("font-weight: bold; color: #ef4444; font-size: 12px;")
        al_layout.addWidget(title)

        desc = QLabel(reason)
        desc.setStyleSheet("color: #fca5a5; font-size: 11px;")
        al_layout.addWidget(desc)

        idx = max(0, self.step_feed_layout.count() - 1)
        self.step_feed_layout.insertWidget(idx, alert_card)

    # Window Dragging Logic
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()


def launch_hud(mock: bool = False, enable_voice: bool = True):
    """Entry point to launch the floating HUD overlay."""
    app = QApplication.instance() or QApplication(sys.argv)
    hud = FloatingHUD(mock_mode=mock, enable_voice=enable_voice)
    hud.show()
    return app.exec_()


if __name__ == "__main__":
    mock_flag = "--mock" in sys.argv
    no_voice = "--no-voice" in sys.argv
    sys.exit(launch_hud(mock=mock_flag, enable_voice=not no_voice))
