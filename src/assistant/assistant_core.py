"""Unified Local Autonomous Voice Assistant coordinating Voice, Reflex, Actuator, and LLM."""

import os
import time
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from rich.console import Console
from rich.panel import Panel

from ..voice.audio_stream import AudioCaptureStream
from ..voice.stt_engine import StreamingWhisperSTT
from ..voice.anticipator import SpeechToIntentAnticipator
from ..perception.accessibility import get_platform_driver, BaseAccessibilityDriver
from ..perception.state_pruner import UIStatePruner
from ..reflex.laya_engine import LayaReflexEngine
from ..reflex.guardrail import SafetyGuardrail
from ..reflex.verifier import StateVerifier
from ..actuator.mouse_keyboard import get_actuator, BaseActuator
from ..generative.llm_fallback import get_generative_provider, BaseGenerativeProvider
from ..orchestrator.models import UIElement, AgentState, ReflexDecision
from ..orchestrator.state_machine import ReflexStateMachine

logger = logging.getLogger(__name__)
console = Console()


class PersonalVoiceAssistant:
    """100% Local Autonomous Desktop Assistant with Streaming Speech Anticipation."""

    def __init__(
        self,
        push_to_talk: bool = True,
        ptt_key: str = "alt_r",
        whisper_model: str = "base",
        llm_model: str = "qwen2.5:3b",
        force_mock_voice: bool = False,
        headless: bool = False,
    ):
        self.push_to_talk = push_to_talk
        self.ptt_key = ptt_key
        self.headless = headless
        self._is_running = False

        # 1. Voice Ingestion & STT
        self.audio_stream = AudioCaptureStream(
            push_to_talk=push_to_talk,
            ptt_key_name=ptt_key,
        )
        self.stt_engine = StreamingWhisperSTT(
            model_size=whisper_model,
            force_mock=force_mock_voice,
        )

        # 2. System 1 Reflex Engine & Anticipator
        self.reflex_engine = LayaReflexEngine(force_mock=True)
        self.anticipator = SpeechToIntentAnticipator(reflex_engine=self.reflex_engine)

        # 3. Perception & Actuation
        self.driver: BaseAccessibilityDriver = get_platform_driver(mock=headless)
        self.pruner = UIStatePruner(max_elements=25)
        self.actuator: BaseActuator = get_actuator(mock=headless, headless=headless)

        # 4. Safety & System 2 Generative LLM
        self.guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
        self.verifier = StateVerifier(max_retries=3)
        self.llm_provider: BaseGenerativeProvider = get_generative_provider(
            provider_type="auto", model_name=llm_model
        )

        # 5. Core Orchestrator
        self.state_machine = ReflexStateMachine(
            driver=self.driver,
            pruner=self.pruner,
            reflex_engine=self.reflex_engine,
            guardrail=self.guardrail,
            verifier=self.verifier,
            actuator=self.actuator,
            generative_provider=self.llm_provider,
        )

    def process_speech_utterance(
        self,
        spoken_text: str,
        was_pre_staged: bool = False,
        confirmed_by_user: bool = False,
    ) -> Dict[str, Any]:
        """Processes a completed voice command through the dual-system pipeline."""
        if not spoken_text.strip():
            return {"status": "EMPTY_INPUT"}

        console.print(f"[bold cyan]Voice Command:[/bold cyan] '[bold yellow]{spoken_text}[/bold yellow]' (Pre-staged: {was_pre_staged})")

        # Check for verbal cancellation
        if self.anticipator.is_cancellation_phrase(spoken_text):
            console.print("[bold red]Action Cancelled by User Voice Command.[/bold red]")
            return {"status": "CANCELLED_BY_VOICE"}

        # 1. Sense environment
        win_title = self.driver.get_active_window_title()
        raw_elements = self.driver.get_ui_elements()
        pruned_elements = self.pruner.prune(raw_elements)

        # 2. Decide via Anticipator or Reflex Engine
        decision, staged = self.anticipator.commit(
            spoken_text, pruned_elements, active_window=win_title
        )

        # Find target element
        target_elem: Optional[UIElement] = None
        if decision.selected_element_id:
            for el in pruned_elements:
                if el.id == decision.selected_element_id:
                    target_elem = el
                    break

        # 3. Safety Guardrail Verification
        decision = self.guardrail.evaluate_and_annotate(
            decision, element=target_elem, context_text=spoken_text
        )

        if decision.is_destructive and not confirmed_by_user:
            elem_name = target_elem.label if target_elem else (decision.selected_element_id or "Öğe")
            msg = (
                f"[bold red]DİKKAT (Güvenlik Uyarısı):[/bold red] "
                f"'{elem_name}' üzerinde '{decision.action_type}' eylemi yıkıcı olarak sınıflandırıldı!\n"
                f"Gerçekleştirmek için komutu 'onaylıyorum' veya 'evet' ile teyit edin."
            )
            console.print(Panel(msg, style="bold red"))
            return {"status": "REQUIRES_CONFIRMATION", "decision": decision}

        # 4. Act (Generative LLM Text Drafting vs Discrete Action)
        if decision.action_type == "CALL_LLM":
            console.print("[bold magenta]System 2 LLM Metin Hazırlıyor...[/bold magenta]")
            generated_text = self.llm_provider.generate(
                prompt=spoken_text,
                context=f"Aktif Pencere: {win_title}",
            )
            console.print(Panel(generated_text, title="[bold green]Hazırlanan Metin[/bold green]"))

            # Automatically inject into active window via clipboard
            console.print("[bold green]Metin aktif pencereye yapıştırılıyor...[/bold green]")
            self.actuator.paste_text(generated_text)
            return {"status": "SUCCESS_GENERATED", "text": generated_text}
        else:
            console.print(f"[bold green]Eylem Gerçekleştiriliyor:[/bold green] {decision.action_type} -> {decision.selected_element_id}")
            self.actuator.execute_decision(decision, element=target_elem)
            return {"status": "SUCCESS_EXECUTED", "decision": decision}

    def start_listening_loop(self) -> None:
        """Starts the interactive real-time voice processing loop."""
        self._is_running = True
        self.audio_stream.start()

        mode_str = f"Bas-Konuş ('{self.ptt_key.upper()}' tuşuna basılı tutun)" if self.push_to_talk else "Sürekli Dinleme (VAD)"
        console.print(Panel(
            f"[bold green]Reflex-Agent Kişisel Asistan Devrede[/bold green]\n"
            f"Mod: [bold cyan]{mode_str}[/bold cyan]\n"
            f"LLM: [bold yellow]{getattr(self.llm_provider, 'model_name', 'Local')}[/bold yellow]\n"
            f"STT: [bold yellow]Local faster-whisper (int8)[/bold yellow]\n"
            f"[dim]Çıkış için Ctrl+C tuşlarına basın.[/dim]"
        ))

        accumulated_audio: List[np.ndarray] = []
        last_speech_time = time.time()

        try:
            while self._is_running:
                chunk = self.audio_stream.get_chunk(timeout=0.1)
                is_speaking = self.audio_stream.is_recording()

                if chunk is not None:
                    accumulated_audio.append(chunk)
                    last_speech_time = time.time()

                    # Real-time speculative anticipation on audio length >= 0.5s
                    total_samples = sum(len(c) for c in accumulated_audio)
                    if total_samples >= 8000:  # 0.5 sec
                        audio_np = np.concatenate(accumulated_audio)
                        partial_transcript = self.stt_engine.transcribe(audio_np)
                        if partial_transcript:
                            # Pre-stage action while speaking
                            raw_elems = self.driver.get_ui_elements()
                            pruned = self.pruner.prune(raw_elems)
                            staged = self.anticipator.feed_partial(
                                partial_transcript,
                                available_elements=pruned,
                                active_window=self.driver.get_active_window_title(),
                            )
                            if staged and not staged.is_cancelled:
                                console.print(
                                    f"[dim cyan]Konuşurken Anlaşıldı:[/dim cyan] '{staged.partial_text}' "
                                    f"-> Ön-Aşama: {staged.decision.action_type} on {staged.decision.selected_element_id}"
                                )

                # Utterance finished condition (PTT release or 0.6s silence)
                if accumulated_audio and not is_speaking:
                    if time.time() - last_speech_time > 0.4:
                        full_audio = np.concatenate(accumulated_audio)
                        accumulated_audio.clear()

                        final_transcript = self.stt_engine.transcribe(full_audio)
                        if final_transcript:
                            self.process_speech_utterance(final_transcript)
                        self.anticipator.reset()

                time.sleep(0.02)
        except KeyboardInterrupt:
            console.print("\n[yellow]Asistan durduruluyor...[/yellow]")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stops the assistant and releases audio devices."""
        self._is_running = False
        self.audio_stream.stop()
        console.print("[dim]Ses kaynakları serbest bırakıldı.[/dim]")
