"""Unit tests for Voice Capture, Streaming Speech Anticipation, and Desktop OS Assistant."""

import time
import pytest
import numpy as np

from src.orchestrator.models import UIElement, AgentState, ReflexDecision
from src.reflex.laya_engine import LayaReflexEngine
from src.voice.audio_stream import AudioCaptureStream
from src.voice.stt_engine import StreamingWhisperSTT
from src.voice.anticipator import SpeechToIntentAnticipator
from src.actuator.mouse_keyboard import MockActuator
from src.assistant.assistant_core import PersonalVoiceAssistant


@pytest.fixture
def mock_elements():
    return [
        UIElement(id="btn_login", label="Login", control_type="Button", bbox=(10, 20, 80, 30)),
        UIElement(id="txt_user", label="Username", control_type="Edit", bbox=(10, 60, 150, 30)),
        UIElement(id="btn_delete", label="Delete Account", control_type="Button", bbox=(10, 100, 120, 30)),
    ]


def test_audio_stream_simulation():
    """Verifies that AudioCaptureStream properly buffers and yields audio chunks."""
    stream = AudioCaptureStream(push_to_talk=False)
    dummy_audio = np.zeros(1600, dtype=np.float32)

    stream.simulate_audio_input(dummy_audio)
    retrieved = stream.get_chunk(timeout=0.1)

    assert retrieved is not None
    assert len(retrieved) == 1600


def test_streaming_speech_anticipation_partial(mock_elements):
    """Verifies that Laya pre-stages action on partial streaming speech."""
    engine = LayaReflexEngine(force_mock=True)
    anticipator = SpeechToIntentAnticipator(reflex_engine=engine)

    # User starts speaking: "Lütfen login..."
    partial_text = "Lütfen login"
    staged = anticipator.feed_partial(
        partial_text, available_elements=mock_elements, active_window="Test App"
    )

    assert staged is not None
    assert staged.decision.selected_element_id == "btn_login"
    assert staged.decision.action_type == "CLICK"

    # User finishes speaking: "Lütfen login butonuna bas"
    final_text = "Lütfen login butonuna bas"
    decision, was_pre_staged = anticipator.commit(
        final_text, available_elements=mock_elements, active_window="Test App"
    )

    assert decision.selected_element_id == "btn_login"
    assert was_pre_staged is True


def test_voice_barge_in_cancellation(mock_elements):
    """Verifies that verbal 'vazgeçtim' / 'dur' aborts staged action."""
    engine = LayaReflexEngine(force_mock=True)
    anticipator = SpeechToIntentAnticipator(reflex_engine=engine)

    # User starts speaking: "Delete Account..."
    anticipator.feed_partial("Delete Account", mock_elements, "Test App")
    assert anticipator.staged_action is not None

    # User yells: "Hayır dur vazgeçtim!"
    staged_cancel = anticipator.feed_partial("dur vazgeçtim", mock_elements, "Test App")
    assert staged_cancel is None
    assert anticipator.is_interrupted is True

    decision, was_pre_staged = anticipator.commit("dur vazgeçtim", mock_elements, "Test App")
    assert decision.action_type == "WAIT"
    assert was_pre_staged is False


def test_expanded_desktop_actuator_controls():
    """Verifies that MockActuator properly records clipboard pasting, app launch, and hotkeys."""
    actuator = MockActuator()

    # Test hotkey
    actuator.hotkey("ctrl", "shift", "esc")
    assert any(e["action"] == "hotkey" and "ctrl" in e["keys"] for e in actuator.events)

    # Test clipboard paste
    long_text = "Bu bir uzun test paragrafıdır ve doğrudan panoya kopyalanıp yapıştırılmalıdır."
    actuator.paste_text(long_text)
    assert any(e["action"] == "paste_text" and e["text"] == long_text for e in actuator.events)

    # Test app launch
    actuator.launch_app("notepad.exe")
    assert any(e["action"] == "launch_app" and e["app"] == "notepad.exe" for e in actuator.events)


def test_voice_assistant_generative_drafting_dispatch(mock_elements):
    """Verifies that text preparation requests activate System 2 LLM and clipboard paste."""
    assistant = PersonalVoiceAssistant(
        force_mock_voice=True,
        headless=True,
    )
    # Mock elements in driver
    assistant.driver.set_elements(mock_elements)

    res = assistant.process_speech_utterance("Yarınki toplantı için bir özet hazırla")
    assert res["status"] == "SUCCESS_GENERATED"
    assert "text" in res
    assert len(res["text"]) > 20


def test_voice_guardrail_confirmation_interception(mock_elements):
    """Verifies that vocal destructive commands trigger REQUIRES_CONFIRMATION."""
    assistant = PersonalVoiceAssistant(
        force_mock_voice=True,
        headless=True,
    )
    assistant.driver.set_elements(mock_elements)

    # Without confirmation
    res_unconfirmed = assistant.process_speech_utterance("Delete Account", confirmed_by_user=False)
    assert res_unconfirmed["status"] == "REQUIRES_CONFIRMATION"

    # With confirmation
    res_confirmed = assistant.process_speech_utterance("Delete Account", confirmed_by_user=True)
    assert res_confirmed["status"] == "SUCCESS_EXECUTED"


def test_voice_assistant_email_drafting(mock_elements):
    """Verifies that 'e-posta yaz' triggers System 2 LLM drafting."""
    assistant = PersonalVoiceAssistant(
        force_mock_voice=True,
        headless=True,
    )
    assistant.driver.set_elements(mock_elements)

    res = assistant.process_speech_utterance("Yarınki yönetim toplantısı için 2 maddelik e-posta yaz")
    assert res["status"] == "SUCCESS_GENERATED"
    assert "text" in res
    assert len(res["text"]) > 20


def test_voice_assistant_turkish_typing_dispatch(mock_elements):
    """Verifies that Turkish typing phrasing correctly types into targeted field."""
    assistant = PersonalVoiceAssistant(
        force_mock_voice=True,
        headless=True,
    )
    assistant.driver.set_elements(mock_elements)

    res = assistant.process_speech_utterance("kullanıcı adı alanına admin yaz")
    assert res["status"] == "SUCCESS_EXECUTED"
    assert res["decision"].action_type == "TYPE"
    assert res["decision"].text_to_type == "admin"
    assert res["decision"].selected_element_id == "txt_user"

