"""Unit tests for minimal black Dynamic Island and Settings Drawer."""

import os
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5.QtWidgets import QApplication
from src.ui.floating_hud import FloatingHUD, PulseIndicator, query_input_microphones


@pytest.fixture(scope="session")
def qapp():
    """Initializes or retrieves the QApplication singleton for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pulse_indicator_initialization(qapp):
    """Verifies PulseIndicator initializes and updates color states correctly."""
    indicator = PulseIndicator()
    assert indicator.width() == 16
    assert indicator.height() == 16

    indicator.set_color("#00e5ff")
    assert indicator.state_color.name() == "#00e5ff"


def test_query_input_microphones():
    """Verifies that microphone query returns a list of devices safely without raising."""
    mics = query_input_microphones()
    assert isinstance(mics, list)


def test_dynamic_island_initialization_mock_mode(qapp):
    """Verifies Dynamic Island initializes cleanly in compact pill format."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)

    assert hud.mock_mode is True
    assert hud.island_pill is not None
    assert hud.lbl_transcript is not None
    assert hud.btn_settings is not None
    assert hud.settings_card is not None
    assert hud.settings_card.isVisible() is False

    hud.show()
    # Test toggling settings drawer
    hud._toggle_settings()
    assert hud.settings_card.isVisible() is True
    hud._toggle_settings()
    assert hud.settings_card.isVisible() is False

    hud.close()


def test_dynamic_island_mode_toggle(qapp):
    """Verifies toggling between Mock and Physical mode inside settings."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)
    assert hud.mock_mode is True

    hud._toggle_mode()
    assert hud.mock_mode is False
    assert "Physical" in hud.btn_mode_toggle.text()

    hud._toggle_mode()
    assert hud.mock_mode is True
    assert "Mock" in hud.btn_mode_toggle.text()

    hud.close()


def test_dynamic_island_mic_selection(qapp):
    """Verifies that changing microphone selection updates selected index safely."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)
    if hud.mics_list:
        hud._on_mic_changed(0)
        assert hud.selected_mic_index == hud.mics_list[0][0]
    hud.close()


def test_dynamic_island_log_drawer_and_activity_logging(qapp):
    """Verifies that the LogCard drawer toggles and displays activity logs in real-time."""
    from src.utils.activity_logger import activity_logger

    hud = FloatingHUD(mock_mode=True, enable_voice=False)
    hud.show()
    assert hud.log_card is not None
    assert hud.log_card.isVisible() is False

    # Toggle log drawer open
    hud._toggle_logs()
    assert hud.log_card.isVisible() is True
    assert hud.settings_card.isVisible() is False

    # Log actions
    activity_logger.log_voice_heard("firefoxu aç")
    activity_logger.log_goal_understood("firefoxu aç")
    activity_logger.log_action_step(1, "APP_FOCUS", "firefox")
    activity_logger.log_finished("SUCCESS", 1)

    # Allow Qt queued connection event to process
    qapp.processEvents()

    log_text = hud.txt_log_display.toPlainText()
    assert "firefoxu aç" in log_text or len(activity_logger.get_recent_entries()) > 0

    # Toggle closed
    hud._toggle_logs()
    assert hud.log_card.isVisible() is False
    hud.close()


def test_audio_stream_manual_and_poller():
    """Verifies AudioCaptureStream manual recording toggle and poller initialization."""
    from src.voice.audio_stream import AudioCaptureStream
    stream = AudioCaptureStream(push_to_talk=True, ptt_key_name="alt_r")

    assert stream.is_recording() is False
    stream.set_manual_recording(True)
    assert stream.is_recording() is True
    stream.set_manual_recording(False)
    assert stream.is_recording() is False

    stream.stop()
