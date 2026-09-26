"""Unit tests for PyQt5 Floating HUD and Glassmorphic Overlay."""

import os
import pytest

# Ensure offscreen Qt platform for headless CI / unit test environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5.QtWidgets import QApplication
from src.ui.floating_hud import FloatingHUD, PulseIndicator


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
    assert indicator.width() == 18
    assert indicator.height() == 18

    indicator.set_color("#00e5ff")
    assert indicator.state_color.name() == "#00e5ff"


def test_floating_hud_initialization_mock_mode(qapp):
    """Verifies FloatingHUD initializes with correct frameless flags and components."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)

    assert hud.mock_mode is True
    assert hud.lbl_title.text() == "Reflex-Agent OS"
    assert "ModernBERT" in hud.lbl_model_badge.text()
    assert hud.omnibar is not None
    assert hud.anticipation_box is not None

    # Test preset injection
    test_goal = "open firefox and open a new tab and go to youtube in that new tab"
    hud.omnibar.setText(test_goal)
    assert hud.omnibar.text() == test_goal

    hud.close()


def test_floating_hud_speculative_pre_dispatch_trigger(qapp):
    """Verifies that typing a goal triggers instant speculative pre-dispatching."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)

    # Type goal
    hud.omnibar.setText("open firefox")
    assert "⚡" in hud.lbl_anticipate_action.text() or "firefox" in hud.lbl_anticipate_action.text().lower()

    # Clear feed
    hud._clear_feed()
    assert "Hazır" in hud.lbl_anticipate_action.text()

    hud.close()


def test_floating_hud_mode_toggle(qapp):
    """Verifies mode toggling between Mock and Physical execution."""
    hud = FloatingHUD(mock_mode=True, enable_voice=False)
    assert hud.mock_mode is True

    hud._toggle_mode()
    assert hud.mock_mode is False
    assert "Physical" in hud.btn_mode.text()

    hud._toggle_mode()
    assert hud.mock_mode is True
    assert "Mock" in hud.btn_mode.text()

    hud.close()
