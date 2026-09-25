"""Automated test suite verifying the Reflex-Agent against a Tkinter Mock GUI application."""

import sys
import time
import pytest
import tkinter as tk
from typing import Optional

from src.perception.accessibility import MockAccessibilityDriver
from src.perception.state_pruner import UIStatePruner
from src.reflex.laya_engine import LayaReflexEngine
from src.reflex.guardrail import SafetyGuardrail
from src.reflex.verifier import StateVerifier
from src.actuator.mouse_keyboard import MockActuator
from src.orchestrator.models import UIElement, AgentState, ReflexDecision
from src.orchestrator.state_machine import ReflexStateMachine


class MockGUIApp:
    """Tkinter Demo Application containing text inputs, normal buttons, and dangerous actions."""

    def __init__(self, root: Optional[tk.Tk] = None):
        self.root = root or tk.Tk()
        self.root.title("Reflex-Agent Mock Application")
        self.root.geometry("320x240")

        # Username Label & Entry
        self.lbl_user = tk.Label(self.root, text="Username:")
        self.lbl_user.custom_id = "lbl_username"
        self.lbl_user.pack(pady=4)

        self.entry_user = tk.Entry(self.root)
        self.entry_user.custom_id = "txt_username"
        self.entry_user.pack(pady=4)

        # Submit / Login Button
        self.btn_login = tk.Button(self.root, text="Login", command=self.on_login)
        self.btn_login.custom_id = "btn_login"
        self.btn_login.pack(pady=6)

        # Destructive Action Button
        self.btn_delete = tk.Button(
            self.root, text="Delete Account", fg="red", command=self.on_delete
        )
        self.btn_delete.custom_id = "btn_delete"
        self.btn_delete.pack(pady=6)

        # Status Label
        self.status_text = tk.StringVar(value="Ready")
        self.lbl_status = tk.Label(
            self.root, textvariable=self.status_text, relief=tk.SUNKEN, bd=1
        )
        self.lbl_status.custom_id = "lbl_status"
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=4)

        self.root.update_idletasks()

    def on_login(self) -> None:
        username = self.entry_user.get().strip() or "Guest"
        self.status_text.set(f"Logged in successfully as {username}")
        self.root.update_idletasks()

    def on_delete(self) -> None:
        self.status_text.set("Account Deleted!")
        self.root.update_idletasks()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


@pytest.fixture
def mock_gui():
    """Fixture providing an instantiated Tkinter MockGUIApp."""
    root = tk.Tk()
    root.withdraw()  # Headless mode for CI/test stability
    app = MockGUIApp(root=root)
    yield app
    app.destroy()


def test_laya_selects_login_for_turkish_goal(mock_gui):
    """Validates that Laya selects the 'Login' button when given the Turkish goal 'Giriş yap'."""
    driver = MockAccessibilityDriver()
    driver.bind_tkinter_root(mock_gui.root)

    elements = driver.get_ui_elements()
    pruner = UIStatePruner(max_elements=25)
    pruned = pruner.prune(elements)

    engine = LayaReflexEngine(force_mock=True)

    state = AgentState(
        user_goal="Giriş yap",
        active_window="Reflex-Agent Mock Application",
        available_elements=pruned,
    )

    t0 = time.perf_counter()
    decision = engine.predict(state)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert decision.selected_element_id == "btn_login", (
        f"Expected 'btn_login' for goal 'Giriş yap', but got '{decision.selected_element_id}'"
    )
    assert decision.action_type == "CLICK"
    assert latency_ms <= 50.0, f"Latency ({latency_ms:.2f} ms) exceeded 50 ms budget"


def test_destructive_delete_account_raises_permission_error(mock_gui):
    """Validates that attempting to click 'Delete Account' raises PermissionError via SafetyGuardrail."""
    driver = MockAccessibilityDriver()
    driver.bind_tkinter_root(mock_gui.root)

    elements = driver.get_ui_elements()
    pruner = UIStatePruner(max_elements=25)
    pruned = pruner.prune(elements)

    engine = LayaReflexEngine(force_mock=True)
    guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
    actuator = MockActuator()

    state = AgentState(
        user_goal="Delete Account",
        active_window="Reflex-Agent Mock Application",
        available_elements=pruned,
    )

    decision = engine.predict(state)
    assert decision.selected_element_id == "btn_delete"

    # Evaluate safety guardrail
    target_elem = next((e for e in pruned if e.id == decision.selected_element_id), None)
    annotated_decision = guardrail.evaluate_and_annotate(decision, element=target_elem)

    assert annotated_decision.is_destructive is True

    # Assert that execution is blocked without explicit confirmation
    with pytest.raises(PermissionError) as exc_info:
        guardrail.enforce(annotated_decision, element=target_elem, confirmed_by_user=False)

    assert "Blocked Destructive Action" in str(exc_info.value)
    # Ensure actuator was never called
    assert len(actuator.events) == 0


@pytest.mark.asyncio
async def test_form_filling_and_login_flow(mock_gui):
    """Tests the state machine executing form input and subsequent login."""
    driver = MockAccessibilityDriver()
    driver.bind_tkinter_root(mock_gui.root)

    pruner = UIStatePruner(max_elements=25)
    engine = LayaReflexEngine(force_mock=True)
    guardrail = SafetyGuardrail()
    verifier = StateVerifier(max_retries=3)
    actuator = MockActuator()

    sm = ReflexStateMachine(
        driver=driver,
        pruner=pruner,
        reflex_engine=engine,
        guardrail=guardrail,
        verifier=verifier,
        actuator=actuator,
    )

    # Step 1: Type admin into username
    res1 = await sm.step("Type admin into Username")
    assert res1["decision"].action_type == "TYPE"
    assert res1["decision"].selected_element_id == "txt_username"

    # Simulate UI update
    mock_gui.entry_user.delete(0, tk.END)
    mock_gui.entry_user.insert(0, "admin")
    mock_gui.root.update_idletasks()

    # Step 2: Click Login
    res2 = await sm.step("Giriş yap")
    assert res2["decision"].action_type == "CLICK"
    assert res2["decision"].selected_element_id == "btn_login"

    # Trigger click on GUI
    mock_gui.on_login()
    assert "Logged in successfully" in mock_gui.status_text.get()


def test_decision_dispatch_latency_under_50ms(mock_gui):
    """Validates that System 1 decisions are consistently dispatched in <= 50 ms."""
    driver = MockAccessibilityDriver()
    driver.bind_tkinter_root(mock_gui.root)
    pruned = UIStatePruner().prune(driver.get_ui_elements())

    engine = LayaReflexEngine(force_mock=True)
    state = AgentState(
        user_goal="Login",
        active_window="Reflex-Agent Mock Application",
        available_elements=pruned,
    )

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        decision = engine.predict(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert decision.execution_latency_ms is not None

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    assert avg_latency <= 50.0, f"Average latency {avg_latency:.2f} ms > 50 ms"
    assert p95_latency <= 50.0, f"P95 latency {p95_latency:.2f} ms > 50 ms"
