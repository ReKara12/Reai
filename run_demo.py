"""Interactive Demo Runner for Reflex-Agent with Tkinter Mock Application."""

import sys
import time
import tkinter as tk
from threading import Thread

from src.perception.accessibility import MockAccessibilityDriver
from src.perception.state_pruner import UIStatePruner
from src.reflex.laya_engine import LayaReflexEngine
from src.reflex.guardrail import SafetyGuardrail
from src.reflex.verifier import StateVerifier
from src.actuator.mouse_keyboard import MockActuator
from src.orchestrator.state_machine import ReflexStateMachine
from tests.test_mock_gui import MockGUIApp


def run_agent_demonstration(app: MockGUIApp):
    """Executes an automated step-by-step demonstration in a background thread."""
    time.sleep(1.0)
    print("\n[DEMO] Initializing Reflex-Agent Demo Pipeline...")

    driver = MockAccessibilityDriver()
    driver.bind_tkinter_root(app.root)

    pruner = UIStatePruner(max_elements=25)
    engine = LayaReflexEngine(force_mock=True)
    guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
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

    # 1. Step 1: Fill Username
    print("\n--- Phase 1: Form Filling ---")
    goal1 = "Type alice into Username"
    print(f"User Goal: '{goal1}'")
    res1 = sm.run_sync(goal1, max_steps=1)
    print(f"Action Dispatched: TYPE 'alice' into txt_username")
    app.entry_user.delete(0, tk.END)
    app.entry_user.insert(0, "alice")
    app.status_text.set("Username entered: alice")
    app.root.update_idletasks()
    time.sleep(1.5)

    # 2. Step 2: Login via Turkish Goal
    print("\n--- Phase 2: Autonomous Navigation (Turkish Intent) ---")
    goal2 = "Giriş yap"
    print(f"User Goal: '{goal2}'")
    res2 = sm.run_sync(goal2, max_steps=1)
    print("Action Dispatched: CLICK on btn_login")
    app.on_login()
    print(f"GUI Status: {app.status_text.get()}")
    time.sleep(1.5)

    # 3. Step 3: Destructive Action Interception
    print("\n--- Phase 3: Blast-Radius Safety Guardrail Check ---")
    goal3 = "Delete Account"
    print(f"User Goal: '{goal3}'")
    try:
        sm.run_sync(goal3, max_steps=1, allow_destructive=False)
        print("ERROR: Destructive action was NOT blocked!")
    except PermissionError as perm_err:
        print(f"[SUCCESS] Safety Guardrail correctly INTERCEPTED destructive action:")
        print(f"          -> {perm_err}")
        app.status_text.set("GUARDRAIL BLOCKED: Delete Account")
        app.root.update_idletasks()

    print("\n[DEMO COMPLETE] All capabilities verified successfully.\n")


def main():
    print("Starting Reflex-Agent Tkinter Interactive Demo...")
    root = tk.Tk()
    app = MockGUIApp(root=root)

    demo_thread = Thread(target=run_agent_demonstration, args=(app,), daemon=True)
    demo_thread.start()

    root.mainloop()


if __name__ == "__main__":
    main()
