"""Unit tests for Laya Reflex Engine decisions, action selection, and latency benchmarking."""

import time
import pytest
from src.orchestrator.models import UIElement, AgentState, ReflexDecision
from src.reflex.laya_engine import LayaReflexEngine
from src.perception.state_pruner import UIStatePruner


@pytest.fixture
def sample_elements():
    return [
        UIElement(id="btn_submit", label="Submit Form", control_type="Button", bbox=(100, 200, 80, 30)),
        UIElement(id="txt_email", label="Email Address", control_type="Edit", bbox=(100, 100, 200, 30)),
        UIElement(id="txt_pass", label="Password", control_type="Edit", bbox=(100, 150, 200, 30)),
        UIElement(id="chk_remember", label="Remember Me", control_type="CheckBox", bbox=(100, 180, 120, 20)),
        UIElement(id="btn_cancel", label="Cancel", control_type="Button", bbox=(190, 200, 80, 30)),
    ]


def test_action_selection_click(sample_elements):
    """Tests discrete CLICK action selection on Submit button."""
    engine = LayaReflexEngine(force_mock=True)
    state = AgentState(
        user_goal="Click on the Submit Form button",
        active_window="Registration Window",
        available_elements=sample_elements,
    )

    decision = engine.predict(state)
    assert decision.selected_element_id == "btn_submit"
    assert decision.action_type == "CLICK"
    assert decision.confidence >= 0.50
    assert decision.is_task_completed is False


def test_action_selection_type(sample_elements):
    """Tests TYPE action selection and text extraction."""
    engine = LayaReflexEngine(force_mock=True)
    state = AgentState(
        user_goal="Type 'user@example.com' into Email Address",
        active_window="Registration Window",
        available_elements=sample_elements,
    )

    decision = engine.predict(state)
    assert decision.selected_element_id == "txt_email"
    assert decision.action_type == "TYPE"
    assert decision.text_to_type == "user@example.com"


def test_action_selection_call_llm(sample_elements):
    """Tests fallback to System 2 CALL_LLM when complex text synthesis is required."""
    engine = LayaReflexEngine(force_mock=True)
    state = AgentState(
        user_goal="Draft a creative story about robots for the user",
        active_window="Story Editor",
        available_elements=sample_elements,
    )

    decision = engine.predict(state)
    assert decision.action_type == "CALL_LLM"


def test_task_completion_check():
    """Tests task completion recognition via noul / status evaluation."""
    engine = LayaReflexEngine(force_mock=True)
    completed_elements = [
        UIElement(id="lbl_msg", label="Operation completed successfully", control_type="Text", bbox=(10, 10, 200, 20)),
        UIElement(id="btn_ok", label="OK", control_type="Button", bbox=(10, 40, 60, 25)),
    ]

    state = AgentState(
        user_goal="Complete user registration",
        active_window="Success Dialog",
        available_elements=completed_elements,
    )

    decision = engine.predict(state)
    assert decision.is_task_completed is True


def test_ui_state_pruning_token_budget():
    """Verifies that dense UI trees are pruned to <= 25 nodes and containers are removed."""
    dense_elements = []
    # Add 10 non-interactive containers
    for i in range(10):
        dense_elements.append(
            UIElement(id=f"pane_{i}", label="", control_type="Pane", bbox=(0, i * 20, 500, 20))
        )
    # Add 10 empty decorative labels
    for i in range(10):
        dense_elements.append(
            UIElement(id=f"dec_{i}", label="", control_type="Text", bbox=(0, 0, 0, 0))
        )
    # Add 30 interactive buttons
    for i in range(30):
        dense_elements.append(
            UIElement(id=f"btn_{i}", label=f"Action {i}", control_type="Button", bbox=(10, i * 25, 80, 20))
        )

    pruner = UIStatePruner(max_elements=25)
    pruned = pruner.prune(dense_elements, focus_point=(10, 10))

    assert len(pruned) <= 25
    # All preserved elements must be interactive
    for elem in pruned:
        assert elem.control_type.lower() == "button"


def test_latency_benchmark_50_iterations(sample_elements):
    """Benchmarks 50 decision iterations to verify SLA <= 50 ms."""
    engine = LayaReflexEngine(force_mock=True)
    state = AgentState(
        user_goal="Submit Form",
        active_window="Registration Window",
        available_elements=sample_elements,
    )

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        decision = engine.predict(state)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        assert decision.execution_latency_ms is not None

    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    avg = sum(latencies) / len(latencies)

    assert avg <= 50.0, f"Average latency ({avg:.2f} ms) exceeded 50 ms limit"
    assert p95 <= 50.0, f"P95 latency ({p95:.2f} ms) exceeded 50 ms limit"


def test_app_focus_and_multi_stage_sequence():
    """Verifies that background app focusing and multi-stage button clicking work without LLM latency."""
    engine = LayaReflexEngine(force_mock=True)
    goal = "önüme antigravityi açıp quick start ile yeni bir proje başlat"

    # Stage 1: Antigravity in background -> FOCUS_WINDOW
    s1 = AgentState(user_goal=goal, active_window="PowerShell", available_elements=[])
    d1 = engine.predict(s1)
    assert d1.action_type == "FOCUS_WINDOW"
    assert d1.text_to_type == "antigravity"

    # Stage 2: Antigravity focused -> Clicks Create New Project
    s2 = AgentState(
        user_goal=goal,
        active_window="Antigravity",
        available_elements=[
            UIElement(id="btn_create", label="Create New Project", control_type="Button", bbox=(10, 10, 100, 30)),
            UIElement(id="btn_open", label="Open Folder", control_type="Button", bbox=(10, 50, 100, 30)),
        ],
    )
    d2 = engine.predict(s2)
    assert d2.action_type == "CLICK"
    assert d2.selected_element_id == "btn_create"

    # Stage 3: Modal opens -> Clicks Quick Start
    s3 = AgentState(
        user_goal=goal,
        active_window="Antigravity - Create Project",
        available_elements=[
            UIElement(id="btn_quick", label="Quick Start", control_type="Button", bbox=(20, 20, 100, 30)),
            UIElement(id="btn_custom", label="Custom", control_type="Button", bbox=(20, 60, 100, 30)),
        ],
    )
    d3 = engine.predict(s3)
    assert d3.action_type == "CLICK"
    assert d3.selected_element_id == "btn_quick"

