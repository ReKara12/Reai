"""Unified CLI Entry Point for Reflex-Agent."""

import argparse
import asyncio
import logging
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.perception.accessibility import get_platform_driver, MockAccessibilityDriver
from src.perception.state_pruner import UIStatePruner
from src.reflex.laya_engine import LayaReflexEngine
from src.reflex.guardrail import SafetyGuardrail, BlastRadiusScorer
from src.reflex.verifier import StateVerifier
from src.actuator.mouse_keyboard import get_actuator, MockActuator
from src.generative.llm_fallback import get_generative_provider
from src.orchestrator.models import UIElement
from src.orchestrator.state_machine import ReflexStateMachine

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_benchmark(iterations: int = 50, force_mock: bool = False) -> None:
    """Runs latency benchmarking for System 1 Laya Reflex predictions."""
    console.print(Panel("[bold green]Starting System 1 Reflex Latency Benchmark[/bold green]"))

    mock_driver = MockAccessibilityDriver()
    mock_driver.set_elements(
        [
            UIElement(id="btn_login", label="Login", control_type="Button", bbox=(10, 20, 80, 30)),
            UIElement(id="txt_user", label="Username", control_type="Edit", bbox=(10, 60, 150, 30)),
            UIElement(id="txt_pass", label="Password", control_type="Edit", bbox=(10, 100, 150, 30)),
            UIElement(id="btn_delete", label="Delete Account", control_type="Button", bbox=(10, 140, 120, 30)),
            UIElement(id="lbl_status", label="Ready", control_type="Text", bbox=(10, 180, 200, 20)),
        ]
    )

    pruner = UIStatePruner(max_elements=25)
    engine = LayaReflexEngine(force_mock=force_mock)

    elements = pruner.prune(mock_driver.get_ui_elements())
    goals = [
        "Giriş yap",
        "Click the Login button",
        "Type admin into Username field",
        "Hesabı sil",
        "Wait for processing",
    ]

    latencies = []
    for i in range(iterations):
        goal = goals[i % len(goals)]
        from src.orchestrator.models import AgentState
        state = AgentState(
            user_goal=goal,
            active_window="Benchmark Window",
            available_elements=elements,
        )
        t0 = time.perf_counter()
        decision = engine.predict(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]

    table = Table(title="System 1 Reflex Latency Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold magenta")
    table.add_column("Target SLA", style="green")

    table.add_row("Iterations", str(iterations), "-")
    table.add_row("Average Latency", f"{avg_lat:.2f} ms", "<= 50 ms")
    table.add_row("Min Latency", f"{min_lat:.2f} ms", "<= 50 ms")
    table.add_row("Max Latency", f"{max_lat:.2f} ms", "<= 50 ms")
    table.add_row("P95 Latency", f"{p95_lat:.2f} ms", "<= 50 ms")
    table.add_row("Status SLA Check", "[bold green]PASSED[/bold green]" if p95_lat <= 50.0 else "[bold red]FAILED[/bold red]", "<= 50 ms")

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex-Agent: Hybrid OS Controller")
    parser.add_argument("--goal", type=str, default="Giriş yap", help="User navigation goal")
    parser.add_argument("--window", type=str, default=None, help="Target window title filter")
    parser.add_argument("--mock", action="store_true", help="Force mock perception and actuator")
    parser.add_argument("--headless", action="store_true", help="Run actuator in safe headless mode")
    parser.add_argument("--allow-destructive", action="store_true", help="Confirm execution of destructive actions")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum agent loop iterations")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark suite")
    parser.add_argument("--voice", action="store_true", help="Launch interactive local voice assistant mode")
    parser.add_argument("--continuous", action="store_true", help="Use continuous VAD listening instead of push-to-talk")
    parser.add_argument("--ptt-key", type=str, default="alt_r", help="Push-to-talk key (default: alt_r)")
    parser.add_argument("--llm", type=str, default="qwen2.5:3b", help="Local LLM model name (default: qwen2.5:3b)")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(iterations=100, force_mock=args.mock)
        return

    if args.voice:
        from src.assistant.assistant_core import PersonalVoiceAssistant
        assistant = PersonalVoiceAssistant(
            push_to_talk=not args.continuous,
            ptt_key=args.ptt_key,
            llm_model=args.llm,
            headless=args.headless,
        )
        assistant.start_listening_loop()
        return

    console.print(Panel(f"[bold cyan]Reflex-Agent Initializing[/bold cyan]\nGoal: [bold yellow]{args.goal}[/bold yellow]"))

    driver = get_platform_driver(mock=args.mock)
    actuator = get_actuator(mock=args.mock, headless=args.headless)
    pruner = UIStatePruner(max_elements=25)
    engine = LayaReflexEngine(force_mock=args.mock)
    guardrail = SafetyGuardrail()
    verifier = StateVerifier(max_retries=3)
    llm = get_generative_provider("auto", model_name=args.llm)

    agent = ReflexStateMachine(
        driver=driver,
        pruner=pruner,
        reflex_engine=engine,
        guardrail=guardrail,
        verifier=verifier,
        actuator=actuator,
        generative_provider=llm,
    )

    try:
        result = agent.run_sync(
            user_goal=args.goal,
            max_steps=args.max_steps,
            allow_destructive=args.allow_destructive,
        )
        console.print(f"[bold green]Execution Result:[/bold green] {result}")
    except PermissionError as perm_err:
        console.print(f"[bold red]Execution Halted by Safety Guardrail:[/bold red] {perm_err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
