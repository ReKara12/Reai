"""Autonomous State Machine loop coordinating Sense -> Decide -> Guardrail -> Act -> Verify."""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Tuple

from .models import UIElement, AgentState, ReflexDecision
from ..perception.accessibility import BaseAccessibilityDriver, get_platform_driver
from ..perception.state_pruner import UIStatePruner
from ..reflex.laya_engine import LayaReflexEngine
from ..reflex.guardrail import SafetyGuardrail
from ..reflex.verifier import StateVerifier
from ..actuator.mouse_keyboard import BaseActuator, get_actuator
from ..generative.llm_fallback import BaseGenerativeProvider, get_generative_provider

logger = logging.getLogger(__name__)


class ReflexStateMachine:
    """Coordinates the dual-system OS control pipeline with sub-second feedback loop."""

    def __init__(
        self,
        driver: Optional[BaseAccessibilityDriver] = None,
        pruner: Optional[UIStatePruner] = None,
        reflex_engine: Optional[LayaReflexEngine] = None,
        guardrail: Optional[SafetyGuardrail] = None,
        verifier: Optional[StateVerifier] = None,
        actuator: Optional[BaseActuator] = None,
        generative_provider: Optional[BaseGenerativeProvider] = None,
        step_delay: float = 0.05,
    ):
        self.driver = driver or get_platform_driver()
        self.pruner = pruner or UIStatePruner(max_elements=25)
        self.reflex_engine = reflex_engine or LayaReflexEngine()
        self.guardrail = guardrail or SafetyGuardrail()
        self.verifier = verifier or StateVerifier(max_retries=3)
        self.actuator = actuator or get_actuator()
        self.generative_provider = generative_provider or get_generative_provider()
        self.step_delay = step_delay

        self.history: List[str] = []
        self.step_count: int = 0
        self.last_focus: Optional[Tuple[int, int]] = None

    def sense_and_prune(self, user_goal: str) -> Tuple[AgentState, str]:
        """Sense active window and UI elements, prunes to top 25, and computes pre-state hash."""
        win_title = self.driver.get_active_window_title()
        raw_elements = self.driver.get_ui_elements()
        pruned_elements = self.pruner.prune(raw_elements, focus_point=self.last_focus)

        state = AgentState(
            user_goal=user_goal,
            active_window=win_title,
            available_elements=pruned_elements,
            history=list(self.history),
            step_count=self.step_count,
        )
        pre_hash = self.verifier.compute_state_hash(pruned_elements, win_title)
        return state, pre_hash

    async def step(
        self,
        user_goal: str,
        allow_destructive: bool = False,
    ) -> Dict[str, Any]:
        """Executes a single Sense -> Prune -> Decide -> Guardrail -> Act -> Verify cycle."""
        self.step_count += 1
        logger.info("--- Step %d Starting for Goal: '%s' ---", self.step_count, user_goal)

        # 1. Sense & Prune
        state, pre_hash = self.sense_and_prune(user_goal)

        # 2. Decide (System 1 Reflex)
        decision = self.reflex_engine.predict(state)
        logger.info(
            "Decision: action=%s, elem=%s, conf=%.2f, latency=%.1fms",
            decision.action_type,
            decision.selected_element_id,
            decision.confidence,
            decision.execution_latency_ms or 0.0,
        )

        # If task is already completed without any pending action, return immediately
        if decision.is_task_completed and (decision.action_type == "WAIT" or not decision.selected_element_id):
            logger.info("Task completed according to System 1.")
            return {
                "step": self.step_count,
                "status": "COMPLETED",
                "decision": decision,
                "mutated": False,
            }

        # Find target element
        target_element: Optional[UIElement] = None
        if decision.selected_element_id:
            for el in state.available_elements:
                if el.id == decision.selected_element_id:
                    target_element = el
                    break

        # 3. Guardrail Check
        decision = self.guardrail.evaluate_and_annotate(
            decision,
            element=target_element,
            context_text=user_goal,
            laya_agent=self.reflex_engine.agent,
        )

        # Enforce safety policy (will raise PermissionError if destructive and not confirmed)
        self.guardrail.enforce(
            decision,
            element=target_element,
            confirmed_by_user=allow_destructive,
        )

        # 4. Act
        generated_text: Optional[str] = None
        if decision.action_type == "CALL_LLM":
            generated_text = self.generative_provider.generate(
                prompt=user_goal,
                context=f"Active Window: {state.active_window}",
            )
            # If target input element is focused, type the generated response
            if target_element:
                decision.text_to_type = generated_text
                self.actuator.execute_decision(
                    ReflexDecision(
                        action_type="TYPE",
                        selected_element_id=target_element.id,
                        text_to_type=generated_text,
                    ),
                    element=target_element,
                )
            else:
                self.actuator.paste_text(generated_text)

            decision.is_task_completed = True
            return {
                "step": self.step_count,
                "status": "COMPLETED",
                "decision": decision,
                "mutated": True,
                "is_stuck": False,
                "generated_text": generated_text,
                "latency_ms": decision.execution_latency_ms,
            }
        else:
            self.actuator.execute_decision(decision, element=target_element)

        if target_element:
            self.last_focus = target_element.center

        # Brief settle time for UI event loop
        await asyncio.sleep(self.step_delay)

        # 5. Verify Mutation & Check Stuck Loops
        post_raw = self.driver.get_ui_elements()
        post_pruned = self.pruner.prune(post_raw, focus_point=self.last_focus)
        post_title = self.driver.get_active_window_title()
        post_hash = self.verifier.compute_state_hash(post_pruned, post_title)

        mutated = self.verifier.verify_mutation(pre_hash, post_hash)
        is_stuck = self.verifier.track_action(
            pre_hash, decision.action_type, decision.selected_element_id
        )

        summary_event = (
            f"Step {self.step_count}: {decision.action_type} on "
            f"'{decision.selected_element_id}' (mutated={mutated})"
        )
        self.history.append(summary_event)

        return {
            "step": self.step_count,
            "status": "STUCK" if is_stuck else "OK",
            "decision": decision,
            "mutated": mutated,
            "is_stuck": is_stuck,
            "generated_text": generated_text,
            "latency_ms": decision.execution_latency_ms,
        }

    async def run(
        self,
        user_goal: str,
        max_steps: int = 20,
        allow_destructive: bool = False,
    ) -> Dict[str, Any]:
        """Runs the loop until goal completion, max steps, or stuck loop detection."""
        self.verifier.reset()
        self.history.clear()
        self.step_count = 0

        results = []
        for _ in range(max_steps):
            step_result = await self.step(user_goal, allow_destructive=allow_destructive)
            results.append(step_result)

            if step_result["status"] == "COMPLETED":
                return {"status": "SUCCESS", "steps": self.step_count, "history": self.history}
            if step_result["status"] == "STUCK":
                return {"status": "FAILED_STUCK", "steps": self.step_count, "history": self.history}

        return {"status": "MAX_STEPS_REACHED", "steps": self.step_count, "history": self.history}

    def run_sync(
        self,
        user_goal: str,
        max_steps: int = 20,
        allow_destructive: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for running the agent loop."""
        return asyncio.run(self.run(user_goal, max_steps=max_steps, allow_destructive=allow_destructive))
