"""Streaming Speech-to-Intent Anticipator with speculative execution and voice barge-in."""

import re
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..orchestrator.models import UIElement, AgentState, ReflexDecision
from ..reflex.laya_engine import LayaReflexEngine

logger = logging.getLogger(__name__)

BARGE_IN_CANCEL_KEYWORDS = [
    "dur", "vazgeçtim", "iptal", "bekle", "hayır", "yanlış",
    "stop", "cancel", "abort", "wait", "undo",
]


@dataclass
class StagedAction:
    """Represents an anticipated action staged while the user was still speaking."""
    partial_text: str
    decision: ReflexDecision
    target_element: Optional[UIElement]
    staged_timestamp: float
    is_cancelled: bool = False


class SpeechToIntentAnticipator:
    """Monitors streaming partial transcripts and speculatively pre-stages actions using Laya."""

    def __init__(self, reflex_engine: Optional[LayaReflexEngine] = None):
        self.reflex_engine = reflex_engine or LayaReflexEngine(force_mock=True)
        self.staged_action: Optional[StagedAction] = None
        self.is_interrupted = False

    def is_cancellation_phrase(self, text: str) -> bool:
        """Detects voice barge-in or verbal cancellation triggers."""
        text_lower = text.lower().strip()
        tokens = set(re.findall(r"\w+", text_lower))
        return bool(tokens.intersection(BARGE_IN_CANCEL_KEYWORDS))

    def feed_partial(
        self,
        partial_text: str,
        available_elements: List[UIElement],
        active_window: str = "",
    ) -> Optional[StagedAction]:
        """Processes partial streaming utterance (e.g. 'Chrome'u açıp...') in <1ms."""
        if not partial_text or len(partial_text.strip()) < 3:
            return None

        # Check for immediate barge-in interruption
        if self.is_cancellation_phrase(partial_text):
            logger.warning("[BARGE-IN] Interruption detected from voice: '%s'. Aborting staged work.", partial_text)
            self.is_interrupted = True
            if self.staged_action:
                self.staged_action.is_cancelled = True
            return None

        self.is_interrupted = False

        # Run ultra-fast Laya System 1 on partial text
        partial_state = AgentState(
            user_goal=partial_text,
            active_window=active_window,
            available_elements=available_elements,
        )

        t0 = time.perf_counter()
        speculative_decision = self.reflex_engine.predict(partial_state)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        target_elem = None
        if speculative_decision.selected_element_id:
            for el in available_elements:
                if el.id == speculative_decision.selected_element_id:
                    target_elem = el
                    break

        self.staged_action = StagedAction(
            partial_text=partial_text,
            decision=speculative_decision,
            target_element=target_elem,
            staged_timestamp=time.time(),
        )

        logger.debug(
            "[ANTICIPATION] Partial '%s' -> Staged action: %s on %s (latency: %.2fms)",
            partial_text,
            speculative_decision.action_type,
            speculative_decision.selected_element_id,
            latency_ms,
        )
        return self.staged_action

    def commit(
        self,
        final_text: str,
        available_elements: List[UIElement],
        active_window: str = "",
    ) -> Tuple[ReflexDecision, bool]:
        """Commits the final utterance. Returns (ReflexDecision, was_pre_staged)."""
        if self.is_interrupted or self.is_cancellation_phrase(final_text):
            logger.info("[ANTICIPATION] Utterance was cancelled by user voice command.")
            return (
                ReflexDecision(action_type="WAIT", reasoning="User cancelled action via voice"),
                False,
            )

        # Check if already staged accurately
        if (
            self.staged_action
            and not self.staged_action.is_cancelled
            and self.staged_action.decision.selected_element_id
        ):
            # Verify target still exists in current elements
            staged_id = self.staged_action.decision.selected_element_id
            if any(el.id == staged_id for el in available_elements):
                logger.info(
                    "[ANTICIPATION] Zero-lag commit: Action '%s' on '%s' was already staged while speaking!",
                    self.staged_action.decision.action_type,
                    staged_id,
                )
                return self.staged_action.decision, True

        # Fallback to full inference if not pre-staged
        final_state = AgentState(
            user_goal=final_text,
            active_window=active_window,
            available_elements=available_elements,
        )
        decision = self.reflex_engine.predict(final_state)
        return decision, False

    def reset(self) -> None:
        """Clears the anticipation state."""
        self.staged_action = None
        self.is_interrupted = False
