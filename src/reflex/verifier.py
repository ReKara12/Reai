"""State mutation verifier and stuck-loop detector for OS control actions."""

import hashlib
import json
import logging
from typing import List, Optional, Tuple, Any

from ..orchestrator.models import UIElement

logger = logging.getLogger(__name__)


class StateVerifier:
    """Verifies that executed actions actually mutate the UI state and prevents infinite loops."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.stuck_counter = 0
        self.last_action_signature: Optional[Tuple[str, str, Optional[str]]] = None
        self.action_history: List[Tuple[str, str, Optional[str]]] = []

    def compute_state_hash(
        self, elements: List[UIElement], window_title: str = ""
    ) -> str:
        """Computes a deterministic SHA-256 fingerprint of the current UI state."""
        normalized_items = []
        for elem in sorted(elements, key=lambda e: e.id):
            normalized_items.append(
                (
                    elem.id,
                    elem.label.strip(),
                    elem.control_type.lower(),
                    elem.value or "",
                    elem.is_enabled,
                    elem.bbox,
                )
            )

        payload = {
            "window": window_title.strip(),
            "elements": normalized_items,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify_mutation(self, pre_hash: str, post_hash: str) -> bool:
        """Returns True if the UI mutated as a consequence of the action."""
        mutated = pre_hash != post_hash
        if not mutated:
            logger.warning("UI state did NOT mutate after action execution. (Hash: %s)", pre_hash[:10])
        else:
            logger.info("UI state successfully mutated from %s to %s", pre_hash[:8], post_hash[:8])
        return mutated

    def track_action(
        self,
        state_hash: str,
        action_type: str,
        element_id: Optional[str] = None,
    ) -> bool:
        """Tracks consecutive identical actions.

        Returns True if a stuck infinite loop is detected (exceeding max_retries).
        """
        signature = (action_type, element_id)
        self.action_history.append(signature)

        if self.last_action_signature == signature:
            self.stuck_counter += 1
            logger.warning(
                "Repeated identical action detected (%d/%d): action=%s, elem=%s",
                self.stuck_counter,
                self.max_retries,
                action_type,
                element_id,
            )
        else:
            self.stuck_counter = 1
            self.last_action_signature = signature

        if self.stuck_counter >= self.max_retries:
            logger.error("STUCK LOOP DETECTED: Action '%s' on '%s' repeated %d times without goal progress.", action_type, element_id, self.stuck_counter)
            return True

        return False

        return False

    def reset(self) -> None:
        """Resets the state verifier counters and history."""
        self.stuck_counter = 0
        self.last_action_signature = None
        self.action_history.clear()
