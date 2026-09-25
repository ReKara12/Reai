"""Safety Guardrail and Blast-Radius Risk Scorer for Reflex-Agent."""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..orchestrator.models import UIElement, ReflexDecision

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "safety_rules.yaml"


class BlastRadiusScorer:
    """Calculates risk score (0.0 to 1.0) and blast radius for a planned OS action."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_RULES_PATH
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning("Could not read safety rules from %s: %s", self.config_path, exc)

        # Fallback default rules
        return {
            "destructive_keywords": [
                "delete", "remove", "destroy", "format", "wipe", "drop",
                "purge", "erase", "terminate", "reset", "uninstall", "sil",
                "hesabı sil", "kaldır", "temizle", "sıfırla",
            ],
            "financial_keywords": [
                "pay", "checkout", "transfer", "wire", "charge",
                "authorize payment", "öde", "satın al",
            ],
            "dangerous_commands": [
                "rm -rf", "rmdir /s", "del /f", "format ", "drop database", "mkfs",
            ],
            "risk_thresholds": {
                "high_risk_min": 0.65,
                "medium_risk_max": 0.64,
                "low_risk_max": 0.30,
            },
        }

    def score(
        self,
        decision: ReflexDecision,
        element: Optional[UIElement] = None,
        context_text: Optional[str] = None,
    ) -> float:
        """Computes a risk score between 0.0 (safe) and 1.0 (extremely hazardous)."""
        risk = 0.05  # Base benign action risk

        # Inspect targeted element
        elem_text = ""
        if element:
            elem_text = f"{element.label} {element.id}".lower()

        text_to_type = (decision.text_to_type or "").lower()
        full_text = f"{elem_text} {text_to_type} {context_text or ''}".lower()

        # Check dangerous terminal/shell commands
        for cmd in self.rules.get("dangerous_commands", []):
            if cmd.lower() in full_text:
                risk = max(risk, 0.95)
                break

        # Check destructive keywords
        for kw in self.rules.get("destructive_keywords", []):
            if kw.lower() in full_text:
                risk = max(risk, 0.85)
                break

        # Check financial keywords
        for kw in self.rules.get("financial_keywords", []):
            if kw.lower() in full_text:
                risk = max(risk, 0.75)
                break

        return min(1.0, risk)


class SafetyGuardrail:
    """Enforces destructive action blocks and confirmation requirements."""

    def __init__(
        self,
        blast_scorer: Optional[BlastRadiusScorer] = None,
        strict_mode: bool = True,
        threshold: float = 0.65,
    ):
        self.scorer = blast_scorer or BlastRadiusScorer()
        self.strict_mode = strict_mode
        self.threshold = threshold

    def evaluate_and_annotate(
        self,
        decision: ReflexDecision,
        element: Optional[UIElement] = None,
        context_text: Optional[str] = None,
        laya_agent: Any = None,
    ) -> ReflexDecision:
        """Annotates decision.is_destructive via fast heuristic or secondary Laya noul inference."""
        risk_score = self.scorer.score(decision, element=element, context_text=context_text)

        # Secondary fast Laya inference if available
        if laya_agent and hasattr(laya_agent, "predict") and element:
            try:
                question = {
                    "is_destructive": {
                        "type": "noul",
                        "instructions": (
                            f"Does executing '{decision.action_type}' on '{element.label}' "
                            f"pose an irreversible, destructive, or high-blast-radius operation?"
                        ),
                    }
                }
                res = laya_agent.predict({"element": element.to_compact_str()}, question)
                noul_prob = float(res.get("answers", {}).get("is_destructive", {}).get("noul", 0.0))
                # Weighted blend
                risk_score = 0.5 * risk_score + 0.5 * noul_prob
            except Exception as exc:
                logger.debug("Secondary Laya safety check fallback: %s", exc)

        is_destructive = risk_score >= self.threshold
        decision.is_destructive = is_destructive

        if is_destructive:
            logger.warning(
                "SafetyGuardrail flagged destructive action: %s on element '%s' (Risk: %.2f)",
                decision.action_type,
                element.label if element else "None",
                risk_score,
            )

        return decision

    def enforce(
        self,
        decision: ReflexDecision,
        element: Optional[UIElement] = None,
        confirmed_by_user: bool = False,
    ) -> None:
        """Enforces guardrail policy. Raises PermissionError if destructive and not confirmed."""
        if not decision.is_destructive:
            return

        if confirmed_by_user:
            logger.info("Destructive action explicitly confirmed by user. Proceeding.")
            return

        elem_name = element.label if element else (decision.selected_element_id or "unknown")
        err_msg = (
            f"Safety Guardrail Blocked Destructive Action: Action '{decision.action_type}' "
            f"on element '{elem_name}' is classified as HIGH-RISK/DESTRUCTIVE. "
            f"Execution halted to prevent unintended data loss."
        )
        logger.error(err_msg)
        raise PermissionError(err_msg)
