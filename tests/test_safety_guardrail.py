"""Unit tests verifying the Safety Guardrail, blast-radius scoring, and stuck-loop detection."""

import pytest
from src.orchestrator.models import UIElement, ReflexDecision
from src.reflex.guardrail import SafetyGuardrail, BlastRadiusScorer
from src.reflex.verifier import StateVerifier


def test_blast_radius_scorer_benign_action():
    """Verifies that normal non-destructive navigation receives a low risk score."""
    scorer = BlastRadiusScorer()
    decision = ReflexDecision(
        action_type="CLICK",
        selected_element_id="btn_settings",
    )
    element = UIElement(id="btn_settings", label="Settings", control_type="Button", bbox=(10, 10, 80, 25))

    risk = scorer.score(decision, element=element)
    assert risk <= 0.30, f"Expected low risk, got {risk}"


def test_blast_radius_scorer_destructive_keywords():
    """Verifies that delete, wipe, and drop keywords trigger a high risk score."""
    scorer = BlastRadiusScorer()

    test_cases = [
        ("btn_del", "Delete Account", "CLICK", None),
        ("btn_wipe", "Wipe All Data", "CLICK", None),
        ("btn_sil", "Hesabı Sil", "CLICK", None),
        ("txt_cmd", "Terminal", "TYPE", "rm -rf /var/data"),
        ("txt_sql", "Query Box", "TYPE", "DROP TABLE users;"),
    ]

    for elem_id, label, action, text in test_cases:
        elem = UIElement(id=elem_id, label=label, control_type="Button", bbox=(0, 0, 50, 20))
        dec = ReflexDecision(action_type=action, selected_element_id=elem_id, text_to_type=text)
        risk = scorer.score(dec, element=elem)
        assert risk >= 0.65, f"Expected high risk for '{label}' / '{text}', got {risk}"


def test_guardrail_blocks_unconfirmed_destructive_action():
    """Verifies that SafetyGuardrail raises PermissionError on destructive action without confirmation."""
    guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
    elem = UIElement(id="btn_purge", label="Purge Database", control_type="Button", bbox=(0, 0, 100, 30))
    decision = ReflexDecision(action_type="CLICK", selected_element_id="btn_purge")

    annotated = guardrail.evaluate_and_annotate(decision, element=elem)
    assert annotated.is_destructive is True

    # Must raise PermissionError when unconfirmed
    with pytest.raises(PermissionError) as exc_info:
        guardrail.enforce(annotated, element=elem, confirmed_by_user=False)

    assert "Blocked Destructive Action" in str(exc_info.value)


def test_guardrail_allows_confirmed_destructive_action():
    """Verifies that SafetyGuardrail permits destructive action when explicitly confirmed."""
    guardrail = SafetyGuardrail(strict_mode=True, threshold=0.65)
    elem = UIElement(id="btn_format", label="Format Partition", control_type="Button", bbox=(0, 0, 100, 30))
    decision = ReflexDecision(action_type="CLICK", selected_element_id="btn_format")

    annotated = guardrail.evaluate_and_annotate(decision, element=elem)
    assert annotated.is_destructive is True

    # Explicit confirmation should not raise an exception
    guardrail.enforce(annotated, element=elem, confirmed_by_user=True)


def test_state_verifier_hash_and_mutation():
    """Tests that state hash uniquely captures changes in elements and detects mutation."""
    verifier = StateVerifier()
    elem1 = UIElement(id="e1", label="Status", control_type="Text", bbox=(0, 0, 10, 10), value="Pending")
    elem2 = UIElement(id="e1", label="Status", control_type="Text", bbox=(0, 0, 10, 10), value="Finished")

    hash1 = verifier.compute_state_hash([elem1], "Window A")
    hash2 = verifier.compute_state_hash([elem1], "Window A")
    hash3 = verifier.compute_state_hash([elem2], "Window A")

    # Identical state produces same hash
    assert hash1 == hash2
    # Modified value produces distinct hash
    assert hash1 != hash3

    assert verifier.verify_mutation(hash1, hash3) is True
    assert verifier.verify_mutation(hash1, hash2) is False


def test_state_verifier_stuck_loop_detection():
    """Verifies that repeating identical actions 3 times triggers stuck loop alert."""
    verifier = StateVerifier(max_retries=3)
    elem = UIElement(id="btn_stubborn", label="Retry", control_type="Button", bbox=(0, 0, 50, 20))
    state_hash = verifier.compute_state_hash([elem], "Error Window")

    assert verifier.track_action(state_hash, "CLICK", "btn_stubborn") is False
    assert verifier.track_action(state_hash, "CLICK", "btn_stubborn") is False
    # Third consecutive repetition without mutation triggers stuck flag
    is_stuck = verifier.track_action(state_hash, "CLICK", "btn_stubborn")
    assert is_stuck is True
