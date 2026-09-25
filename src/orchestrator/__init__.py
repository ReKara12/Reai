"""Orchestrator package for Reflex-Agent."""

from .models import UIElement, AgentState, ReflexDecision


def __getattr__(name: str):
    if name == "ReflexStateMachine":
        from .state_machine import ReflexStateMachine
        return ReflexStateMachine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["UIElement", "AgentState", "ReflexDecision", "ReflexStateMachine"]
