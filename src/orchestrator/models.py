"""Strict Pydantic v2 data models for Reflex-Agent."""

from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class UIElement(BaseModel):
    """Represents an accessible UI element detected in the active window."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    label: str
    control_type: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    is_enabled: bool = True
    is_visible: bool = True
    value: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates center coordinates (cx, cy) from bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    def to_compact_str(self) -> str:
        """Returns a token-efficient string representation for decision models."""
        val_str = f"='{self.value}'" if self.value else ""
        return f"[{self.id}] {self.control_type}: '{self.label}'{val_str}"


class AgentState(BaseModel):
    """Encapsulates the complete perception state of the agent at a given step."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_goal: str
    active_window: str
    available_elements: List[UIElement] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    step_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def format_candidates_prompt(self, max_elements: int = 25) -> str:
        """Formats the candidate UI elements into a compact list for Laya."""
        lines = [f"Goal: {self.user_goal}", f"Active Window: {self.active_window}", "Elements:"]
        for elem in self.available_elements[:max_elements]:
            lines.append(f"  - {elem.to_compact_str()}")
        return "\n".join(lines)


class ReflexDecision(BaseModel):
    """The discrete System 1 decision produced by the Reflex Engine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    selected_element_id: Optional[str] = None
    action_type: str  # CLICK, DOUBLE_CLICK, TYPE, SCROLL, WAIT, CALL_LLM
    text_to_type: Optional[str] = None
    confidence: float = 1.0
    is_destructive: bool = False
    is_task_completed: bool = False
    reasoning: Optional[str] = None
    execution_latency_ms: Optional[float] = None
