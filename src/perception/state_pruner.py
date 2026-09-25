"""Prunes dense UI accessibility trees to fit within Laya's 512-token context window."""

import math
from typing import List, Optional, Tuple, Set
from ..orchestrator.models import UIElement

# Common interactive control types across Windows UIA, Win32, and GUI frameworks
INTERACTIVE_CONTROL_TYPES: Set[str] = {
    "button",
    "edit",
    "combobox",
    "checkbox",
    "radiobutton",
    "hyperlink",
    "menuitem",
    "tabitem",
    "listitem",
    "treeitem",
    "slider",
    "spinner",
    "textbox",
    "input",
}

# Containers and decorative types to aggressively filter out
CONTAINER_CONTROL_TYPES: Set[str] = {
    "window",
    "pane",
    "group",
    "titlebar",
    "menubar",
    "toolbar",
    "scrollbar",
    "separator",
    "tooltip",
    "statusbar",
    "header",
    "headeritem",
}


class UIStatePruner:
    """Filters and prioritizes UI elements to keep context compact and token-efficient."""

    def __init__(
        self,
        max_elements: int = 25,
        prune_non_interactive: bool = True,
        prune_empty_text: bool = True,
        sort_by_focus: bool = True,
    ):
        self.max_elements = max_elements
        self.prune_non_interactive = prune_non_interactive
        self.prune_empty_text = prune_empty_text
        self.sort_by_focus = sort_by_focus

    def is_interactive(self, elem: UIElement) -> bool:
        """Determines whether an element represents an interactive actionable control."""
        ctrl_lower = elem.control_type.lower()
        if ctrl_lower in INTERACTIVE_CONTROL_TYPES:
            return True
        if ctrl_lower in CONTAINER_CONTROL_TYPES:
            return False
        # If unknown control type, check if it has a clickable label or value
        return bool(elem.label and elem.is_enabled)

    def is_decorative_or_empty(self, elem: UIElement) -> bool:
        """Checks if the element is empty, zero-sized, or purely decorative."""
        x, y, w, h = elem.bbox
        # Zero-sized or offscreen bounding boxes
        if (w <= 0 or h <= 0) and not elem.label:
            return True

        # Non-actionable text labels with no content
        if elem.control_type.lower() in ("text", "label") and not elem.label.strip():
            return True

        return False

    def calculate_distance_to_focus(
        self, elem: UIElement, focus_point: Tuple[int, int]
    ) -> float:
        """Calculates Euclidean distance between an element center and the focus point."""
        cx, cy = elem.center
        fx, fy = focus_point
        return math.hypot(cx - fx, cy - fy)

    def prune(
        self,
        elements: List[UIElement],
        focus_point: Optional[Tuple[int, int]] = None,
    ) -> List[UIElement]:
        """Prunes the dense element list to at most `max_elements` relevant nodes."""
        filtered: List[UIElement] = []

        for elem in elements:
            # Check visibility and enabled status
            if not elem.is_visible:
                continue

            # Filter decorative or empty nodes
            if self.prune_empty_text and self.is_decorative_or_empty(elem):
                continue

            # Filter non-interactive containers
            if self.prune_non_interactive and not self.is_interactive(elem):
                continue

            filtered.append(elem)

        # Spatial sorting: prioritize elements closest to focus or top-to-bottom reading order
        if self.sort_by_focus and focus_point is not None:
            filtered.sort(key=lambda el: self.calculate_distance_to_focus(el, focus_point))
        else:
            # Top-to-bottom, left-to-right reading order
            filtered.sort(key=lambda el: (el.bbox[1], el.bbox[0]))

        return filtered[: self.max_elements]
