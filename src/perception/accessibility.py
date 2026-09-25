"""OS UI accessibility tree reader and cross-platform drivers."""

import sys
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import tkinter as tk

from ..orchestrator.models import UIElement

logger = logging.getLogger(__name__)


class BaseAccessibilityDriver(ABC):
    """Abstract Base Class for OS-level Accessibility Tree inspection."""

    @abstractmethod
    def get_active_window_title(self) -> str:
        """Retrieves the title of the currently focused foreground window."""
        pass

    @abstractmethod
    def get_ui_elements(self, window_title: Optional[str] = None) -> List[UIElement]:
        """Extracts accessible interactive elements from the targeted window."""
        pass

    def get_element_by_id(self, element_id: str, window_title: Optional[str] = None) -> Optional[UIElement]:
        """Finds an element by its identifier."""
        elements = self.get_ui_elements(window_title=window_title)
        for elem in elements:
            if elem.id == element_id:
                return elem
        return None


class WindowsAccessibilityDriver(BaseAccessibilityDriver):
    """Native Windows UI Automation reader backed by pywinauto."""

    def __init__(self, backend: str = "uia"):
        self.backend = backend
        self._desktop = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from pywinauto import Desktop
            self._desktop = Desktop(backend=self.backend)
        except Exception as exc:
            logger.warning("Could not initialize pywinauto Desktop: %s", exc)
            self._desktop = None

    def get_active_window_title(self) -> str:
        if sys.platform != "win32":
            return "Active Window"

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            hwnd = user32.GetForegroundWindow()
            if not hwnd or not user32.IsWindowVisible(hwnd):
                # Query interactive input desktop if foreground window is not set in current subshell
                h_input = user32.OpenInputDesktop(0, False, 0x01FF)
                if h_input:
                    top_wins = []
                    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                    ignore_titles = {
                        "program manager", "task switching", "medal overlay",
                        "nvidia geforce overlay", "microsoft text input application"
                    }
                    def enum_cb(h, _):
                        if user32.IsWindowVisible(h):
                            length = user32.GetWindowTextLengthW(h)
                            if length > 0:
                                buff = ctypes.create_unicode_buffer(length + 1)
                                user32.GetWindowTextW(h, buff, length + 1)
                                title = buff.value.strip()
                                if title and title.lower() not in ignore_titles:
                                    top_wins.append(h)
                                    return False
                        return True
                    user32.EnumDesktopWindows(h_input, WNDENUMPROC(enum_cb), 0)
                    if top_wins:
                        hwnd = top_wins[0]

            if not hwnd:
                return "Desktop Window"

            length = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value.strip()

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            proc_name = ""
            if pid.value:
                h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
                if h_proc:
                    buf_size = wintypes.DWORD(1024)
                    proc_buf = ctypes.create_unicode_buffer(1024)
                    if kernel32.QueryFullProcessImageNameW(h_proc, 0, proc_buf, ctypes.byref(buf_size)):
                        proc_name = proc_buf.value.split("\\")[-1]
                    kernel32.CloseHandle(h_proc)

            if proc_name and title:
                return f"{title} [{proc_name}]"
            elif title:
                return title
            elif proc_name:
                return proc_name
            return "Active Window"
        except Exception as exc:
            logger.debug("Active window resolution failed: %s", exc)
            return "Active Window"

    def get_ui_elements(self, window_title: Optional[str] = None) -> List[UIElement]:
        results: List[UIElement] = []
        if not self._desktop:
            self._init_backend()
            if not self._desktop:
                return results

        try:
            target_window = None
            if window_title:
                try:
                    target_window = self._desktop.window(title_re=f".*{window_title}.*")
                except Exception:
                    pass

            if target_window is None:
                # 1. Try active foreground window via win32gui
                try:
                    import win32gui
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd and win32gui.IsWindowVisible(hwnd):
                        target_window = self._desktop.window(handle=hwnd)
                except Exception:
                    pass

            if target_window is None:
                # 2. Try visible top-level windows from desktop
                try:
                    windows = self._desktop.windows(visible_only=True)
                    if windows:
                        target_window = windows[0]
                except Exception:
                    pass

            if target_window is None:
                return results

            descendants = target_window.descendants()
            # Limit traversal to prevent UI lag on huge desktop windows
            for idx, ctrl in enumerate(descendants[:80]):
                try:
                    props = ctrl.get_properties()
                    ctrl_type = props.get("control_type", "Control")
                    text = props.get("texts", [""])[0] if props.get("texts") else ""
                    rect = props.get("rectangle")
                    is_enabled = props.get("is_enabled", True)
                    is_visible = props.get("is_visible", True)

                    bbox = (0, 0, 0, 0)
                    if rect:
                        bbox = (
                            int(rect.left),
                            int(rect.top),
                            int(rect.width()),
                            int(rect.height()),
                        )

                    elem_id = f"win_{idx}_{ctrl_type.lower()}"
                    results.append(
                        UIElement(
                            id=elem_id,
                            label=text.strip(),
                            control_type=ctrl_type,
                            bbox=bbox,
                            is_enabled=is_enabled,
                            is_visible=is_visible,
                            metadata={"class_name": props.get("class_name", "")},
                        )
                    )
                except Exception as elem_err:
                    logger.debug("Failed reading element index %d: %s", idx, elem_err)
                    continue

        except Exception as exc:
            logger.warning("Error traversing Windows UI elements: %s", exc)

        return results


class MockAccessibilityDriver(BaseAccessibilityDriver):
    """In-memory accessibility driver for headless CI, Linux tests, and Tkinter harnesses."""

    def __init__(self, initial_title: str = "Mock Test Window"):
        self.active_window_title = initial_title
        self._elements: List[UIElement] = []
        self._tkinter_root: Optional[tk.Tk] = None

    def set_active_window(self, title: str) -> None:
        self.active_window_title = title

    def set_elements(self, elements: List[UIElement]) -> None:
        self._elements = list(elements)

    def add_element(self, element: UIElement) -> None:
        self._elements.append(element)

    def clear(self) -> None:
        self._elements.clear()

    def bind_tkinter_root(self, root: tk.Tk) -> None:
        """Binds a Tkinter root window to reflect live widgets into the perception tree."""
        self._tkinter_root = root
        self.active_window_title = root.title() or "Tkinter Application"

    def get_active_window_title(self) -> str:
        if self._tkinter_root and self._tkinter_root.winfo_exists():
            return self._tkinter_root.title() or self.active_window_title
        return self.active_window_title

    def get_ui_elements(self, window_title: Optional[str] = None) -> List[UIElement]:
        if self._tkinter_root and self._tkinter_root.winfo_exists():
            return self._extract_from_tkinter()
        return list(self._elements)

    def _extract_from_tkinter(self) -> List[UIElement]:
        results: List[UIElement] = []
        if not self._tkinter_root:
            return results

        def walk(widget: tk.Widget, counter: List[int]):
            counter[0] += 1
            idx = counter[0]
            w_class = widget.winfo_class()
            widget_name = getattr(widget, "custom_id", f"widget_{idx}")

            text = ""
            val = None
            if hasattr(widget, "cget"):
                try:
                    text = str(widget.cget("text"))
                except Exception:
                    text = ""

            # Check Entry / Text values
            if isinstance(widget, tk.Entry):
                val = widget.get()
                w_class = "Edit"
            elif isinstance(widget, tk.Button):
                w_class = "Button"
            elif isinstance(widget, tk.Label):
                w_class = "Text"

            is_enabled = True
            if hasattr(widget, "cget"):
                try:
                    state = widget.cget("state")
                    if str(state) == "disabled":
                        is_enabled = False
                except Exception:
                    pass

            try:
                x = widget.winfo_rootx()
                y = widget.winfo_rooty()
                w = widget.winfo_width()
                h = widget.winfo_height()
            except Exception:
                x, y, w, h = (0, 0, 0, 0)

            # In withdrawn test mode, unmapped widgets may have default 1x1 or 0x0
            is_managed = bool(widget.winfo_manager() != "")
            is_visible = bool(widget.winfo_ismapped() or is_managed)
            if w <= 0 or h <= 0:
                w, h = (100, 30)

            # Determine best human-readable label
            custom_label = getattr(widget, "label", None)
            clean_name = widget_name
            for prefix in ("txt_", "btn_", "lbl_", "chk_"):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):].replace("_", " ").title()
                    break

            elem_label = text or custom_label or clean_name

            results.append(
                UIElement(
                    id=widget_name,
                    label=elem_label,
                    control_type=w_class,
                    bbox=(x, y, w, h),
                    is_enabled=is_enabled,
                    is_visible=is_visible,
                    value=val,
                )
            )

            for child in widget.winfo_children():
                walk(child, counter)

        counter = [0]
        for child in self._tkinter_root.winfo_children():
            walk(child, counter)

        return results


def get_platform_driver(mock: bool = False, initial_title: str = "Desktop") -> BaseAccessibilityDriver:
    """Factory helper to obtain the suitable platform driver."""
    if mock:
        return MockAccessibilityDriver(initial_title=initial_title)
    if sys.platform == "win32":
        try:
            return WindowsAccessibilityDriver()
        except Exception:
            return MockAccessibilityDriver(initial_title=initial_title)
    return MockAccessibilityDriver(initial_title=initial_title)
