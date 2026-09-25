"""Cross-platform mouse and keyboard event actuator with safe headless mode and full OS controls."""

import os
import sys
import time
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple

from ..orchestrator.models import UIElement, ReflexDecision

logger = logging.getLogger(__name__)

# Initialize High-DPI Awareness on Windows to prevent coordinate scaling drift
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class BaseActuator(ABC):
    """Abstract Base Class for OS-level mouse, keyboard, and window actuators."""

    @abstractmethod
    def click(self, x: int, y: int) -> None:
        """Executes a single left mouse click at (x, y)."""
        pass

    @abstractmethod
    def right_click(self, x: int, y: int) -> None:
        """Executes a single right mouse click at (x, y) to invoke context menus."""
        pass

    @abstractmethod
    def double_click(self, x: int, y: int) -> None:
        """Executes a double left mouse click at (x, y)."""
        pass

    @abstractmethod
    def drag_and_drop(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Drags mouse from (start_x, start_y) to (end_x, end_y)."""
        pass

    @abstractmethod
    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Types the specified text sequence."""
        pass

    @abstractmethod
    def paste_text(self, text: str) -> None:
        """Instantly injects text block via system clipboard (avoids character lag)."""
        pass

    @abstractmethod
    def hotkey(self, *keys: str) -> None:
        """Executes a combination of keyboard keys (e.g. 'ctrl', 'c' or 'alt', 'tab')."""
        pass

    @abstractmethod
    def scroll(self, clicks: int) -> None:
        """Scrolls the mouse wheel by the specified amount."""
        pass

    @abstractmethod
    def launch_app(self, app_name: str) -> None:
        """Launches a desktop application or shell command."""
        pass

    @abstractmethod
    def focus_window(self, title_pattern: str) -> bool:
        """Brings an open window to the foreground by matching title pattern."""
        pass

    @abstractmethod
    def wait(self, duration: float = 0.1) -> None:
        """Pauses execution."""
        pass

    def execute_decision(self, decision: ReflexDecision, element: Optional[UIElement] = None) -> None:
        """Translates a high-level ReflexDecision into concrete actuator calls."""
        action = decision.action_type.upper()

        if action == "WAIT":
            self.wait(0.2)
            return

        if action in ("FOCUS_WINDOW", "SWITCH_APP", "OPEN_APP"):
            target = decision.text_to_type or (element.label if element else "")
            success = self.focus_window(target)
            if not success:
                logger.info("Window '%s' not found to focus. Attempting launch.", target)
                self.launch_app(target)
            self.wait(0.2)
            return

        if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            if not element:
                logger.warning("Action %s requested but no target element provided.", action)
                return
            cx, cy = element.center
            if action == "CLICK":
                self.click(cx, cy)
            elif action == "DOUBLE_CLICK":
                self.double_click(cx, cy)
            elif action == "RIGHT_CLICK":
                self.right_click(cx, cy)
            return

        if action == "DRAG":
            # Drag if bounding box provided
            if element:
                cx, cy = element.center
                self.drag_and_drop(cx, cy, cx + 100, cy + 100)
            return

        if action == "TYPE":
            if element:
                # Click on the input field first to establish keyboard focus
                cx, cy = element.center
                self.click(cx, cy)
                self.wait(0.05)
            if decision.text_to_type:
                # If large text block (>30 chars), use instant clipboard pasting
                if len(decision.text_to_type) > 30:
                    self.paste_text(decision.text_to_type)
                else:
                    self.type_text(decision.text_to_type)
            return

        if action == "PASTE":
            if element:
                cx, cy = element.center
                self.click(cx, cy)
                self.wait(0.05)
            if decision.text_to_type:
                self.paste_text(decision.text_to_type)
            return

        if action == "LAUNCH_APP":
            app = decision.text_to_type or (element.label if element else "notepad")
            self.launch_app(app)
            return

        if action == "HOTKEY":
            if decision.text_to_type:
                keys = decision.text_to_type.split("+")
                self.hotkey(*[k.strip().lower() for k in keys])
            return

        if action == "SCROLL":
            self.scroll(-3)
            return

        if action == "CALL_LLM":
            logger.info("System 2 LLM invocation specified. Actuator passes control to Generative layer.")
            return

        logger.warning("Unrecognized action type: %s", action)


COMMON_APP_MAP = {
    "antigravity": os.path.expandvars(r"%LOCALAPPDATA%\Programs\Antigravity\Antigravity.exe"),
    "notepad": "notepad.exe",
    "not defteri": "notepad.exe",
    "chrome": os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
    "calc": "calc.exe",
    "hesap makinesi": "calc.exe",
    "spotify": os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
    "code": os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
    "vscode": os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
}


class PyAutoGUIActuator(BaseActuator):
    """Native OS actuator powered by PyAutoGUI, pyperclip, and Win32 APIs."""

    def __init__(self, headless: Optional[bool] = None):
        self.headless = (
            headless
            if headless is not None
            else bool(os.environ.get("HEADLESS", "0") == "1")
        )
        self._pyautogui = None
        if not self.headless:
            try:
                import pyautogui
                pyautogui.FAILSAFE = False
                pyautogui.PAUSE = 0.05
                self._pyautogui = pyautogui
            except Exception as exc:
                logger.warning("Could not initialize pyautogui (%s). Switching to headless safe mode.", exc)
                self.headless = True

    def click(self, x: int, y: int) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Left click at (%d, %d)", x, y)
            return
        try:
            self._pyautogui.click(x, y)
        except Exception as exc:
            logger.error("PyAutoGUI click error at (%d, %d): %s", x, y, exc)

    def right_click(self, x: int, y: int) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Right click at (%d, %d)", x, y)
            return
        try:
            self._pyautogui.rightClick(x, y)
        except Exception as exc:
            logger.error("PyAutoGUI rightClick error at (%d, %d): %s", x, y, exc)

    def double_click(self, x: int, y: int) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Double click at (%d, %d)", x, y)
            return
        try:
            self._pyautogui.doubleClick(x, y)
        except Exception as exc:
            logger.error("PyAutoGUI doubleClick error at (%d, %d): %s", x, y, exc)

    def drag_and_drop(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Drag from (%d, %d) to (%d, %d)", start_x, start_y, end_x, end_y)
            return
        try:
            self._pyautogui.moveTo(start_x, start_y)
            self._pyautogui.dragTo(end_x, end_y, button="left", duration=0.3)
        except Exception as exc:
            logger.error("PyAutoGUI drag error: %s", exc)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Typing text: '%s'", text)
            return
        try:
            self._pyautogui.write(text, interval=interval)
        except Exception as exc:
            logger.error("PyAutoGUI write error: %s", exc)

    def paste_text(self, text: str) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Pasting text: '%s'...", text[:30])
            return
        try:
            import pyperclip
            pyperclip.copy(text)
            self.hotkey("ctrl", "v")
        except Exception as exc:
            logger.warning("Clipboard paste failed (%s), falling back to write.", exc)
            self.type_text(text)

    def hotkey(self, *keys: str) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Hotkey pressed: %s", "+".join(keys))
            return
        try:
            self._pyautogui.hotkey(*keys)
        except Exception as exc:
            logger.error("PyAutoGUI hotkey error: %s", exc)

    def scroll(self, clicks: int) -> None:
        if self.headless or not self._pyautogui:
            logger.info("[HEADLESS ACTUATOR] Mouse scroll: %d clicks", clicks)
            return
        try:
            self._pyautogui.scroll(clicks)
        except Exception as exc:
            logger.error("PyAutoGUI scroll error: %s", exc)

    def launch_app(self, app_name: str) -> None:
        if self.headless:
            logger.info("[HEADLESS ACTUATOR] Launch application: '%s'", app_name)
            return

        clean_name = app_name.lower().strip()
        resolved_path = COMMON_APP_MAP.get(clean_name, app_name)

        try:
            logger.info("Launching desktop application: '%s' (Resolved: '%s')", app_name, resolved_path)
            if sys.platform == "win32":
                if os.path.exists(resolved_path):
                    os.startfile(resolved_path)
                else:
                    os.startfile(app_name)
            else:
                subprocess.Popen([resolved_path])
        except Exception as exc:
            logger.warning("Direct launch failed (%s). Attempting shell run.", exc)
            try:
                subprocess.Popen(resolved_path, shell=True)
            except Exception as shell_err:
                logger.error("Could not launch app '%s': %s", app_name, shell_err)

    def focus_window(self, title_pattern: str) -> bool:
        """Finds a window by title or process name and brings it to the foreground."""
        if self.headless:
            logger.info("[HEADLESS ACTUATOR] Focus window matching: '%s'", title_pattern)
            return True
        if sys.platform != "win32":
            return False

        try:
            import subprocess, csv
            import win32gui
            import win32con
            import win32process
            import ctypes

            user32 = ctypes.windll.user32
            target_lower = title_pattern.lower().strip()

            # 1. Inspect running processes from tasklist to match process name or MainWindowTitle
            matched_pids = set()
            try:
                out = subprocess.check_output(['tasklist', '/v', '/fo', 'csv'], text=True, errors='ignore')
                for row in csv.reader(out.splitlines()):
                    if len(row) > 8:
                        pname = row[0].lower()
                        wtitle = row[8].lower()
                        if target_lower in pname or target_lower in wtitle:
                            try:
                                matched_pids.add(int(row[1]))
                            except ValueError:
                                pass
            except Exception as proc_err:
                logger.debug("Process lookup via tasklist failed: %s", proc_err)

            found_hwnds = []
            def enum_cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    try:
                        pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                    except Exception:
                        pid = 0
                    if pid in matched_pids or (title and target_lower in title.lower()):
                        found_hwnds.append((hwnd, title or f"PID_{pid}"))

            win32gui.EnumWindows(enum_cb, None)

            if not found_hwnds:
                logger.info("No visible window found matching '%s'", title_pattern)
                return False

            target_hwnd, window_title = found_hwnds[0]
            logger.info("Bringing window to front: '%s' (HWND: %d)", window_title, target_hwnd)

            # Bypass Windows focus-stealing restrictions
            fg_hwnd = win32gui.GetForegroundWindow()
            if fg_hwnd != target_hwnd:
                curr_thread = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
                target_thread = win32process.GetWindowThreadProcessId(target_hwnd)[0]
                if curr_thread and curr_thread != target_thread:
                    user32.AttachThreadInput(curr_thread, target_thread, True)
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(target_hwnd)
                    user32.AttachThreadInput(curr_thread, target_thread, False)
                else:
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(target_hwnd)
            return True
        except Exception as exc:
            logger.error("Failed to focus window '%s': %s", title_pattern, exc)
            return False

    def wait(self, duration: float = 0.1) -> None:
        time.sleep(duration)


class MockActuator(BaseActuator):
    """Deterministic mock actuator recording all events for unit tests and CI."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def click(self, x: int, y: int) -> None:
        logger.info("[MOCK ACTUATOR] Click at (%d, %d)", x, y)
        self.events.append({"action": "click", "x": x, "y": y, "timestamp": time.time()})

    def right_click(self, x: int, y: int) -> None:
        logger.info("[MOCK ACTUATOR] Right click at (%d, %d)", x, y)
        self.events.append({"action": "right_click", "x": x, "y": y, "timestamp": time.time()})

    def double_click(self, x: int, y: int) -> None:
        logger.info("[MOCK ACTUATOR] Double Click at (%d, %d)", x, y)
        self.events.append({"action": "double_click", "x": x, "y": y, "timestamp": time.time()})

    def drag_and_drop(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        logger.info("[MOCK ACTUATOR] Drag from (%d, %d) to (%d, %d)", start_x, start_y, end_x, end_y)
        self.events.append({"action": "drag", "from": (start_x, start_y), "to": (end_x, end_y), "timestamp": time.time()})

    def type_text(self, text: str, interval: float = 0.02) -> None:
        logger.info("[MOCK ACTUATOR] Type text: '%s'", text)
        self.events.append({"action": "type_text", "text": text, "interval": interval, "timestamp": time.time()})

    def paste_text(self, text: str) -> None:
        logger.info("[MOCK ACTUATOR] Paste text: '%s'", text[:30])
        self.events.append({"action": "paste_text", "text": text, "timestamp": time.time()})

    def hotkey(self, *keys: str) -> None:
        logger.info("[MOCK ACTUATOR] Hotkey: %s", "+".join(keys))
        self.events.append({"action": "hotkey", "keys": keys, "timestamp": time.time()})

    def scroll(self, clicks: int) -> None:
        logger.info("[MOCK ACTUATOR] Scroll: %d", clicks)
        self.events.append({"action": "scroll", "clicks": clicks, "timestamp": time.time()})

    def launch_app(self, app_name: str) -> None:
        logger.info("[MOCK ACTUATOR] Launch app: '%s'", app_name)
        self.events.append({"action": "launch_app", "app": app_name, "timestamp": time.time()})

    def focus_window(self, title_pattern: str) -> bool:
        logger.info("[MOCK ACTUATOR] Focus window: '%s'", title_pattern)
        self.events.append({"action": "focus_window", "title": title_pattern, "timestamp": time.time()})
        return True

    def wait(self, duration: float = 0.1) -> None:
        self.events.append({"action": "wait", "duration": duration, "timestamp": time.time()})

    def clear(self) -> None:
        self.events.clear()


def get_actuator(mock: bool = False, headless: bool = False) -> BaseActuator:
    """Factory function for actuator creation."""
    if mock:
        return MockActuator()
    return PyAutoGUIActuator(headless=headless)
