"""Activity Logger for Reflex-Agent.

Logs user voice transcripts, parsed goals, speculative decisions, executed actions,
and task results to both a persistent log file (logs/agent_activity.log) and
in-memory buffer for real-time display in the UI.
"""

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Optional, Callable


LOG_FILE_PATH = Path("logs/agent_activity.log")


class ActivityLogger:
    """Thread-safe activity logger for voice transcripts and executed actions."""
    _instance: Optional["ActivityLogger"] = None
    _lock = threading.Lock()

    def __init__(self, log_path: Path = LOG_FILE_PATH, max_history: int = 200):
        self.log_path = log_path
        self.max_history = max_history
        self._history = deque(maxlen=max_history)
        self._callbacks: List[Callable[[str, str, str], None]] = []
        self._file_lock = threading.Lock()

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "ActivityLogger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_callback(self, callback: Callable[[str, str, str], None]):
        """Registers a callback(timestamp, category, message) invoked on each new log."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def log(self, category: str, message: str) -> str:
        """Logs an event with category and message."""
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        short_time = now.strftime("%H:%M:%S")
        formatted = f"[{time_str}] [{category:<14}] {message}"

        # In-memory history
        entry = (short_time, category, message)
        self._history.append(entry)

        # Write to disk
        with self._file_lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception as e:
                pass

        # Notify callbacks (e.g. Qt UI slots)
        for cb in self._callbacks:
            try:
                cb(short_time, category, message)
            except Exception:
                pass

        return formatted

    # Semantic Helpers
    def log_voice_heard(self, raw_transcript: str):
        """Logs what the user said according to the speech-to-text model."""
        return self.log("SES_ALGILANDI", f'🎙️ Söylenen: "{raw_transcript}"')

    def log_goal_understood(self, goal: str):
        """Logs what intent or goal the agent derived from the speech."""
        return self.log("HEDEF_ANALIZI", f'🧠 Anlaşılan Amaç: "{goal}"')

    def log_anticipation(self, action: str, target: str, latency_ms: float):
        """Logs the sub-35ms reflex prediction before execution."""
        return self.log("REFLEKS_TAHMİN", f'⚡ Öngörülen Eylem: [{action}] ➔ {target} ({latency_ms:.1f}ms)')

    def log_action_step(self, step_num: int, action: str, target: str, status: str = "Başarılı"):
        """Logs an executed action step on the desktop."""
        return self.log(f"ADIM_{step_num}", f'🚀 Yapılan İşlem: [{action}] ➔ {target} ({status})')

    def log_finished(self, status: str, total_steps: int):
        """Logs completion status of the task."""
        if status == "SUCCESS":
            return self.log("SONUÇ", f'✓ Görev {total_steps} adımda başarıyla tamamlandı.')
        elif status == "GUARDRAIL_BLOCKED":
            return self.log("GÜVENLİK", f'🛡️ Güvenlik Engeli: Yıkıcı eyleme izin verilmedi.')
        else:
            return self.log("SONUÇ", f'⚠️ İşlem Sonlandı: {status} ({total_steps} adım)')

    def log_error(self, error_message: str):
        """Logs an error encountered during perception, model inference, or actuation."""
        return self.log("HATA", f'❌ {error_message}')

    def get_recent_entries(self) -> List[tuple]:
        """Returns list of (short_time, category, message) entries."""
        return list(self._history)

    def get_full_log_path(self) -> str:
        """Returns absolute path to the log file."""
        return str(self.log_path.resolve())

    def clear_history(self):
        """Clears in-memory history."""
        self._history.clear()


# Global convenience accessor
activity_logger = ActivityLogger.get_instance()
