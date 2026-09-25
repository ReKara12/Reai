"""Laya-based System 1 Reflex Decision Engine for low-latency discrete OS control."""

import os
import re
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
import math

from ..orchestrator.models import UIElement, AgentState, ReflexDecision

logger = logging.getLogger(__name__)

# Goal synonym mapping for multi-lingual intent recognition
INTENT_SYNONYMS = {
    "login": ["giriş", "giris", "login", "sign in", "signin", "submit", "oturum aç", "enter"],
    "submit": ["submit", "onayla", "gönder", "save", "kaydet", "ok"],
    "delete": ["delete", "remove", "sil", "destroy", "kaldır", "drop", "wipe"],
    "type": ["type", "yaz", "enter", "fill", "gir", "input"],
    "cancel": ["cancel", "iptal", "close", "vazgeç", "dismiss"],
    "create": ["yeni", "olustur", "olusturma", "baslat", "create", "new", "add", "ekle"],
    "project": ["proje", "project", "workspace", "repo", "calisma alani"],
    "start": ["baslat", "start", "quick", "hizli", "launch", "run"],
    "llm": [
        "generate", "synthesize", "write story", "draft", "creative", "summarize",
        "özet", "ozet", "taslak", "hazırla", "hazirla", "rapor", "metin", "makale",
        "e-posta", "eposta", "email", "mail", "mektup", "yazı", "hikaye"
    ],
}


def normalize_text(text: str) -> str:
    """Normalizes Turkish and accented characters for robust matching."""
    t = text.lower()
    mapping = {
        'ş': 's', 'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ş': 's', 'Ğ': 'g', 'Ü': 'u', 'Ö': 'o', 'Ç': 'c'
    }
    for tr, asc in mapping.items():
        t = t.replace(tr, asc)
    # Replace unicode replacement characters if terminal/editor corrupted utf-8
    t = re.sub(r'[\ufffd\?]', 's', t)
    return t


class LayaReflexEngine:
    """System 1 Reflex Decision Engine utilizing Laya ModernBERT typed decisions

    with an ultra-fast deterministic fallback mock when offline or CPU-only.
    """

    def __init__(
        self,
        checkpoint: str = "convaiinnovations/laya",
        subfolder: str = "typed-decisions",
        fallback_to_mock: bool = True,
        force_mock: Optional[bool] = None,
    ):
        self.checkpoint = checkpoint
        self.subfolder = subfolder
        self.fallback_to_mock = fallback_to_mock
        self.force_mock = (
            force_mock
            if force_mock is not None
            else bool(os.environ.get("REFLEX_FORCE_MOCK", "0") == "1")
        )
        self.agent = None
        self._init_laya()

    def _init_laya(self) -> None:
        """Attempts to load the Laya Agent checkpoint, falling back to mock if unavailable."""
        if self.force_mock:
            logger.info("REFLEX_FORCE_MOCK is set. Operating in ultra-fast Mock mode.")
            return

        try:
            import laya
            logger.info("Initializing Laya Agent with checkpoint: %s (subfolder: %s)", self.checkpoint, self.subfolder)
            self.agent = laya.Agent(
                model_id_or_path=self.checkpoint,
                subfolder=self.subfolder,
                fast=True,
            )
            logger.info("Laya Agent initialized successfully.")
        except Exception as exc:
            if self.fallback_to_mock:
                logger.warning(
                    "Laya Agent could not be loaded (%s). Seamlessly falling back to high-performance Reflex mock.",
                    exc,
                )
                self.agent = None
            else:
                raise exc

    def build_query_questions(
        self, state: AgentState
    ) -> Dict[str, Dict[str, Any]]:
        """Constructs Laya-typed decision query structure:

        - choice: candidate element_id selection
        - choice: action_type (CLICK, TYPE, CALL_LLM, WAIT, DOUBLE_CLICK, SCROLL)
        - noul: is_task_completed
        """
        element_criteria = {}
        for elem in state.available_elements:
            element_criteria[elem.id] = f"{elem.control_type} labeled '{elem.label}'"

        if not element_criteria:
            element_criteria["none"] = "No interactive elements available"

        questions = {
            "selected_element_id": {
                "type": "choice",
                "instructions": f"Which element best matches the user's intent: '{state.user_goal}'?",
                "criteria": element_criteria,
            },
            "action_type": {
                "type": "choice",
                "instructions": f"What discrete OS action should be performed for goal '{state.user_goal}'?",
                "criteria": {
                    "CLICK": "Single left mouse click on the targeted interactive control",
                    "DOUBLE_CLICK": "Double left click to open or select",
                    "TYPE": "Type input text into the active edit/input field",
                    "SCROLL": "Scroll viewport up or down",
                    "WAIT": "Pause briefly for UI asynchronous transition",
                },
            },
            "is_task_completed": {
                "type": "noul",
                "instructions": f"Has the goal '{state.user_goal}' been fully and successfully completed based on current state?",
            },
        }
        return questions

    def predict(self, state: AgentState) -> ReflexDecision:
        """Executes the System 1 inference pass with sub-50ms latency tracking."""
        t_start = time.perf_counter()

        if self.agent is not None:
            try:
                decision = self._predict_with_laya(state)
            except Exception as exc:
                logger.warning("Laya inference failed (%s). Falling back to mock.", exc)
                decision = self._predict_with_fallback(state)
        else:
            decision = self._predict_with_fallback(state)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        decision.execution_latency_ms = round(t_elapsed_ms, 2)
        return decision

    def _extract_active_subgoal(self, goal: str, active_window: str) -> str:
        """Strips completed app-launch prefixes when target app is already in the foreground."""
        norm = normalize_text(goal)
        act_lower = active_window.lower() if active_window else ""

        # English patterns: 'open <app> and/then <subgoal>'
        m_en = re.search(r"^(?:open|launch|switch to)\s+([a-zA-Z0-9_\-]+)\s+(?:and|then)\s+(.+)$", goal, re.IGNORECASE)
        if m_en:
            app_name, remaining = m_en.group(1).lower(), m_en.group(2).strip()
            if app_name in act_lower:
                return remaining

        # Turkish patterns: 'önüme <app> açıp <subgoal>' / '<app>'i aç ve <subgoal>'
        m_tr = re.search(r"^(?:onume\s+)?([a-zA-Z0-9_\-]+)(?:'i|'ı|'yi|'yı|i|ı)?\s+(?:acip|ac\s+ve|getir\s+ve)\s+(.+)$", norm)
        if m_tr:
            app_name, remaining = m_tr.group(1).lower(), m_tr.group(2).strip()
            if app_name in act_lower:
                return remaining

        return goal

    def _predict_with_laya(self, state: AgentState) -> ReflexDecision:
        """Performs inference via the real Laya agent."""
        raw_goal = state.user_goal.strip()
        goal_norm = normalize_text(raw_goal)

        # 1. Fast-path: Check for Generative LLM synthesis requirement
        if self._is_generative_drafting(goal_norm):
            return ReflexDecision(
                selected_element_id=None,
                action_type="CALL_LLM",
                confidence=0.99,
                is_task_completed=False,
                reasoning="Generative drafting intent detected; routed to System 2 LLM.",
            )

        # 2. Fast-path: Check if application focus is needed
        app_to_focus = self._extract_app_focus_intent(goal_norm, state)
        if app_to_focus:
            return ReflexDecision(
                selected_element_id=None,
                action_type="FOCUS_WINDOW",
                text_to_type=app_to_focus,
                confidence=0.99,
                is_task_completed=False,
                reasoning=f"Detected application '{app_to_focus}' in goal while current window is '{state.active_window}'. Bringing to front.",
            )

        # 3. Fast-path: Check for hotkey intent (e.g. new tab ctrl+t, close tab ctrl+w)
        hotkey_info = self._extract_hotkey_intent(goal_norm, state)
        if hotkey_info:
            hotkey_str, hotkey_reason = hotkey_info
            return ReflexDecision(
                selected_element_id=None,
                action_type="HOTKEY",
                text_to_type=hotkey_str,
                confidence=0.99,
                is_task_completed=False,
                reasoning=hotkey_reason,
            )

        # 4. Fast-path: Check for direct URL navigation (e.g. go to youtube, visit website)
        target_url = self._extract_url_navigation(raw_goal)
        if target_url:
            hist_str = " ".join(state.history)
            if "NAVIGATE_URL" in hist_str or target_url in hist_str:
                return ReflexDecision(
                    selected_element_id=None,
                    action_type="WAIT",
                    confidence=0.99,
                    is_task_completed=True,
                    reasoning=f"URL navigation to '{target_url}' successfully executed.",
                )
            return ReflexDecision(
                selected_element_id=None,
                action_type="NAVIGATE_URL",
                text_to_type=target_url,
                confidence=0.99,
                is_task_completed=False,
                reasoning=f"Opening URL '{target_url}' directly via OS browser actuator.",
            )

        # 5. Fast-path: Standalone hotkey completion check
        if any(w in goal_norm for w in ["new tab", "yeni sekme", "close tab", "sekme kapat", "refresh page", "sayfayi yenile"]):
            hist_str = " ".join(state.history).lower()
            if any(hk in hist_str for hk in ["ctrl+t", "ctrl+w", "ctrl+r"]):
                return ReflexDecision(
                    selected_element_id=None,
                    action_type="WAIT",
                    confidence=0.99,
                    is_task_completed=True,
                    reasoning="Hotkey shortcut completed successfully.",
                )

        active_subgoal = self._extract_active_subgoal(state.user_goal, state.active_window)
        questions = self.build_query_questions(state)
        if active_subgoal != state.user_goal:
            questions["selected_element_id"]["instructions"] = f"Which element best matches the user's intent: '{active_subgoal}'?"
            questions["action_type"]["instructions"] = f"What discrete OS action should be performed for goal '{active_subgoal}'?"

        compact_state = {
            "goal": active_subgoal,
            "window": state.active_window,
            "elements": [el.to_compact_str() for el in state.available_elements],
            "history": state.history[-3:] if state.history else [],
        }

        result = self.agent.predict(compact_state, questions)
        answers = result.get("answers", {})

        selected_id = answers.get("selected_element_id", {}).get("choice")
        action_type = answers.get("action_type", {}).get("choice", "CLICK")
        confidence = float(answers.get("selected_element_id", {}).get("confidence", 0.95))
        noul_val = float(answers.get("is_task_completed", {}).get("noul", 0.0))
        is_task_completed = noul_val >= 0.5

        if selected_id == "none":
            selected_id = None

        # Robust Decision Fusion:
        # If Laya selected a container matching the active window title (e.g. 'Antigravity')
        # while an interactive control matches the subgoal, fuse with the action control!
        fallback_decision = self._predict_with_fallback(state)
        if fallback_decision.selected_element_id and fallback_decision.selected_element_id != selected_id:
            chosen_elem = next((e for e in state.available_elements if e.id == selected_id), None)
            act_win_base = state.active_window.split("[")[0].strip().lower()
            if chosen_elem and (
                chosen_elem.label.lower() in act_win_base
                or chosen_elem.control_type.lower() in ("window", "control", "pane", "titlebar")
            ):
                logger.info(
                    "Laya selected inert container '%s'; fusing with high-confidence action control '%s'.",
                    selected_id,
                    fallback_decision.selected_element_id,
                )
                selected_id = fallback_decision.selected_element_id
                action_type = fallback_decision.action_type
                is_task_completed = fallback_decision.is_task_completed
                confidence = fallback_decision.confidence

        # CRITICAL SAFETY: An active actuation (CLICK/TYPE/FOCUS) can NEVER be marked as completed before execution!
        if selected_id is not None and action_type in ("CLICK", "DOUBLE_CLICK", "TYPE", "FOCUS_WINDOW"):
            is_task_completed = False

        text_to_type = self._extract_text_to_type(state.user_goal, action_type)

        return ReflexDecision(
            selected_element_id=selected_id,
            action_type=action_type,
            text_to_type=text_to_type,
            confidence=confidence,
            is_task_completed=is_task_completed,
            reasoning=f"Laya inference: action={action_type}, element={selected_id}",
        )

    def _predict_with_fallback(self, state: AgentState) -> ReflexDecision:
        """Deterministic, ultra-fast System 1 decision engine fallback (<5ms).

        Accurately parses multilingual goals, matching candidate elements and action types.
        """
        active_subgoal = self._extract_active_subgoal(state.user_goal, state.active_window)
        raw_goal = active_subgoal.strip()
        goal_norm = normalize_text(raw_goal)
        elements = state.available_elements

        # 1. Check if the task is already completed (via UI status elements or history)
        is_completed = False
        for elem in elements:
            label_norm = normalize_text(elem.label)
            if "logged in" in label_norm or "giris yapildi" in label_norm:
                if any(syn in goal_norm for syn in INTENT_SYNONYMS["login"]):
                    is_completed = True
                    break
            elif "deleted" in label_norm or "silindi" in label_norm:
                if any(syn in goal_norm for syn in INTENT_SYNONYMS["delete"]):
                    is_completed = True
                    break
            elif any(term in label_norm for term in ["success", "basarili", "completed"]):
                # Generic success is only applicable if not conflicting with an active action
                if not any(k in goal_norm for k in ["delete", "sil", "remove", "type", "yaz"]):
                    is_completed = True
                    break

        if is_completed:
            return ReflexDecision(
                selected_element_id=None,
                action_type="WAIT",
                confidence=0.99,
                is_task_completed=True,
                reasoning="Task completion detected via active UI state label.",
            )

        # 2. Determine Action Type
        action_type = "CLICK"
        text_to_type = None
        target_hint = None

        # Check for Generative LLM synthesis vs UI control actions
        if self._is_generative_drafting(goal_norm):
            action_type = "CALL_LLM"
        elif app_to_focus := self._extract_app_focus_intent(goal_norm, state):
            return ReflexDecision(
                selected_element_id=None,
                action_type="FOCUS_WINDOW",
                text_to_type=app_to_focus,
                confidence=0.99,
                is_task_completed=False,
                reasoning=f"Detected application '{app_to_focus}' in goal while current window is '{state.active_window}'. Bringing to front.",
            )
        elif hotkey_info := self._extract_hotkey_intent(goal_norm, state):
            hotkey_str, hotkey_reason = hotkey_info
            return ReflexDecision(
                selected_element_id=None,
                action_type="HOTKEY",
                text_to_type=hotkey_str,
                confidence=0.99,
                is_task_completed=False,
                reasoning=hotkey_reason,
            )
        elif target_url := self._extract_url_navigation(raw_goal):
            hist_str = " ".join(state.history)
            if "NAVIGATE_URL" in hist_str or target_url in hist_str:
                return ReflexDecision(
                    selected_element_id=None,
                    action_type="WAIT",
                    confidence=0.99,
                    is_task_completed=True,
                    reasoning=f"URL navigation to '{target_url}' successfully executed.",
                )
            return ReflexDecision(
                selected_element_id=None,
                action_type="NAVIGATE_URL",
                text_to_type=target_url,
                confidence=0.99,
                is_task_completed=False,
                reasoning=f"Opening URL '{target_url}' directly via OS browser actuator.",
            )
        elif any(w in goal_norm for w in ["new tab", "yeni sekme", "close tab", "sekme kapat", "refresh page", "sayfayi yenile"]):
            hist_str = " ".join(state.history).lower()
            if any(hk in hist_str for hk in ["ctrl+t", "ctrl+w", "ctrl+r"]):
                return ReflexDecision(
                    selected_element_id=None,
                    action_type="WAIT",
                    confidence=0.99,
                    is_task_completed=True,
                    reasoning="Hotkey shortcut completed successfully.",
                )
        elif any(term in goal_norm for term in ["scroll", "kaydir"]):
            action_type = "SCROLL"
        elif any(term in goal_norm for term in ["double click", "cift tikla"]):
            action_type = "DOUBLE_CLICK"
        elif any(term in goal_norm for term in ["wait", "bekle"]):
            action_type = "WAIT"
        else:
            # Check for typing intent or edit control target
            is_type_intent = bool(
                re.search(r"\b(type|write|yaz)\b", goal_norm)
                or (re.search(r"\b(gir|enter)\b", goal_norm) and "giris" not in goal_norm and "login" not in goal_norm)
            )
            if is_type_intent:
                action_type = "TYPE"
                extracted_tuple = self._extract_type_info(raw_goal)
                text_to_type, target_hint = extracted_tuple

        # 3. Match Element Candidates using bi-directional relevance scoring
        best_element: Optional[UIElement] = None
        highest_score = -1.0
        scores: Dict[str, float] = {}

        for elem in elements:
            score = self._compute_relevance_score(goal_norm, elem, action_type, target_hint)
            scores[elem.id] = score
            if score > highest_score:
                highest_score = score
                best_element = elem

        # Calculate calibrated confidence
        total_score = sum(math.exp(s) for s in scores.values()) if scores else 1.0
        confidence = (
            math.exp(highest_score) / total_score
            if total_score > 0 and highest_score > 0
            else 0.85
        )
        confidence = min(0.99, max(0.50, confidence))

        selected_id = best_element.id if best_element and highest_score > 0.1 else None

        # If user target is an input field and action was CLICK without text, keep CLICK to focus
        if action_type != "CALL_LLM" and best_element and best_element.control_type.lower() in ("edit", "textbox", "input") and text_to_type:
            action_type = "TYPE"

        return ReflexDecision(
            selected_element_id=selected_id,
            action_type=action_type,
            text_to_type=text_to_type,
            confidence=round(confidence, 4),
            is_task_completed=False,
            reasoning=f"Reflex matched element '{selected_id}' (score: {highest_score:.2f}) with action '{action_type}'.",
        )

    def _is_generative_drafting(self, goal_norm: str) -> bool:
        """Detects if goal is a generative text synthesis / drafting request."""
        direct_keywords = [
            "generate", "synthesize", "creative", "summarize", "ozetle", "ozet cikar",
            "taslak hazirla", "taslak olustur", "taslak cikar", "taslak",
            "hikaye yaz", "siir yaz", "fikra anlat", "dilekce yaz", "acikla",
            "kod yaz", "program yaz", "script yaz"
        ]
        if any(kw in goal_norm for kw in direct_keywords):
            return True

        doc_types = [
            "eposta", "e-posta", "email", "mail", "rapor", "metin", "makale",
            "mektup", "yazi", "paragraf", "icerik", "dokuman", "belge", "ozet"
        ]
        draft_verbs = [
            "yaz", "hazirla", "olustur", "uret", "draft", "compose", "write", "create"
        ]
        has_doc = any(doc in goal_norm for doc in doc_types)
        has_verb = any(v in goal_norm for v in draft_verbs)
        if has_doc and has_verb:
            # Check if it's explicitly targeting an input field (e.g. 'e-posta alanina admin@test.com yaz')
            field_indicators = ["alanina", "kutusuna", "kismina", "input", "textbox", "into", " in "]
            if any(f in goal_norm for f in field_indicators):
                return False
            return True

        return False

    def _extract_app_focus_intent(self, goal_norm: str, state: AgentState) -> Optional[str]:
        """Detects if goal mentions an application to open/focus when not already active."""
        # 1. Do not repeat app focus if already executed in this session
        if any("FOCUS_WINDOW" in h for h in state.history):
            return None

        apps = [
            "antigravity", "chrome", "firefox", "edge", "brave", "notepad", "not defteri",
            "spotify", "vscode", "code", "calc", "hesap makinesi", "terminal", "powershell",
            "explorer", "dosya gezgini", "word", "excel"
        ]

        detected_app = None
        for app in apps:
            if re.search(rf"\b{re.escape(app)}(?:'i|'ı|'yi|'yı|i|ı|'e|'a|e|a)?\b", goal_norm):
                if re.search(r"\b(ac|acip|getir|baslat|one getir|gec|open|launch|switch|focus)\b", goal_norm):
                    detected_app = "notepad" if "not defter" in app else app
                    break

        if not detected_app:
            m = re.search(r"(?:onume|önüme)?\s*([a-zA-Z0-9_\-]+)(?:'i|'ı|'yi|'yı|i|ı)?\s+(?:ac|acip|getir|baslat|one getir)\b", goal_norm)
            if m:
                detected_app = m.group(1).strip()
            else:
                m_en = re.search(r"\b(?:open|launch|switch to|focus)\s+([a-zA-Z0-9_\-]+)\b", goal_norm)
                if m_en:
                    detected_app = m_en.group(1).strip()

        if detected_app:
            act_lower = state.active_window.lower() if state.active_window else ""
            if detected_app.lower() not in act_lower:
                return detected_app

        return None

    def _extract_hotkey_intent(self, goal_norm: str, state: AgentState) -> Optional[Tuple[str, str]]:
        """Detects hotkey shortcuts required by the goal if not already performed in session."""
        hist = " ".join(state.history).lower()

        # 1. New Tab: 'new tab', 'yeni sekme', 'tab ac'
        if any(w in goal_norm for w in ["new tab", "yeni sekme", "yeni bir sekme", "baska sekme"]):
            if "ctrl+t" not in hist:
                return "ctrl+t", "Opening a new browser tab via ctrl+t"

        # 2. Close Tab: 'close tab', 'sekme kapat', 'sekmeyi kapat'
        if any(w in goal_norm for w in ["close tab", "sekme kapat", "sekmeyi kapat", "tabi kapat"]):
            if "ctrl+w" not in hist:
                return "ctrl+w", "Closing active browser tab via ctrl+w"

        # 3. New Window: 'new window', 'yeni pencere'
        if any(w in goal_norm for w in ["new window", "yeni pencere"]):
            if "ctrl+n" not in hist:
                return "ctrl+n", "Opening a new window via ctrl+n"

        # 4. Refresh / Reload: 'refresh page', 'sayfayi yenile', 'yenile'
        if any(w in goal_norm for w in ["refresh page", "sayfayi yenile", "sayfa yenile", "reload"]):
            if "ctrl+r" not in hist and "f5" not in hist:
                return "ctrl+r", "Refreshing page via ctrl+r"

        # 5. Address Bar Focus: 'address bar', 'adres cubugu', 'url cubugu'
        if any(w in goal_norm for w in ["address bar", "adres cubugu", "url cubugu"]):
            if "ctrl+l" not in hist:
                return "ctrl+l", "Focusing browser address bar via ctrl+l"

        return None

    def _extract_url_navigation(self, raw_goal: str) -> Optional[str]:
        """Extracts web URL or website target from natural language navigation goal."""
        # Never treat explicit text typing into fields as URL navigation
        if re.search(r"\b(type|write|yaz|gir)\b.*(?:into|in|to|alanina|kutusuna|kismina)", raw_goal, re.IGNORECASE):
            return None

        norm = normalize_text(raw_goal)

        # 1. Direct explicit URL match
        url_match = re.search(r"https?://[^\s]+", raw_goal)
        if url_match:
            return url_match.group(0).rstrip(".,;)'\"")

        # 2. Domain pattern (must not be an email address like user@example.com)
        domain_match = re.search(r"(?<![@\w])([a-zA-Z0-9_\-]+\.(?:com|org|net|io|edu|gov|co|ai|dev|app|me|tr)(?:/[^\s]*)?)\b", raw_goal, re.IGNORECASE)
        if domain_match:
            candidate = domain_match.group(1).rstrip(".,;')\"'")
            idx = raw_goal.find(candidate)
            if idx > 0 and raw_goal[idx - 1] == "@":
                pass
            else:
                return f"https://{candidate}"

        # 3. Search query intent (e.g. search cats on google, google'da kediler ara)
        search_match = re.search(r"(?:search|ara)\s+['\"]?(.+?)['\"]?\s+(?:on|in|from|uzerinde)\s+(?:google|web)\b", raw_goal, re.IGNORECASE)
        if search_match:
            q = search_match.group(1).strip()
            return f"https://www.google.com/search?q={q.replace(' ', '+')}"
        tr_search = re.search(r"(?:google|web)(?:'da|'de|da|de)\s+['\"]?(.+?)['\"]?\s+ara\b", norm, re.IGNORECASE)
        if tr_search:
            q = tr_search.group(1).strip()
            return f"https://www.google.com/search?q={q.replace(' ', '+')}"

        # 4. Known popular websites with navigation verbs
        SITE_MAP = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://github.com",
            "twitter": "https://x.com",
            "x.com": "https://x.com",
            "reddit": "https://www.reddit.com",
            "wikipedia": "https://www.wikipedia.org",
            "amazon": "https://www.amazon.com",
            "linkedin": "https://www.linkedin.com",
            "netflix": "https://www.netflix.com",
            "chatgpt": "https://chatgpt.com",
            "openai": "https://chatgpt.com",
        }
        for site, url in SITE_MAP.items():
            if re.search(rf"\b{re.escape(site)}(?:'a|'e|'ye|'ya|a|e)?\b", norm):
                # Check for navigation or visit intent
                if any(w in norm for w in ["git", "ac", "acip", "bak", "go", "open", "visit", "navigate", "load", "launch"]):
                    return url
                # Or if site is explicitly named with tab context
                if "tab" in norm or "sekme" in norm:
                    return url

        return None

    def _extract_type_info(self, goal: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (text_to_type, target_field_hint) supporting Turkish and English formats."""
        norm = normalize_text(goal)

        # 1. Turkish: <field> alanına/kutusuna/kısmına/yerine <text> yaz/gir
        tr_field = re.search(
            r"(.+?)\s+(?:alanina|kutusuna|kismina|yerine)\s+['\"]?(.+?)['\"]?\s+(?:yaz|gir|ekle)$",
            norm,
            re.IGNORECASE,
        )
        if tr_field:
            return tr_field.group(2).strip(), tr_field.group(1).strip()

        # 2. English: type <text> into/in/to <field>
        into_pat = re.search(
            r"(?:type|write|enter|fill)\s+['\"]?(.+?)['\"]?\s+(?:into|in|to)\s+['\"]?(.+?)['\"]?$",
            goal,
            re.IGNORECASE,
        )
        if into_pat:
            return into_pat.group(1).strip(), into_pat.group(2).strip()

        # 3. Pattern: type 'text' or 'text' yaz
        quote_pat = re.search(r"['\"]([^'\"]+)['\"]", goal)
        if quote_pat:
            return quote_pat.group(1).strip(), None

        # 4. Simple English: type <text>
        simple_pat = re.search(r"(?:type|write|enter)\s+(\S+)", goal, re.IGNORECASE)
        if simple_pat:
            return simple_pat.group(1).strip(), None

        # 5. Simple Turkish: <text> yaz/gir
        tr_simple = re.search(r"(\S+)\s+(?:yaz|gir)$", norm, re.IGNORECASE)
        if tr_simple:
            return tr_simple.group(1).strip(), None

        return None, None

    def _compute_relevance_score(
        self, goal_norm: str, elem: UIElement, action: str, target_hint: Optional[str] = None
    ) -> float:
        """Computes semantic relevance score between goal intent and UI element."""
        label_norm = normalize_text(elem.label)
        id_norm = normalize_text(elem.id)
        ctrl_lower = elem.control_type.lower()
        score = 0.0

        # Direct target hint match (e.g. into Username)
        if target_hint:
            hint_norm = normalize_text(target_hint)
            if hint_norm in label_norm or hint_norm in id_norm:
                score += 8.0

        # Exact and substring matches
        if label_norm and label_norm in goal_norm:
            score += 4.0
        if id_norm and id_norm in goal_norm:
            score += 3.0

        # Cleaned ID prefix matches (e.g., txt_username -> username, btn_login -> login)
        clean_id = re.sub(r"^(txt_|btn_|lbl_|chk_)", "", id_norm)
        if clean_id and clean_id in goal_norm:
            score += 4.0

        # Word-level overlap
        goal_tokens = set(re.findall(r"\w+", goal_norm))
        label_tokens = set(re.findall(r"\w+", label_norm))
        overlap = goal_tokens.intersection(label_tokens)
        if overlap:
            score += 3.0 * len(overlap)

        # Synonym matches
        for intent, synonyms in INTENT_SYNONYMS.items():
            norm_synonyms = [normalize_text(s) for s in synonyms]
            goal_has_intent = any(syn in goal_norm for syn in norm_synonyms)
            label_has_intent = any(
                syn in label_norm or syn in id_norm or syn in clean_id for syn in norm_synonyms
            )

            if goal_has_intent and label_has_intent:
                score += 5.0

        # Control type appropriateness
        if action in ("CLICK", "DOUBLE_CLICK") and ctrl_lower == "button":
            score += 1.5
        elif action == "TYPE" and ctrl_lower in ("edit", "textbox", "input"):
            score += 2.0

        # Penalize non-interactive container controls that merely label the active window/app
        if label_norm and any(app in label_norm for app in ["antigravity", "chrome", "notepad", "terminal"]):
            if ctrl_lower not in ("button", "menuitem", "listitem", "edit"):
                score -= 6.0

        # Boost enabled elements
        if elem.is_enabled:
            score += 0.5
        else:
            score -= 3.0

        return max(0.0, score)

    def _extract_text_to_type(self, goal: str, action_type: str) -> Optional[str]:
        if action_type != "TYPE":
            return None
        text, _ = self._extract_type_info(goal)
        return text
