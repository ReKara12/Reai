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
                    "CALL_LLM": "Activate System 2 LLM fallback for free-form generative text synthesis",
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

        questions = self.build_query_questions(state)
        compact_state = {
            "goal": state.user_goal,
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
        raw_goal = state.user_goal.strip()
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
            "antigravity", "chrome", "notepad", "not defteri", "spotify", "vscode",
            "code", "calc", "hesap makinesi", "terminal", "powershell", "explorer",
            "dosya gezgini", "word", "excel"
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
