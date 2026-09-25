# Reai: Anticipatory Local Voice OS Controller

Reai is a production-grade, 100% local, dual-system autonomous OS navigation and voice control framework engineered for sub-second anticipatory actuation, determinism, and safety.

Inspired by real-time streaming AI assistants, Reai goes beyond basic intent routers by combining sub-speech anticipatory discrete decision-making (<1ms inference) with native OS Accessibility Tree perception (Windows UIA) and local generative LLMs.

---

## 1. System Architecture

```text
               +-------------------------------------------+
               |        Streaming Voice (Whisper)          |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |         Perception & UI Pruner            |
               |  (OS Accessibility Tree -> Top 25 Nodes)  |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |        System 1: Reflex Engine            |
               |    (Laya ModernBERT Typed Decisions)      |
               |       Latency: ~1-30ms (SLA <= 50ms)      |
               +-------------------------------------------+
                        /                         \
           Discrete Actions                      Free-form Text
         (CLICK, TYPE, FOCUS)                 (action == CALL_LLM)
                    |                                     |
                    v                                     v
  +-----------------------------------+   +-------------------------------+
  |       Safety Guardrail            |   |   System 2: Local LLM         |
  |  (Blast-Radius Risk Scorer)       |   |   (Qwen 2.5 3B via Ollama)    |
  +-----------------------------------+   +-------------------------------+
                    |                                     |
                    +------------------+------------------+
                                       |
                                       v
               +-------------------------------------------+
               |             Actuator Layer                |
               |   (PyAutoGUI / Safe Headless Runner)      |
               +-------------------------------------------+
                                       |
                                       v
               +-------------------------------------------+
               |          State Mutation Verifier          |
               |   (SHA-256 Hash Delta & Stuck Loop Stop)  |
               +-------------------------------------------+
```

### Core Architecture
1. **System 1 (Laya Reflex Decision Engine):** Uses `laya` (ModernBERT-based bi-directional decision engine) for ultra-low latency (~1-30ms) discrete action choices (`CLICK`, `DOUBLE_CLICK`, `TYPE`, `FOCUS_WINDOW`, `SCROLL`, `WAIT`). Eliminates hallucinations and slow token-by-token decoding for OS navigation.
2. **Perception Layer:** Extracts native UI Accessibility Trees (Windows UI Automation via `pywinauto` / cross-platform mock) and prunes them to fit Laya's context budget.
3. **Actuator Layer:** Translates chosen element IDs into OS-level mouse/keyboard events (`pyautogui`, Win32 API) or headless simulated events.
4. **System 2 (Local LLM Synthesis):** Activates a local Ollama model (Qwen 2.5 3B) ONLY when free-form text or creative drafting is needed (`action_type == "CALL_LLM"`).
5. **Safety Guardrail & Blast Radius Scorer:** Evaluates risk on high-consequence operations (delete, financial checkout, format commands) and halts execution with a `PermissionError` unless explicitly confirmed.
6. **State Mutation Verifier:** Computes SHA-256 UI state fingerprints before and after each action to ensure that the view actually mutated, stopping infinite loops if repeated actions yield zero state changes.

---

## 2. Directory Structure

```text
Reai/
├── config/
│   ├── settings.yaml          # Global settings (thresholds, pruning, model configs)
│   └── safety_rules.yaml      # Blast-radius rules, dangerous commands & keywords
├── src/
│   ├── actuator/
│   │   └── mouse_keyboard.py  # Cross-platform PyAutoGUI & safe headless actuator
│   ├── assistant/
│   │   └── assistant_core.py  # Voice assistant orchestration loop & push-to-talk
│   ├── generative/
│   │   └── llm_fallback.py    # System 2 free-form text generator (Local Ollama / Mock)
│   ├── orchestrator/
│   │   ├── models.py          # Strict Pydantic v2 schemas (UIElement, AgentState, ReflexDecision)
│   │   └── state_machine.py   # Sense -> Decide -> Guardrail -> Act -> Verify async loop
│   ├── perception/
│   │   ├── accessibility.py   # Windows UIA & Mock accessibility drivers
│   │   └── state_pruner.py    # Spatial & semantic pruner (limits UI tree to top 25 nodes)
│   ├── reflex/
│   │   ├── guardrail.py       # Blast-radius risk scorer & destructive action blocker
│   │   ├── laya_engine.py     # Laya typed-decisions engine with deterministic fallback
│   │   └── verifier.py        # SHA-256 UI state hash verifier & stuck-loop detector
│   ├── voice/
│   │   ├── audio_stream.py    # Low-latency microphone streaming & push-to-talk listener
│   │   └── stt_engine.py      # Fast local Whisper STT (faster-whisper)
│   └── main.py                # Unified CLI entry point
├── tests/
│   ├── test_mock_gui.py       # Tkinter test harness with live widgets & login flows
│   ├── test_reflex_engine.py  # Action prediction and <=50ms latency benchmarking
│   ├── test_safety_guardrail.py# Destructive action blocking and stuck-loop verification
│   └── test_voice_assistant.py# Voice loop unit tests
├── pytest.ini
├── requirements.txt
├── run.py                     # Convenience runner script
├── run_demo.py                # Interactive Tkinter GUI demo
└── README.md
```

---

## 3. Installation & Setup

```bash
# Clone the repository
git clone git@github.com:ReKara12/Reai.git
cd Reai

# Install dependencies
pip install -r requirements.txt
```

### System 2 Local LLM Setup (Optional for Free-form Text)
For creative drafting and free-form text entry, install Ollama and pull Qwen 2.5:
```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## 4. Running Reai

```bash
# Push-to-Talk Voice Assistant (Press SPACE to talk)
python run.py voice

# Run with a user goal via CLI
python run.py cli --goal "Open Antigravity and start a quick project"

# Run latency benchmark suite
python run.py benchmark

# Run interactive Tkinter GUI demo
python run.py demo

# Run in safe headless mode
python run.py cli --goal "Click submit" --headless --mock
```

---

## 5. Running the Test Suite

```bash
python run.py test
```
All unit tests validate:
- Sub-50ms System 1 Laya decision latency
- Destructive operation interception and blast-radius safety checks
- UI tree spatial & semantic pruning
- SHA-256 state mutation checking and stuck-loop prevention
- Dual-system voice and OS execution flow
