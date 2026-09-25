"""Synthetic and real-world multi-turn OS reflex dataset generator for Laya / ModernBERT fine-tuning."""

import os
import json
import random
from typing import List, Dict, Any, Tuple

# Predefined templates across operating domains
DOMAINS = {
    "antigravity": {
        "apps": ["Antigravity", "Antigravity.exe"],
        "scenarios": [
            {
                "goals_tr": ["antigravity'de yeni bir proje başlat", "önüme antigravityi açıp quick start ile yeni proje oluştur", "yeni proje başlat"],
                "goals_en": ["open antigravity and start a quick project", "create a new project in antigravity", "start quick project"],
                "steps": [
                    {"window": "PowerShell [powershell.exe]", "action": "FOCUS_WINDOW", "target": "none", "text": "antigravity", "completed": False},
                    {
                        "window": "Antigravity - Welcome [Antigravity.exe]",
                        "elements": [
                            {"id": "btn_create_proj", "label": "Create New Project", "type": "Button"},
                            {"id": "btn_open_folder", "label": "Open Folder", "type": "Button"},
                            {"id": "btn_clone_repo", "label": "Clone Repository", "type": "Button"},
                        ],
                        "action": "CLICK",
                        "target": "btn_create_proj",
                        "text": None,
                        "completed": False,
                    },
                    {
                        "window": "Create Project Modal [Antigravity.exe]",
                        "elements": [
                            {"id": "btn_quick_start", "label": "Quick Start", "type": "Button"},
                            {"id": "btn_empty_proj", "label": "Empty Project", "type": "Button"},
                            {"id": "btn_cancel", "label": "Cancel", "type": "Button"},
                        ],
                        "action": "CLICK",
                        "target": "btn_quick_start",
                        "text": None,
                        "completed": False,
                    },
                    {
                        "window": "Workspace - Antigravity [Antigravity.exe]",
                        "elements": [
                            {"id": "lbl_ready", "label": "Project Ready", "type": "Text"},
                        ],
                        "action": "WAIT",
                        "target": "none",
                        "text": None,
                        "completed": True,
                    }
                ]
            }
        ]
    },
    "browser": {
        "apps": ["Mozilla Firefox [firefox.exe]", "Google Chrome [chrome.exe]"],
        "scenarios": [
            {
                "goals_tr": ["firefox'i aç yeni bir sekme açıp youtube'a git", "yeni sekmede youtube aç", "youtube'a git"],
                "goals_en": ["open firefox and open a new tab and go to youtube in that new tab", "open a new tab and go to youtube", "navigate to youtube"],
                "steps": [
                    {"window": "Desktop", "action": "FOCUS_WINDOW", "target": "none", "text": "firefox", "completed": False},
                    {"window": "Mozilla Firefox [firefox.exe]", "action": "HOTKEY", "target": "none", "text": "ctrl+t", "completed": False},
                    {"window": "New Tab - Mozilla Firefox [firefox.exe]", "action": "NAVIGATE_URL", "target": "none", "text": "https://www.youtube.com", "completed": False},
                    {"window": "YouTube - Mozilla Firefox [firefox.exe]", "action": "WAIT", "target": "none", "text": None, "completed": True},
                ]
            },
            {
                "goals_tr": ["google'da en son yapay zeka haberlerini ara", "google üzerinde modernbert ara"],
                "goals_en": ["search modernbert on google", "search AI news on google"],
                "steps": [
                    {"window": "Mozilla Firefox [firefox.exe]", "action": "NAVIGATE_URL", "target": "none", "text": "https://www.google.com/search?q=modernbert", "completed": False},
                    {"window": "modernbert - Google Search [firefox.exe]", "action": "WAIT", "target": "none", "text": None, "completed": True},
                ]
            },
            {
                "goals_tr": ["bu sekmeyi kapat", "açık sekmeyi kapat"],
                "goals_en": ["close this tab", "close active tab"],
                "steps": [
                    {"window": "Mozilla Firefox [firefox.exe]", "action": "HOTKEY", "target": "none", "text": "ctrl+w", "completed": False},
                    {"window": "Mozilla Firefox [firefox.exe]", "action": "WAIT", "target": "none", "text": None, "completed": True},
                ]
            }
        ]
    },
    "notepad": {
        "apps": ["Untitled - Notepad [notepad.exe]", "Adsız - Not Defteri [notepad.exe]"],
        "scenarios": [
            {
                "goals_tr": ["not defterine toplantı saat 15:00'te yaz", "not defterini aç ve toplantı saat 15:00'te yaz"],
                "goals_en": ["type meeting at 3pm into notepad", "open notepad and type meeting at 3pm"],
                "steps": [
                    {"window": "Desktop", "action": "FOCUS_WINDOW", "target": "none", "text": "notepad", "completed": False},
                    {
                        "window": "Untitled - Notepad [notepad.exe]",
                        "elements": [
                            {"id": "txt_editor", "label": "Text Editor", "type": "Edit"},
                        ],
                        "action": "TYPE",
                        "target": "txt_editor",
                        "text": "meeting at 3pm",
                        "completed": False,
                    },
                    {
                        "window": "Untitled - Notepad [notepad.exe]",
                        "elements": [
                            {"id": "txt_editor", "label": "meeting at 3pm", "type": "Edit"},
                        ],
                        "action": "WAIT",
                        "target": "none",
                        "text": None,
                        "completed": True,
                    }
                ]
            }
        ]
    },
    "vscode": {
        "apps": ["Visual Studio Code [Code.exe]"],
        "scenarios": [
            {
                "goals_tr": ["vscode'da terminali aç", "vscode'u açıp terminal aç"],
                "goals_en": ["open terminal in vscode", "focus vscode and toggle terminal"],
                "steps": [
                    {"window": "Desktop", "action": "FOCUS_WINDOW", "target": "none", "text": "vscode", "completed": False},
                    {"window": "Visual Studio Code [Code.exe]", "action": "HOTKEY", "target": "none", "text": "ctrl+`", "completed": False},
                    {"window": "Terminal - Visual Studio Code [Code.exe]", "action": "WAIT", "target": "none", "text": None, "completed": True},
                ]
            }
        ]
    },
    "generative": {
        "apps": ["Email Client [Thunderbird.exe]", "Gmail - Mozilla Firefox [firefox.exe]"],
        "scenarios": [
            {
                "goals_tr": ["müşteriye gecikme için özür dileyen profesyonel bir e-posta hazırla", "toplantı özeti taslağı çıkar"],
                "goals_en": ["draft a polite apology email for the project delay", "write a summary of the quarterly earnings report"],
                "steps": [
                    {"window": "Email Client [Thunderbird.exe]", "action": "CALL_LLM", "target": "none", "text": None, "completed": False},
                    {"window": "Email Client [Thunderbird.exe]", "action": "WAIT", "target": "none", "text": None, "completed": True},
                ]
            }
        ]
    }
}


def generate_dataset(num_samples: int = 2000, val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates synthetic multi-turn training and validation datasets."""
    samples = []
    
    # Flatten all scenarios
    all_scenarios = []
    for domain_name, domain_data in DOMAINS.items():
        for scen in domain_data["scenarios"]:
            all_scenarios.append(scen)
            
    while len(samples) < num_samples:
        scen = random.choice(all_scenarios)
        lang = random.choice(["tr", "en"])
        goals = scen["goals_tr"] if lang == "tr" else scen["goals_en"]
        goal = random.choice(goals)
        
        # Pick a random step in the scenario
        steps = scen["steps"]
        step_idx = random.randint(0, len(steps) - 1)
        cur_step = steps[step_idx]
        
        # Build history from prior steps
        history = []
        for i in range(step_idx):
            prev = steps[i]
            desc = prev.get("text") or prev.get("target") or "UI"
            history.append(f"Step {i+1}: {prev['action']} on '{desc}' (mutated=True)")
            
        elements = cur_step.get("elements", [])
        elements_str = [f"[{e['id']}] {e['type']} labeled '{e['label']}'" for e in elements]
        if not elements_str:
            elements_str = ["[none] No interactive elements available"]
            
        sample = {
            "instruction": goal,
            "active_window": cur_step["window"],
            "elements": elements_str,
            "history": history,
            "target_action": cur_step["action"],
            "target_element": cur_step["target"],
            "target_text": cur_step.get("text"),
            "is_completed": 1.0 if cur_step["completed"] else 0.0,
        }
        samples.append(sample)
        
    random.shuffle(samples)
    val_split = int(len(samples) * val_ratio)
    train_data = samples[val_split:]
    val_data = samples[:val_split]
    
    return train_data, val_data


def save_datasets(output_dir: str = "data"):
    """Generates and writes train and eval JSON datasets."""
    os.makedirs(output_dir, exist_ok=True)
    train_samples, val_samples = generate_dataset(2000, 0.1)
    
    train_path = os.path.join(output_dir, "train_os_reflex.json")
    val_path = os.path.join(output_dir, "eval_os_reflex.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
        
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(train_samples)} training samples -> {train_path}")
    print(f"Generated {len(val_samples)} validation samples -> {val_path}")


if __name__ == "__main__":
    save_datasets()
