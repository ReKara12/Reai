"""System 2 Local Generative LLM provider with Ollama (Qwen2.5/Llama) and offline fallbacks."""

import os
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class BaseGenerativeProvider(ABC):
    """Abstract interface for System 2 Generative LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Synthesizes free-form text based on user prompt and optional context."""
        pass


class OllamaGenerativeProvider(BaseGenerativeProvider):
    """100% Local LLM provider via Ollama using high-quality local models (e.g. Qwen2.5-3B)."""

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        url = f"{self.host}/api/generate"
        system_instruction = (
            "Sen yardımsever, profesyonel bir masaüstü yapay zeka asistanısın. "
            "Kullanıcının talep ettiği metinleri, e-postaları, özetleri veya kodları "
            "doğrudan uygulamaya yapıştırılmaya hazır, temiz ve eksiksiz bir formatta hazırla."
        )

        full_prompt = prompt
        if context:
            full_prompt = f"Bağlam ve Açık Pencereler:\n{context}\n\nTalep:\n{prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": system_instruction,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result_text = data.get("response", "").strip()
                logger.info("[OLLAMA %s] Generated %d chars.", self.model_name, len(result_text))
                return result_text
        except Exception as exc:
            logger.warning("Ollama generation failed (%s). Falling back to local template synthesizer.", exc)
            return MockGenerativeProvider().generate(prompt, context)


class MockGenerativeProvider(BaseGenerativeProvider):
    """Deterministic local mock provider generating structured answers without network calls."""

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        logger.info("[SYSTEM 2 MOCK LLM] Generating completion for: '%s'", prompt)
        lower_prompt = prompt.lower()

        if "email" in lower_prompt or "e-posta" in lower_prompt or "toplantı" in lower_prompt:
            return (
                "Konu: Yarınki Toplantı ve Gündem Maddeleri\n\n"
                "Merhaba Ekip,\n\n"
                "Yarınki toplantımızda ele alacağımız temel konular aşağıdadır:\n"
                "1. Proje yol haritası ve öncelikli hedefler\n"
                "2. Sistem performans metrikleri ve optimizasyonlar\n"
                "3. Soru-cevap ve sonraki adımların planlanması\n\n"
                "Katılımınız için teşekkürler.\nİyi çalışmalar."
            )
        elif "summarize" in lower_prompt or "özet" in lower_prompt:
            return f"Özet: İlgili metinden ({len(context or '')} karakter) temel eylem maddeleri ve bulgular derlenmiştir."
        elif "kod" in lower_prompt or "code" in lower_prompt:
            return "def process_data(items):\n    return [item.strip() for item in items if item]"

        return f"Talep doğrultusunda hazırlanan metin: {prompt}"


def get_generative_provider(provider_type: str = "auto", **kwargs) -> BaseGenerativeProvider:
    """Factory helper to obtain the optimal local generative provider."""
    if provider_type.lower() == "mock":
        return MockGenerativeProvider()

    if provider_type.lower() in ("ollama", "auto"):
        # Check if Ollama service is reachable on localhost:11434
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    model = kwargs.get("model_name", DEFAULT_OLLAMA_MODEL)
                    logger.info("Using local Ollama Generative Provider with model: %s", model)
                    return OllamaGenerativeProvider(model_name=model)
        except Exception:
            pass

    return MockGenerativeProvider()
