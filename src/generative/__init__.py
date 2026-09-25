"""Generative Layer (System 2) for complex free-form text synthesis."""

from .llm_fallback import (
    BaseGenerativeProvider,
    MockGenerativeProvider,
    OllamaGenerativeProvider,
    get_generative_provider,
)

# Backward-compatibility alias
LiteLLMGenerativeProvider = OllamaGenerativeProvider

__all__ = [
    "BaseGenerativeProvider",
    "MockGenerativeProvider",
    "OllamaGenerativeProvider",
    "LiteLLMGenerativeProvider",
    "get_generative_provider",
]
