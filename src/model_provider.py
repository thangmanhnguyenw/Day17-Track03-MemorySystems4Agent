from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_PROVIDER_ALIASES = {
    "openai": "openai",
    "custom": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""

    normalized = (value or "openai").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
    }

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=config.api_key, **kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
            **kwargs,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=config.api_key, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=config.api_key, **kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=config.base_url, **kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(api_key=config.api_key, **kwargs)

    raise ValueError(f"Unsupported provider: {config.provider}")
