from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    profile_confidence_threshold: float
    memory_decay_half_life: int
    agent_mode: str
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config(prefix: str = "") -> ProviderConfig:
    """Build a ProviderConfig from environment variables."""

    key_prefix = f"{prefix}_" if prefix else ""
    provider = normalize_provider(os.getenv(f"{key_prefix}LLM_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")))
    model_name = os.getenv(
        f"{key_prefix}LLM_MODEL",
        os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"),
    )
    temperature = float(os.getenv(f"{key_prefix}LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.2")))

    api_key = None
    base_url = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # TODO: read env vars for one of the supported providers.
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    # TODO: create `root / "state"`.
    state_dir.mkdir(parents=True, exist_ok=True)

    # TODO: choose sensible defaults for compact memory.
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))
    profile_confidence_threshold = float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.75"))
    memory_decay_half_life = int(os.getenv("MEMORY_DECAY_HALF_LIFE", "30"))
    agent_mode = os.getenv("AGENT_MODE", "offline").strip().lower()

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        profile_confidence_threshold=profile_confidence_threshold,
        memory_decay_half_life=memory_decay_half_life,
        agent_mode=agent_mode,
        model=_provider_config(),
        judge_model=_provider_config("JUDGE"),
    )
