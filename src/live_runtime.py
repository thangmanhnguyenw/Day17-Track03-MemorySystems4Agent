from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from memory_store import estimate_tokens
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def has_live_credentials(model_config: ProviderConfig) -> bool:
    """Return True when the selected provider can attempt a live LLM call."""

    provider = normalize_provider(model_config.provider)
    if provider in {"ollama", "custom"}:
        return True
    return bool(model_config.api_key and model_config.api_key.strip())


def resolve_force_offline(agent_mode: str, model_config: ProviderConfig, force_offline: bool | None) -> bool:
    """Resolve runtime mode: explicit flag wins, else `AGENT_MODE` from config."""

    if force_offline is not None:
        return force_offline

    mode = (agent_mode or "offline").strip().lower()
    if mode == "offline":
        return True
    if mode == "live":
        return False
    # auto: use live when credentials exist
    return not has_live_credentials(model_config)


def build_chat_model_or_none(model_config: ProviderConfig):
    """Build a chat model for live mode, or return None if unavailable."""

    if not has_live_credentials(model_config):
        return None
    try:
        return build_chat_model(model_config)
    except Exception:
        return None


def to_langchain_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def extract_token_usage(response: Any, prompt_text: str, answer: str) -> tuple[int, int]:
    """Read token usage from provider metadata when available, else estimate."""

    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(answer)

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or prompt_tokens)
        completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or completion_tokens)
        return completion_tokens, prompt_tokens

    meta = getattr(response, "response_metadata", None) or {}
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage_metadata") or {}
        if isinstance(token_usage, dict):
            prompt_tokens = int(token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(
                token_usage.get("output_tokens") or token_usage.get("completion_tokens") or completion_tokens
            )

    return completion_tokens, prompt_tokens


def invoke_live_chat(
    model,
    *,
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> tuple[str, Any, int, int]:
    """Call the live chat model and return answer + raw response + token counts."""

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    prompt_messages.extend(to_langchain_messages(history))
    prompt_messages.append(HumanMessage(content=user_message))

    response = model.invoke(prompt_messages)
    answer = str(getattr(response, "content", response)).strip()
    prompt_text = "\n".join(
        f"{type(msg).__name__}: {getattr(msg, 'content', '')}" for msg in prompt_messages
    )
    completion_tokens, prompt_tokens = extract_token_usage(response, prompt_text, answer)
    return answer, response, completion_tokens, prompt_tokens
