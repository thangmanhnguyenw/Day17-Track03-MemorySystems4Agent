from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from live_runtime import build_chat_model_or_none, invoke_live_chat, resolve_force_offline
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool | None = None) -> None:
        self.config = config or load_config()
        self.force_offline = resolve_force_offline(
            self.config.agent_mode,
            self.config.model,
            force_offline,
        )
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not self.force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                pass
        result = self._reply_offline(thread_id, message)
        result["mode"] = "offline"
        return result

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        history = list(session.messages)
        system_prompt = (
            "Bạn là Baseline Agent. Chỉ nhớ thông tin trong thread hiện tại, "
            "không có User.md hay memory xuyên session. "
            "Trả lời ngắn gọn bằng tiếng Việt. "
            "Nếu chưa có thông tin trong thread này, hãy nói rõ là chưa biết."
        )
        answer, _, completion_tokens, prompt_tokens = invoke_live_chat(
            self.langchain_agent,
            system_prompt=system_prompt,
            history=history,
            user_message=message,
        )

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += completion_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "answer": answer,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "mode": "live",
        }

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        context_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in session.messages
        )
        session.prompt_tokens_processed += estimate_tokens(context_text)

        answer = self._generate_offline_answer(session.messages, message)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)

        return {
            "answer": answer,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _generate_offline_answer(self, messages: list[dict[str, str]], message: str) -> str:
        """Answer only from the current thread history."""

        lowered = message.lower()
        if "?" in message or any(
            keyword in lowered
            for keyword in ("nhắc lại", "tên mình", "mình tên gì", "làm nghề gì", "ở đâu")
        ):
            facts = self._extract_thread_facts(messages)
            if not facts:
                return "Mình chưa có đủ thông tin trong phiên này để trả lời chính xác."
            return "Trong phiên hiện tại mình biết: " + "; ".join(f"{k}={v}" for k, v in facts.items())

        return "Mình đã ghi nhận trong phiên hiện tại. Bạn có thể hỏi lại ở cuối phiên này."

    def _extract_thread_facts(self, messages: list[dict[str, str]]) -> dict[str, str]:
        from memory_store import extract_profile_updates

        facts: dict[str, str] = {}
        for item in messages:
            if item["role"] != "user":
                continue
            facts.update(extract_profile_updates(item["content"]))
        return facts

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        return build_chat_model_or_none(self.config.model)
