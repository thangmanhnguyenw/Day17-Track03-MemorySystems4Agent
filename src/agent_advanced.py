from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from live_runtime import build_chat_model_or_none, invoke_live_chat, resolve_force_offline
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_candidates


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool | None = None) -> None:
        self.config = config or load_config()
        self.force_offline = resolve_force_offline(
            self.config.agent_mode,
            self.config.model,
            force_offline,
        )
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.user_turn_counter: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not self.force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        result = self._reply_offline(user_id, thread_id, message)
        result["mode"] = "offline"
        return result

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _persist_profile_updates(self, user_id: str, message: str) -> None:
        self.user_turn_counter[user_id] = self.user_turn_counter.get(user_id, 0) + 1
        turn = self.user_turn_counter[user_id]
        threshold = self.config.profile_confidence_threshold

        for fact in extract_profile_candidates(message):
            if fact.confidence < threshold:
                continue
            self.profile_store.upsert_fact(
                user_id,
                fact.key,
                fact.value,
                confidence=fact.confidence,
                turn=turn,
                is_correction=fact.is_correction,
            )

    def _build_live_system_prompt(self, user_id: str, thread_id: str) -> str:
        profile_text = self.profile_store.read_text(user_id)
        thread_state = self.compact_memory.context(thread_id)
        summary = str(thread_state.get("summary", "")).strip()
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]

        recent_lines = [f"{item['role']}: {item['content']}" for item in messages]
        parts = [
            "Bạn là Advanced Agent với persistent User.md và compact memory.",
            "Ưu tiên facts trong User.md khi trả lời recall cross-session.",
            "Nếu có correction mới, ưu tiên fact mới hơn.",
            "Trả lời ngắn gọn bằng tiếng Việt.",
            "",
            "=== User.md ===",
            profile_text.strip(),
        ]
        if summary:
            parts.extend(["", "=== Compact summary ===", summary])
        if recent_lines:
            parts.extend(["", "=== Recent thread messages ===", "\n".join(recent_lines)])
        return "\n".join(parts)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        thread_state = self.compact_memory.context(thread_id)
        history: list[dict[str, str]] = list(thread_state["messages"])  # type: ignore[assignment]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == message:
            history = history[:-1]

        system_prompt = self._build_live_system_prompt(user_id, thread_id)
        answer, _, completion_tokens, prompt_tokens = invoke_live_chat(
            self.langchain_agent,
            system_prompt=system_prompt,
            history=history,
            user_message=message,
        )

        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + completion_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "answer": answer,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)

        return {
            "answer": answer,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        profile_text = self.profile_store.read_text(user_id)
        thread_state = self.compact_memory.context(thread_id)
        summary = str(thread_state.get("summary", ""))
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]

        parts = [profile_text, summary]
        parts.extend(f"{item['role']}: {item['content']}" for item in messages)
        return estimate_tokens("\n".join(parts))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.get_recall_facts(
            user_id,
            current_turn=self.user_turn_counter.get(user_id, 0),
            threshold=self.config.profile_confidence_threshold,
            half_life=self.config.memory_decay_half_life,
        )
        lowered = message.lower()

        if "?" in message or any(
            keyword in lowered
            for keyword in (
                "nhắc lại",
                "tên mình",
                "mình tên",
                "làm nghề",
                "ở đâu",
                "style trả lời",
                "đồ uống",
                "món ăn",
                "nuôi",
                "tóm tắt",
                "mô tả ngắn",
                "biết",
            )
        ):
            parts: list[str] = []

            if any(key in lowered for key in ("tên", "biết", "tóm tắt", "mô tả")) and facts.get("name"):
                parts.append(f"tên là {facts['name']}")

            if any(key in lowered for key in ("nghề", "làm nghề", "nghề nghiệp", "làm gì")) and facts.get("profession"):
                parts.append(f"nghề nghiệp hiện tại là {facts['profession']}")

            if any(key in lowered for key in ("ở đâu", "nơi ở", "huế", "đà nẵng", "hà nội")) and facts.get("location"):
                parts.append(f"đang ở {facts['location']}")

            if any(key in lowered for key in ("style", "trả lời", "bullet")) and facts.get("response_style"):
                parts.append(f"style trả lời: {facts['response_style']}")

            if "đồ uống" in lowered and facts.get("favorite_drink"):
                parts.append(f"đồ uống yêu thích: {facts['favorite_drink']}")

            if "món ăn" in lowered and facts.get("favorite_food"):
                parts.append(f"món ăn yêu thích: {facts['favorite_food']}")

            if any(key in lowered for key in ("nuôi", "corgi", "con gì")) and facts.get("pet"):
                parts.append(f"mình nuôi {facts['pet']}")

            if "mối quan tâm" in lowered or "tóm tắt" in lowered:
                if facts.get("interests"):
                    parts.append(f"mối quan tâm: {facts['interests']}")
                if "tóm tắt" in lowered and facts.get("profession") and not any("nghề" in part for part in parts):
                    parts.append(f"nghề nghiệp hiện tại là {facts['profession']}")

            if parts:
                return "Theo User.md, " + "; ".join(parts) + "."

            return "Mình chưa có fact ổn định phù hợp trong User.md để trả lời câu này."

        return "Mình đã cập nhật User.md và compact memory cho phiên này."

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        return build_chat_model_or_none(self.config.model)
