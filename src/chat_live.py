"""Quick demo for live Gemini chat (requires valid GEMINI_API_KEY in .env)."""

from __future__ import annotations

import sys
from pathlib import Path

from agent_advanced import AdvancedAgent
from config import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    # Respect AGENT_MODE from .env; override to live for this demo script.
    agent = AdvancedAgent(config=config, force_offline=False)

    user_id = "demo-user"
    thread_id = "demo-thread"
    print(f"Advanced agent ready (offline={agent.force_offline}, model={config.model.model_name})")
    print("Nhập tin nhắn (exit để thoát):\n")

    while True:
        message = input("You: ").strip()
        if not message or message.lower() in {"exit", "quit"}:
            break
        result = agent.reply(user_id, thread_id, message)
        print(f"Agent [{result.get('mode', '?')}]: {result['answer']}\n")


if __name__ == "__main__":
    main()
