from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from live_runtime import resolve_force_offline
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_candidates, extract_profile_updates


def make_config(tmp_path: Path) -> LabConfig:
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    base = load_config(Path(__file__).resolve().parent.parent)
    return replace(
        base,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=2,
        agent_mode="offline",
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test-user"

    path = store.write_text(user_id, "# User Profile\n\n- name: Alice\n")
    assert path.exists()
    assert "Alice" in store.read_text(user_id)

    store.upsert_fact(user_id, "location", "Huế")
    facts = store.facts(user_id)
    assert facts["name"] == "Alice"
    assert facts["location"] == "Huế"

    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True
    assert "Bob" in store.read_text(user_id)
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    thread_id = "compact-thread"

    for index in range(12):
        manager.append(thread_id, "user", f"Tin nhắn dài số {index} với nội dung benchmark stress test.")

    assert manager.compaction_count(thread_id) >= 1
    context = manager.context(thread_id)
    messages = context["messages"]
    assert len(messages) <= 2 + 1  # keep_messages plus possible assistant append in agent
    assert str(context.get("summary", "")).strip() != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = "dungct"
    advanced.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, "session-1", "Mình ở Huế và làm MLOps engineer.")

    advanced_answer = advanced.reply(user_id, "session-2", "Mình tên gì và làm nghề gì?")["answer"]
    assert "DũngCT" in advanced_answer
    assert "MLOps" in advanced_answer

    baseline.reply(user_id, "session-1", "Chào bạn, mình tên là DũngCT.")
    baseline.reply(user_id, "session-1", "Mình ở Huế và làm MLOps engineer.")
    baseline_answer = baseline.reply(user_id, "session-2", "Mình tên gì và làm nghề gì?")["answer"]
    assert "DũngCT" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    thread_id = "long-thread"
    user_id = "stress-user"

    for index in range(20):
        message = f"Lượt {index}: mình đang benchmark compact memory với nội dung dài để tăng token context."
        baseline.reply(user_id, thread_id, message)
        advanced.reply(user_id, thread_id, message)

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)
    assert advanced.compaction_count(thread_id) >= 1
    assert advanced_prompt < baseline_prompt


def test_confidence_threshold_skips_question_only_turns(tmp_path: Path) -> None:
    """Bonus: do not persist facts from low-confidence question turns."""

    question = "Bạn có biết tên mình là DũngCT không?"
    assert extract_profile_updates(question, threshold=0.75) == {}
    assert extract_profile_candidates(question) == []


def test_conflict_handling_prefers_correction(tmp_path: Path) -> None:
    """Bonus: newer correction wins and negated old facts are not re-saved."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "conflict-user"

    agent.reply(user_id, "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply(user_id, "t2", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply(
        user_id,
        "t3",
        "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé, vì đó là thông tin cũ.",
    )

    facts = agent.profile_store.facts(user_id)
    assert facts["profession"] == "MLOps engineer"

    answer = agent.reply(user_id, "t4", "Hiện tại mình làm nghề gì?")["answer"]
    assert "MLOps" in answer
    assert "backend engineer" not in answer.lower()


def test_memory_decay_reduces_stale_low_confidence_facts(tmp_path: Path) -> None:
    """Bonus: stale low-confidence facts fall below the active recall threshold."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "decay-user"
    threshold = 0.75
    half_life = 5

    store.upsert_fact(user_id, "priority", "recall đúng hơn là câu văn quá hoa mỹ", confidence=0.76, turn=1)
    store.upsert_fact(user_id, "name", "DũngCT", confidence=0.94, turn=20, is_correction=True)

    active_early = store.get_recall_facts(user_id, current_turn=1, threshold=threshold, half_life=half_life)
    active_late = store.get_recall_facts(user_id, current_turn=30, threshold=threshold, half_life=half_life)

    assert "priority" in active_early
    assert "name" in active_late
    assert "priority" not in active_late


def test_agent_mode_resolution(tmp_path: Path) -> None:
    """Verify offline/live/auto mode selection."""

    config = make_config(tmp_path)
    assert resolve_force_offline("offline", config.model, None) is True
    assert resolve_force_offline("live", config.model, None) is False
    assert resolve_force_offline("auto", config.model, True) is True


@patch("agent_advanced.invoke_live_chat", side_effect=RuntimeError("simulated live failure"))
@patch("agent_advanced.build_chat_model_or_none", return_value=MagicMock())
def test_live_mode_falls_back_to_offline_without_valid_key(
    _mock_build: MagicMock,
    _mock_invoke: MagicMock,
    tmp_path: Path,
) -> None:
    """Live mode should gracefully fall back when the provider call fails."""

    from dataclasses import replace as dc_replace
    from model_provider import ProviderConfig

    config = make_config(tmp_path)
    config = dc_replace(
        config,
        agent_mode="live",
        model=ProviderConfig(
            provider="gemini",
            model_name="gemini-3.1-flash-lite",
            temperature=0.2,
            api_key="test-key",
        ),
    )
    agent = AdvancedAgent(config=config, force_offline=False)
    result = agent.reply("demo-user", "thread-1", "Chào bạn, mình tên là DũngCT.")
    assert result["mode"] == "offline"
    assert result["answer"]
    _mock_invoke.assert_called_once()
