from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from live_runtime import resolve_force_offline


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0

    normalized_answer = answer.lower()
    hits = sum(1 for item in expected if item.lower() in normalized_answer)
    if hits == 0:
        return 0.0
    if hits < len(expected):
        return 0.5
    return 1.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    if not answer.strip():
        return 0.0

    recall = recall_points(answer, expected)
    length_bonus = 0.2 if 20 <= len(answer) <= 400 else 0.0
    structure_bonus = 0.1 if any(marker in answer for marker in (";", ":", "-")) else 0.0
    return min(1.0, recall * 0.7 + length_bonus + structure_bonus)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_growth = 0
    total_compactions = 0

    for conversation in conversations:
        user_id = conversation["user_id"]
        conv_id = conversation["id"]
        thread_id = f"{conv_id}-main"

        initial_memory = 0
        if hasattr(agent, "memory_file_size"):
            initial_memory = agent.memory_file_size(user_id)

        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, turn)
            total_agent_tokens = max(total_agent_tokens, agent.token_usage(thread_id))
            total_prompt_tokens = max(total_prompt_tokens, agent.prompt_token_usage(thread_id))
            total_compactions = max(total_compactions, agent.compaction_count(thread_id))

        if hasattr(agent, "memory_file_size"):
            memory_growth = max(memory_growth, agent.memory_file_size(user_id) - initial_memory)

        for index, question in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{conv_id}-recall-{index}"
            result = agent.reply(user_id, recall_thread, question["question"])
            answer = result["answer"]
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            total_agent_tokens = max(total_agent_tokens, agent.token_usage(recall_thread))
            total_prompt_tokens = max(total_prompt_tokens, agent.prompt_token_usage(recall_thread))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(title: str, rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    from tabulate import tabulate

    table_rows = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    return f"{title}\n{tabulate(table_rows, headers=headers, tablefmt='github')}"


def build_analysis_section(
    standard_rows: list[BenchmarkRow],
    stress_rows: list[BenchmarkRow],
    *,
    force_offline: bool,
    agent_mode: str,
    model_name: str,
) -> str:
    std_b, std_a = standard_rows
    stress_b, stress_a = stress_rows
    runtime = "offline" if force_offline else "live"
    prompt_delta_std = std_a.prompt_tokens_processed - std_b.prompt_tokens_processed
    prompt_delta_stress = stress_b.prompt_tokens_processed - stress_a.prompt_tokens_processed

    lines = [
        "## Runtime",
        "",
        f"- Mode: **{runtime}** (`AGENT_MODE={agent_mode}`)",
        f"- Model config: `{model_name}`",
        "",
        "## Tóm tắt nhanh",
        "",
        f"- Standard recall: Baseline **{std_b.recall_score:.2f}** vs Advanced **{std_a.recall_score:.2f}**",
        f"- Standard prompt tokens: Baseline {std_b.prompt_tokens_processed:,} vs Advanced {std_a.prompt_tokens_processed:,} "
        f"({'+' if prompt_delta_std >= 0 else ''}{prompt_delta_std:,})",
        f"- Stress prompt tokens: Baseline {stress_b.prompt_tokens_processed:,} vs Advanced {stress_a.prompt_tokens_processed:,} "
        f"(Advanced tiết kiệm {prompt_delta_stress:,} tokens)",
        f"- Stress compactions (Advanced): **{stress_a.compactions}**",
        "",
        "## Nhận xét",
        "",
        "- Advanced recall cao hơn nhờ `User.md` bền vững qua thread mới.",
        "- Advanced có thể tốn prompt hơn ở hội thoại ngắn vì luôn nạp profile.",
        "- Compact memory chủ yếu giúp ở thread dài (giảm `Prompt tokens processed`).",
        "- Bonus guardrails: confidence threshold, conflict handling, memory decay.",
        "",
        "Chi tiết thêm: xem `ANALYSIS.md` ở root repo.",
    ]
    return "\n".join(lines)


def build_markdown_report(
    standard_rows: list[BenchmarkRow],
    stress_rows: list[BenchmarkRow],
    *,
    force_offline: bool,
    agent_mode: str,
    model_name: str,
) -> str:
    """Build a readable markdown report for benchmark results."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        "# Day 17 — Benchmark Report",
        "",
        f"_Generated at {now}_",
        "",
        build_analysis_section(
            standard_rows,
            stress_rows,
            force_offline=force_offline,
            agent_mode=agent_mode,
            model_name=model_name,
        ),
        "",
        "## Standard Benchmark",
        "",
        format_rows("Standard Benchmark", standard_rows).split("\n", 1)[-1],
        "",
        "## Long-Context Stress Benchmark",
        "",
        format_rows("Long-Context Stress Benchmark", stress_rows).split("\n", 1)[-1],
    ]
    return "\n".join(sections)


def save_report(report: str, config: LabConfig) -> tuple[Path, Path]:
    """Write benchmark report to `output/benchmark_report.md` (or `BENCHMARK_OUTPUT`)."""

    output_path = os.getenv("BENCHMARK_OUTPUT", "").strip()
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = config.base_dir / path
    else:
        path = config.base_dir / "output" / "benchmark_report.md"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    stamped = path.parent / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    stamped.write_text(report, encoding="utf-8")
    return path, stamped


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)
    force_offline = resolve_force_offline(config.agent_mode, config.model, None)

    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        for path in profiles_dir.glob("*.md"):
            path.unlink()

    # TODO:
    # - load both datasets from root/data
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"
    standard_conversations = load_conversations(standard_path)
    stress_conversations = load_conversations(stress_path)

    # - initialize baseline and advanced agents
    baseline = BaselineAgent(config=config, force_offline=force_offline)
    advanced = AdvancedAgent(config=config, force_offline=force_offline)

    # - run benchmarks
    standard_rows = [
        run_agent_benchmark("Baseline", baseline, standard_conversations, config),
        run_agent_benchmark("Advanced", advanced, standard_conversations, config),
    ]
    stress_baseline = BaselineAgent(config=config, force_offline=force_offline)
    stress_advanced = AdvancedAgent(config=config, force_offline=force_offline)
    stress_rows = [
        run_agent_benchmark("Baseline", stress_baseline, stress_conversations, config),
        run_agent_benchmark("Advanced", stress_advanced, stress_conversations, config),
    ]

    # - print comparison tables
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    report = build_markdown_report(
        standard_rows,
        stress_rows,
        force_offline=force_offline,
        agent_mode=config.agent_mode,
        model_name=config.model.model_name,
    )
    output_path, stamped_path = save_report(report, config)

    print(format_rows("Standard Benchmark", standard_rows))
    print()
    print(format_rows("Long-Context Stress Benchmark", stress_rows))
    print()
    print("=" * 72)
    print(f"Runtime mode: {'offline' if force_offline else 'live'} (AGENT_MODE={config.agent_mode})")
    print("Analysis (see ANALYSIS.md for full write-up)")
    print("=" * 72)
    std_b, std_a = standard_rows
    stress_b, stress_a = stress_rows
    print(f"- Standard recall: Baseline {std_b.recall_score:.2f} vs Advanced {std_a.recall_score:.2f}")
    print(f"- Standard prompt tokens: Baseline {std_b.prompt_tokens_processed} vs Advanced {std_a.prompt_tokens_processed}")
    print(f"- Stress prompt tokens: Baseline {stress_b.prompt_tokens_processed} vs Advanced {stress_a.prompt_tokens_processed}")
    print(f"- Stress compactions (Advanced): {stress_a.compactions}")
    print("- Compact mainly reduces Prompt tokens processed on long threads, not short chats.")
    print("- Advanced pays extra prompt cost on short chats because User.md is always loaded.")
    print("- Bonus guardrails: confidence threshold, conflict handling, memory decay.")
    print("- Offline mode: NO API key required. See ANALYSIS.md section 5.")
    print()
    print(f"Report saved to: {output_path}")
    print(f"Timestamped copy: {stamped_path}")


if __name__ == "__main__":
    main()
