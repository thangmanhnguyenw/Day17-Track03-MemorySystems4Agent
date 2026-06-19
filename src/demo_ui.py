"""
Day 17 — Memory Systems Visual Demo (Streamlit)

Run:
  cd src
  ..\\.venv\\Scripts\\streamlit.exe run demo_ui.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import (
    build_markdown_report,
    load_conversations,
    recall_points,
    run_agent_benchmark,
    save_report,
)
from config import load_config
from live_runtime import resolve_force_offline
from memory_store import extract_profile_candidates

ROOT = Path(__file__).resolve().parent.parent


def init_state() -> None:
    defaults = {
        "user_id": "demo_user",
        "thread_id": "thread-main",
        "recall_thread_id": "thread-recall",
        "chat_log": [],
        "last_snapshot": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_agents(force_offline: bool) -> tuple[BaselineAgent, AdvancedAgent]:
    config = load_config(ROOT)
    return (
        BaselineAgent(config=config, force_offline=force_offline),
        AdvancedAgent(config=config, force_offline=force_offline),
    )


def memory_snapshot(advanced: AdvancedAgent, user_id: str, thread_id: str) -> dict[str, Any]:
    ctx = advanced.compact_memory.context(thread_id)
    messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
    profile_path = advanced.profile_store.path_for(user_id)
    profile_text = advanced.profile_store.read_text(user_id) if profile_path.exists() else ""
    facts = advanced.profile_store.facts(user_id)
    return {
        "user_md": profile_text,
        "facts": facts,
        "summary": str(ctx.get("summary", "")),
        "recent_messages": messages,
        "compactions": advanced.compaction_count(thread_id),
        "prompt_tokens": advanced.prompt_token_usage(thread_id),
        "agent_tokens": advanced.token_usage(thread_id),
        "memory_bytes": advanced.memory_file_size(user_id),
    }


def render_memory_stack(snapshot: dict[str, Any] | None, agent_label: str = "Advanced") -> None:
    """Visual 3-layer memory stack with fill indicators."""
    if not snapshot:
        st.caption(f"{agent_label}: chưa có dữ liệu memory")
        return

    facts_n = len(snapshot.get("facts", {}))
    recent_n = len(snapshot.get("recent_messages", []))
    summary_len = len(snapshot.get("summary", "") or "")
    compactions = snapshot.get("compactions", 0)

    st.html(
        f"""
        <style>
          .stack {{ font-family: sans-serif; max-width: 420px; }}
          .stack-title {{ font-weight: 700; margin-bottom: 10px; }}
          .layer {{
            border-radius: 10px; padding: 12px 14px; margin: 8px 0;
            border: 2px solid #e5e7eb; position: relative; overflow: hidden;
          }}
          .layer-fill {{
            position: absolute; left: 0; top: 0; bottom: 0; opacity: 0.15; z-index: 0;
          }}
          .layer-body {{ position: relative; z-index: 1; }}
          .l-short {{ border-color: #22c55e; }}
          .l-short .layer-fill {{ background: #22c55e; width: {min(recent_n * 12, 100)}%; }}
          .l-user {{ border-color: #3b82f6; }}
          .l-user .layer-fill {{ background: #3b82f6; width: {min(facts_n * 20, 100)}%; }}
          .l-compact {{ border-color: #a855f7; }}
          .l-compact .layer-fill {{ background: #a855f7; width: {min(summary_len // 10, 100)}%; }}
          .layer small {{ color: #6b7280; }}
        </style>
        <div class="stack">
          <div class="stack-title">{agent_label} — Memory Stack</div>
          <div class="layer l-short">
            <div class="layer-fill"></div>
            <div class="layer-body">
              <b>① Short-term</b><br>
              <small>{recent_n} message gần nhất trong thread</small>
            </div>
          </div>
          <div class="layer l-user">
            <div class="layer-fill"></div>
            <div class="layer-body">
              <b>② User.md</b><br>
              <small>{facts_n} facts · {snapshot.get('memory_bytes', 0)} bytes</small>
            </div>
          </div>
          <div class="layer l-compact">
            <div class="layer-fill"></div>
            <div class="layer-body">
              <b>③ Compact</b><br>
              <small>{compactions} lần compact · summary {summary_len} ký tự</small>
            </div>
          </div>
        </div>
        """
    )


DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "label": "1. Giới thiệu tên",
        "message": "Chào bạn, mình tên là DũngCT, làm backend engineer.",
        "recall": False,
        "hint": "Advanced sẽ ghi fact `name` vào User.md.",
    },
    {
        "label": "2. Recall (thread mới)",
        "message": "Bạn nhớ tên mình không?",
        "recall": True,
        "hint": "Bật recall thread — Baseline quên, Advanced nhớ nhờ User.md.",
    },
    {
        "label": "3. Thread dài (compact)",
        "message": "Hôm nay mình học về memory systems, compact memory, và LangGraph checkpointing. "
        "Thread dài sẽ kích hoạt nén summary để giảm prompt tokens.",
        "recall": False,
        "hint": "Gửi nhiều lần để thấy compactions tăng.",
    },
]


def run_chat_turn(
    baseline: BaselineAgent,
    advanced: AdvancedAgent,
    user_id: str,
    thread_id: str,
    message: str,
) -> None:
    b_res = baseline.reply(user_id, thread_id, message)
    a_res = advanced.reply(user_id, thread_id, message)
    snap = memory_snapshot(advanced, user_id, thread_id)
    st.session_state.chat_log.append(
        {
            "thread": thread_id,
            "user": message,
            "baseline": b_res,
            "advanced": a_res,
            "snapshot": snap,
        }
    )
    st.session_state.last_snapshot = snap


def render_architecture_diagram() -> None:
    st.html(
        """
        <style>
          .mem-wrap { font-family: sans-serif; display: flex; gap: 16px; flex-wrap: wrap; }
          .mem-card { flex: 1; min-width: 280px; border-radius: 12px; padding: 16px; border: 2px solid #ddd; }
          .mem-baseline { background: #fff5f5; border-color: #f87171; }
          .mem-advanced { background: #f0fdf4; border-color: #4ade80; }
          .mem-layer { margin: 8px 0; padding: 10px; border-radius: 8px; background: white; border: 1px solid #e5e7eb; }
          .mem-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 8px; }
          .arrow { text-align: center; font-size: 24px; color: #6b7280; margin: 8px 0; }
        </style>
        <div class="mem-wrap">
          <div class="mem-card mem-baseline">
            <div class="mem-title">Baseline Agent</div>
            <div class="mem-layer">Short-term memory<br><small>Chỉ trong cùng thread_id</small></div>
            <div class="arrow">↓</div>
            <div class="mem-layer" style="border-color:#f87171">Đổi thread → <b>QUÊN</b></div>
            <div class="mem-layer" style="opacity:.6">❌ Không có User.md</div>
            <div class="mem-layer" style="opacity:.6">❌ Không compact</div>
          </div>
          <div class="mem-card mem-advanced">
            <div class="mem-title">Advanced Agent</div>
            <div class="mem-layer" style="border-left:4px solid #22c55e">① Short-term — message gần nhất</div>
            <div class="mem-layer" style="border-left:4px solid #3b82f6">② Persistent — User.md (cross-session)</div>
            <div class="mem-layer" style="border-left:4px solid #a855f7">③ Compact — summary khi thread dài</div>
            <div class="arrow">↓</div>
            <div class="mem-layer" style="border-color:#4ade80">Reply + recall tốt hơn</div>
          </div>
        </div>
        """
    )


def page_overview(force_offline: bool) -> None:
    st.subheader("Kiến trúc Memory System")
    render_architecture_diagram()

    c1, c2, c3 = st.columns(3)
    c1.metric("Runtime", "offline" if force_offline else "live")
    c2.metric("Dataset", "2 file JSON")
    c3.metric("Agents", "Baseline + Advanced")

    st.markdown(
        """
**Luồng reviewer mong đợi:**

1. Baseline chỉ giữ message trong `thread_id` hiện tại → đổi thread là quên.
2. Advanced ghi fact vào `state/profiles/{user_id}.md` → recall cross-session.
3. Thread dài → compact nén message cũ thành summary → giảm prompt cost.
        """
    )

    with st.expander("Bảng so sánh chi tiết"):
        st.markdown(
            """
| Lớp memory | Baseline | Advanced |
|---|---|---|
| Short-term (cùng thread) | Có | Có |
| `User.md` bền vững | Không | Có |
| Compact memory | Không | Có |
| Cross-session recall | Không | Có |
            """
        )


def page_compare_chat(force_offline: bool) -> None:
    st.subheader("So sánh trực tiếp Baseline vs Advanced")

    baseline, advanced = get_agents(force_offline)
    user_id = st.session_state.user_id
    thread_id = st.session_state.thread_id
    recall_thread = st.session_state.recall_thread_id

    st.markdown("##### Kịch bản demo nhanh")
    sc_cols = st.columns(len(DEMO_SCENARIOS))
    for col, scenario in zip(sc_cols, DEMO_SCENARIOS):
        with col:
            st.caption(scenario["hint"])
            if st.button(scenario["label"], width="stretch", key=f"scenario_{scenario['label']}"):
                active = recall_thread if scenario["recall"] else thread_id
                run_chat_turn(baseline, advanced, user_id, active, scenario["message"])
                st.rerun()

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        message = st.text_input("Tin nhắn", placeholder="VD: Chào bạn, mình tên là DũngCT.")
    with col_btn:
        send = st.button("Gửi", type="primary", width="stretch")
        new_thread = st.button("Thread mới", width="stretch")

    if new_thread:
        st.session_state.thread_id = f"thread-{len(st.session_state.chat_log) + 1}"
        st.session_state.recall_thread_id = f"{st.session_state.thread_id}-recall"
        st.rerun()

    use_recall_thread = st.checkbox(
        "Gửi vào thread recall mới (test cross-session)",
        help="Advanced vẫn nhớ nhờ User.md; Baseline sẽ quên.",
    )
    active_thread = recall_thread if use_recall_thread else thread_id

    if send and message.strip():
        run_chat_turn(baseline, advanced, user_id, active_thread, message)
        st.rerun()

    flow_col, stack_col = st.columns([3, 2])
    with flow_col:
        st.markdown("##### Luồng xử lý (lượt gần nhất)")
        if st.session_state.chat_log:
            last = st.session_state.chat_log[-1]
            st.html(
                f"""
                <div style="font-family:sans-serif;line-height:2">
                  <span style="background:#dbeafe;padding:6px 12px;border-radius:8px">👤 User</span>
                  → <span style="background:#fee2e2;padding:6px 12px;border-radius:8px">Baseline (thread only)</span>
                  → <span style="background:#dcfce7;padding:6px 12px;border-radius:8px">Advanced (3 layers)</span>
                  → <span style="background:#f3e8ff;padding:6px 12px;border-radius:8px">💬 Reply</span>
                  <br><small style="color:#6b7280">Thread: <code>{last['thread']}</code></small>
                </div>
                """
            )
        else:
            st.info("Gửi tin nhắn hoặc chọn kịch bản demo để xem luồng.")

    with stack_col:
        render_memory_stack(st.session_state.last_snapshot)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Baseline")
        st.caption("Chỉ nhớ trong thread hiện tại — không có User.md")
        if st.session_state.chat_log:
            last = st.session_state.chat_log[-1]
            st.info(f"**Trả lời:** {last['baseline']['answer']}")
            st.write(f"Mode: `{last['baseline'].get('mode', 'offline')}`")
            st.write(f"Prompt tokens (thread): {baseline.prompt_token_usage(active_thread)}")
        else:
            st.write("_Chưa có tin nhắn._")

    with right:
        st.markdown("#### Advanced")
        st.caption("User.md + compact memory")
        if st.session_state.chat_log:
            last = st.session_state.chat_log[-1]
            st.success(f"**Trả lời:** {last['advanced']['answer']}")
            st.write(f"Mode: `{last['advanced'].get('mode', 'offline')}`")
            st.write(f"Compactions: {last['advanced'].get('compactions', 0)}")
            st.write(f"Prompt tokens: {advanced.prompt_token_usage(active_thread)}")
        else:
            st.write("_Chưa có tin nhắn._")

    if st.session_state.last_snapshot:
        st.markdown("---")
        st.markdown("#### Memory Explorer (Advanced)")
        snap = st.session_state.last_snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compactions", snap["compactions"])
        m2.metric("Prompt tokens", snap["prompt_tokens"])
        m3.metric("User.md bytes", snap["memory_bytes"])
        m4.metric("Facts", len(snap["facts"]))

        t1, t2, t3 = st.tabs(["User.md", "Facts (structured)", "Compact layer"])
        with t1:
            st.code(snap["user_md"] or "(trống)", language="markdown")
        with t2:
            if snap["facts"]:
                st.json(snap["facts"])
            else:
                st.write("Chưa có fact nào được ghi.")
        with t3:
            st.write("**Summary (message cũ đã nén):**")
            st.text(snap["summary"] or "(chưa compact)")
            st.write("**Recent messages (giữ nguyên):**")
            for msg in snap["recent_messages"][-8:]:
                st.markdown(f"- **{msg['role']}:** {msg['content'][:200]}")

    if st.session_state.chat_log:
        st.markdown("---")
        st.markdown("#### Lịch sử hội thoại")
        for i, item in enumerate(reversed(st.session_state.chat_log[-10:]), start=1):
            with st.expander(f"#{len(st.session_state.chat_log) - i + 1} | thread `{item['thread']}`"):
                st.markdown(f"**User:** {item['user']}")
                st.markdown(f"**Baseline:** {item['baseline']['answer']}")
                st.markdown(f"**Advanced:** {item['advanced']['answer']}")


def page_playback(force_offline: bool) -> None:
    st.subheader("Playback dataset — xem agent xử lý từng lượt")

    dataset = st.selectbox(
        "Chọn file data",
        [
            ("Standard", ROOT / "data" / "conversations.json"),
            ("Stress", ROOT / "data" / "advanced_long_context.json"),
        ],
        format_func=lambda x: x[0],
    )
    conversations = load_conversations(dataset[1])
    conv = st.selectbox(
        "Chọn conversation",
        conversations,
        format_func=lambda c: f"{c['id']} ({len(c.get('turns', []))} turns)",
    )

    if st.button("Chạy playback conversation này"):
        baseline, advanced = get_agents(force_offline)
        user_id = conv["user_id"]
        thread_id = f"{conv['id']}-playback"
        steps: list[dict[str, Any]] = []

        for turn in conv.get("turns", []):
            candidates = extract_profile_candidates(turn)
            b = baseline.reply(user_id, thread_id, turn)
            a = advanced.reply(user_id, thread_id, turn)
            steps.append(
                {
                    "turn": turn,
                    "candidates": [
                        {"key": f.key, "value": f.value, "confidence": f.confidence}
                        for f in candidates
                    ],
                    "baseline_answer": b["answer"],
                    "advanced_answer": a["answer"],
                    "compactions": advanced.compaction_count(thread_id),
                    "facts": dict(advanced.profile_store.facts(user_id)),
                }
            )

        recall_results = []
        for i, q in enumerate(conv.get("recall_questions", []), start=1):
            rt = f"{conv['id']}-recall-{i}"
            b = baseline.reply(user_id, rt, q["question"])
            a = advanced.reply(user_id, rt, q["question"])
            recall_results.append(
                {
                    "question": q["question"],
                    "expected": q.get("expected_contains", []),
                    "baseline": b["answer"],
                    "advanced": a["answer"],
                    "baseline_score": recall_points(b["answer"], q.get("expected_contains", [])),
                    "advanced_score": recall_points(a["answer"], q.get("expected_contains", [])),
                }
            )

        st.session_state.playback_steps = steps
        st.session_state.playback_recall = recall_results
        st.session_state.playback_snap = memory_snapshot(advanced, user_id, thread_id)

    if "playback_steps" in st.session_state and st.session_state.playback_steps:
        total = len(st.session_state.playback_steps)
        step_idx = st.slider("Lượt", 0, total - 1, 0)
        st.progress((step_idx + 1) / total, text=f"Lượt {step_idx + 1}/{total}")
        step = st.session_state.playback_steps[step_idx]

        st.markdown(f"**User turn:** {step['turn']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Baseline**")
            st.write(step["baseline_answer"])
        with c2:
            st.markdown("**Advanced**")
            st.write(step["advanced_answer"])

        st.markdown("**Facts trích xuất (confidence threshold):**")
        if step["candidates"]:
            st.dataframe(step["candidates"], width="stretch")
        else:
            st.write("Không có fact mới ở lượt này.")

        st.markdown("**User.md sau lượt này:**")
        st.json(step["facts"])
        st.metric("Compactions tích lũy", step["compactions"])

    if "playback_recall" in st.session_state:
        st.markdown("---")
        st.markdown("#### Recall questions (thread mới)")
        for row in st.session_state.playback_recall:
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"- Baseline ({row['baseline_score']}): {row['baseline']}")
            st.markdown(f"- Advanced ({row['advanced_score']}): {row['advanced']}")
            st.markdown(f"- Expected: `{row['expected']}`")


def page_benchmark(force_offline: bool) -> None:
    st.subheader("Chạy benchmark & xem report")

    if st.button("Chạy full benchmark (2 file data)", type="primary"):
        with st.spinner("Đang chạy benchmark..."):
            config = load_config(ROOT)
            baseline = BaselineAgent(config=config, force_offline=force_offline)
            advanced = AdvancedAgent(config=config, force_offline=force_offline)
            standard = load_conversations(config.data_dir / "conversations.json")
            stress = load_conversations(config.data_dir / "advanced_long_context.json")

            standard_rows = [
                run_agent_benchmark("Baseline", baseline, standard, config),
                run_agent_benchmark("Advanced", advanced, standard, config),
            ]
            sb = BaselineAgent(config=config, force_offline=force_offline)
            sa = AdvancedAgent(config=config, force_offline=force_offline)
            stress_rows = [
                run_agent_benchmark("Baseline", sb, stress, config),
                run_agent_benchmark("Advanced", sa, stress, config),
            ]
            report = build_markdown_report(
                standard_rows,
                stress_rows,
                force_offline=force_offline,
                agent_mode=config.agent_mode,
                model_name=config.model.model_name,
            )
            path, stamped = save_report(report, config)
            st.session_state.benchmark_report = report
            st.session_state.benchmark_paths = (str(path), str(stamped))
            st.session_state.benchmark_rows = {
                "standard": standard_rows,
                "stress": stress_rows,
            }

    if "benchmark_rows" in st.session_state:
        rows = st.session_state.benchmark_rows
        st.markdown("#### Biểu đồ so sánh")
        for label, data in [("Standard", rows["standard"]), ("Stress", rows["stress"])]:
            st.caption(label)
            chart_data = {
                r.agent_name: {
                    "recall": r.recall_score,
                    "prompt_tokens": r.prompt_tokens_processed,
                }
                for r in data
            }
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(
                    {k: v["recall"] for k, v in chart_data.items()},
                    height=200,
                )
                st.caption("Recall score (cao hơn = tốt)")
            with c2:
                st.bar_chart(
                    {k: v["prompt_tokens"] for k, v in chart_data.items()},
                    height=200,
                )
                st.caption("Avg prompt tokens (thấp hơn = rẻ hơn)")

    report_path = ROOT / "output" / "benchmark_report.md"
    if "benchmark_report" in st.session_state:
        st.success(f"Đã lưu: {st.session_state.benchmark_paths[0]}")
        st.markdown(st.session_state.benchmark_report)
    elif report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("Chưa có report. Bấm nút trên để chạy benchmark.")


def main() -> None:
    st.set_page_config(
        page_title="Day 17 — Memory Systems Demo",
        page_icon="🧠",
        layout="wide",
    )
    init_state()

    st.title("Day 17 — Memory Systems for AI Agent")
    st.caption("Dashboard trực quan: Baseline vs Advanced, User.md, Compact Memory")

    with st.sidebar:
        st.header("Cấu hình")
        mode = st.radio("Runtime mode", ["offline", "live", "auto"], index=0)
        force_offline = mode == "offline"
        if mode == "live":
            force_offline = False
        if mode == "auto":
            config = load_config(ROOT)
            force_offline = resolve_force_offline("auto", config.model, None)

        st.session_state.user_id = st.text_input("User ID", st.session_state.user_id)
        st.session_state.thread_id = st.text_input("Thread ID", st.session_state.thread_id)

        if st.button("Reset agents & memory demo"):
            demo_profiles = load_config(ROOT).state_dir / "profiles"
            if demo_profiles.exists():
                for p in demo_profiles.glob("demo_user.md"):
                    p.unlink()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
**Offline:** không cần API  
**Live:** cần `GEMINI_API_KEY`  
**Auto:** có key → live
            """
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tổng quan", "So sánh Chat", "Playback Data", "Benchmark"]
    )
    with tab1:
        page_overview(force_offline)
    with tab2:
        page_compare_chat(force_offline)
    with tab3:
        page_playback(force_offline)
    with tab4:
        page_benchmark(force_offline)


if __name__ == "__main__":
    main()
