# Student Scaffold

This `src/` folder is the student version of the lab.

- It keeps the same high-level structure
- The Python files are intentionally incomplete and contain pseudocode / TODOs
- The benchmark structure should include: standard benchmark + long-context stress benchmark
- The runtime should support these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Suggested flow:

1. Start with `config.py`
2. Implement `memory_store.py`
3. Finish `agent_baseline.py`
4. Finish `agent_advanced.py`
5. Implement `benchmark.py`
6. Make `test_agents.py` pass

Datasets are available at the repo root in `data/`.

## Runtime modes

| Mode | API key | Command |
|---|---|---|
| Offline benchmark / tests | **Not required** | `AGENT_MODE=offline` (default) |
| Live Gemini chat | `GEMINI_API_KEY` in `.env` | `AGENT_MODE=live` + `python chat_live.py` |
| Auto | Uses live if key exists | `AGENT_MODE=auto` |

Bonus guardrails implemented: confidence threshold, conflict handling, memory decay. See `../ANALYSIS.md`.

## Visual demo UI

```powershell
..\.venv\Scripts\streamlit.exe run demo_ui.py
```
