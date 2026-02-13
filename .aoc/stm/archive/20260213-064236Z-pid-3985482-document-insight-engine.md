- [2026-02-13 07:42] Objective: Stabilize hybrid local-dev workflow (Docker infra + local API/UI), enable streaming UX, and reduce LLM latency for POC.
Task IDs: none explicitly tracked in Taskmaster during this thread.
- [2026-02-13 07:42] Done:
- Diagnosed earlier UI network error as timeout-related behavior; verified container connectivity and API/Ollama paths.
- Implemented streaming pipeline end-to-end (`POST /ask/stream`), including API NDJSON events, local LLM token streaming, frontend client stream consumption, and UI incremental rendering.
- Added fallback in UI to non-stream `/ask` when `/ask/stream` is unavailable.
- Added tests for stream behavior in API/local engine and ran targeted pytest successfully.
- [2026-02-13 07:42] Done (dev workflow):
- Added `docker-compose.dev.yml` overlay for fast iteration (bind mounts + reload behavior) and documented usage in README.
- Created and iterated custom AOC Zellij hybrid layout (`aoc.hybrid`) with panes: API Dev, UI Dev, Infra Logs, Dev Shell (removed Explorer/Agent per user request).
- Hardened hybrid layout startup: auto `.venv` bootstrap, dependency install, python3.12 preference, Streamlit first-run telemetry prompt mitigation, timeout env exports.
- [2026-02-13 07:42] Done (performance tuning):
- Switched default local model to `llama3.2:1b` in settings/compose/dev overlay/hybrid env defaults.
- Pulled `llama3.2:1b` model in Ollama and removed large model `llama3.2:latest` as requested.
- Verified API trace model is `llama3.2:1b`; first token latency improved (approx 126s -> 45s in direct stream probe).
- [2026-02-13 07:42] In progress / current state:
- System is operational in hybrid mode; streaming emits token events after first-token delay.
- Remaining latency is mostly prefill/CPU bound on prompt+context size; not a transport/streaming plumbing failure.
- [2026-02-13 07:42] Blockers / risks:
- CPU-only inference still has high first-token latency on larger prompts/doc contexts.
- If timeouts are lowered or env vars drift, requests may regress to 504/abort.
- Local `.venv` is untracked; environment reproducibility depends on setup commands/layout bootstrap.
- [2026-02-13 07:42] Files touched (high impact):
- Repo: `src/api/main.py`, `src/engine/local_llm.py`, `frontend/app.py`, `frontend/client.py`, `src/config/settings.py`, `docker-compose.yml`, `docker-compose.dev.yml`, `README.md`, `tests/test_api.py`, `tests/test_local_llm.py`, `tests/test_compose.py`.
- User config: `~/.config/zellij/layouts/aoc.hybrid.kdl`.
- [2026-02-13 07:42] Last command outcomes:
- `pytest tests/test_local_llm.py tests/test_api.py` passed earlier after streaming implementation.
- `pytest tests/test_settings.py tests/test_compose.py` passed after model/default updates.
- Direct stream probes confirmed token events and final event; first token currently around tens of seconds on 1b model for doc-grounded prompts.
- [2026-02-13 07:42] Open decisions / assumptions:
- Assumed POC priority favors speed over quality; 1b model retained by default.
- Assumed acceptable to keep timeout env high (300s) for local dev resilience.
- Potential next tuning (not yet applied): reduce retrieval top-k/context size to further cut first-token latency.
- [2026-02-13 07:42] Next 3-5 steps:
1) Keep hybrid tab as default and restart panes from fresh tab after any layout/env changes.
2) If latency remains too high, reduce retrieval context (e.g., top-k from 5 to 2-3) and re-measure first token.
3) Add lightweight UI progress event before first token ("retrieving"/"generating") for perceived responsiveness.
4) Run one end-to-end user flow (upload -> ingest indexed -> ask stream) and capture baseline timings for POC demo notes.
