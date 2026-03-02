- [2026-02-20 08:06] Handoff (2026-02-20):
Objective:
- Stabilize API/local model routing and deep-mode reliability; then add ingestion UX visibility.

Completed this session:
- Added robust local deep parser handling mixed JSON/prose outputs in `src/engine/local_agent_client.py` (tool-call extraction + safer insufficient_evidence fallback).
- Added separate deep local model setting (`LOCAL_DEEP_MODEL`) and wired usage/diagnostics (`src/config/settings.py`, `src/engine/local_agent_client.py`, `src/api/main.py`).
- Implemented per-request chat model routing via headers (`x-model-backend`, `x-api-key`, `x-api-model`) and UI controls (`frontend/app.py`, `frontend/client.py`, `frontend/state.py`).
- Added Gemini fast-generation client and wired fast/deep API-model usage with local fallback selection (`src/engine/local_llm.py`, `src/api/main.py`).
- Added AUTO runtime fallback: when API model fails, retry local backend for both fast/deep and stream endpoints (`src/api/main.py`).

Validation:
- `pytest tests/test_api.py` -> pass
- `pytest tests/test_frontend_client.py tests/test_frontend_state.py tests/test_local_agent_client.py tests/test_cloud_agent.py tests/test_settings.py tests/test_local_llm.py` -> pass

Runtime findings relevant to ingestion delay:
- `/healthz` reports queue workers running (2), backend RedisVL ready.
- Target document `6687fc186628b8216f0057c5af08ea20` now returns `indexed` from `/ingest/{id}`.
- Delay source is primarily embedding stage fanout (many sequential embedding calls); status text remains generic `processing` until completion.
- Redis counts observed for that doc: tier1=38, tier4=11 vectors.

User’s latest request:
- Add a visible embedding progress bar in UI during ingestion.

In-progress / next session plan:
1) Add ingestion progress metadata from worker/orchestrator (stage + counts, especially embedding processed/total).
2) Expose via `/ingest/{document_id}` response fields or companion endpoint.
3) Update Streamlit upload/status panel to render progress bar + stage label.
4) Add tests for progress payload + frontend rendering behavior.

Notes:
- Repo has many unrelated dirty files in `.aoc/*`, `.taskmaster/*`, etc.; avoid reverting unrelated changes.
- No commit created yet in this session.
