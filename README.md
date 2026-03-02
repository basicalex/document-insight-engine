# Document Insight Engine

Grounded document QA + extraction service (FastAPI + Redis + Ollama + Streamlit).

---

## What you get

- **Upload + ingest** PDFs/images into a searchable index.
- **Grounded Q&A** with citations (`/ask`, `/ask/stream`).
- **Agentic deep modes** with filesystem tools over parsed markdown.
- **Structured extraction** via `/extract` (schema-driven, provenance-aware).
- **Operational visibility** via `/healthz`, `/readyz`, `/metrics`.

This repo is optimized for practical local development and Docker deployment.

---

## Architecture (high level)

1. **Ingestion**: upload -> parse -> chunk -> embed -> index.
2. **Fast mode**: local retrieval + grounded generation.
3. **Deep modes**: tool-using agent loop (`list_sections`, `read_section`, `keyword_grep`, `structured_extract`).
4. **State**: Redis-backed API state when available (sessions, idempotency, queue metadata).

Core modules:

- `src/api/main.py` — API routes + orchestration
- `src/ingestion/*` — pipeline stages
- `src/engine/local_llm.py` — fast grounded QA
- `src/engine/cloud_agent.py` — deep/deep-lite agent loop
- `src/engine/extractor.py` — structured extraction
- `src/tools/fs_tools.py` — deterministic markdown tools
- `frontend/app.py` — Streamlit chat UI

---

## Quickstart

### Prereqs

- Python **3.11+**
- Docker + Docker Compose
- (Recommended) `jq` for shell examples

### Recommended dev flow (local API/UI + Docker infra)

```bash
./scripts/dev-lite-up.sh
```

This starts:

- Redis + Ollama in Docker
- API on `http://localhost:8000`
- UI on `http://localhost:8501`

Use full profile when you want Docling-enabled parser stack:

```bash
./scripts/dev-full-up.sh
```

### Full Docker flow

```bash
./scripts/docker-lite-up.sh
# or
./scripts/docker-full-up.sh
```

Profiles:

- `.env.profile.lite` (default): LangExtract on, no Docling
- `.env.profile.full`: Docling + LangExtract
- `.env`: secrets only (e.g. `CLOUD_AGENT_API_KEY`)

---

## 2-minute smoke test

### 1) Upload

```bash
curl -sS -X POST http://localhost:8000/upload \
  -F "files=@tests/data/documents/dummy_invoice.pdf"
```

### 2) Ask (fast, local)

```bash
curl -sS -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "x-model-backend: local" \
  -d '{
    "question": "What is the total due?",
    "mode": "fast"
  }'
```

### 3) Readiness

```bash
curl -sS http://localhost:8000/healthz | jq
curl -sS http://localhost:8000/readyz
```

---

## Modes and routing

### Answer modes

- `fast` — retrieval-first local QA
- `deep-lite` — agentic loop with tighter cap
- `deep` — agentic loop (full configured cap)

### Backend routing (`x-model-backend`)

- `auto` (recommended): API model when key is present, local fallback when needed
- `api`: force Gemini path
- `local`: force Ollama/local path

### API model policy

API model is pinned to:

- `gemini-2.5-flash`

`x-api-model` overrides are intentionally ignored.

---

## API surface

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `POST /ingest`
- `GET /ingests`
- `GET /ingest/{document_id}`
- `POST /upload`
- `POST /ask`
- `POST /ask/stream`
- `POST /extract`

Dummy test fixtures are in `tests/data/documents/`.

---

## Structured extraction

`/extract` accepts:

- `document_id`
- JSON schema
- extraction prompt

Deep/deep-lite agents can also call structured extraction as a tool.

---

## Configuration notes

Common env knobs:

- `DEEP_MODE_ENABLED=true`
- `CLOUD_AGENT_API_KEY=...` (or `GOOGLE_API_KEY`)
- `CLOUD_AGENT_TIMEOUT_SECONDS` (profile defaults tuned)
- `API_STATE_BACKEND=auto|redis|memory`
- `API_STATE_SESSION_MAX_TURNS` (history window)

Session history semantics:

- persisted by `session_id`
- scoped by `session_id::document_id`

If document scope changes, history scope changes too.

---

## Local model prerequisites (Ollama)

Typical models used in this project:

```bash
ollama pull hadad/LFM2.5-1.2B:Q8_0
ollama pull nomic-embed-text:v1.5
ollama pull qwen2.5:7b-instruct
```

---

## Developer workflow

```bash
ruff check
pytest -q
```

For full-stack validation, use the provided scripts instead of ad-hoc compose commands.

---

## Troubleshooting (practical)

- **Seeing 429 from Gemini**: this is often short-window quota throttling, not necessarily daily budget exhaustion.
  - Use `x-model-backend: auto` or `local`.
- **Insufficient evidence**:
  - select a specific document (avoid all-doc scope for precise Q&A)
  - ask document-grounded questions
- **No conversation continuity**:
  - ensure stable `session_id` in API clients
  - keep document scope stable
- **Deep mode provider unavailable**:
  - verify `/healthz` deep provider readiness and API key config

---

## License / data

Dummy docs in `tests/data/documents/` are synthetic and contain no sensitive data.
