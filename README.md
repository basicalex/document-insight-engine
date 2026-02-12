# Document Insight Engine

Python service for document upload, extraction, and question answering.

## Assignment alignment

This repository is being built for the following core assignment goals:

- `POST /upload`: accept one or more PDF/image docs and store them.
- `POST /ask`: answer questions from uploaded docs.
- Dockerized deployment.
- Repository-committed dummy docs for testing.

Current state:

- Foundation + infrastructure are implemented.
- Ingestion pipeline is implemented (`/ingest` currently available).
- `/ingest` now executes extract -> parse -> chunk -> embed -> index before returning status.
- Engine internals for local QA, agent loop guard, and Tier 4 extraction are implemented.
- API contracts for `/ingest` and `/ask` are implemented with validation and guardrails.
- `POST /upload` alias is available for assignment-compatible upload calls.
- Streamlit chat UI is available in `frontend/app.py` with mode toggle and trace viewer.

## Included dummy test docs (committed)

See `tests/data/documents/`:

- `dummy_invoice.pdf`
- `dummy_contract.pdf`
- `dummy_policy.pdf`
- `dummy_scanned_snippet.png`

These are synthetic fixtures and contain no sensitive data.

## Setup (manual)

```bash
python -m pip install -e .[dev]
python -m pytest
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Run Streamlit UI (in another terminal):

```bash
python -m pip install -e .[ui]
streamlit run frontend/app.py
```

Health check:

```bash
curl http://localhost:8000/healthz
```

## Setup (Docker)

```bash
docker compose up --build
```

Services started by compose:

- `api` (FastAPI)
- `redis` (persistent volume)
- `ollama` (model cache mounted under `./models`)

## API status and contracts

Current API endpoints:

- `GET /healthz`
- `POST /ingest`
- `GET /ingest/{document_id}`
- `POST /upload`
- `POST /ask`

Planned/next API endpoints:

- Real deep-mode provider integration beyond fallback scripted behavior

Deep mode capability:

- Deep mode is disabled by default at runtime.
- Enable with env vars: `DEEP_MODE_ENABLED=true` and `CLOUD_AGENT_PROVIDER=fallback`.

Contract examples:

Upload request:

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@tests/data/documents/dummy_invoice.pdf"
```

Expected upload response shape:

```json
{
  "document_id": "doc_123",
  "file_path": "data/uploads/dummy_invoice.pdf",
  "status": "uploaded",
  "message": "queued for processing"
}
```

Ask request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total due?",
    "mode": "fast",
    "document_id": "doc_123"
  }'
```

Expected ask response shape:

```json
{
  "answer": "Total due is 1234.00 USD.",
  "mode": "fast",
  "document_id": "doc_123",
  "insufficient_evidence": false,
  "citations": [
    {
      "chunk_id": "chunk-42",
      "page": 1,
      "text": "Total Due: 1234.00 USD",
      "start_offset": 120,
      "end_offset": 141
    }
  ]
}
```

## Approach

- **Framework**: FastAPI (typed contracts, async-ready API development).
- **Extraction**: PyMuPDF first, OCR fallback path for scanned/image-like docs.
- **QA/RAG**: Local QA baseline + optional deeper retrieval/agentic path.
- **Infra**: Docker Compose with Redis + Ollama + API, with persistence and limits.

## Docker manual verification checklist

Use this checklist to validate end-to-end behavior manually.

1) Start services and wait for health checks:

```bash
docker compose up --build -d
docker compose ps
```

2) Validate API and Redis reachability:

```bash
curl http://localhost:8000/healthz
docker compose exec redis redis-cli ping
```

3) Upload one document with assignment endpoint:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@tests/data/documents/dummy_invoice.pdf;type=application/pdf"
```

4) Upload multiple documents in one request:

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@tests/data/documents/dummy_invoice.pdf;type=application/pdf" \
  -F "files=@tests/data/documents/dummy_contract.pdf;type=application/pdf"
```

5) Query ingestion status for a document:

```bash
curl http://localhost:8000/ingest/<document_id>
```

6) Ask grounded question in fast mode:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total due?",
    "mode": "fast",
    "document_id": "<document_id>"
  }'
```

7) Restart API and re-check persistence:

```bash
docker compose restart api
curl http://localhost:8000/ingest/<document_id>
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total due?",
    "mode": "fast",
    "document_id": "<document_id>"
  }'
```

## Engine components (implemented)

- `src/engine/local_llm.py`: grounded local QA flow with retrieval-first prompting, insufficient-evidence fallback, and trace metadata.
- `src/engine/cloud_agent.py`: strict tool-allowlisted agent orchestration (`list_sections`, `read_section`, `keyword_grep`) with hard 5-iteration guard.
- `src/engine/extractor.py`: Tier 4 structured extraction adapter with schema validation, per-field provenance offset checks, and token-budget preflight guards.
- `src/tools/fs_tools.py`: deterministic markdown filesystem tools used by agent reasoning.
