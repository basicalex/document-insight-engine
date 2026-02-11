# Document Insight Engine

Python service for document upload, extraction, and question answering.

## Assignment alignment

This repository is being built for the following core assignment goals:

- `POST /upload`: accept one or more PDF/image docs and store them.
- `POST /ask`: answer questions from uploaded docs.
- Dockerized deployment.
- Repository-committed dummy docs for testing.

Current state: foundation and infrastructure are implemented; `/upload` and `/ask` are next.

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

## API examples (target contracts)

Upload request (planned):

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

Ask request (planned):

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
