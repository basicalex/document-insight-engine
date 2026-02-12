# Agent Ops Cockpit (AOC) Global Context

## Core Philosophy
This machine uses the **Agent Ops Cockpit (AOC)** system. All agents (Gemini, Claude, OpenCode) running here share a unified set of tools for **Memory** and **Task Management**.

## 1. Project Structure
```
/home/ceii/dev/document-insight-engine
├── AGENTS.md
├── data
├── docker-compose.yml
├── Dockerfile
├── models
├── pyproject.toml
├── README.md
├── src
│   ├── api
│   ├── config
│   ├── engine
│   ├── ingestion
│   ├── __init__.py
│   ├── models
│   ├── __pycache__
│   └── tools
└── tests
    ├── data
    ├── __pycache__
    ├── test_chunking.py
    ├── test_compose.py
    ├── test_extraction.py
    ├── test_fs_tools.py
    ├── test_indexing.py
    ├── test_ingest.py
    ├── test_local_llm.py
    ├── test_orchestration.py
    ├── test_parsing.py
    ├── test_schemas.py
    └── test_settings.py

14 directories, 17 files
```

## 2. Long-Term Memory (`aoc-mem`)
**Purpose:** Persistent storage of architectural decisions.
**Commands:** `aoc-mem read`, `aoc-mem add "fact"`.

## 3. Short-Term Memory (`aoc-stm`)
**Purpose:** STM diary state for short-term continuity and long-term auditability.
**Commands:** `aoc-stm add "note"`, `aoc-stm archive`, `aoc-stm`.

## 4. Task Management (`aoc-task`)
**Purpose:** Granular tracking of work.
**Commands:** `aoc-task list`, `aoc-task add "Task"`.

## 5. Operational Rules
- **No Amnesia:** Always check `aoc-mem` first.
- **No Ghost Work:** Track all work in `aoc-task` (or `task-master`).

## 6. README Content
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

## 7. Active Workstreams (Tags)
```
api-ui (2)
engine (4)
infra (2)
ingestion (6)
master (1)
qa (1)
```

## 8. RLM Skill (Large Codebase Analysis)
When you need to analyze more files than fit in your context:
1. **Scan:** Run `aoc-rlm scan` (or `rlm scan`) to see the scale of the codebase.
2. **Peek:** Run `aoc-rlm peek "search_term"` (or `rlm peek`) to find relevant snippets and file paths.
3. **Slice:** Run `aoc-rlm chunk --pattern "src/relevant/*.rs"` (or `rlm chunk`) to get JSON chunks.
4. **Process:** Use your available sub-agent tools (like `Task`) to process chunks in parallel.
5. **Reduce:** Synthesize the sub-agent outputs into a final answer.
