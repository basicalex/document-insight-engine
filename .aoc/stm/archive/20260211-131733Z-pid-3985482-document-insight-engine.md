Objective + active task/subtask IDs (tag)
- Complete ingestion lane and transition to engine lane.
- Active next task: Task 9 (tag: engine), no subtask started yet.

Done
- Ingestion tasks 3-8 marked done (upload intake, Tier 1 extraction, Docling parsing, parent-child chunking, hybrid indexing, orchestration lifecycle).
- Added tests for each ingestion module; full suite currently green.
- Created commits:
  - e010ba2 (upload + Tier 1 extraction)
  - 2f08d5f (parsing + chunking + indexing)
  - 004d5b6 (orchestration)
  - afc3eec (project scaffold/config/tests/fixtures + AOC/Taskmaster files)

In progress
- No code currently in progress; next execution target is Task 9 (local QA grounded RAG).

Blockers / risks
- Runtime deps for optional paths not installed in this environment (`docling`, `redisvl`, OCR libs) but covered with adapters/tests and graceful errors.
- API surface is still partial vs assignment wording (`/ingest` implemented; `/ask` pending in later tasks).

Files touched
- Ingestion core: `src/ingestion/uploads.py`, `src/ingestion/extraction.py`, `src/ingestion/parsing.py`, `src/ingestion/chunking.py`, `src/ingestion/indexing.py`, `src/ingestion/orchestration.py`, `src/ingestion/__init__.py`
- API/tests: `src/api/main.py`, `tests/test_ingest.py`, `tests/test_extraction.py`, `tests/test_parsing.py`, `tests/test_chunking.py`, `tests/test_indexing.py`, `tests/test_orchestration.py`
- Baseline scaffold committed as well (`README.md`, `pyproject.toml`, compose/docker/config/models/tests fixtures).

Last command outcomes (tests/lint/build)
- `pytest -q` -> pass (`36 passed`).
- No dedicated lint/build command run after last commit batch.

Open decisions / assumptions
- Assume embedding dims remain Tier1=384 (MiniLM) and Tier4=3072 (Gemini embedding) until model choices are changed.
- Assume pipeline idempotency is keyed by explicit idempotency key or document-id-derived fallback.
- Promote to aoc-mem: if we finalize `/upload` alias vs `/ingest` contract direction for external API compatibility.

Next 3-5 concrete steps
- Start Task 9 (engine): implement local QA flow with grounding-first prompt policy and insufficient-evidence fallback.
- Add retrieval trace payload fields (prompt version, chunk IDs, latency) to QA response path.
- Add unit/integration tests for prompt assembly + fallback behavior with mocked LLM/retrieval.
- Wire QA engine into API ask routing scaffold for upcoming Task 13 alignment.
