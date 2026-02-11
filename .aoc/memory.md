# Agent Memory for Project: document-insight-engine
This file contains persistent context, decisions, and knowledge for the AI agent.
Agents should read this to understand project history and append new decisions here.

## Core Decisions
- [2026-02-11 12:13] PRD intake executed in append mode for tag master using .taskmaster/docs/prd.txt; generated 15 normalized tasks (IDs 1-15) spanning foundation->ingestion->engine->API->frontend->validation with explicit dependency DAG and PRD links set at task level.
- [2026-02-11 12:16] Performed task-breakdown on master tasks 8,11,12,13,15: added 5 subtasks each with dependency chains for implementation order and test-first closure.
- [2026-02-11 12:18] Ran prd-align on master tasks: tightened Task 12 and Task 15 to explicitly enforce PRD metrics (Tier1 <2s, Tier4 >98% with character-level grounding), added Tier4 token-estimation/budget guard subtask (12.6), and aligned validation planning to 60/30/10 test pyramid.
- [2026-02-11 12:20] Applied tag-aligned workflow lanes: moved tasks from master into infra(1-2), ingestion(3-8), engine(9-12), api-ui(13-14), qa(15) to support lane-based execution and testing.
- [2026-02-11 13:34] Completed ingestion task 3: added upload intake service with MIME/size validation, filename sanitization, deterministic document IDs, idempotent file persistence under data/uploads, FastAPI /ingest endpoint, and pytest coverage for allowed/blocked MIME, traversal-safe names, oversized uploads, and idempotency.
- [2026-02-11 13:35] Completed ingestion task 4: added Tier1TextExtractor supporting PDF text-layer extraction via PyMuPDF with OCR fallback for scanned PDFs/images, page-level provenance metadata, and tests covering fallback triggering and page-order output semantics.
- [2026-02-11 13:47] Completed ingestion task 5: implemented Tier2DoclingParser with pluggable Docling backend, markdown export handling, heading-aware section paths, stable block IDs, and markdown table atomicity guarantees; added parser tests covering table atomic blocks, section lineage, deterministic IDs, and backend-provided page references.
- [2026-02-11 13:49] Completed ingestion task 6: implemented ParentChildChunker producing ~1024-token parent and ~256-token child chunks (configurable), preserving block order/lineage metadata (section path, page refs, block IDs), enforcing child->parent linkage IDs, and keeping table blocks atomic during child generation; added chunking tests for boundaries, linkage, and table unsplit behavior.
