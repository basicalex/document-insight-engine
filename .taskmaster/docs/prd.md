# Repository Planning Graph (RPG) - Intrface Document Insight Engine

<rpg-method>

# Repository Planning Graph (RPG) Method - PRD

This document follows the RPG methodology to structure the **Intrface Document Insight Engine**. It separates functional capabilities from structural implementation and defines explicit dependencies to allow Task Master agents to execute without circular logic.

## Core Principles Applied

1. **Dual-Semantics**: Capabilities (Ingestion, Reasoning) are mapped to Modules (`src/ingestion`, `src/engine`).
2. **Explicit Dependencies**: Tier 3 (Agent) cannot be built before Tier 2 (Docling Storage) is established.
3. **Topological Order**: Infrastructure -> Ingestion -> Local Search -> Cloud Agents -> Frontend.

</rpg-method>

---

<overview>

## Problem Statement

Complex business documents (contracts, invoices, compliance reports) are opaque binary blobs.

* **Local RAG** is fast but often fails on complex layouts (tables) and lacks reasoning.
* **Cloud LLMs** are smart but privacy-invasive, expensive for simple queries, and often hallucinate data without strict grounding.
* **Users** (Legal/Finance) need a system that adapts: "Fast & Private" for quick questions, "Deep & Auditable" for high-stakes extraction.

## Target Users

* **Legal Analysts**: Need to compare clauses across documents (Tier 3).
* **Finance Officers**: Need to extract validated JSON data from invoices (Tier 4).
* **General Staff**: Need quick answers from handbooks without API costs (Tier 1).

## Success Metrics

* **Tier 1 Latency**: < 2 seconds (End-to-end).
* **Tier 4 Accuracy**: > 98% extraction accuracy with character-level source grounding.
* **Reliability**: 0% data loss on restart (Persistent Docker Volumes).
* **Safety**: Agentic loops hard-capped at 5 iterations.

</overview>

---

<functional-decomposition>

## Capability Tree

### Capability: Data Ingestion & Parsing

*Handles the conversion of raw files into machine-understandable formats.*

#### Feature: Raw File Handling

* **Description**: Validates file types (PDF, Image), saves to persistent disk, and generates unique IDs.
* **Inputs**: `UploadFile` (FastAPI).
* **Outputs**: `file_path`, `document_id`.
* **Behavior**: Checks MIME types, sanitizes filenames, streams to `./data/uploads`.

#### Feature: Basic Text Extraction (Tier 1)

* **Description**: Extracts raw text for the MVP fast path.
* **Inputs**: PDF Path.
* **Outputs**: String content.
* **Behavior**: Uses PyMuPDF. Fallback to EasyOCR if text layer is empty.

#### Feature: Layout-Aware Parsing (Tier 2-4)

* **Description**: Converts PDF to structured Markdown, preserving tables and headers.
* **Inputs**: PDF Path.
* **Outputs**: `doc_123.md` (Markdown file).
* **Behavior**: Uses **Docling**. Must isolate tables as atomic units.

### Capability: Knowledge Indexing

*Transforms text into searchable vectors.*

#### Feature: Parent-Child Chunking

* **Description**: optimizing retrieval by indexing small chunks but retrieving large contexts.
* **Inputs**: Markdown text.
* **Outputs**: List of `VectorNode` (Child) and `ParentDocument`.
* **Behavior**: Splits into 256-token children linked to 1024-token parent sections.

#### Feature: Hybrid Embedding

* **Description**: Generates vectors based on Tier selection.
* **Inputs**: Text chunk, Mode (`fast` or `deep`).
* **Outputs**: Float Vector.
* **Behavior**:
* `fast`: Local `all-MiniLM-L6-v2`.
* `deep`: Cloud `gemini-embedding-001` (Multimodal support).



#### Feature: Vector Persistence

* **Description**: Stores and retrieves vectors.
* **Inputs**: Vectors, Metadata.
* **Outputs**: Search results.
* **Behavior**: Uses Redis with **RedisVL**. Segregates indices (`tier1_idx` vs `tier4_idx`).

### Capability: Inference Engine (The Brain)

*Routes queries to the appropriate reasoning logic.*

#### Feature: Local QA (Tier 1/2)

* **Description**: Private, offline generation.
* **Inputs**: Context chunks, User Query.
* **Outputs**: Answer string.
* **Behavior**: Calls **Ollama** (Llama 3.2). Uses standard RAG prompt.

#### Feature: FileSystem Agent (Tier 3)

* **Description**: Agentic scavenging of documents using tool use.
* **Inputs**: User Query, Document ID.
* **Outputs**: Answer + reasoning trace.
* **Behavior**: Uses **Gemini 3 Flash**. Tools: `list_sections`, `read_section`, `keyword_grep`. Loop limit: 5.

#### Feature: Structured Extraction (Tier 4)

* **Description**: Turns documents into grounded JSON.
* **Inputs**: Document, Extraction Schema.
* **Outputs**: Validated JSON with source offsets.
* **Behavior**: Uses **Google LangExtract**.

### Capability: User Interface

*Streamlit frontend for interaction.*

#### Feature: Chat & Control

* **Description**: Chat interface with mode toggle.
* **Inputs**: User text, Radio selection.
* **Outputs**: UI render.
* **Behavior**: Maintains session state, displays traces in expanders.

</functional-decomposition>

---

<structural-decomposition>

## Repository Structure

```
intrface-engine/
├── src/
│   ├── config/             # Configuration & Env
│   ├── models/             # Pydantic Schemas (Data Layer)
│   ├── ingestion/          # Parsers (PyMuPDF, Docling)
│   ├── storage/            # Redis/Vector Logic
│   ├── engine/             # LLM Logic (Ollama, Gemini, Agent)
│   ├── tools/              # fs-explorer tools for Agent
│   └── api/                # FastAPI Routes
├── frontend/               # Streamlit App
├── tests/                  # Pytest suite
└── docker-compose.yml

```

## Module Definitions

### Module: `src/config`

* **Maps to**: Foundation
* **Responsibility**: Environment variables, path constants, logging setup.
* **Exports**: `settings`, `logger`.

### Module: `src/models`

* **Maps to**: Data Layer
* **Responsibility**: Pydantic models for API contracts and DB schemas.
* **Exports**: `Document`, `ChatRequest`, `ChatResponse`, `AgentTrace`.

### Module: `src/ingestion`

* **Maps to**: Data Ingestion Capability
* **Responsibility**: Routing files to parsers.
* **Structure**:
* `simple_parser.py` (PyMuPDF)
* `layout_parser.py` (Docling)


* **Exports**: `parse_document(file, mode)`

### Module: `src/storage`

* **Maps to**: Knowledge Indexing Capability
* **Responsibility**: Redis interactions.
* **Structure**:
* `redis_client.py` (Connection)
* `vector_store.py` (RedisVL Index management)


* **Exports**: `index_document()`, `search_hybrid()`, `save_session()`.

### Module: `src/tools`

* **Maps to**: Tier 3 Agent Tools
* **Responsibility**: Defining the `fs-explorer` functions.
* **Exports**: `get_fs_tools(doc_id)` returning list of callables.

### Module: `src/engine`

* **Maps to**: Inference Engine Capability
* **Responsibility**: Calling LLMs.
* **Structure**:
* `local_llm.py` (Ollama interaction)
* `cloud_agent.py` (Gemini + LangGraph/Loop)
* `extractor.py` (LangExtract)


* **Exports**: `generate_local()`, `run_agent()`, `extract_structured()`.

### Module: `src/api`

* **Maps to**: API Capability
* **Responsibility**: HTTP Endpoints.
* **Exports**: `app` (FastAPI instance).

</structural-decomposition>

---

<dependency-graph>

## Dependency Chain

### Foundation Layer (Phase 0)

*No dependencies.*

* **src/config**: Loads `.env`, checks API keys.
* **src/models**: Defines `Document` and `ChatRequest` schemas.

### Persistence Layer (Phase 1)

* **src/storage**: Depends on [src/config, src/models]. Needs Redis connection.

### Ingestion Layer (Phase 2)

* **src/ingestion**: Depends on [src/config, src/models].
* *Note:* Needs `storage` to save parsed results, but strictly speaking, parsing can happen before storage. However, we usually pipe Ingest -> Storage.

### Logic Layer (Phase 3)

* **src/tools**: Depends on [src/ingestion] (Needs Markdown structure).
* **src/engine**: Depends on [src/storage, src/tools, src/config].
* `local_llm` needs `storage` for RAG.
* `cloud_agent` needs `tools`.



### Interface Layer (Phase 4)

* **src/api**: Depends on [src/engine, src/ingestion].
* **frontend**: Depends on [src/api] (via HTTP calls).

</dependency-graph>

---

<implementation-roadmap>

## Development Phases

### Phase 0: Foundation & Infrastructure

**Goal**: Docker environment with persistent storage and basic config.
**Entry Criteria**: Empty Repo.
**Tasks**:

* [ ] Create `docker-compose.yml` with Redis, Ollama, and API services.
* [ ] Configure Docker Volumes for `./data` and `./models` (Critical).
* [ ] Implement `src/config/settings.py`.
* [ ] Implement `src/models/schemas.py`.
**Exit Criteria**: `docker-compose up` runs without errors; Redis and Ollama are reachable.

### Phase 1: Ingestion & Local Search (Tier 1/2)

**Goal**: Functional MVP. Upload PDF -> Search -> Answer (Local).
**Entry Criteria**: Phase 0 complete.
**Tasks**:

* [ ] Implement `src/ingestion/simple_parser.py` (PyMuPDF).
* [ ] Implement `src/ingestion/layout_parser.py` (Docling).
* [ ] Implement `src/storage/vector_store.py` (RedisVL setup for `tier1_idx`).
* [ ] Implement `src/engine/local_llm.py` (Ollama hook).
* [ ] Create API Endpoint `POST /ingest` and `POST /ask`.
**Exit Criteria**: Can upload a PDF and get an answer from Llama 3.2.

### Phase 2: Agentic Capabilities (Tier 3/4)

**Goal**: Cloud powered deep reasoning.
**Entry Criteria**: Phase 1 complete.
**Tasks**:

* [ ] Implement `src/tools/fs_tools.py` (List/Read/Grep logic on Markdown).
* [ ] Implement `src/engine/cloud_agent.py` (Gemini loop with 5-step guard).
* [ ] Implement `src/engine/extractor.py` (LangExtract integration).
* [ ] Update `src/models` to support `trace_logs`.
**Exit Criteria**: Agent can answer "Find the contradiction between Section A and B".

### Phase 3: Frontend & Validation

**Goal**: User-ready artifact.
**Entry Criteria**: Phase 2 complete.
**Tasks**:

* [ ] Build `frontend/app.py` (Streamlit).
* [ ] Add Sidebar Mode Toggle.
* [ ] Implement Agent Trace visualization.
* [ ] Write `tests/test_api.py`.
* [ ] Create `tests/data/` with dummy contracts.
**Exit Criteria**: UI works, tests pass.

</implementation-roadmap>

---

<test-strategy>

## Test Pyramid

* **E2E**: 10% (Streamlit -> API -> LLM full flow).
* **Integration**: 30% (API -> Redis, API -> Ollama).
* **Unit**: 60% (Parsers, Text Splitters, Pydantic validation).

## Critical Test Scenarios

### Ingestion Module

* **Happy Path**: PDF uploads, Docling returns Markdown.
* **Edge Case**: Corrupt PDF file.
* **Edge Case**: Image-only PDF (triggers OCR).

### Agent Engine

* **Loop Guard**: Mock an agent that never finds the answer; assert loop breaks at 5.
* **Tool Use**: Verify `read_section` returns specific text from the Markdown file.

### API

* **Persistence**: Restart Docker container; assert previous `document_id` is still searchable.

## Test Generation Guidelines

* Use `pytest-asyncio` for FastAPI tests.
* Mock external APIs (Gemini) in Unit tests; use real APIs in Integration tests (require `.env`).

</test-strategy>

---

<architecture>

## System Components

1. **Orchestrator**: FastAPI (Python).
2. **Vector DB**: Redis (Stack) with RedisVL.
3. **Local Inference**: Ollama (Llama 3.2).
4. **Cloud Inference**: Google AI Studio (Gemini 3 Flash).

## Data Models

* **Document**: `{id, filename, filepath, chunks, metadata}`.
* **Session**: `{session_id, history, mode}`.

## Technology Stack

**Decision: Redis over Chroma/FAISS**

* **Rationale**: We need both Vector Search AND Session Caching. Redis does both in one container.
* **Trade-offs**: Slightly higher RAM usage.

**Decision: Docling over Unstructured**

* **Rationale**: Superior table structure preservation which is critical for "Insight" engine.

</architecture>

---

<risks>

## Technical Risks

**Risk**: Docker Memory Exhaustion (Ollama).

* **Impact**: High. Container crashes.
* **Mitigation**: Set explicit `deploy: resources: limits` in docker-compose.
* **Fallback**: Fallback to Cloud-only mode if local hardware fails.

**Risk**: Gemini API Cost/Rate Limits.

* **Impact**: Medium. Tier 3 fails.
* **Mitigation**: Loop Guard (5 max). Token estimation before Tier 4 run.

## Scope Risks

**Risk**: UI Complexity.

* **Mitigation**: Keep Streamlit simple. No custom JS components, standard widgets only.

</risks>
