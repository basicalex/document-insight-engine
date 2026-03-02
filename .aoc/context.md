# Project Context Snapshot

## Repository
- Name: document-insight-engine
- Root: /home/ceii/dev/document-insight-engine
- Git branch: main

## Key Files
- README.md
- pyproject.toml
- Dockerfile
- docker-compose.yml

## Project Structure (tree -L 2)
```
/home/ceii/dev/document-insight-engine
├── AGENTS.md
├── data
├── docker-compose.dev.yml
├── docker-compose.yml
├── Dockerfile
├── frontend
│   ├── app.py
│   ├── client.py
│   ├── __init__.py
│   ├── progress.py
│   ├── readiness.py
│   └── state.py
├── models
├── pyproject.toml
├── README.md
├── scripts
│   ├── dev-api.sh
│   ├── dev-full-up.sh
│   ├── dev-lite-up.sh
│   ├── dev-ui.sh
│   ├── dev-up.sh
│   ├── docker-full-up.sh
│   ├── docker-lite-up.sh
│   ├── docker-up.sh
│   └── _profile_common.sh
├── src
│   ├── api
│   ├── config
│   ├── engine
│   ├── evals
│   ├── ingestion
│   ├── __init__.py
│   ├── models
│   └── tools
└── tests
    ├── data
    ├── test_api.py
    ├── test_api_state_store.py
    ├── test_chunking.py
    ├── test_cloud_agent.py
    ├── test_compose.py
    ├── test_embeddings.py
    ├── test_evaluation_harness.py
    ├── test_extraction.py
    ├── test_extractor.py
    ├── test_frontend_app.py
    ├── test_frontend_client.py
    ├── test_frontend_readiness.py
    ├── test_frontend_state.py
    ├── test_fs_tools.py
    ├── test_gemini_client.py
    ├── test_google_parser.py
    ├── test_indexing.py
    ├── test_ingest.py
    ├── test_ingests_endpoint.py
    ├── test_local_agent_client.py
    ├── test_local_llm.py
    ├── test_orchestration.py
    ├── test_parsing.py
    ├── test_phase_validation.py
    ├── test_recent_ingestions.py
    ├── test_runtime_pipeline.py
    ├── test_schemas.py
    └── test_settings.py

15 directories, 50 files
```

## README Headings
# Document Insight Engine
## Assignment alignment
## Included dummy test docs (committed)
## Runtime profiles
## Setup (hybrid local API/UI + Docker infra)
## Setup (full Docker)
## API status and contracts
## Approach
## Docker manual verification checklist
## Engine components (implemented)
## UI coverage

## Current Task Tag
```
completion
```

## Active Workstreams (Tags)
```
api-ui (2)
completion (10)
engine (4)
infra (3)
ingestion (6)
integration (1)
master (1)
qa (1)
```

## Task PRD Location
- Directory: .taskmaster/docs/prds
- Resolve tag PRD default with: aoc-task tag prd show --tag <tag>
- Resolve task PRD override with: aoc-task prd show <id> --tag <tag>
- Effective precedence: task PRD override -> tag PRD default
