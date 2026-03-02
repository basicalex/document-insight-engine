- [2026-02-15 05:18] Objective: unblock deep mode in no-Docling dev profile by ensuring fs-explorer can read parsed markdown artifacts.
Active task context: no-Docling runtime validation + deep-mode fs-explorer handoff (post-task 28 infra improvements).

Done:
- Switched dev/default runtime to no-Docling profile (DOCLING_ENABLED=false, GOOGLE_PARSER_ENABLED=false, PARSER_ROUTING_MODE=fallback_only) in docker-compose.yml and docker-compose.dev.yml.
- Docker builds now default to ai-lite (LangExtract) with optional INSTALL_DOCLING=true path; tests and README updated accordingly.
- Rebuilt and started api/ui with INSTALL_DOCLING=false; readiness is healthy (readyz=200).
- Pulled Ollama embedding model nomic-embed-text and aligned compose embedding envs (model/dimension 768) plus EMBEDDING_FILTER_STRICT=false to avoid vector filter failures.
- Verified upload+ingest reaches indexed; fast /ask now returns 200 in local profile.

In progress:
- Deep /ask still fails for indexed docs because fs tools cannot find parsed markdown artifact in /app/data/parsed.

Blockers / risks:
- Root cause appears architectural: ingestion/orchestration parses content but does not persist parsed markdown files used by fs tools.
- Deep agent depends on get_fs_tools(document_id) -> /app/data/parsed/<document_id>.md lookup and fails hard when file missing.

Files touched:
- Dockerfile
- docker-compose.yml
- docker-compose.dev.yml
- pyproject.toml
- README.md
- src/api/main.py
- tests/test_api.py
- tests/test_compose.py

Last command outcomes:
- docker compose up -d --build api ui (INSTALL_DOCLING=false): success, services healthy.
- healthz: docling disabled, google parser disabled, langextract ready.
- smoke upload/ingest: indexed.
- fast ask: 200 (insufficient_evidence response).
- deep ask: 500 with error no parsed markdown found for document_id in /app/data/parsed.
- pytest -q: 162 passed.

Open decisions / assumptions:
- Assumption: deep mode should work in fallback-only parser mode without Docling.
- Decision needed: persist parsed markdown during ingest parse stage vs parse-on-demand in deep path when artifact missing.
- Prefer persist-on-ingest for deterministic fs-tool behavior and simpler deep path latency/profile.

Next steps (3-5):
1) Implement parsed artifact persistence in ingestion flow (after parse stage) to cfg.parsed_dir/<document_id>.md (or <document_id>_<name>.md consistently).
2) Add/adjust tests to assert parsed markdown file exists after successful ingest and deep ask can access fs tools without Docling.
3) Add fallback guard in deep path: if markdown artifact missing, return deterministic actionable error or trigger safe parse-on-demand (decision above).
4) Re-run docker smoke: upload -> ingest indexed -> ask fast/deep both 200 in no-Docling profile.
5) Promote durable decision to aoc-mem once approach is chosen and validated.
