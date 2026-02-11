---
name: prd-intake
description: Parse a project PRD into initial tasks, then refine with sub-agents.
---

## Goal
Turn a project PRD (usually `.taskmaster/docs/prd.txt`) into a high-quality initial task graph.

## Workflow
1. Verify PRD exists and has actionable sections (goals, stories, requirements, acceptance criteria).
2. Run a dry parse first:
   - `aoc-task prd parse --dry-run --json`
   - Or `aoc-task prd parse <path> --dry-run --json`
3. Review candidates for coverage, duplicates, and scope mistakes.
4. In OpenCode, fan out sub-agents by domain/epic for refinement (parallel where possible).
5. Apply generation:
   - Append mode: `aoc-task prd parse <path> --apply append`
   - Replace mode: `aoc-task prd parse <path> --apply replace --force`
6. Run follow-up alignment:
   - `task-breakdown` for large tasks
   - `tag-align` for tags/status/dependencies
   - `prd-align` to ensure details and testStrategy match the PRD

## Guardrails
- Never edit `.taskmaster/tasks/tasks.json` directly.
- Use `--dry-run` before destructive `--apply replace`.
- PRD links are task-level only; subtasks must not include PRD links.
- Keep generated tasks actionable and testable.

## Notes
- Default parse source is `.taskmaster/docs/prd.txt` (fallback `.taskmaster/docs/prd.md`).
- Generated tasks are linked to the parsed PRD path via `aocPrd`.
