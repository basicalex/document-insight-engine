# Agent Ops Cockpit (AOC) Global Context

## Core Philosophy
This machine uses the **Agent Ops Cockpit (AOC)** system. All agents (Gemini, Claude, OpenCode) running here share a unified set of tools for **Memory** and **Task Management**.

## 1. Project Structure
```
/home/ceii/dev/document-insight-engine
└── AGENTS.md

1 directory, 1 file
```

## 2. Long-Term Memory (`aoc-mem`)
**Purpose:** Persistent storage of architectural decisions.
**Commands:** `aoc-mem read`, `aoc-mem add "fact"`.

## 3. Short-Term Memory (`aoc-stm`)
**Purpose:** Ephemeral handoff state for context-window continuation.
**Commands:** `aoc-stm add "note"`, `aoc-stm --last`, `aoc-stm history`.

## 4. Task Management (`aoc-task`)
**Purpose:** Granular tracking of work.
**Commands:** `aoc-task list`, `aoc-task add "Task"`.

## 5. Operational Rules
- **No Amnesia:** Always check `aoc-mem` first.
- **No Ghost Work:** Track all work in `aoc-task` (or `task-master`).

## 7. Active Workstreams (Tags)
```
master (0)
```

## 8. RLM Skill (Large Codebase Analysis)
When you need to analyze more files than fit in your context:
1. **Scan:** Run `aoc-rlm scan` (or `rlm scan`) to see the scale of the codebase.
2. **Peek:** Run `aoc-rlm peek "search_term"` (or `rlm peek`) to find relevant snippets and file paths.
3. **Slice:** Run `aoc-rlm chunk --pattern "src/relevant/*.rs"` (or `rlm chunk`) to get JSON chunks.
4. **Process:** Use your available sub-agent tools (like `Task`) to process chunks in parallel.
5. **Reduce:** Synthesize the sub-agent outputs into a final answer.
