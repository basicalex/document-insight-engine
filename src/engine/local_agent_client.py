from __future__ import annotations

import json
from typing import Any

from src.config.settings import Settings, settings
from src.engine.cloud_agent import CloudAgentProviderError, DeepProviderErrorCode
from src.engine.local_llm import OllamaGenerateError, OllamaHTTPClient
from src.models.schemas import Mode


class LocalDeepModelClient:
    def __init__(
        self,
        *,
        cfg: Settings = settings,
        model_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self._client = OllamaHTTPClient(self.cfg.ollama_base_url)

    def next_step(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        model = self.model_name or self.cfg.local_deep_model or self.cfg.local_llm_model
        prompt = _build_local_turn_prompt(
            question=question,
            mode=mode,
            document_id=document_id,
            iteration=iteration,
            history=history,
            allowed_tools=allowed_tools,
        )

        try:
            raw = self._client.generate(
                model=model,
                prompt=prompt,
                timeout_seconds=self.cfg.cloud_agent_timeout_seconds,
            )
        except OllamaGenerateError as exc:
            raise CloudAgentProviderError(
                code=DeepProviderErrorCode.UNAVAILABLE,
                message=f"local deep provider request failed: {exc}",
            ) from exc

        return _parse_local_response(raw)


def _build_local_turn_prompt(
    *,
    question: str,
    mode: Mode,
    document_id: str,
    iteration: int,
    history: list[dict[str, Any]],
    allowed_tools: list[str],
) -> str:
    history_json = json.dumps(history, ensure_ascii=False)
    tools_json = json.dumps(allowed_tools)
    return (
        "You are a tool-using deep reasoning planner for document QA. "
        "Return exactly one JSON object and no extra prose.\n"
        "Allowed responses:\n"
        "1) Tool call: "
        '{"action":"tool_call","tool_name":"<allowed>","arguments":{...}}\n'
        "2) Final answer: "
        '{"action":"final","answer":"...","insufficient_evidence":true|false}\n\n'
        "Rules:\n"
        "- Use only allowed tools.\n"
        "- If unsure, prefer insufficient_evidence=true in final answer.\n"
        "- Keep answer concise and grounded in tool outputs.\n\n"
        f"question: {question}\n"
        f"mode: {mode.value}\n"
        f"document_id: {document_id}\n"
        f"iteration: {iteration}\n"
        f"allowed_tools: {tools_json}\n"
        f"history: {history_json}\n"
    )


def _parse_local_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = _strip_markdown_fence(text)

    parsed = _parse_decision_object(text)
    if parsed is None:
        return {
            "action": "final",
            "answer": text,
            "insufficient_evidence": True,
        }

    action = str(parsed.get("action", "")).strip().lower()
    if action == "tool_call":
        tool_name = parsed.get("tool_name")
        arguments = parsed.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            return {
                "action": "final",
                "answer": "I could not form a valid tool request.",
                "insufficient_evidence": True,
            }
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "action": "tool_call",
            "tool_name": tool_name.strip(),
            "arguments": arguments,
        }

    answer = str(parsed.get("answer", "")).strip()
    insufficient_raw = parsed.get("insufficient_evidence", not answer)
    insufficient = (
        insufficient_raw if isinstance(insufficient_raw, bool) else (not answer)
    )
    return {
        "action": "final",
        "answer": answer,
        "insufficient_evidence": insufficient,
    }


def _parse_decision_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            return None

        try:
            candidate, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue

        cursor = end
        if isinstance(candidate, dict) and _looks_like_decision(candidate):
            return candidate

    return None


def _looks_like_decision(candidate: dict[str, Any]) -> bool:
    action = str(candidate.get("action", "")).strip().lower()
    if action in {"tool_call", "final"}:
        return True
    return "answer" in candidate


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
