from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from src.config.settings import Settings, settings
from src.models.schemas import AgentTrace, ChatResponse, Mode, TraceEvent
from src.tools import get_fs_tools


ALLOWED_TOOL_NAMES = ("list_sections", "read_section", "keyword_grep")


class DeepProviderAction(str, Enum):
    FINAL = "final"
    TOOL_CALL = "tool_call"


class DeepProviderErrorCode(str, Enum):
    NOT_CONFIGURED = "provider_not_configured"
    AUTHENTICATION_FAILED = "provider_auth_failed"
    RATE_LIMITED = "provider_rate_limited"
    TIMEOUT = "provider_timeout"
    UNAVAILABLE = "provider_unavailable"
    MALFORMED_RESPONSE = "provider_malformed_response"


@dataclass(frozen=True)
class DeepProviderRetryPolicy:
    attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 8.0


@dataclass(frozen=True)
class DeepProviderDecision:
    action: DeepProviderAction
    answer: str = ""
    insufficient_evidence: bool = False
    tool_name: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ProviderDecisionError:
    code: str
    message: str


class CloudAgentProviderError(Exception):
    def __init__(self, *, code: DeepProviderErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CloudAgentModelClient(Protocol):
    def next_step(
        self,
        *,
        question: str,
        mode: Mode,
        document_id: str,
        iteration: int,
        history: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> DeepProviderDecision | dict[str, Any]: ...


ToolCallable = Callable[..., dict[str, Any]]
ToolProvider = Callable[[str], dict[str, ToolCallable]]


@dataclass(frozen=True)
class _ToolValidationError:
    code: str
    message: str


class CloudAgentEngine:
    def __init__(
        self,
        model_client: CloudAgentModelClient,
        tool_provider: ToolProvider | None = None,
        cfg: Settings = settings,
        max_iterations: int = 5,
        model_name: str = "gemini-3-flash",
        prompt_version: str = "cloud-agent-v1",
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.model_client = model_client
        self.tool_provider = tool_provider or (
            lambda document_id: get_fs_tools(document_id)
        )
        self.cfg = cfg
        self.max_iterations = max_iterations
        self.model_name = model_name
        self.prompt_version = prompt_version

    def ask(self, question: str, mode: Mode, document_id: str | None) -> ChatResponse:
        if not document_id:
            trace = AgentTrace(
                model=self.model_name,
                prompt_version=self.prompt_version,
                iterations=0,
                termination_reason="missing_document_id",
            )
            return ChatResponse(
                answer="Deep mode requires a document_id for filesystem reasoning.",
                mode=mode,
                document_id=document_id,
                insufficient_evidence=True,
                citations=[],
                trace=trace,
            )

        started = time.perf_counter()
        tools = self.tool_provider(document_id)
        history: list[dict[str, Any]] = []
        trace_events: list[TraceEvent] = []
        retrieved_keys: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            turn_started = time.perf_counter()
            try:
                raw_decision = self.model_client.next_step(
                    question=question,
                    mode=mode,
                    document_id=document_id,
                    iteration=iteration,
                    history=history,
                    allowed_tools=list(ALLOWED_TOOL_NAMES),
                )
            except CloudAgentProviderError as exc:
                trace_events.append(
                    TraceEvent(
                        stage="agent",
                        message="provider request failed",
                        latency_ms=_latency_ms(turn_started),
                        metadata={
                            "iteration": str(iteration),
                            "code": exc.code.value,
                        },
                    )
                )
                return self._terminal_response(
                    mode=mode,
                    document_id=document_id,
                    answer="Deep reasoning is temporarily unavailable from the configured provider.",
                    insufficient_evidence=True,
                    retrieved_keys=retrieved_keys,
                    trace_events=trace_events,
                    iterations=iteration,
                    termination_reason=exc.code.value,
                    started=started,
                )

            normalized_decision = _normalize_provider_decision(raw_decision)
            if isinstance(normalized_decision, _ProviderDecisionError):
                return self._terminal_response(
                    mode=mode,
                    document_id=document_id,
                    answer=(
                        "Deep reasoning failed because the model returned an invalid step."
                    ),
                    insufficient_evidence=True,
                    retrieved_keys=retrieved_keys,
                    trace_events=trace_events,
                    iterations=iteration,
                    termination_reason=normalized_decision.code,
                    started=started,
                )

            if normalized_decision.action == DeepProviderAction.FINAL:
                answer = normalized_decision.answer.strip()
                insufficient = normalized_decision.insufficient_evidence
                if not answer:
                    answer = (
                        "I do not have enough grounded evidence in the document to "
                        "answer that question."
                    )
                    insufficient = True

                trace_events.append(
                    TraceEvent(
                        stage="agent",
                        message="final answer emitted",
                        latency_ms=_latency_ms(turn_started),
                        metadata={"iteration": str(iteration)},
                    )
                )
                return self._terminal_response(
                    mode=mode,
                    document_id=document_id,
                    answer=answer,
                    insufficient_evidence=insufficient,
                    retrieved_keys=retrieved_keys,
                    trace_events=trace_events,
                    iterations=iteration,
                    termination_reason="completed"
                    if not insufficient
                    else "insufficient_evidence",
                    started=started,
                )

            tool_name = normalized_decision.tool_name
            arguments = normalized_decision.arguments or {}
            validated = _validate_tool_invocation(
                tool_name=tool_name, arguments=arguments
            )
            if isinstance(validated, _ToolValidationError):
                trace_events.append(
                    TraceEvent(
                        stage="tool_call",
                        message="tool invocation rejected",
                        latency_ms=_latency_ms(turn_started),
                        metadata={
                            "iteration": str(iteration),
                            "tool_name": tool_name or "(missing)",
                            "code": validated.code,
                        },
                    )
                )
                return self._terminal_response(
                    mode=mode,
                    document_id=document_id,
                    answer="Deep reasoning stopped due to an invalid tool request.",
                    insufficient_evidence=True,
                    retrieved_keys=retrieved_keys,
                    trace_events=trace_events,
                    iterations=iteration,
                    termination_reason=validated.code,
                    started=started,
                )

            tool = tools.get(tool_name)
            if tool is None:
                trace_events.append(
                    TraceEvent(
                        stage="tool_call",
                        message="tool unavailable for document",
                        latency_ms=_latency_ms(turn_started),
                        metadata={
                            "iteration": str(iteration),
                            "tool_name": tool_name,
                            "code": "tool_unavailable",
                        },
                    )
                )
                return self._terminal_response(
                    mode=mode,
                    document_id=document_id,
                    answer="Deep reasoning could not access the requested tool.",
                    insufficient_evidence=True,
                    retrieved_keys=retrieved_keys,
                    trace_events=trace_events,
                    iterations=iteration,
                    termination_reason="tool_unavailable",
                    started=started,
                )

            call_started = time.perf_counter()
            result = tool(**validated)
            call_latency = _latency_ms(call_started)

            if isinstance(result, dict) and result.get("ok") is False:
                code = str(result.get("error", {}).get("code", "tool_error"))
                message = f"{tool_name} returned error"
            else:
                code = "ok"
                message = f"{tool_name} completed"

            trace_events.append(
                TraceEvent(
                    stage="tool_call",
                    message=message,
                    latency_ms=call_latency,
                    metadata={
                        "iteration": str(iteration),
                        "tool_name": tool_name,
                        "status": code,
                    },
                )
            )

            retrieved_keys.extend(
                _extract_retrieved_keys(tool_name=tool_name, result=result)
            )
            history.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "arguments": validated,
                    "result": result,
                }
            )

        trace_events.append(
            TraceEvent(
                stage="agent",
                message="loop cap reached",
                latency_ms=_latency_ms(started),
                metadata={"max_iterations": str(self.max_iterations)},
            )
        )
        return self._terminal_response(
            mode=mode,
            document_id=document_id,
            answer=(
                "I could not reach a grounded answer within the allowed reasoning "
                "steps."
            ),
            insufficient_evidence=True,
            retrieved_keys=retrieved_keys,
            trace_events=trace_events,
            iterations=self.max_iterations,
            termination_reason="max_iterations_reached",
            started=started,
        )

    def _terminal_response(
        self,
        *,
        mode: Mode,
        document_id: str | None,
        answer: str,
        insufficient_evidence: bool,
        retrieved_keys: list[str],
        trace_events: list[TraceEvent],
        iterations: int,
        termination_reason: str,
        started: float,
    ) -> ChatResponse:
        trace = AgentTrace(
            model=self.model_name,
            prompt_version=self.prompt_version,
            iterations=min(iterations, 5),
            retrieved_chunk_ids=list(dict.fromkeys(retrieved_keys)),
            tool_calls=trace_events,
            termination_reason=termination_reason,
            total_latency_ms=_latency_ms(started),
        )
        return ChatResponse(
            answer=answer,
            mode=mode,
            document_id=document_id,
            insufficient_evidence=insufficient_evidence,
            citations=[],
            trace=trace,
        )


def run_agent(
    *,
    question: str,
    mode: Mode,
    document_id: str,
    model_client: CloudAgentModelClient,
    tool_provider: ToolProvider | None = None,
    cfg: Settings = settings,
) -> ChatResponse:
    engine = CloudAgentEngine(
        model_client=model_client,
        tool_provider=tool_provider,
        cfg=cfg,
    )
    return engine.ask(question=question, mode=mode, document_id=document_id)


def _normalize_provider_decision(
    raw: DeepProviderDecision | dict[str, Any],
) -> DeepProviderDecision | _ProviderDecisionError:
    if isinstance(raw, DeepProviderDecision):
        return raw
    if not isinstance(raw, dict):
        return _ProviderDecisionError(
            code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
            message="provider decision must be an object",
        )

    action_raw = str(raw.get("action", "")).strip().lower()
    if action_raw == DeepProviderAction.FINAL.value:
        answer = raw.get("answer", "")
        if answer is None:
            answer = ""
        if not isinstance(answer, str):
            return _ProviderDecisionError(
                code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
                message="final.answer must be a string",
            )
        insufficient = raw.get("insufficient_evidence", False)
        if not isinstance(insufficient, bool):
            return _ProviderDecisionError(
                code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
                message="final.insufficient_evidence must be a boolean",
            )
        return DeepProviderDecision(
            action=DeepProviderAction.FINAL,
            answer=answer,
            insufficient_evidence=insufficient,
        )

    if action_raw == DeepProviderAction.TOOL_CALL.value:
        tool_name = raw.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return _ProviderDecisionError(
                code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
                message="tool_call.tool_name must be a non-empty string",
            )
        arguments = raw.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _ProviderDecisionError(
                code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
                message="tool_call.arguments must be an object",
            )
        return DeepProviderDecision(
            action=DeepProviderAction.TOOL_CALL,
            tool_name=tool_name.strip(),
            arguments=arguments,
        )

    return _ProviderDecisionError(
        code=DeepProviderErrorCode.MALFORMED_RESPONSE.value,
        message="action must be 'final' or 'tool_call'",
    )


def _validate_tool_invocation(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any] | _ToolValidationError:
    if tool_name not in ALLOWED_TOOL_NAMES:
        return _ToolValidationError(
            code="tool_not_allowed",
            message=f"tool '{tool_name}' is not in the allowlist",
        )
    if not isinstance(arguments, dict):
        return _ToolValidationError(
            code="invalid_tool_request",
            message="tool arguments must be an object",
        )

    if tool_name == "list_sections":
        allowed = {"limit"}
        if set(arguments.keys()) - allowed:
            return _ToolValidationError(
                code="invalid_tool_request",
                message="list_sections received unsupported arguments",
            )
        normalized: dict[str, Any] = {}
        if "limit" in arguments:
            if not isinstance(arguments["limit"], int):
                return _ToolValidationError(
                    code="invalid_tool_request",
                    message="list_sections.limit must be an integer",
                )
            normalized["limit"] = arguments["limit"]
        return normalized

    if tool_name == "read_section":
        allowed = {"section_key", "max_chars"}
        if set(arguments.keys()) - allowed:
            return _ToolValidationError(
                code="invalid_tool_request",
                message="read_section received unsupported arguments",
            )
        section_key = arguments.get("section_key")
        if not isinstance(section_key, str) or not section_key.strip():
            return _ToolValidationError(
                code="invalid_tool_request",
                message="read_section.section_key must be a non-empty string",
            )
        normalized = {"section_key": section_key.strip()}
        if "max_chars" in arguments:
            if not isinstance(arguments["max_chars"], int):
                return _ToolValidationError(
                    code="invalid_tool_request",
                    message="read_section.max_chars must be an integer",
                )
            normalized["max_chars"] = arguments["max_chars"]
        return normalized

    allowed = {"keyword", "section_key", "max_matches", "context_chars"}
    if set(arguments.keys()) - allowed:
        return _ToolValidationError(
            code="invalid_tool_request",
            message="keyword_grep received unsupported arguments",
        )

    keyword = arguments.get("keyword")
    if not isinstance(keyword, str) or not keyword.strip():
        return _ToolValidationError(
            code="invalid_tool_request",
            message="keyword_grep.keyword must be a non-empty string",
        )

    normalized = {"keyword": keyword.strip()}
    if "section_key" in arguments:
        if arguments["section_key"] is not None and not isinstance(
            arguments["section_key"], str
        ):
            return _ToolValidationError(
                code="invalid_tool_request",
                message="keyword_grep.section_key must be a string when provided",
            )
        normalized["section_key"] = arguments["section_key"]
    if "max_matches" in arguments:
        if not isinstance(arguments["max_matches"], int):
            return _ToolValidationError(
                code="invalid_tool_request",
                message="keyword_grep.max_matches must be an integer",
            )
        normalized["max_matches"] = arguments["max_matches"]
    if "context_chars" in arguments:
        if not isinstance(arguments["context_chars"], int):
            return _ToolValidationError(
                code="invalid_tool_request",
                message="keyword_grep.context_chars must be an integer",
            )
        normalized["context_chars"] = arguments["context_chars"]
    return normalized


def _extract_retrieved_keys(tool_name: str, result: Any) -> list[str]:
    if not isinstance(result, dict) or result.get("ok") is False:
        return []

    if tool_name == "read_section":
        section = result.get("section")
        if isinstance(section, dict):
            key = section.get("key")
            if isinstance(key, str) and key:
                return [key]
        return []

    if tool_name == "keyword_grep":
        matches = result.get("matches")
        if not isinstance(matches, list):
            return []
        keys = [
            str(item.get("section_key"))
            for item in matches
            if isinstance(item, dict) and isinstance(item.get("section_key"), str)
        ]
        return [key for key in keys if key]

    return []


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
