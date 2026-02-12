from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Protocol

from src.config.settings import Settings, settings


@dataclass(frozen=True)
class ValidationDiagnostic:
    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class FieldProvenance:
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True)
class StructuredExtractionResult:
    document_id: str
    model: str
    prompt_version: str
    data: dict[str, Any]
    provenance: dict[str, FieldProvenance]
    accepted_fields: list[str]
    rejected_fields: list[str]
    diagnostics: list[ValidationDiagnostic]
    token_usage: dict[str, int]
    latency_ms: int


@dataclass(frozen=True)
class StructuredExtractionError:
    code: str
    message: str
    diagnostics: list[ValidationDiagnostic] = dataclass_field(default_factory=list)
    token_usage: dict[str, int] = dataclass_field(default_factory=dict)
    latency_ms: int = 0


@dataclass(frozen=True)
class StructuredExtractionEnvelope:
    ok: bool
    result: StructuredExtractionResult | None = None
    error: StructuredExtractionError | None = None


class LangExtractProviderError(Exception):
    pass


class LangExtractClient(Protocol):
    def extract(
        self,
        *,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None,
        model_name: str,
    ) -> dict[str, Any]: ...


class RuntimeLangExtractClient:
    def extract(
        self,
        *,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None,
        model_name: str,
    ) -> dict[str, Any]:
        try:
            import langextract
        except ModuleNotFoundError as exc:
            raise LangExtractProviderError(
                "LangExtract provider is unavailable. Install the 'langextract' package."
            ) from exc

        if not hasattr(langextract, "extract"):
            raise LangExtractProviderError(
                "LangExtract package does not expose an 'extract' function"
            )

        payload = langextract.extract(
            document=document_text,
            schema=schema,
            prompt=prompt,
            model=model_name,
        )
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()  # pragma: no cover
        elif hasattr(payload, "dict"):
            payload = payload.dict()  # pragma: no cover

        if not isinstance(payload, dict):
            raise LangExtractProviderError("LangExtract returned an invalid payload")
        return payload


class Tier4StructuredExtractor:
    def __init__(
        self,
        cfg: Settings = settings,
        client: LangExtractClient | None = None,
        *,
        max_input_tokens: int = 12000,
        max_output_tokens: int = 4000,
        strict_schema: bool = True,
        model_name: str = "langextract",
        prompt_version: str = "tier4-extraction-v1",
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        self.cfg = cfg
        self.client = client or RuntimeLangExtractClient()
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.strict_schema = strict_schema
        self.model_name = model_name
        self.prompt_version = prompt_version

    def extract_structured(
        self,
        *,
        document_id: str,
        document_text: str,
        schema: dict[str, Any],
        prompt: str | None = None,
    ) -> StructuredExtractionEnvelope:
        started = time.perf_counter()
        schema_error = _validate_schema_contract(schema)
        if schema_error is not None:
            return self._error(
                code="invalid_schema",
                message="extraction schema is invalid",
                diagnostics=[schema_error],
                token_usage={"input_estimate": 0, "output_estimate": 0, "total": 0},
                started=started,
            )

        input_tokens = _estimate_input_tokens(
            document_text=document_text, schema=schema, prompt=prompt
        )
        if input_tokens > self.max_input_tokens:
            return self._error(
                code="token_budget_exceeded",
                message="estimated input tokens exceed Tier 4 budget",
                diagnostics=[
                    ValidationDiagnostic(
                        code="input_budget_exceeded",
                        message="preflight token estimate exceeded configured budget",
                        details={
                            "input_estimate": input_tokens,
                            "max_input_tokens": self.max_input_tokens,
                        },
                    )
                ],
                token_usage={
                    "input_estimate": input_tokens,
                    "output_estimate": 0,
                    "total": input_tokens,
                },
                started=started,
            )

        try:
            payload = self.client.extract(
                document_text=document_text,
                schema=schema,
                prompt=prompt,
                model_name=self.model_name,
            )
        except LangExtractProviderError as exc:
            return self._error(
                code="provider_error",
                message=str(exc),
                diagnostics=[],
                token_usage={
                    "input_estimate": input_tokens,
                    "output_estimate": 0,
                    "total": input_tokens,
                },
                started=started,
            )

        output_tokens = _estimate_tokens(json.dumps(payload, sort_keys=True))
        if output_tokens > self.max_output_tokens:
            return self._error(
                code="token_budget_exceeded",
                message="estimated output tokens exceed Tier 4 budget",
                diagnostics=[
                    ValidationDiagnostic(
                        code="output_budget_exceeded",
                        message="provider payload estimate exceeded configured output budget",
                        details={
                            "output_estimate": output_tokens,
                            "max_output_tokens": self.max_output_tokens,
                        },
                    )
                ],
                token_usage={
                    "input_estimate": input_tokens,
                    "output_estimate": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                started=started,
            )

        report = _validate_payload(
            payload=payload,
            schema=schema,
            document_text=document_text,
            strict_schema=self.strict_schema,
        )
        token_usage = {
            "input_estimate": input_tokens,
            "output_estimate": output_tokens,
            "total": input_tokens + output_tokens,
        }

        if report.fatal:
            return self._error(
                code="validation_failed",
                message="structured extraction failed validation",
                diagnostics=report.diagnostics,
                token_usage=token_usage,
                started=started,
            )

        return StructuredExtractionEnvelope(
            ok=True,
            result=StructuredExtractionResult(
                document_id=document_id,
                model=self.model_name,
                prompt_version=self.prompt_version,
                data=report.accepted_data,
                provenance=report.accepted_provenance,
                accepted_fields=sorted(report.accepted_data.keys()),
                rejected_fields=sorted(report.rejected_fields),
                diagnostics=report.diagnostics,
                token_usage=token_usage,
                latency_ms=_latency_ms(started),
            ),
            error=None,
        )

    def _error(
        self,
        *,
        code: str,
        message: str,
        diagnostics: list[ValidationDiagnostic],
        token_usage: dict[str, int],
        started: float,
    ) -> StructuredExtractionEnvelope:
        return StructuredExtractionEnvelope(
            ok=False,
            result=None,
            error=StructuredExtractionError(
                code=code,
                message=message,
                diagnostics=diagnostics,
                token_usage=token_usage,
                latency_ms=_latency_ms(started),
            ),
        )


def extract_structured(
    *,
    document_id: str,
    document_text: str,
    schema: dict[str, Any],
    prompt: str | None = None,
    cfg: Settings = settings,
    client: LangExtractClient | None = None,
) -> StructuredExtractionEnvelope:
    extractor = Tier4StructuredExtractor(cfg=cfg, client=client)
    return extractor.extract_structured(
        document_id=document_id,
        document_text=document_text,
        schema=schema,
        prompt=prompt,
    )


@dataclass(frozen=True)
class _ValidationReport:
    accepted_data: dict[str, Any]
    accepted_provenance: dict[str, FieldProvenance]
    rejected_fields: set[str]
    diagnostics: list[ValidationDiagnostic]
    fatal: bool


def _validate_schema_contract(schema: dict[str, Any]) -> ValidationDiagnostic | None:
    if not isinstance(schema, dict):
        return ValidationDiagnostic(
            code="schema_not_object",
            message="schema must be a JSON object",
        )

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return ValidationDiagnostic(
            code="schema_missing_properties",
            message="schema.properties must be a non-empty object",
        )

    required = schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, list):
        return ValidationDiagnostic(
            code="schema_required_invalid",
            message="schema.required must be a list when provided",
        )

    for field_name, field_schema in properties.items():
        if not isinstance(field_name, str) or not field_name:
            return ValidationDiagnostic(
                code="schema_field_invalid",
                message="schema property names must be non-empty strings",
            )
        if not isinstance(field_schema, dict):
            return ValidationDiagnostic(
                code="schema_field_invalid",
                message=f"schema for field '{field_name}' must be an object",
                field=field_name,
            )
        field_type = field_schema.get("type")
        if field_type not in {
            "string",
            "number",
            "integer",
            "boolean",
            "object",
            "array",
            "null",
        }:
            return ValidationDiagnostic(
                code="schema_field_type_invalid",
                message=f"field '{field_name}' has unsupported type '{field_type}'",
                field=field_name,
            )

    for field_name in required:
        if field_name not in properties:
            return ValidationDiagnostic(
                code="schema_required_unknown_field",
                message=f"required field '{field_name}' not found in schema.properties",
                field=field_name,
            )

    return None


def _validate_payload(
    *,
    payload: dict[str, Any],
    schema: dict[str, Any],
    document_text: str,
    strict_schema: bool,
) -> _ValidationReport:
    diagnostics: list[ValidationDiagnostic] = []
    rejected_fields: set[str] = set()

    data = payload.get("data")
    provenance = payload.get("provenance")
    if not isinstance(data, dict):
        diagnostics.append(
            ValidationDiagnostic(
                code="payload_data_invalid",
                message="payload.data must be an object",
            )
        )
        return _ValidationReport({}, {}, rejected_fields, diagnostics, True)
    if not isinstance(provenance, dict):
        diagnostics.append(
            ValidationDiagnostic(
                code="payload_provenance_invalid",
                message="payload.provenance must be an object",
            )
        )
        return _ValidationReport({}, {}, rejected_fields, diagnostics, True)

    properties = schema["properties"]
    required = set(schema.get("required", []) or [])
    accepted_data: dict[str, Any] = {}
    accepted_provenance: dict[str, FieldProvenance] = {}

    for field_name, value in data.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            if strict_schema:
                rejected_fields.add(field_name)
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unsupported_field",
                        message="field is not part of extraction schema",
                        field=field_name,
                    )
                )
            continue

        if not _matches_schema_type(value=value, schema_type=field_schema["type"]):
            rejected_fields.add(field_name)
            diagnostics.append(
                ValidationDiagnostic(
                    code="invalid_field_type",
                    message=(
                        f"field value does not match schema type '{field_schema['type']}'"
                    ),
                    field=field_name,
                    details={"value_type": type(value).__name__},
                )
            )
            continue

        raw_provenance = provenance.get(field_name)
        field_provenance = _parse_provenance(
            field_name=field_name,
            raw_provenance=raw_provenance,
            document_text=document_text,
        )
        if isinstance(field_provenance, ValidationDiagnostic):
            rejected_fields.add(field_name)
            diagnostics.append(field_provenance)
            continue

        accepted_data[field_name] = value
        accepted_provenance[field_name] = field_provenance

    fatal = False
    for field_name in sorted(required):
        if field_name not in accepted_data:
            fatal = True
            diagnostics.append(
                ValidationDiagnostic(
                    code="required_field_missing",
                    message="required schema field missing valid grounded value",
                    field=field_name,
                )
            )

    if not accepted_data:
        fatal = True
        diagnostics.append(
            ValidationDiagnostic(
                code="no_grounded_fields",
                message="no schema fields passed grounding validation",
            )
        )

    return _ValidationReport(
        accepted_data=accepted_data,
        accepted_provenance=accepted_provenance,
        rejected_fields=rejected_fields,
        diagnostics=diagnostics,
        fatal=fatal,
    )


def _parse_provenance(
    *,
    field_name: str,
    raw_provenance: Any,
    document_text: str,
) -> FieldProvenance | ValidationDiagnostic:
    if not isinstance(raw_provenance, dict):
        return ValidationDiagnostic(
            code="missing_provenance",
            message="field provenance is required",
            field=field_name,
        )

    start = raw_provenance.get("start_offset")
    end = raw_provenance.get("end_offset")
    if not isinstance(start, int) or not isinstance(end, int):
        return ValidationDiagnostic(
            code="invalid_provenance_offsets",
            message="provenance offsets must be integers",
            field=field_name,
        )
    if start < 0 or end <= start or end > len(document_text):
        return ValidationDiagnostic(
            code="invalid_provenance_offsets",
            message="provenance offsets are out of bounds",
            field=field_name,
            details={
                "start_offset": start,
                "end_offset": end,
                "text_len": len(document_text),
            },
        )

    extracted_text = document_text[start:end]
    claimed_text = raw_provenance.get("text")
    if claimed_text is not None and str(claimed_text) != extracted_text:
        return ValidationDiagnostic(
            code="provenance_text_mismatch",
            message="provenance text does not match document offsets",
            field=field_name,
            details={"claimed": str(claimed_text), "actual": extracted_text},
        )

    return FieldProvenance(start_offset=start, end_offset=end, text=extracted_text)


def _matches_schema_type(*, value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "null":
        return value is None
    return False


def _estimate_input_tokens(
    *,
    document_text: str,
    schema: dict[str, Any],
    prompt: str | None,
) -> int:
    schema_text = json.dumps(schema, sort_keys=True)
    prompt_text = prompt or ""
    return (
        _estimate_tokens(document_text)
        + _estimate_tokens(schema_text)
        + _estimate_tokens(prompt_text)
    )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4)))


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
