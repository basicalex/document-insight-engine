from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from src.models.schemas import ChatResponse, Citation, IngestionStatus


@dataclass(frozen=True)
class ObservabilitySLOs:
    http_request_p95_ms: int = 1500
    retrieval_p95_ms: int = 1000
    generation_p95_ms: int = 2500
    insufficient_evidence_rate_max: float = 0.35
    citation_completeness_min: float = 0.90
    grounding_gap_rate_max: float = 0.20


@dataclass
class _RollingDistribution:
    max_samples: int = 5000

    def __post_init__(self) -> None:
        self._samples: deque[float] = deque(maxlen=max(32, self.max_samples))

    def observe(self, value: float) -> None:
        self._samples.append(float(max(0.0, value)))

    @property
    def count(self) -> int:
        return len(self._samples)

    def percentile(self, ratio: float) -> float:
        if not self._samples:
            return 0.0

        bounded = min(1.0, max(0.0, ratio))
        samples = sorted(self._samples)
        if len(samples) == 1:
            return float(samples[0])

        rank = max(0, min(len(samples) - 1, math.ceil(bounded * len(samples)) - 1))
        return float(samples[rank])


@dataclass
class ObservabilityRegistry:
    slos: ObservabilitySLOs = field(default_factory=ObservabilitySLOs)

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._http_requests_total = 0
        self._http_errors_total = 0
        self._http_route_totals: dict[str, int] = {}
        self._http_latency_ms = _RollingDistribution()

        self._ingestion_status_totals: dict[str, int] = {}
        self._ingestion_retries_total = 0
        self._ingestion_dead_letters_total = 0

        self._qa_requests_total = 0
        self._qa_insufficient_evidence_total = 0
        self._qa_citation_required_total = 0
        self._qa_citation_complete_total = 0
        self._qa_grounding_gap_total = 0
        self._qa_retrieval_latency_ms = _RollingDistribution()
        self._qa_generation_latency_ms = _RollingDistribution()

        self._trace_links: deque[dict[str, str]] = deque(maxlen=100)

    def record_http_request(
        self,
        *,
        route: str,
        method: str,
        status_code: int,
        latency_ms: int,
    ) -> None:
        route_label = route.strip() or "(unknown)"
        method_label = method.strip().upper() or "GET"
        status_label = f"{max(100, min(599, int(status_code))) // 100}xx"
        key = f"{method_label} {route_label} {status_label}"

        with self._lock:
            self._http_requests_total += 1
            if int(status_code) >= 400:
                self._http_errors_total += 1
            self._http_latency_ms.observe(latency_ms)
            self._http_route_totals[key] = self._http_route_totals.get(key, 0) + 1

    def record_ingestion_status(self, status: IngestionStatus | str) -> None:
        label = status.value if isinstance(status, IngestionStatus) else str(status)
        label = label.strip() or "unknown"
        with self._lock:
            self._ingestion_status_totals[label] = (
                self._ingestion_status_totals.get(label, 0) + 1
            )

    def record_ingestion_retry(self) -> None:
        with self._lock:
            self._ingestion_retries_total += 1

    def record_ingestion_dead_letter(self) -> None:
        with self._lock:
            self._ingestion_dead_letters_total += 1

    def record_chat_response(self, response: ChatResponse) -> None:
        retrieval_latencies: list[int] = []
        generation_latencies: list[int] = []
        if response.trace is not None:
            for event in response.trace.tool_calls:
                if event.latency_ms is None:
                    continue
                if event.stage == "retrieval":
                    retrieval_latencies.append(int(event.latency_ms))
                elif event.stage in {"generation", "agent"}:
                    generation_latencies.append(int(event.latency_ms))

        with self._lock:
            self._qa_requests_total += 1
            if response.insufficient_evidence:
                self._qa_insufficient_evidence_total += 1
            else:
                self._qa_citation_required_total += 1
                if _citations_complete(response.citations):
                    self._qa_citation_complete_total += 1
                else:
                    self._qa_grounding_gap_total += 1

            for latency in retrieval_latencies:
                self._qa_retrieval_latency_ms.observe(latency)
            for latency in generation_latencies:
                self._qa_generation_latency_ms.observe(latency)

    def record_trace_link(
        self, *, correlation_id: str | None, trace_id: str | None
    ) -> None:
        if not correlation_id or not trace_id:
            return

        now = int(time.time())
        with self._lock:
            self._trace_links.append(
                {
                    "correlation_id": correlation_id,
                    "trace_id": trace_id,
                    "timestamp_unix": str(now),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            qa_requests = self._qa_requests_total
            citation_required = self._qa_citation_required_total

            insufficient_rate = _safe_rate(
                self._qa_insufficient_evidence_total, qa_requests
            )
            citation_completeness = _safe_rate(
                self._qa_citation_complete_total, citation_required
            )
            grounding_gap_rate = _safe_rate(
                self._qa_grounding_gap_total, citation_required
            )

            report = {
                "http": {
                    "requests_total": self._http_requests_total,
                    "errors_total": self._http_errors_total,
                    "latency_ms": {
                        "p95": round(self._http_latency_ms.percentile(0.95), 2),
                        "samples": self._http_latency_ms.count,
                    },
                    "routes": dict(sorted(self._http_route_totals.items())),
                },
                "ingestion": {
                    "status_totals": dict(
                        sorted(self._ingestion_status_totals.items())
                    ),
                    "retries_total": self._ingestion_retries_total,
                    "dead_letters_total": self._ingestion_dead_letters_total,
                },
                "qa": {
                    "requests_total": qa_requests,
                    "insufficient_evidence_total": self._qa_insufficient_evidence_total,
                    "insufficient_evidence_rate": round(insufficient_rate, 4),
                    "citation_required_total": citation_required,
                    "citation_complete_total": self._qa_citation_complete_total,
                    "citation_completeness_rate": round(citation_completeness, 4),
                    "grounding_gap_total": self._qa_grounding_gap_total,
                    "grounding_gap_rate": round(grounding_gap_rate, 4),
                    "retrieval_latency_ms": {
                        "p95": round(self._qa_retrieval_latency_ms.percentile(0.95), 2),
                        "samples": self._qa_retrieval_latency_ms.count,
                    },
                    "generation_latency_ms": {
                        "p95": round(
                            self._qa_generation_latency_ms.percentile(0.95), 2
                        ),
                        "samples": self._qa_generation_latency_ms.count,
                    },
                },
                "trace_links_recent": list(self._trace_links),
            }

        report["slo"] = _evaluate_slos(report=report, slos=self.slos)
        return report

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP die_http_requests_total Total HTTP requests.",
            "# TYPE die_http_requests_total counter",
            f"die_http_requests_total {snapshot['http']['requests_total']}",
            "# HELP die_http_errors_total Total HTTP error responses.",
            "# TYPE die_http_errors_total counter",
            f"die_http_errors_total {snapshot['http']['errors_total']}",
            "# HELP die_http_request_latency_p95_ms HTTP request latency p95 in milliseconds.",
            "# TYPE die_http_request_latency_p95_ms gauge",
            f"die_http_request_latency_p95_ms {snapshot['http']['latency_ms']['p95']}",
            "# HELP die_qa_requests_total Total QA responses.",
            "# TYPE die_qa_requests_total counter",
            f"die_qa_requests_total {snapshot['qa']['requests_total']}",
            "# HELP die_qa_insufficient_evidence_rate Ratio of insufficient evidence responses.",
            "# TYPE die_qa_insufficient_evidence_rate gauge",
            f"die_qa_insufficient_evidence_rate {snapshot['qa']['insufficient_evidence_rate']}",
            "# HELP die_qa_citation_completeness_rate Citation completeness rate.",
            "# TYPE die_qa_citation_completeness_rate gauge",
            f"die_qa_citation_completeness_rate {snapshot['qa']['citation_completeness_rate']}",
            "# HELP die_qa_grounding_gap_rate Citation grounding gap rate.",
            "# TYPE die_qa_grounding_gap_rate gauge",
            f"die_qa_grounding_gap_rate {snapshot['qa']['grounding_gap_rate']}",
            "# HELP die_ingestion_retries_total Total ingestion retries.",
            "# TYPE die_ingestion_retries_total counter",
            f"die_ingestion_retries_total {snapshot['ingestion']['retries_total']}",
            "# HELP die_ingestion_dead_letters_total Total ingestion dead-letter moves.",
            "# TYPE die_ingestion_dead_letters_total counter",
            f"die_ingestion_dead_letters_total {snapshot['ingestion']['dead_letters_total']}",
        ]

        for label, value in snapshot["ingestion"]["status_totals"].items():
            escaped = _escape_label_value(label)
            lines.append(
                f'die_ingestion_status_total{{status="{escaped}"}} {int(value)}'
            )

        for key, value in snapshot["http"]["routes"].items():
            escaped = _escape_label_value(key)
            lines.append(f'die_http_route_total{{route="{escaped}"}} {int(value)}')

        return "\n".join(lines) + "\n"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _citations_complete(citations: list[Citation]) -> bool:
    if not citations:
        return False

    for citation in citations:
        if not citation.chunk_id.strip():
            return False
        if not citation.text.strip():
            return False
        if citation.start_offset is None or citation.end_offset is None:
            return False
        if citation.end_offset < citation.start_offset:
            return False
    return True


def _evaluate_slos(report: dict[str, Any], slos: ObservabilitySLOs) -> dict[str, Any]:
    checks = [
        _slo_check(
            name="http_request_p95_ms",
            value=float(report["http"]["latency_ms"]["p95"]),
            target=float(slos.http_request_p95_ms),
            comparator="<=",
        ),
        _slo_check(
            name="retrieval_p95_ms",
            value=float(report["qa"]["retrieval_latency_ms"]["p95"]),
            target=float(slos.retrieval_p95_ms),
            comparator="<=",
        ),
        _slo_check(
            name="generation_p95_ms",
            value=float(report["qa"]["generation_latency_ms"]["p95"]),
            target=float(slos.generation_p95_ms),
            comparator="<=",
        ),
        _slo_check(
            name="insufficient_evidence_rate",
            value=float(report["qa"]["insufficient_evidence_rate"]),
            target=float(slos.insufficient_evidence_rate_max),
            comparator="<=",
        ),
        _slo_check(
            name="citation_completeness_rate",
            value=float(report["qa"]["citation_completeness_rate"]),
            target=float(slos.citation_completeness_min),
            comparator=">=",
        ),
        _slo_check(
            name="grounding_gap_rate",
            value=float(report["qa"]["grounding_gap_rate"]),
            target=float(slos.grounding_gap_rate_max),
            comparator="<=",
        ),
    ]

    return {
        "overall": "pass" if all(item["pass"] for item in checks) else "fail",
        "checks": checks,
    }


def _slo_check(
    *,
    name: str,
    value: float,
    target: float,
    comparator: str,
) -> dict[str, Any]:
    if comparator == "<=":
        passed = value <= target
    elif comparator == ">=":
        passed = value >= target
    else:
        raise ValueError(f"unsupported comparator: {comparator}")

    return {
        "name": name,
        "value": round(value, 4),
        "target": round(target, 4),
        "comparator": comparator,
        "pass": passed,
    }


def _escape_label_value(value: str) -> str:
    return json.dumps(value)[1:-1]
