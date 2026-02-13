from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    require_citations: bool = True
    max_latency_ms: int = 2500


@dataclass(frozen=True)
class EvaluationPrediction:
    case_id: str
    answer: str
    insufficient_evidence: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0


@dataclass(frozen=True)
class EvaluationThresholds:
    min_grounded_accuracy: float = 0.90
    max_hallucination_rate: float = 0.10
    min_citation_completeness: float = 0.90
    max_p95_latency_ms: int = 2500


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    passed: bool
    grounded: bool
    hallucinated: bool
    citation_complete: bool
    latency_ok: bool
    answer: str


@dataclass(frozen=True)
class EvaluationReport:
    totals: dict[str, Any]
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    threshold_checks: list[dict[str, Any]]
    overall_pass: bool
    cases: list[EvaluationCaseResult]


def evaluate_predictions(
    *,
    cases: list[EvaluationCase],
    predictions: list[EvaluationPrediction],
    thresholds: EvaluationThresholds,
) -> EvaluationReport:
    case_by_id = {item.case_id: item for item in cases}
    prediction_by_id = {item.case_id: item for item in predictions}

    results: list[EvaluationCaseResult] = []
    latencies: list[int] = []

    grounded_count = 0
    hallucinated_count = 0
    citation_required = 0
    citation_complete_count = 0

    for case_id, case in case_by_id.items():
        prediction = prediction_by_id.get(case_id)
        if prediction is None:
            result = EvaluationCaseResult(
                case_id=case_id,
                passed=False,
                grounded=False,
                hallucinated=True,
                citation_complete=False,
                latency_ok=False,
                answer="",
            )
            results.append(result)
            hallucinated_count += 1
            if case.require_citations:
                citation_required += 1
            continue

        latencies.append(max(0, int(prediction.latency_ms)))
        answer = prediction.answer.strip()
        grounded = _contains_all(answer=answer, needles=case.expected_substrings)
        hallucinated = _contains_any(answer=answer, needles=case.forbidden_substrings)

        citation_complete = True
        if case.require_citations and not prediction.insufficient_evidence:
            citation_required += 1
            citation_complete = _citations_complete(prediction.citations)
            if citation_complete:
                citation_complete_count += 1
            else:
                hallucinated = True

        latency_ok = int(prediction.latency_ms) <= int(case.max_latency_ms)
        passed = grounded and not hallucinated and citation_complete and latency_ok

        if grounded:
            grounded_count += 1
        if hallucinated:
            hallucinated_count += 1

        results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                passed=passed,
                grounded=grounded,
                hallucinated=hallucinated,
                citation_complete=citation_complete,
                latency_ok=latency_ok,
                answer=answer,
            )
        )

    total_cases = len(case_by_id)
    pass_count = sum(1 for item in results if item.passed)
    grounded_accuracy = _safe_rate(grounded_count, total_cases)
    hallucination_rate = _safe_rate(hallucinated_count, total_cases)
    citation_completeness = _safe_rate(citation_complete_count, citation_required)
    p95_latency_ms = _percentile_95(latencies)

    metrics = {
        "pass_rate": round(_safe_rate(pass_count, total_cases), 4),
        "grounded_accuracy": round(grounded_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "citation_completeness": round(citation_completeness, 4),
        "p95_latency_ms": p95_latency_ms,
    }

    checks = [
        _check(
            name="grounded_accuracy",
            value=grounded_accuracy,
            target=thresholds.min_grounded_accuracy,
            comparator=">=",
        ),
        _check(
            name="hallucination_rate",
            value=hallucination_rate,
            target=thresholds.max_hallucination_rate,
            comparator="<=",
        ),
        _check(
            name="citation_completeness",
            value=citation_completeness,
            target=thresholds.min_citation_completeness,
            comparator=">=",
        ),
        _check(
            name="p95_latency_ms",
            value=float(p95_latency_ms),
            target=float(thresholds.max_p95_latency_ms),
            comparator="<=",
        ),
    ]

    return EvaluationReport(
        totals={
            "cases": total_cases,
            "passed": pass_count,
            "failed": total_cases - pass_count,
        },
        metrics=metrics,
        thresholds=asdict(thresholds),
        threshold_checks=checks,
        overall_pass=all(item["pass"] for item in checks),
        cases=results,
    )


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate QA predictions against a corpus."
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to prediction JSON",
    )
    parser.add_argument(
        "--report-path",
        required=False,
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--assert-thresholds",
        action="store_true",
        help="Exit non-zero when thresholds fail",
    )
    args = parser.parse_args(argv)

    cases, default_thresholds = _load_corpus(Path(args.corpus))
    predictions = _load_predictions(Path(args.predictions))

    report = evaluate_predictions(
        cases=cases,
        predictions=predictions,
        thresholds=default_thresholds,
    )

    payload = {
        "totals": report.totals,
        "metrics": report.metrics,
        "thresholds": report.thresholds,
        "threshold_checks": report.threshold_checks,
        "overall_pass": report.overall_pass,
        "cases": [asdict(item) for item in report.cases],
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    if args.assert_thresholds and not report.overall_pass:
        return 1
    return 0


def _load_corpus(path: Path) -> tuple[list[EvaluationCase], EvaluationThresholds]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corpus must be a JSON object")

    case_rows = raw.get("cases", [])
    if not isinstance(case_rows, list):
        raise ValueError("corpus.cases must be an array")

    threshold_row = raw.get("thresholds", {})
    if not isinstance(threshold_row, dict):
        raise ValueError("corpus.thresholds must be an object")

    thresholds = EvaluationThresholds(
        min_grounded_accuracy=float(threshold_row.get("min_grounded_accuracy", 0.90)),
        max_hallucination_rate=float(threshold_row.get("max_hallucination_rate", 0.10)),
        min_citation_completeness=float(
            threshold_row.get("min_citation_completeness", 0.90)
        ),
        max_p95_latency_ms=int(threshold_row.get("max_p95_latency_ms", 2500)),
    )

    cases: list[EvaluationCase] = []
    for row in case_rows:
        if not isinstance(row, dict):
            continue
        cases.append(
            EvaluationCase(
                case_id=str(row.get("case_id", "")).strip(),
                question=str(row.get("question", "")).strip(),
                expected_substrings=[
                    str(item) for item in row.get("expected_substrings", [])
                ],
                forbidden_substrings=[
                    str(item) for item in row.get("forbidden_substrings", [])
                ],
                require_citations=bool(row.get("require_citations", True)),
                max_latency_ms=int(
                    row.get("max_latency_ms", thresholds.max_p95_latency_ms)
                ),
            )
        )

    return [item for item in cases if item.case_id], thresholds


def _load_predictions(path: Path) -> list[EvaluationPrediction]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("predictions must be a JSON object")

    rows = raw.get("predictions", [])
    if not isinstance(rows, list):
        raise ValueError("predictions.predictions must be an array")

    predictions: list[EvaluationPrediction] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        citations = row.get("citations", [])
        predictions.append(
            EvaluationPrediction(
                case_id=str(row.get("case_id", "")).strip(),
                answer=str(row.get("answer", "")).strip(),
                insufficient_evidence=bool(row.get("insufficient_evidence", False)),
                citations=list(citations) if isinstance(citations, list) else [],
                latency_ms=int(row.get("latency_ms", 0)),
            )
        )
    return [item for item in predictions if item.case_id]


def _contains_all(*, answer: str, needles: list[str]) -> bool:
    text = answer.lower()
    for needle in needles:
        if needle.strip().lower() not in text:
            return False
    return True


def _contains_any(*, answer: str, needles: list[str]) -> bool:
    text = answer.lower()
    for needle in needles:
        if needle.strip().lower() in text:
            return True
    return False


def _citations_complete(citations: list[dict[str, Any]]) -> bool:
    if not citations:
        return False

    for citation in citations:
        if not isinstance(citation, dict):
            return False
        chunk_id = str(citation.get("chunk_id", "")).strip()
        text = str(citation.get("text", "")).strip()
        start_offset = citation.get("start_offset")
        end_offset = citation.get("end_offset")
        if not chunk_id or not text:
            return False
        if start_offset is None or end_offset is None:
            return False
        if int(end_offset) < int(start_offset):
            return False
    return True


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _percentile_95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(max(0, int(value)) for value in values)
    rank = max(0, min(len(ordered) - 1, math_ceil(0.95 * len(ordered)) - 1))
    return int(ordered[rank])


def _check(
    *, name: str, value: float, target: float, comparator: str
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


def math_ceil(value: float) -> int:
    integer = int(value)
    if float(integer) == float(value):
        return integer
    return integer + 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
