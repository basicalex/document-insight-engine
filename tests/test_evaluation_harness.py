from __future__ import annotations

import json
from pathlib import Path

from src.evals.harness import (
    EvaluationPrediction,
    evaluate_predictions,
    run_cli,
    _load_corpus,
    _load_predictions,
)


def test_evaluation_harness_passes_curated_fixture() -> None:
    corpus_path = Path("tests/data/eval/qa_corpus.json")
    predictions_path = Path("tests/data/eval/qa_predictions.json")

    cases, thresholds = _load_corpus(corpus_path)
    predictions = _load_predictions(predictions_path)
    report = evaluate_predictions(
        cases=cases,
        predictions=predictions,
        thresholds=thresholds,
    )

    assert report.overall_pass is True
    assert report.metrics["grounded_accuracy"] == 1.0
    assert report.metrics["hallucination_rate"] == 0.0
    assert report.metrics["citation_completeness"] == 1.0


def test_evaluation_harness_fails_thresholds_on_hallucination(tmp_path: Path) -> None:
    corpus_path = Path("tests/data/eval/qa_corpus.json")
    predictions_path = tmp_path / "predictions.json"
    predictions_payload = json.loads(
        Path("tests/data/eval/qa_predictions.json").read_text(encoding="utf-8")
    )
    predictions_payload["predictions"][0]["answer"] = "The total due is 9999 USD."
    predictions_path.write_text(
        json.dumps(predictions_payload, indent=2),
        encoding="utf-8",
    )

    exit_code = run_cli(
        [
            "--corpus",
            str(corpus_path),
            "--predictions",
            str(predictions_path),
            "--assert-thresholds",
        ]
    )

    assert exit_code == 1


def test_evaluation_harness_handles_missing_prediction() -> None:
    cases, thresholds = _load_corpus(Path("tests/data/eval/qa_corpus.json"))
    report = evaluate_predictions(
        cases=cases,
        predictions=[
            EvaluationPrediction(
                case_id="invoice_total_due",
                answer="The total due is 1234.00 USD.",
                insufficient_evidence=False,
                latency_ms=300,
                citations=[
                    {
                        "chunk_id": "chunk-1",
                        "text": "Total Due: 1234.00 USD",
                        "start_offset": 11,
                        "end_offset": 31,
                    }
                ],
            )
        ],
        thresholds=thresholds,
    )

    assert report.totals["failed"] == 1
    assert report.metrics["hallucination_rate"] >= 0.5
