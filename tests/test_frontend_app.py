from __future__ import annotations

from frontend.progress import normalize_ingest_progress


def test_normalize_ingest_progress_returns_percent_and_message() -> None:
    percent, message = normalize_ingest_progress(
        {
            "stage": "embed",
            "processed_items": 3,
            "total_items": 10,
        }
    )

    assert percent == 30
    assert message == "embed: 3/10 - 30%"


def test_normalize_ingest_progress_clamps_overflow_percent() -> None:
    percent, message = normalize_ingest_progress(
        {
            "stage": "index",
            "processed_items": 25,
            "total_items": 20,
        }
    )

    assert percent == 100
    assert message == "index: 25/20 - 100%"


def test_normalize_ingest_progress_falls_back_to_stage_without_totals() -> None:
    percent, message = normalize_ingest_progress(
        {
            "stage": "parse",
            "processed_items": "not-a-number",
            "total_items": 0,
        }
    )

    assert percent is None
    assert message == "parse"
