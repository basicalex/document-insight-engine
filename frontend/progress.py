from __future__ import annotations

from typing import Any


def normalize_ingest_progress(progress: Any) -> tuple[int | None, str | None]:
    if not isinstance(progress, dict):
        return None, None

    stage = str(progress.get("stage") or "unknown")
    processed = _safe_non_negative_int(progress.get("processed_items"))
    total = _safe_non_negative_int(progress.get("total_items"))
    if total > 0:
        percent = min(100, int((processed / total) * 100))
        return percent, f"{stage}: {processed}/{total} - {percent}%"

    if stage:
        return None, stage
    return None, None


def _safe_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)
