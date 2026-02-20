from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any
from fastapi.testclient import TestClient

from src.models.schemas import IngestionStatus

from tests.test_api import _make_services, _client_with_services, StubOrchestrator

def test_get_recent_ingests_returns_list(tmp_path: Path) -> None:
    orchestrator = StubOrchestrator(status=IngestionStatus.INDEXED)
    services = _make_services(tmp_path=tmp_path, orchestrator=orchestrator)

    with _client_with_services(services) as client:
        # Upload a doc
        client.post(
            "/upload",
            files={"file": ("invoice1.pdf", b"%PDF-1.4\ninvoice1", "application/pdf")},
        )
        
        # Upload another doc
        client.post(
            "/upload",
            files={"file": ("invoice2.pdf", b"%PDF-1.4\ninvoice2", "application/pdf")},
        )
        
        response = client.get("/ingests?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "count" in data
        
        assert data["count"] == 2
        assert len(data["documents"]) == 2
        
        # Verify chronological ordering - last uploaded should be first in results
        # assuming the state store handles recent tracking properly
        assert "invoice2" in data["documents"][0]["file_path"]
        assert "invoice1" in data["documents"][1]["file_path"]

