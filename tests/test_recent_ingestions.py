from __future__ import annotations

import asyncio
from src.api.state_store import InMemoryApiStateStore
from src.models.schemas import IngestResponse, IngestionStatus

def test_in_memory_state_store_recent_ingestions() -> None:
    store = InMemoryApiStateStore(
        idempotency_ttl_seconds=60,
        ingestion_ttl_seconds=60,
        session_ttl_seconds=60,
        idempotency_claim_ttl_seconds=5,
    )

    async def scenario() -> None:
        for i in range(10):
            await store.put_ingestion_record(
                IngestResponse(
                    document_id=f"doc-{i}",
                    file_path=f"data/uploads/doc-{i}.pdf",
                    status=IngestionStatus.INDEXED,
                    message="indexed",
                )
            )
            await asyncio.sleep(0.01)

        recent = await store.get_recent_ingestions(limit=3)
        assert len(recent) == 3
        assert recent[0].document_id == "doc-9"
        assert recent[1].document_id == "doc-8"
        assert recent[2].document_id == "doc-7"

    asyncio.run(scenario())
