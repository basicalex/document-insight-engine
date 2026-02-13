from __future__ import annotations

import asyncio

from src.api.state_store import InMemoryApiStateStore
from src.models.schemas import IngestResponse, IngestionStatus


def test_in_memory_state_store_idempotency_claim_and_replay() -> None:
    store = InMemoryApiStateStore(
        idempotency_ttl_seconds=60,
        ingestion_ttl_seconds=60,
        session_ttl_seconds=60,
        idempotency_claim_ttl_seconds=5,
    )

    async def scenario() -> None:
        claimed = await store.claim_idempotency_key("idem-1")
        assert claimed is True

        second_claim = await store.claim_idempotency_key("idem-1")
        assert second_claim is False

        response = IngestResponse(
            document_id="doc-1",
            file_path="data/uploads/doc-1.pdf",
            status=IngestionStatus.INDEXED,
            message="indexed",
        )
        await store.put_idempotency_response("idem-1", response)

        replay = await store.get_idempotency_response("idem-1")
        assert replay is not None
        assert replay.document_id == "doc-1"

    asyncio.run(scenario())


def test_in_memory_state_store_session_history_is_trimmed() -> None:
    store = InMemoryApiStateStore(
        idempotency_ttl_seconds=60,
        ingestion_ttl_seconds=60,
        session_ttl_seconds=60,
        idempotency_claim_ttl_seconds=5,
    )

    async def scenario() -> None:
        for index in range(5):
            await store.append_session_turn(
                key="session-1::*",
                question=f"q{index}",
                answer=f"a{index}",
                max_turns=3,
            )

        history = await store.get_session_history("session-1::*")
        assert [row["question"] for row in history] == ["q2", "q3", "q4"]

    asyncio.run(scenario())


def test_in_memory_state_store_ingestion_record_expires() -> None:
    store = InMemoryApiStateStore(
        idempotency_ttl_seconds=60,
        ingestion_ttl_seconds=1,
        session_ttl_seconds=60,
        idempotency_claim_ttl_seconds=5,
    )

    async def scenario() -> None:
        await store.put_ingestion_record(
            IngestResponse(
                document_id="doc-expire",
                file_path="data/uploads/doc-expire.pdf",
                status=IngestionStatus.UPLOADED,
                message="queued",
            )
        )
        assert await store.get_ingestion_record("doc-expire") is not None

        await asyncio.sleep(1.05)
        assert await store.get_ingestion_record("doc-expire") is None

    asyncio.run(scenario())
