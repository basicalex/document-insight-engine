from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.models.schemas import IngestResponse


STATE_SCHEMA_VERSION = 1


class ApiStateStoreError(RuntimeError):
    pass


class ApiStateStore(Protocol):
    async def get_idempotency_response(self, key: str) -> IngestResponse | None: ...

    async def claim_idempotency_key(self, key: str) -> bool: ...

    async def put_idempotency_response(
        self, key: str, response: IngestResponse
    ) -> None: ...

    async def release_idempotency_key(self, key: str) -> None: ...

    async def get_ingestion_record(self, document_id: str) -> IngestResponse | None: ...

    async def put_ingestion_record(self, response: IngestResponse) -> None: ...

    async def get_session_history(self, key: str) -> list[dict[str, str]]: ...

    async def append_session_turn(
        self,
        *,
        key: str,
        question: str,
        answer: str,
        max_turns: int,
    ) -> None: ...

    async def close(self) -> None: ...


def _encode_model_payload(response: IngestResponse) -> str:
    return json.dumps(
        {
            "version": STATE_SCHEMA_VERSION,
            "payload": response.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )


def _decode_model_payload(raw: str) -> IngestResponse | None:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(decoded, dict):
        return None
    if int(decoded.get("version", 0)) != STATE_SCHEMA_VERSION:
        return None

    payload = decoded.get("payload")
    if not isinstance(payload, dict):
        return None

    try:
        return IngestResponse.model_validate(payload)
    except Exception:
        return None


@dataclass
class InMemoryApiStateBackend:
    idempotency_responses: dict[str, tuple[str, float | None]] = field(
        default_factory=dict
    )
    idempotency_claims: dict[str, float] = field(default_factory=dict)
    ingestion_records: dict[str, tuple[str, float | None]] = field(default_factory=dict)
    session_histories: dict[str, tuple[list[dict[str, str]], float | None]] = field(
        default_factory=dict
    )


@dataclass
class InMemoryApiStateStore:
    idempotency_ttl_seconds: int
    ingestion_ttl_seconds: int
    session_ttl_seconds: int
    idempotency_claim_ttl_seconds: int
    backend: InMemoryApiStateBackend = field(default_factory=InMemoryApiStateBackend)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def get_idempotency_response(self, key: str) -> IngestResponse | None:
        async with self._lock:
            self._prune_expired()
            stored = self.backend.idempotency_responses.get(key)
            if stored is None:
                return None
            return _decode_model_payload(stored[0])

    async def claim_idempotency_key(self, key: str) -> bool:
        async with self._lock:
            self._prune_expired()
            if key in self.backend.idempotency_claims:
                return False
            if key in self.backend.idempotency_responses:
                return False

            self.backend.idempotency_claims[key] = self._expires_at(
                self.idempotency_claim_ttl_seconds
            ) or (time.monotonic() + 1.0)
            return True

    async def put_idempotency_response(
        self, key: str, response: IngestResponse
    ) -> None:
        async with self._lock:
            self._prune_expired()
            self.backend.idempotency_responses[key] = (
                _encode_model_payload(response),
                self._expires_at(self.idempotency_ttl_seconds),
            )
            self.backend.idempotency_claims.pop(key, None)

    async def release_idempotency_key(self, key: str) -> None:
        async with self._lock:
            self.backend.idempotency_claims.pop(key, None)

    async def get_ingestion_record(self, document_id: str) -> IngestResponse | None:
        async with self._lock:
            self._prune_expired()
            stored = self.backend.ingestion_records.get(document_id)
            if stored is None:
                return None
            return _decode_model_payload(stored[0])

    async def put_ingestion_record(self, response: IngestResponse) -> None:
        async with self._lock:
            self._prune_expired()
            self.backend.ingestion_records[response.document_id] = (
                _encode_model_payload(response),
                self._expires_at(self.ingestion_ttl_seconds),
            )

    async def get_session_history(self, key: str) -> list[dict[str, str]]:
        async with self._lock:
            self._prune_expired()
            stored = self.backend.session_histories.get(key)
            if stored is None:
                return []
            turns = stored[0]
            return [dict(item) for item in turns]

    async def append_session_turn(
        self,
        *,
        key: str,
        question: str,
        answer: str,
        max_turns: int,
    ) -> None:
        async with self._lock:
            self._prune_expired()
            turns: list[dict[str, str]] = []
            existing = self.backend.session_histories.get(key)
            if existing is not None:
                turns = [dict(item) for item in existing[0]]
            turns.append({"question": question, "answer": answer})
            if max_turns > 0 and len(turns) > max_turns:
                turns = turns[-max_turns:]
            self.backend.session_histories[key] = (
                turns,
                self._expires_at(self.session_ttl_seconds),
            )

    async def close(self) -> None:
        return None

    def _prune_expired(self) -> None:
        now = time.monotonic()
        self._prune_map(self.backend.idempotency_responses, now)
        self._prune_map(self.backend.ingestion_records, now)
        self._prune_map(self.backend.session_histories, now)

        expired_claims = [
            key
            for key, expires_at in self.backend.idempotency_claims.items()
            if expires_at <= now
        ]
        for key in expired_claims:
            self.backend.idempotency_claims.pop(key, None)

    def _prune_map(
        self, target: dict[str, tuple[Any, float | None]], now: float
    ) -> None:
        expired_keys = [
            key
            for key, (_, expires_at) in target.items()
            if expires_at is not None and expires_at <= now
        ]
        for key in expired_keys:
            target.pop(key, None)

    def _expires_at(self, ttl_seconds: int) -> float | None:
        if ttl_seconds <= 0:
            return None
        return time.monotonic() + float(ttl_seconds)


class RedisApiStateStore:
    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        idempotency_ttl_seconds: int,
        ingestion_ttl_seconds: int,
        session_ttl_seconds: int,
        idempotency_claim_ttl_seconds: int,
    ) -> None:
        try:
            from redis.asyncio import Redis
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ApiStateStoreError(
                "Redis state store requires the 'redis' package"
            ) from exc

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix.strip() or "die:state"
        self._idempotency_ttl_seconds = max(0, int(idempotency_ttl_seconds))
        self._ingestion_ttl_seconds = max(0, int(ingestion_ttl_seconds))
        self._session_ttl_seconds = max(0, int(session_ttl_seconds))
        self._idempotency_claim_ttl_seconds = max(1, int(idempotency_claim_ttl_seconds))

    async def get_idempotency_response(self, key: str) -> IngestResponse | None:
        raw = await self._client.get(self._idempotency_response_key(key))
        if not raw:
            return None
        return _decode_model_payload(raw)

    async def claim_idempotency_key(self, key: str) -> bool:
        claimed = await self._client.set(
            self._idempotency_claim_key(key),
            "1",
            nx=True,
            ex=self._idempotency_claim_ttl_seconds,
        )
        return bool(claimed)

    async def put_idempotency_response(
        self, key: str, response: IngestResponse
    ) -> None:
        payload = _encode_model_payload(response)
        response_key = self._idempotency_response_key(key)
        claim_key = self._idempotency_claim_key(key)
        pipeline = self._client.pipeline(transaction=True)
        if self._idempotency_ttl_seconds > 0:
            pipeline.set(response_key, payload, ex=self._idempotency_ttl_seconds)
        else:
            pipeline.set(response_key, payload)
        pipeline.delete(claim_key)
        await pipeline.execute()

    async def release_idempotency_key(self, key: str) -> None:
        await self._client.delete(self._idempotency_claim_key(key))

    async def get_ingestion_record(self, document_id: str) -> IngestResponse | None:
        raw = await self._client.get(self._ingestion_record_key(document_id))
        if not raw:
            return None
        return _decode_model_payload(raw)

    async def put_ingestion_record(self, response: IngestResponse) -> None:
        key = self._ingestion_record_key(response.document_id)
        payload = _encode_model_payload(response)
        if self._ingestion_ttl_seconds > 0:
            await self._client.set(key, payload, ex=self._ingestion_ttl_seconds)
            return
        await self._client.set(key, payload)

    async def get_session_history(self, key: str) -> list[dict[str, str]]:
        rows = await self._client.lrange(self._session_history_key(key), 0, -1)
        history: list[dict[str, str]] = []
        for row in rows:
            turn = _decode_session_turn(row)
            if turn is not None:
                history.append(turn)
        return history

    async def append_session_turn(
        self,
        *,
        key: str,
        question: str,
        answer: str,
        max_turns: int,
    ) -> None:
        turns_limit = max(1, int(max_turns))
        payload = _encode_session_turn(question=question, answer=answer)
        history_key = self._session_history_key(key)
        pipeline = self._client.pipeline(transaction=True)
        pipeline.rpush(history_key, payload)
        pipeline.ltrim(history_key, -turns_limit, -1)
        if self._session_ttl_seconds > 0:
            pipeline.expire(history_key, self._session_ttl_seconds)
        await pipeline.execute()

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()
            return
        self._client.close()

    def _idempotency_response_key(self, key: str) -> str:
        return f"{self._key_prefix}:idem:response:{key}"

    def _idempotency_claim_key(self, key: str) -> str:
        return f"{self._key_prefix}:idem:claim:{key}"

    def _ingestion_record_key(self, document_id: str) -> str:
        return f"{self._key_prefix}:ingest:{document_id}"

    def _session_history_key(self, key: str) -> str:
        return f"{self._key_prefix}:session:{key}"


def _encode_session_turn(*, question: str, answer: str) -> str:
    return json.dumps(
        {
            "version": STATE_SCHEMA_VERSION,
            "question": question,
            "answer": answer,
        },
        ensure_ascii=False,
    )


def _decode_session_turn(raw: str) -> dict[str, str] | None:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(decoded, dict):
        return None
    if int(decoded.get("version", 0)) != STATE_SCHEMA_VERSION:
        return None

    question = str(decoded.get("question", "")).strip()
    answer = str(decoded.get("answer", "")).strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer}
