from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from src.api.state_store import ApiStateStore, ApiStateStoreError
from src.api.telemetry import ObservabilityRegistry
from src.ingestion.orchestration import IngestionOrchestrator
from src.models.schemas import IngestResponse, IngestionStatus


logger = logging.getLogger(__name__)


QUEUE_SCHEMA_VERSION = 1


class IngestionQueueBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestionQueueJob:
    document_id: str
    file_path: str
    mime_type: str
    idempotency_key: str | None = None
    attempt: int = 0


class IngestionQueueBackend(Protocol):
    async def enqueue(self, job: IngestionQueueJob) -> None: ...

    async def dequeue(self, timeout_seconds: float) -> IngestionQueueJob | None: ...

    async def push_dead_letter(self, job: IngestionQueueJob, error: str) -> None: ...

    async def close(self) -> None: ...


@dataclass
class InMemoryIngestionQueueBackend:
    max_dead_letters: int = 500

    def __post_init__(self) -> None:
        self._queue: asyncio.Queue[IngestionQueueJob] = asyncio.Queue()
        self._dead_letters: list[dict[str, Any]] = []

    async def enqueue(self, job: IngestionQueueJob) -> None:
        await self._queue.put(job)

    async def dequeue(self, timeout_seconds: float) -> IngestionQueueJob | None:
        try:
            return await asyncio.wait_for(
                self._queue.get(), timeout=max(0.01, timeout_seconds)
            )
        except asyncio.TimeoutError:
            return None

    async def push_dead_letter(self, job: IngestionQueueJob, error: str) -> None:
        self._dead_letters.append({"job": asdict(job), "error": error})
        if len(self._dead_letters) > self.max_dead_letters:
            self._dead_letters = self._dead_letters[-self.max_dead_letters :]

    async def close(self) -> None:
        return None


class RedisIngestionQueueBackend:
    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        max_dead_letters: int,
    ) -> None:
        try:
            from redis.asyncio import Redis
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise IngestionQueueBackendError(
                "Redis ingestion queue requires the 'redis' package"
            ) from exc

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._jobs_key = f"{key_prefix}:jobs"
        self._dead_key = f"{key_prefix}:dead"
        self._max_dead_letters = max(1, int(max_dead_letters))

    async def enqueue(self, job: IngestionQueueJob) -> None:
        await self._client.rpush(self._jobs_key, _encode_job(job))

    async def dequeue(self, timeout_seconds: float) -> IngestionQueueJob | None:
        timeout = max(1, int(timeout_seconds))
        row = await self._client.blpop(self._jobs_key, timeout=timeout)
        if not row:
            return None
        _, payload = row
        job = _decode_job(payload)
        if job is None:
            logger.warning("Discarding malformed ingestion queue payload")
            return None
        return job

    async def push_dead_letter(self, job: IngestionQueueJob, error: str) -> None:
        payload = json.dumps(
            {
                "version": QUEUE_SCHEMA_VERSION,
                "job": asdict(job),
                "error": error,
            },
            ensure_ascii=False,
        )
        pipeline = self._client.pipeline(transaction=True)
        pipeline.lpush(self._dead_key, payload)
        pipeline.ltrim(self._dead_key, 0, self._max_dead_letters - 1)
        await pipeline.execute()

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()
            return
        self._client.close()


@dataclass
class IngestionWorkerPool:
    backend: IngestionQueueBackend
    orchestrator: IngestionOrchestrator
    state_store: ApiStateStore
    worker_concurrency: int
    max_retries: int
    retry_backoff_seconds: float
    poll_timeout_seconds: float
    ingest_timeout_seconds: int
    telemetry: ObservabilityRegistry | None = None

    def __post_init__(self) -> None:
        self._stop_event = asyncio.Event()
        self._workers: list[asyncio.Task[Any]] = []
        self._is_running = False
        self._pending_jobs = 0
        self._dead_letter_jobs = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def enqueue(
        self,
        *,
        document_id: str,
        file_path: Path,
        mime_type: str,
        idempotency_key: str | None,
    ) -> None:
        job = IngestionQueueJob(
            document_id=document_id,
            file_path=str(file_path),
            mime_type=mime_type,
            idempotency_key=idempotency_key,
            attempt=0,
        )
        await self.backend.enqueue(job)
        self._pending_jobs += 1

    def start(self) -> None:
        if self._is_running:
            return
        self._stop_event.clear()
        workers = max(1, int(self.worker_concurrency))
        self._workers = [
            asyncio.create_task(self._worker_loop(worker_id=index + 1))
            for index in range(workers)
        ]
        self._is_running = True

    async def stop(self) -> None:
        self._stop_event.set()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        self._is_running = False
        await self.backend.close()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "running": self._is_running,
            "workers": len(self._workers),
            "pending_jobs": max(0, self._pending_jobs),
            "dead_letter_jobs": max(0, self._dead_letter_jobs),
            "max_retries": self.max_retries,
        }

    async def _worker_loop(self, *, worker_id: int) -> None:
        while not self._stop_event.is_set():
            job: IngestionQueueJob | None = None
            try:
                job = await self.backend.dequeue(self.poll_timeout_seconds)
                if job is None:
                    continue
                self._pending_jobs = max(0, self._pending_jobs - 1)
                await self._process_job(job=job, worker_id=worker_id)
            except asyncio.CancelledError:  # pragma: no cover - task cancellation path
                return
            except Exception as exc:
                logger.exception("Ingestion worker %s crashed on job loop", worker_id)
                if job is None:
                    continue
                await self._handle_worker_loop_failure(
                    job=job,
                    worker_id=worker_id,
                    error=exc,
                )

    async def _process_job(self, *, job: IngestionQueueJob, worker_id: int) -> None:
        lock_key = f"queue:doc:{job.document_id}"
        lock_acquired = await self.state_store.claim_idempotency_key(lock_key)
        if not lock_acquired:
            await self._requeue(job=job, delay_seconds=self.retry_backoff_seconds)
            return

        try:
            existing = await self.state_store.get_ingestion_record(job.document_id)
            if existing is not None and existing.status in {
                IngestionStatus.INDEXED,
                IngestionStatus.PARTIAL,
                IngestionStatus.FAILED,
            }:
                return

            await self._persist_status(
                document_id=job.document_id,
                file_path=job.file_path,
                status=IngestionStatus.PROCESSING,
                message="ingestion is processing",
            )

            loop = asyncio.get_running_loop()
            process_task = loop.run_in_executor(
                None,
                self.orchestrator.process,
                job.document_id,
                Path(job.file_path),
                job.mime_type,
                job.idempotency_key,
            )

            while not process_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(process_task), timeout=0.5)
                except asyncio.TimeoutError:
                    current_record = self.orchestrator.get_record(job.document_id)
                    if current_record and current_record.progress:
                        await self._persist_status(
                            document_id=job.document_id,
                            file_path=job.file_path,
                            status=current_record.status,
                            message=f"ingestion is processing ({current_record.progress.stage})",
                            progress=current_record.progress,
                        )

            record = process_task.result()
            await self._persist_status(
                document_id=job.document_id,
                file_path=job.file_path,
                status=record.status,
                message=_ingestion_status_message(record.status, record.error_message),
                progress=record.progress,
            )

        except Exception as exc:
            if job.attempt < self.max_retries:
                next_attempt = job.attempt + 1
                await self._persist_status(
                    document_id=job.document_id,
                    file_path=job.file_path,
                    status=IngestionStatus.PROCESSING,
                    message=(
                        "ingestion retrying "
                        f"(attempt {next_attempt}/{self.max_retries})"
                    ),
                )
                await self._requeue(
                    job=replace(job, attempt=next_attempt),
                    delay_seconds=self.retry_backoff_seconds * max(1, next_attempt),
                )
                if self.telemetry is not None:
                    self.telemetry.record_ingestion_retry()
                return

            message = f"ingestion failed after retries: {exc}"
            await self._persist_status(
                document_id=job.document_id,
                file_path=job.file_path,
                status=IngestionStatus.FAILED,
                message=message,
            )
            await self.backend.push_dead_letter(job, error=str(exc))
            self._dead_letter_jobs += 1
            if self.telemetry is not None:
                self.telemetry.record_ingestion_dead_letter()
            logger.warning(
                "Ingestion worker %s moved document %s to dead letter: %s",
                worker_id,
                job.document_id,
                exc,
            )
        finally:
            await self.state_store.release_idempotency_key(lock_key)

    async def _requeue(self, *, job: IngestionQueueJob, delay_seconds: float) -> None:
        delay = max(0.0, float(delay_seconds))
        if delay > 0:
            await asyncio.sleep(delay)
        await self.backend.enqueue(job)
        self._pending_jobs += 1

    async def _persist_status(
        self,
        *,
        document_id: str,
        file_path: str,
        status: IngestionStatus,
        message: str,
        progress: Any = None,
    ) -> None:
        kwargs = {
            "document_id": document_id,
            "file_path": file_path,
            "status": status,
            "message": message,
        }
        if progress is not None:
            from src.models.schemas import IngestionProgress as SchemaProgress

            kwargs["progress"] = SchemaProgress(
                stage=progress.stage,
                processed_items=progress.processed_items,
                total_items=progress.total_items,
            )

        await self.state_store.put_ingestion_record(IngestResponse(**kwargs))
        if self.telemetry is not None:
            self.telemetry.record_ingestion_status(status)

    async def _handle_worker_loop_failure(
        self,
        *,
        job: IngestionQueueJob,
        worker_id: int,
        error: Exception,
    ) -> None:
        if job.attempt < self.max_retries:
            next_attempt = job.attempt + 1
            await self._persist_status(
                document_id=job.document_id,
                file_path=job.file_path,
                status=IngestionStatus.PROCESSING,
                message=(
                    f"ingestion retrying (attempt {next_attempt}/{self.max_retries})"
                ),
            )
            await self._requeue(
                job=replace(job, attempt=next_attempt),
                delay_seconds=self.retry_backoff_seconds * max(1, next_attempt),
            )
            if self.telemetry is not None:
                self.telemetry.record_ingestion_retry()
            return

        message = f"ingestion failed after retries: {error}"
        await self._persist_status(
            document_id=job.document_id,
            file_path=job.file_path,
            status=IngestionStatus.FAILED,
            message=message,
        )
        await self.backend.push_dead_letter(job, error=str(error))
        self._dead_letter_jobs += 1
        if self.telemetry is not None:
            self.telemetry.record_ingestion_dead_letter()
        logger.warning(
            "Ingestion worker %s moved document %s to dead letter after worker crash: %s",
            worker_id,
            job.document_id,
            error,
        )


def build_ingestion_queue_backend(
    *,
    backend: str,
    redis_url: str,
    key_prefix: str,
    dead_letter_max_items: int,
) -> IngestionQueueBackend:
    if backend == "memory":
        return InMemoryIngestionQueueBackend(max_dead_letters=dead_letter_max_items)

    try:
        return RedisIngestionQueueBackend(
            redis_url=redis_url,
            key_prefix=key_prefix,
            max_dead_letters=dead_letter_max_items,
        )
    except IngestionQueueBackendError:
        raise
    except Exception as exc:
        raise IngestionQueueBackendError(
            f"unable to connect ingestion queue backend: {exc}"
        ) from exc


def build_worker_pool(
    *,
    queue_backend: str,
    redis_url: str,
    key_prefix: str,
    dead_letter_max_items: int,
    state_store: ApiStateStore,
    orchestrator: IngestionOrchestrator,
    worker_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    poll_timeout_seconds: float,
    ingest_timeout_seconds: int,
    telemetry: ObservabilityRegistry | None = None,
) -> IngestionWorkerPool:
    backend = build_ingestion_queue_backend(
        backend=queue_backend,
        redis_url=redis_url,
        key_prefix=key_prefix,
        dead_letter_max_items=dead_letter_max_items,
    )
    return IngestionWorkerPool(
        backend=backend,
        orchestrator=orchestrator,
        state_store=state_store,
        worker_concurrency=worker_concurrency,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        ingest_timeout_seconds=ingest_timeout_seconds,
        telemetry=telemetry,
    )


def _encode_job(job: IngestionQueueJob) -> str:
    return json.dumps(
        {
            "version": QUEUE_SCHEMA_VERSION,
            "job": asdict(job),
        },
        ensure_ascii=False,
    )


def _decode_job(raw: str) -> IngestionQueueJob | None:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(decoded, dict):
        return None
    if int(decoded.get("version", 0)) != QUEUE_SCHEMA_VERSION:
        return None

    payload = decoded.get("job")
    if not isinstance(payload, dict):
        return None

    document_id = str(payload.get("document_id", "")).strip()
    file_path = str(payload.get("file_path", "")).strip()
    mime_type = str(payload.get("mime_type", "")).strip()
    idempotency_key = payload.get("idempotency_key")
    attempt = int(payload.get("attempt", 0))
    if not document_id or not file_path or not mime_type or attempt < 0:
        return None
    if idempotency_key is not None:
        idempotency_key = str(idempotency_key).strip() or None

    return IngestionQueueJob(
        document_id=document_id,
        file_path=file_path,
        mime_type=mime_type,
        idempotency_key=idempotency_key,
        attempt=attempt,
    )


def build_worker_pool_with_fallback(
    *,
    requested_backend: str,
    redis_url: str,
    key_prefix: str,
    dead_letter_max_items: int,
    state_store: ApiStateStore,
    orchestrator: IngestionOrchestrator,
    worker_concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    poll_timeout_seconds: float,
    ingest_timeout_seconds: int,
    telemetry: ObservabilityRegistry | None = None,
) -> IngestionWorkerPool:
    if requested_backend == "memory":
        return build_worker_pool(
            queue_backend="memory",
            redis_url=redis_url,
            key_prefix=key_prefix,
            dead_letter_max_items=dead_letter_max_items,
            state_store=state_store,
            orchestrator=orchestrator,
            worker_concurrency=worker_concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
            ingest_timeout_seconds=ingest_timeout_seconds,
            telemetry=telemetry,
        )

    try:
        return build_worker_pool(
            queue_backend="redis",
            redis_url=redis_url,
            key_prefix=key_prefix,
            dead_letter_max_items=dead_letter_max_items,
            state_store=state_store,
            orchestrator=orchestrator,
            worker_concurrency=worker_concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
            ingest_timeout_seconds=ingest_timeout_seconds,
            telemetry=telemetry,
        )
    except (IngestionQueueBackendError, ApiStateStoreError) as exc:
        if requested_backend == "redis":
            raise
        logger.warning(
            "Falling back to in-memory ingestion queue backend: %s",
            exc,
        )
        return build_worker_pool(
            queue_backend="memory",
            redis_url=redis_url,
            key_prefix=key_prefix,
            dead_letter_max_items=dead_letter_max_items,
            state_store=state_store,
            orchestrator=orchestrator,
            worker_concurrency=worker_concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
            ingest_timeout_seconds=ingest_timeout_seconds,
            telemetry=telemetry,
        )


def _ingestion_status_message(
    status: IngestionStatus, error_message: str | None
) -> str:
    if status == IngestionStatus.INDEXED:
        return "indexed and ready for retrieval"
    if status == IngestionStatus.PARTIAL:
        return error_message or "partially indexed; some stages failed"
    if status == IngestionStatus.FAILED:
        return error_message or "ingestion failed"
    if status == IngestionStatus.PROCESSING:
        return "ingestion is processing"
    return "queued for processing"
