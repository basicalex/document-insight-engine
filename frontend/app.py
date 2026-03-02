from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import streamlit as st

from frontend.client import ApiError, DocumentInsightApi
from frontend.progress import normalize_ingest_progress
from frontend.readiness import classify_runtime_readiness
from frontend.state import (
    DEFAULT_API_BASE_URL,
    append_assistant_message,
    append_user_message,
    clear_chat,
    initialize_session_state,
    set_document_id,
    set_mode,
)


def initialize_session_state_with_url_params() -> None:
    initialize_session_state(st.session_state)

    # Try to load doc from query params
    if "doc" in st.query_params and not st.session_state.get("active_document_id"):
        st.session_state["active_document_id"] = st.query_params["doc"]

    if not st.session_state.get("runtime_bootstrapped", False):
        _refresh_runtime_data(silent=True)
        # Fetch initial ingest history
        try:
            with DocumentInsightApi(base_url=st.session_state["api_base_url"]) as api:
                recent = api.get_recent_ingestions(limit=20)
                if isinstance(recent, dict) and "documents" in recent:
                    for doc in reversed(recent["documents"]):  # Keep chronological
                        _upsert_ingest_history_record(
                            document_id=doc.get("document_id", ""),
                            status=doc.get("status", ""),
                            message=doc.get("message", ""),
                            file_path=doc.get("file_path", ""),
                        )
        except Exception:
            pass  # Fail silently on initial load if backend is down
        st.session_state["runtime_bootstrapped"] = True


def main() -> None:
    st.set_page_config(
        page_title="Document Insight Chat",
        page_icon=":page_facing_up:",
        layout="wide",
    )
    initialize_session_state_with_url_params()
    _apply_compact_sidebar_spacing()

    st.title("Document Insight Engine")
    st.caption("Upload a document, switch answer depth, and inspect grounded traces.")

    _render_sidebar()
    _render_status_bar()
    _render_runtime_readiness_banner()
    _render_observability_panel()
    _render_chat_history()
    _handle_chat_prompt()


def _apply_compact_sidebar_spacing() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0.25rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 0.25rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    st.sidebar.header("Controls")

    current_base_url = st.session_state["api_base_url"]
    base_url = st.sidebar.text_input("API base URL", value=current_base_url)
    base_url_value = str(base_url or "").strip()
    st.session_state["api_base_url"] = base_url_value or DEFAULT_API_BASE_URL

    mode_options = ["fast", "deep-lite", "deep"]
    current_mode = str(st.session_state.get("chat_mode", "fast"))
    mode_index = (
        mode_options.index(current_mode) if current_mode in mode_options else 0
    )
    selected_mode = st.sidebar.radio(
        "Answer mode",
        options=mode_options,
        index=mode_index,
        horizontal=True,
    )
    set_mode(st.session_state, str(selected_mode))

    model_backend = st.sidebar.radio(
        "Model backend",
        options=["auto", "api", "local"],
        index=["auto", "api", "local"].index(
            str(st.session_state.get("model_backend", "auto"))
            if str(st.session_state.get("model_backend", "auto"))
            in {"auto", "api", "local"}
            else "auto"
        ),
        horizontal=True,
        help="auto uses API model when key is available (with local fallback on failures), otherwise local model. api forces API only.",
    )
    st.session_state["model_backend"] = str(model_backend)

    st.session_state["api_model"] = "gemini-2.5-flash"
    st.sidebar.caption("API model is pinned to gemini-2.5-flash")

    api_key = st.sidebar.text_input(
        "API key (optional)",
        value=st.session_state["api_key"],
        type="password",
        help="Used only for chat calls in this UI session",
    )
    st.session_state["api_key"] = str(api_key or "").strip()

    # Prepare options for the document selectbox
    # The history contains dicts with 'document_id', 'file_path', 'status', etc.
    history = st.session_state.get("ingest_history", [])
    recent_docs = []
    seen = set()
    labels_by_doc_id: dict[str, str] = {}
    for item in reversed(history):
        if not isinstance(item, dict):
            continue

        doc_id = str(item.get("document_id") or "").strip()
        if not doc_id:
            continue

        file_path = str(item.get("file_path") or "").strip()
        file_name = Path(file_path).name if file_path else ""
        if file_name.startswith(f"{doc_id}_"):
            file_name = file_name[len(doc_id) + 1 :]
        labels_by_doc_id[doc_id] = file_name or doc_id

        if doc_id not in seen:
            seen.add(doc_id)
            recent_docs.append(doc_id)

    # Include the active document if it's not in the history
    active_doc = st.session_state.get("active_document_id", "")
    if active_doc and active_doc not in seen:
        recent_docs.insert(0, active_doc)

    options = [""] + recent_docs
    index = options.index(active_doc) if active_doc in options else 0

    def _doc_option_label(doc_id: str) -> str:
        if not doc_id:
            return "All indexed documents"
        title = labels_by_doc_id.get(doc_id, doc_id)
        short_id = doc_id[:8]
        return f"{title} · {short_id}"

    document_id_value = st.sidebar.selectbox(
        "Document scope",
        options=options,
        index=index,
        format_func=_doc_option_label,
    )

    if document_id_value != active_doc:
        set_document_id(st.session_state, str(document_id_value or ""))
        if document_id_value:
            st.query_params["doc"] = str(document_id_value)
        else:
            if "doc" in st.query_params:
                del st.query_params["doc"]

    uploaded_files = st.sidebar.file_uploader(
        "Upload file(s)",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True,
    )
    if st.sidebar.button(
        "Upload + Ingest", use_container_width=True, disabled=not uploaded_files
    ):
        _handle_upload(uploaded_files)

    if st.session_state["active_document_id"] and st.sidebar.button(
        "Refresh status", use_container_width=True
    ):
        _refresh_ingest_status(st.session_state["active_document_id"])

    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_chat(st.session_state)


def _render_status_bar() -> None:
    status_cols = st.columns(4)
    status_cols[0].metric("Mode", st.session_state["chat_mode"])

    active_document_id = str(st.session_state["active_document_id"] or "").strip()
    active_document = active_document_id or "none"
    status_cols[1].metric("Document", active_document)

    status_cols[2].metric("Turns", str(len(st.session_state["messages"])))
    latest_ingest_status = _latest_ingest_status_for_document(active_document)
    runtime = st.session_state.get("runtime_health") or {}
    runtime_status = str(
        runtime.get("readiness", {}).get("overall")
        or runtime.get("status")
        or latest_ingest_status
    )
    status_cols[3].metric("Runtime", runtime_status)

    if not active_document_id:
        st.info(
            "Document scope is set to all indexed documents. Answers and citations may mix content across files. "
            "Select a specific document in the sidebar for focused retrieval."
        )


def _handle_upload(uploaded_files: list[Any]) -> None:
    if not uploaded_files:
        return

    files: list[tuple[str, bytes, str]] = []
    for uploaded_file in uploaded_files:
        content_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
        if not content_type:
            content_type = "application/octet-stream"
        files.append((uploaded_file.name, uploaded_file.getvalue(), content_type))

    with st.sidebar:
        with st.spinner("Uploading and ingesting..."):
            try:
                with DocumentInsightApi(
                    base_url=st.session_state["api_base_url"]
                ) as api:
                    response = api.upload(files=files)
            except ApiError as exc:
                st.error(_format_api_error(exc))
                return

    records = _normalize_upload_response(response)
    for item in records:
        _upsert_ingest_history_record(
            document_id=str(item.get("document_id") or "").strip(),
            status=str(item.get("status") or "").strip(),
            message=str(item.get("message") or "").strip(),
            file_path=str(item.get("file_path") or "").strip(),
        )

    if records:
        latest = records[-1]
        document_id = str(latest.get("document_id", "")).strip()
        if document_id:
            st.session_state["active_document_id"] = document_id
            st.query_params["doc"] = document_id

            status = str(latest.get("status") or "").strip().lower()
            if status in {"uploaded", "processing"}:
                _monitor_uploaded_document_progress(document_id=document_id)
        st.sidebar.success(f"Ingested {len(records)} document(s)")


def _monitor_uploaded_document_progress(
    *,
    document_id: str,
    timeout_seconds: float = 180.0,
) -> None:
    if not document_id:
        return

    with st.sidebar:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
    status_placeholder.markdown("_Waiting for indexing to start..._")

    try:
        with DocumentInsightApi(base_url=st.session_state["api_base_url"]) as api:
            latest = _wait_for_document_indexed(
                api=api,
                document_id=document_id,
                placeholder=status_placeholder,
                progress_placeholder=progress_placeholder,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=1.0,
            )
    except ApiError as exc:
        progress_placeholder.empty()
        status_placeholder.warning(
            "Upload succeeded, but status polling failed: " + _format_api_error(exc)
        )
        return

    status = str(latest.get("status") or "").strip().lower()
    progress_placeholder.empty()
    if status == "indexed":
        status_placeholder.success(
            "Ingestion completed: indexed and ready for retrieval."
        )
    elif status in {"partial", "failed"}:
        status_placeholder.warning(f"Ingestion finished with status: {status}.")
    else:
        status_placeholder.info(
            "Ingestion is still running. You can continue and refresh status as needed."
        )


def _refresh_ingest_status(document_id: str) -> None:
    with st.sidebar:
        with st.spinner("Refreshing ingestion status..."):
            try:
                with DocumentInsightApi(
                    base_url=st.session_state["api_base_url"]
                ) as api:
                    response = api.get_ingest_status(document_id=document_id)
            except ApiError as exc:
                st.error(_format_api_error(exc))
                return

    _upsert_ingest_history_record(
        document_id=str(response.get("document_id") or "").strip(),
        status=str(response.get("status") or "").strip(),
        message=str(response.get("message") or "").strip(),
        file_path=str(response.get("file_path") or "").strip(),
    )


def _refresh_runtime_data(*, silent: bool) -> None:
    try:
        with DocumentInsightApi(base_url=st.session_state["api_base_url"]) as api:
            health = api.healthz()
            metrics_text = api.metrics()
    except ApiError as exc:
        if not silent:
            st.sidebar.error(_format_api_error(exc))
        return

    st.session_state["runtime_health"] = health
    st.session_state["metrics_text"] = metrics_text
    if not silent:
        st.sidebar.success("Runtime and metrics refreshed")


def _run_structured_extract(*, document_id: str, schema_text: str, prompt: str) -> None:
    try:
        schema_payload = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"Schema JSON is invalid: {exc.msg}")
        return

    if not isinstance(schema_payload, dict):
        st.sidebar.error("Schema JSON must be an object")
        return

    with st.sidebar:
        with st.spinner("Running structured extraction..."):
            try:
                with DocumentInsightApi(
                    base_url=st.session_state["api_base_url"]
                ) as api:
                    result = api.extract(
                        document_id=document_id,
                        extraction_schema=schema_payload,
                        prompt=prompt,
                    )
            except ApiError as exc:
                st.error(_format_api_error(exc))
                return

    st.session_state["last_extract_result"] = result
    st.sidebar.success("Structured extraction completed")


def _render_runtime_readiness_banner() -> None:
    runtime = st.session_state.get("runtime_health") or {}
    if not runtime:
        st.info("Runtime status not loaded yet.")
        return

    chat_mode = st.session_state.get("chat_mode", "fast")
    blocking_issues, optional_notices = classify_runtime_readiness(
        runtime=runtime,
        chat_mode=str(chat_mode),
    )

    if blocking_issues:
        st.error("Blocking issues for current action: " + "; ".join(blocking_issues))

    if optional_notices:
        st.info("Optional capability notices: " + "; ".join(optional_notices))

    if not blocking_issues and not optional_notices:
        st.success("Runtime readiness looks good for active capabilities.")


def _render_observability_panel() -> None:
    with st.expander("Runtime observability", expanded=False):
        runtime = st.session_state.get("runtime_health") or {}
        observability = runtime.get("observability")
        if isinstance(observability, dict):
            st.json(observability)
        else:
            st.caption("Observability snapshot is not available yet.")

        metrics_text = str(st.session_state.get("metrics_text") or "").strip()
        if metrics_text:
            st.code(metrics_text, language="text")
        else:
            st.caption("Metrics payload is not available yet.")


def _render_extraction_result() -> None:
    payload = st.session_state.get("last_extract_result")
    if not isinstance(payload, dict):
        return

    with st.expander("Structured extraction result", expanded=False):
        st.json(payload)


def _render_chat_history() -> None:
    if st.session_state["ingest_history"]:
        with st.expander("Recent ingest events", expanded=False):
            st.json(st.session_state["ingest_history"][-5:])

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                backend_label = str(message.get("backend_label") or "").strip()
                if backend_label:
                    st.caption(backend_label)
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_assistant_details(message)


def _handle_chat_prompt() -> None:
    prompt = st.chat_input("Ask a question about the active document")
    if not prompt:
        return

    append_user_message(st.session_state, prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        progress_placeholder = st.empty()
        answer_placeholder.markdown("_Starting response..._")
        active_document_id = st.session_state["active_document_id"] or None
        try:
            with DocumentInsightApi(base_url=st.session_state["api_base_url"]) as api:
                try:
                    response = _request_chat_response(
                        api=api,
                        prompt=prompt,
                        document_id=active_document_id,
                        session_id=st.session_state["session_id"] or None,
                        chat_mode=st.session_state["chat_mode"],
                        model_backend=st.session_state.get("model_backend"),
                        api_key=st.session_state.get("api_key"),
                        api_model=st.session_state.get("api_model"),
                        placeholder=answer_placeholder,
                    )
                except ApiError as chat_error:
                    if active_document_id and _is_document_not_ready_error(chat_error):
                        answer_placeholder.markdown(
                            "_Document is still indexing. Waiting for readiness..._"
                        )
                        status_payload = _wait_for_document_indexed(
                            api=api,
                            document_id=active_document_id,
                            placeholder=answer_placeholder,
                            progress_placeholder=progress_placeholder,
                        )
                        status = str(status_payload.get("status", "")).strip().lower()
                        if status == "indexed":
                            response = _request_chat_response(
                                api=api,
                                prompt=prompt,
                                document_id=active_document_id,
                                session_id=st.session_state["session_id"] or None,
                                chat_mode=st.session_state["chat_mode"],
                                model_backend=st.session_state.get("model_backend"),
                                api_key=st.session_state.get("api_key"),
                                api_model=st.session_state.get("api_model"),
                                placeholder=answer_placeholder,
                            )
                        else:
                            status_text = str(status_payload.get("status") or "unknown")
                            raise ApiError(
                                status_code=chat_error.status_code,
                                code=chat_error.code,
                                message=(
                                    "document status is "
                                    f"{status_text}; ready status is indexed"
                                ),
                                correlation_id=chat_error.correlation_id,
                                details=chat_error.details,
                            ) from chat_error
                    else:
                        raise
        except ApiError as exc:
            progress_placeholder.empty()
            error_text = _format_api_error(exc)
            append_assistant_message(
                st.session_state,
                content=error_text,
                mode=st.session_state["chat_mode"],
                insufficient_evidence=False,
                citations=[],
                trace=None,
            )
            st.error(error_text)
            return

        progress_placeholder.empty()
        answer = str(response.get("answer", ""))
        mode = str(response.get("mode", st.session_state["chat_mode"]))
        insufficient_evidence = bool(response.get("insufficient_evidence", False))
        citations = response.get("citations") if isinstance(response, dict) else []
        trace = response.get("trace") if isinstance(response, dict) else None
        backend_label = _infer_response_backend_label(
            response=response,
            model_backend=st.session_state.get("model_backend"),
            api_key=st.session_state.get("api_key"),
            api_model=st.session_state.get("api_model"),
        )

        if backend_label:
            st.caption(backend_label)

        append_assistant_message(
            st.session_state,
            content=answer,
            mode=mode,
            insufficient_evidence=insufficient_evidence,
            citations=citations if isinstance(citations, list) else [],
            trace=trace if isinstance(trace, dict) else None,
            backend_label=backend_label,
        )

        _render_assistant_details(st.session_state["messages"][-1])


def _events_with_wait_feedback(*, events: Any, mode: str) -> Iterator[dict[str, Any]]:
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    completed = threading.Event()

    def _pump() -> None:
        try:
            for event in events:
                event_queue.put(("event", event))
        except Exception as exc:
            event_queue.put(("error", exc))
        finally:
            completed.set()

    threading.Thread(target=_pump, name="chat-stream-pump", daemon=True).start()

    if mode == "deep":
        waiting_message = "Deep mode is reasoning through the document"
    elif mode == "deep-lite":
        waiting_message = "Deep-lite mode is synthesizing grounded evidence"
    else:
        waiting_message = "Generating answer"
    started = time.monotonic()
    first_event_received = False
    last_reported_second = -1

    while not completed.is_set() or not event_queue.empty():
        try:
            item_type, payload = event_queue.get(timeout=1.0)
        except queue.Empty:
            if first_event_received:
                continue
            elapsed_seconds = int(time.monotonic() - started)
            if elapsed_seconds <= 0 or elapsed_seconds == last_reported_second:
                continue
            last_reported_second = elapsed_seconds
            yield {
                "type": "status",
                "phase": "generation",
                "message": f"{waiting_message} ({elapsed_seconds}s elapsed)",
            }
            continue

        if item_type == "error":
            raise payload
        first_event_received = True
        if isinstance(payload, dict):
            yield payload


def _normalized_model_backend(model_backend: str | None) -> str:
    normalized = str(model_backend or "").strip().lower()
    if normalized in {"auto", "api", "local"}:
        return normalized
    return "auto"


def _should_retry_with_local_model(
    *,
    error: ApiError,
    chat_mode: str,
    model_backend: str | None,
) -> bool:
    if chat_mode not in {"deep", "deep-lite"}:
        return False
    if _normalized_model_backend(model_backend) == "local":
        return False

    retryable_codes = {
        "provider_not_configured",
        "provider_auth_failed",
        "provider_rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "provider_malformed_response",
        "ask_timeout",
    }
    if error.code in retryable_codes:
        return True
    return error.status_code in {429, 503, 504}


def _backend_marker_from_trace(
    *,
    response: dict[str, Any],
    api_model: str | None,
) -> str | None:
    trace = response.get("trace")
    if not isinstance(trace, dict):
        return None

    model_name = str(trace.get("model") or "").strip().lower()
    if not model_name:
        return None

    normalized_api_model = str(api_model or "").strip().lower()
    if "gemini" in model_name:
        return "api"
    if normalized_api_model and model_name == normalized_api_model:
        return "api"
    return "local"


def _infer_response_backend_label(
    *,
    response: dict[str, Any],
    model_backend: str | None,
    api_key: str | None,
    api_model: str | None,
) -> str:
    explicit_marker = str(response.get("_response_backend") or "").strip().lower()
    if explicit_marker == "local-fallback":
        return "Backend: local (fallback)"
    if explicit_marker == "api":
        return "Backend: api"
    if explicit_marker == "local":
        return "Backend: local"

    traced = _backend_marker_from_trace(response=response, api_model=api_model)
    if traced == "api":
        return "Backend: api"
    if traced == "local":
        return "Backend: local"

    requested_backend = _normalized_model_backend(model_backend)
    if requested_backend == "api":
        return "Backend: api"
    if requested_backend == "local":
        return "Backend: local"

    if str(api_key or "").strip():
        return "Backend: api (auto)"
    return "Backend: local (auto)"


def _request_chat_response(
    *,
    api: DocumentInsightApi,
    prompt: str,
    document_id: str | None,
    session_id: str | None,
    chat_mode: str,
    model_backend: str | None,
    api_key: str | None,
    api_model: str | None,
    placeholder: Any,
) -> dict[str, Any]:
    try:
        response = _consume_streamed_response(
            events=_events_with_wait_feedback(
                events=api.ask_stream_events(
                    question=prompt,
                    mode=chat_mode,
                    document_id=document_id,
                    session_id=session_id,
                    model_backend=model_backend,
                    api_key=api_key,
                    api_model=api_model,
                ),
                mode=chat_mode,
            ),
            placeholder=placeholder,
            mode=chat_mode,
            document_id=document_id,
        )
        return response
    except ApiError as stream_error:
        if stream_error.status_code == 404:
            response = api.ask(
                question=prompt,
                mode=chat_mode,
                document_id=document_id,
                session_id=session_id,
                model_backend=model_backend,
                api_key=api_key,
                api_model=api_model,
            )
            placeholder.markdown(str(response.get("answer", "")))
            return response

        if _should_retry_with_local_model(
            error=stream_error,
            chat_mode=chat_mode,
            model_backend=model_backend,
        ):
            placeholder.markdown(
                "_API deep model unavailable; retrying with local model..._"
            )
            response = _consume_streamed_response(
                events=_events_with_wait_feedback(
                    events=api.ask_stream_events(
                        question=prompt,
                        mode=chat_mode,
                        document_id=document_id,
                        session_id=session_id,
                        model_backend="local",
                        api_key=None,
                        api_model=None,
                    ),
                    mode=chat_mode,
                ),
                placeholder=placeholder,
                mode=chat_mode,
                document_id=document_id,
            )
            response["_response_backend"] = "local-fallback"
            return response

        raise


def _is_document_not_ready_error(error: ApiError) -> bool:
    return error.status_code == 409 and error.code == "document_not_ready"


def _wait_for_document_indexed(
    *,
    api: DocumentInsightApi,
    document_id: str,
    placeholder: Any,
    progress_placeholder: Any | None = None,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] = {}

    while time.monotonic() < deadline:
        latest = api.get_ingest_status(document_id=document_id)
        status = str(latest.get("status") or "unknown")
        normalized_status = status.lower()
        message = str(latest.get("message") or "").strip()
        progress = latest.get("progress")

        _upsert_ingest_history_record(
            document_id=document_id,
            status=status,
            message=message,
            file_path=str(latest.get("file_path") or "").strip(),
        )

        progress_percent, progress_message = normalize_ingest_progress(progress)
        if progress_message:
            placeholder.markdown(f"_Indexing status: {status} ({progress_message})_")
        else:
            suffix = f": {message}" if message else ""
            placeholder.markdown(f"_Indexing status: {status}{suffix}_")

        if progress_placeholder is not None:
            if progress_percent is not None:
                progress_placeholder.progress(
                    progress_percent,
                    text=f"Indexing progress: {progress_percent}%",
                )
            elif normalized_status == "indexed":
                progress_placeholder.progress(100, text="Indexing complete")
            elif normalized_status in {"failed", "partial"}:
                progress_placeholder.empty()

        if normalized_status == "indexed":
            return latest
        if normalized_status in {"failed", "partial"}:
            return latest
        time.sleep(poll_interval_seconds)

    return latest


def _consume_streamed_response(
    *,
    events: Any,
    placeholder: Any,
    mode: str,
    document_id: str | None,
) -> dict[str, Any]:
    chunks: list[str] = []
    final_payload: dict[str, Any] | None = None

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).strip().lower()
        if event_type == "status":
            status_message = str(event.get("message", "")).strip()
            if status_message:
                placeholder.markdown(f"_{status_message}_")
            continue

        if event_type == "token":
            delta = str(event.get("delta", ""))
            if not delta:
                continue
            chunks.append(delta)
            placeholder.markdown("".join(chunks))
            continue

        if event_type == "final":
            payload = event.get("response")
            if isinstance(payload, dict):
                final_payload = payload

    if final_payload is None:
        final_payload = {
            "answer": "".join(chunks).strip(),
            "mode": mode,
            "document_id": document_id,
            "insufficient_evidence": False,
            "citations": [],
            "trace": None,
        }

    answer = str(final_payload.get("answer", "")).strip()
    if not answer:
        answer = "I could not produce a response. Please retry."
        final_payload["answer"] = answer
    placeholder.markdown(answer)
    return final_payload


def _render_assistant_details(message: dict[str, Any]) -> None:
    if message.get("insufficient_evidence"):
        st.info("The model reported insufficient evidence for a grounded answer.")

    citations = message.get("citations") or []
    if citations:
        with st.expander("Citations", expanded=False):
            st.json(citations)

    trace = message.get("trace")
    if trace:
        with st.expander("Trace", expanded=False):
            st.json(trace)


def _format_api_error(error: ApiError) -> str:
    if error.correlation_id:
        return (
            f"{error.message} (code={error.code}, "
            f"status={error.status_code}, correlation_id={error.correlation_id})"
        )
    return f"{error.message} (code={error.code}, status={error.status_code})"


def _normalize_upload_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    documents = response.get("documents")
    if isinstance(documents, list):
        normalized: list[dict[str, Any]] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "document_id": item.get("document_id"),
                    "status": item.get("status"),
                    "message": item.get("message"),
                    "file_path": item.get("file_path"),
                }
            )
        return normalized

    return [
        {
            "document_id": response.get("document_id"),
            "status": response.get("status"),
            "message": response.get("message"),
            "file_path": response.get("file_path"),
        }
    ]


def _latest_ingest_status_for_document(document_id: str) -> str:
    normalized = str(document_id or "").strip()
    history = st.session_state.get("ingest_history") or []
    if not normalized or not isinstance(history, list):
        if history:
            return str(history[-1].get("status") or "n/a")
        return "n/a"

    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("document_id") or "").strip() == normalized:
            return str(item.get("status") or "n/a")

    if history:
        return str(history[-1].get("status") or "n/a")
    return "n/a"


def _upsert_ingest_history_record(
    *,
    document_id: str,
    status: str,
    message: str,
    file_path: str,
) -> None:
    if not document_id:
        return

    history = st.session_state.get("ingest_history")
    if not isinstance(history, list):
        history = []
        st.session_state["ingest_history"] = history

    record = {
        "document_id": document_id,
        "status": status,
        "message": message,
        "file_path": file_path,
    }

    for index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("document_id") or "").strip() == document_id:
            history[index] = record
            return

    history.append(record)


if __name__ == "__main__":
    main()
