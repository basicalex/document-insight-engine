from __future__ import annotations

import json
import mimetypes
import time
from typing import Any

import streamlit as st

from frontend.client import ApiError, DocumentInsightApi
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


def main() -> None:
    st.set_page_config(
        page_title="Document Insight Chat",
        page_icon=":page_facing_up:",
        layout="wide",
    )
    initialize_session_state(st.session_state)
    if not st.session_state.get("runtime_bootstrapped", False):
        _refresh_runtime_data(silent=True)
        st.session_state["runtime_bootstrapped"] = True

    st.title("Document Insight Engine")
    st.caption("Upload a document, switch answer depth, and inspect grounded traces.")

    _render_sidebar()
    _render_status_bar()
    _render_runtime_readiness_banner()
    _render_observability_panel()
    _render_extraction_result()
    _render_chat_history()
    _handle_chat_prompt()


def _render_sidebar() -> None:
    st.sidebar.header("Controls")

    current_base_url = st.session_state["api_base_url"]
    base_url = st.sidebar.text_input("API base URL", value=current_base_url)
    base_url_value = str(base_url or "").strip()
    st.session_state["api_base_url"] = base_url_value or DEFAULT_API_BASE_URL

    mode_index = 0 if st.session_state["chat_mode"] == "fast" else 1
    selected_mode = st.sidebar.radio(
        "Answer mode",
        options=["fast", "deep"],
        index=mode_index,
        horizontal=True,
    )
    set_mode(st.session_state, str(selected_mode))

    document_id_value = st.sidebar.text_input(
        "Document ID (optional)",
        value=st.session_state["active_document_id"],
        placeholder="doc_123",
    )
    set_document_id(st.session_state, str(document_id_value or ""))

    session_id_value = st.sidebar.text_input(
        "Session ID (optional)",
        value=st.session_state["session_id"],
        placeholder="chat_session_1",
    )
    st.session_state["session_id"] = str(session_id_value or "").strip()

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

    st.sidebar.divider()
    st.sidebar.subheader("Runtime")
    if st.sidebar.button("Refresh runtime + metrics", use_container_width=True):
        _refresh_runtime_data(silent=False)

    st.sidebar.divider()
    st.sidebar.subheader("Structured extract")
    schema_text = st.sidebar.text_area(
        "Schema JSON",
        value=st.session_state["extract_schema_text"],
        height=180,
    )
    st.session_state["extract_schema_text"] = schema_text
    extract_prompt = st.sidebar.text_input(
        "Extract prompt",
        value=st.session_state["extract_prompt"],
        placeholder="Extract requested fields with provenance",
    )
    st.session_state["extract_prompt"] = str(extract_prompt or "")

    if st.sidebar.button(
        "Run structured extract",
        use_container_width=True,
        disabled=not st.session_state["active_document_id"],
    ):
        _run_structured_extract(
            document_id=st.session_state["active_document_id"],
            schema_text=st.session_state["extract_schema_text"],
            prompt=st.session_state["extract_prompt"],
        )

    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_chat(st.session_state)


def _render_status_bar() -> None:
    status_cols = st.columns(4)
    status_cols[0].metric("Mode", st.session_state["chat_mode"])

    active_document = st.session_state["active_document_id"] or "none"
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
        st.sidebar.success(f"Ingested {len(records)} document(s)")


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
        st.info(
            "Runtime status not loaded yet. Use 'Refresh runtime + metrics' in sidebar."
        )
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
                        )
                        status = str(status_payload.get("status", "")).strip().lower()
                        if status == "indexed":
                            response = _request_chat_response(
                                api=api,
                                prompt=prompt,
                                document_id=active_document_id,
                                session_id=st.session_state["session_id"] or None,
                                chat_mode=st.session_state["chat_mode"],
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
            error_text = _format_api_error(exc)
            append_assistant_message(
                st.session_state,
                content=error_text,
                mode=st.session_state["chat_mode"],
                insufficient_evidence=True,
                citations=[],
                trace=None,
            )
            st.error(error_text)
            return

        answer = str(response.get("answer", ""))
        mode = str(response.get("mode", st.session_state["chat_mode"]))
        insufficient_evidence = bool(response.get("insufficient_evidence", False))
        citations = response.get("citations") if isinstance(response, dict) else []
        trace = response.get("trace") if isinstance(response, dict) else None

        append_assistant_message(
            st.session_state,
            content=answer,
            mode=mode,
            insufficient_evidence=insufficient_evidence,
            citations=citations if isinstance(citations, list) else [],
            trace=trace if isinstance(trace, dict) else None,
        )

        _render_assistant_details(st.session_state["messages"][-1])


def _request_chat_response(
    *,
    api: DocumentInsightApi,
    prompt: str,
    document_id: str | None,
    session_id: str | None,
    chat_mode: str,
    placeholder: Any,
) -> dict[str, Any]:
    try:
        return _consume_streamed_response(
            events=api.ask_stream_events(
                question=prompt,
                mode=chat_mode,
                document_id=document_id,
                session_id=session_id,
            ),
            placeholder=placeholder,
            mode=chat_mode,
            document_id=document_id,
        )
    except ApiError as stream_error:
        if stream_error.status_code != 404:
            raise
        response = api.ask(
            question=prompt,
            mode=chat_mode,
            document_id=document_id,
            session_id=session_id,
        )
        placeholder.markdown(str(response.get("answer", "")))
        return response


def _is_document_not_ready_error(error: ApiError) -> bool:
    return error.status_code == 409 and error.code == "document_not_ready"


def _wait_for_document_indexed(
    *,
    api: DocumentInsightApi,
    document_id: str,
    placeholder: Any,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] = {}

    while time.monotonic() < deadline:
        latest = api.get_ingest_status(document_id=document_id)
        status = str(latest.get("status") or "unknown")
        message = str(latest.get("message") or "").strip()
        _upsert_ingest_history_record(
            document_id=document_id,
            status=status,
            message=message,
            file_path=str(latest.get("file_path") or "").strip(),
        )
        suffix = f": {message}" if message else ""
        placeholder.markdown(f"_Indexing status: {status}{suffix}_")

        normalized_status = status.lower()
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
