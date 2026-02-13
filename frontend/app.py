from __future__ import annotations

import mimetypes
from typing import Any

import streamlit as st

from frontend.client import ApiError, DocumentInsightApi
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

    st.title("Document Insight Engine")
    st.caption("Upload a document, switch answer depth, and inspect grounded traces.")

    _render_sidebar()
    _render_status_bar()
    _render_chat_history()
    _handle_chat_prompt()


def _render_sidebar() -> None:
    st.sidebar.header("Controls")

    current_base_url = st.session_state["api_base_url"]
    base_url = st.sidebar.text_input("API base URL", value=current_base_url)
    st.session_state["api_base_url"] = base_url.strip() or DEFAULT_API_BASE_URL

    mode_index = 0 if st.session_state["chat_mode"] == "fast" else 1
    selected_mode = st.sidebar.radio(
        "Answer mode",
        options=["fast", "deep"],
        index=mode_index,
        horizontal=True,
    )
    set_mode(st.session_state, selected_mode)

    document_id_value = st.sidebar.text_input(
        "Document ID (optional)",
        value=st.session_state["active_document_id"],
        placeholder="doc_123",
    )
    set_document_id(st.session_state, document_id_value)

    session_id_value = st.sidebar.text_input(
        "Session ID (optional)",
        value=st.session_state["session_id"],
        placeholder="chat_session_1",
    )
    st.session_state["session_id"] = session_id_value.strip()

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

    active_document = st.session_state["active_document_id"] or "none"
    status_cols[1].metric("Document", active_document)

    status_cols[2].metric("Turns", str(len(st.session_state["messages"])))
    latest_status = (
        st.session_state["ingest_history"][-1]["status"]
        if st.session_state["ingest_history"]
        else "n/a"
    )
    status_cols[3].metric("Ingest status", latest_status)


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
        st.session_state["ingest_history"].append(item)

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

    st.session_state["ingest_history"].append(
        {
            "document_id": response.get("document_id"),
            "status": response.get("status"),
            "message": response.get("message"),
            "file_path": response.get("file_path"),
        }
    )


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
        try:
            with DocumentInsightApi(base_url=st.session_state["api_base_url"]) as api:
                try:
                    response = _consume_streamed_response(
                        events=api.ask_stream_events(
                            question=prompt,
                            mode=st.session_state["chat_mode"],
                            document_id=st.session_state["active_document_id"] or None,
                            session_id=st.session_state["session_id"] or None,
                        ),
                        placeholder=answer_placeholder,
                        mode=st.session_state["chat_mode"],
                        document_id=st.session_state["active_document_id"] or None,
                    )
                except ApiError as stream_error:
                    if stream_error.status_code == 404:
                        response = api.ask(
                            question=prompt,
                            mode=st.session_state["chat_mode"],
                            document_id=st.session_state["active_document_id"] or None,
                            session_id=st.session_state["session_id"] or None,
                        )
                        answer_placeholder.markdown(str(response.get("answer", "")))
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


if __name__ == "__main__":
    main()
