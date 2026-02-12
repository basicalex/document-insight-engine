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

    uploaded_file = st.sidebar.file_uploader(
        "Upload a file",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
    )
    if st.sidebar.button(
        "Ingest", use_container_width=True, disabled=uploaded_file is None
    ):
        _handle_ingest(uploaded_file)

    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_chat(st.session_state)


def _render_status_bar() -> None:
    status_cols = st.columns(3)
    status_cols[0].metric("Mode", st.session_state["chat_mode"])

    active_document = st.session_state["active_document_id"] or "none"
    status_cols[1].metric("Document", active_document)

    status_cols[2].metric("Turns", str(len(st.session_state["messages"])))


def _handle_ingest(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return

    content_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
    if not content_type:
        content_type = "application/octet-stream"

    with st.sidebar:
        with st.spinner("Ingesting document..."):
            try:
                with DocumentInsightApi(
                    base_url=st.session_state["api_base_url"]
                ) as api:
                    response = api.ingest(
                        file_name=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                        content_type=content_type,
                    )
            except ApiError as exc:
                st.error(_format_api_error(exc))
                return

    document_id = str(response.get("document_id", "")).strip()
    if document_id:
        st.session_state["active_document_id"] = document_id

    st.session_state["ingest_history"].append(
        {
            "document_id": response.get("document_id"),
            "status": response.get("status"),
            "message": response.get("message"),
            "file_path": response.get("file_path"),
        }
    )
    st.sidebar.success(f"Ingested document: {document_id}")


def _render_chat_history() -> None:
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
        with st.spinner("Thinking..."):
            try:
                with DocumentInsightApi(
                    base_url=st.session_state["api_base_url"]
                ) as api:
                    response = api.ask(
                        question=prompt,
                        mode=st.session_state["chat_mode"],
                        document_id=st.session_state["active_document_id"] or None,
                        session_id=st.session_state["session_id"] or None,
                    )
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

        st.markdown(answer)
        _render_assistant_details(st.session_state["messages"][-1])


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


if __name__ == "__main__":
    main()
