import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Planner Frontend", page_icon="🧠", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def post_json(url: str, payload: dict):
    """Send a POST request with JSON payload and return (ok, data_or_error)."""
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except ValueError:
        return False, "Backend returned a non-JSON response."

def get_json(url: str):
    """Send a GET request and return (ok, data_or_error)."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except ValueError:
        return False, "Backend returned a non-JSON response."

def render_sources(sources):
    """Render retrieved source chunks in expandable boxes."""
    if not sources:
        st.info("No sources returned.")
        return

    st.subheader("Sources")
    for i, source in enumerate(sources, start=1):
        document_id = source.get("document_id", "unknown")
        chunk_index = source.get("chunk_index", "unknown")
        text = source.get("text", "")

        with st.expander(f"Source {i} — {document_id} (chunk {chunk_index})"):
            st.markdown(f"**Document ID:** `{document_id}`")
            st.markdown(f"**Chunk Index:** `{chunk_index}`")
            st.write(text)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Settings")
backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.write(
    "This frontend interacts with a FastAPI backend for a planner-routed "
    "Retrieval-Augmented Generation (RAG) system."
)

st.sidebar.markdown("### Example Questions")
st.sidebar.code("How does MVCC work in PostgreSQL?")
st.sidebar.code("Compare PostgreSQL MVCC with SQL Server snapshot isolation.")
st.sidebar.code("What is the capital of France?")

# -----------------------------
# Main UI
# -----------------------------
st.title("🧠 AI Agent Platform - RAG Planner Demo")
st.write(
    "Test document ingestion and planner-routed question answering from your RAG backend."
)

tab_ask, tab_ingest, tab_docs = st.tabs(
    ["Ask Questions", "Ingest Document", "Knowledge Base"]
)
# -----------------------------
# Ask Tab
# -----------------------------
with tab_ask:
    st.header("Ask a Question")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Compare PostgreSQL MVCC with SQL Server snapshot isolation.",
        height=120,
    )

    if st.button("Ask", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {"question": question.strip()}

            with st.spinner("Querying backend..."):
                ok, result = post_json(f"{backend_url}/ask_routed", payload)

            if not ok:
                st.error(f"Request failed: {result}")
            else:
                answer = result.get("answer", "No answer returned.")
                route = result.get("route", "unknown")
                sources = result.get("sources", [])
                reason = result.get("reason") or result.get("planner_reason")

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Route")
                st.code(route)

                if reason:
                    st.subheader("Planner Reason")
                    st.write(reason)

                render_sources(sources)

# -----------------------------
# Ingest Tab
# -----------------------------
with tab_ingest:
    st.header("Ingest a Document")

    document_id = st.text_input(
        "Document ID",
        placeholder="Example: db_postgres"
    )
    document_text = st.text_area(
        "Document Text",
        placeholder="Paste the document content here...",
        height=250,
    )

    if st.button("Ingest", use_container_width=True):
        if not document_id.strip():
            st.warning("Please enter a document_id.")
        elif not document_text.strip():
            st.warning("Please enter document text.")
        else:
            payload = {
                "document_id": document_id.strip(),
                "text": document_text.strip(),
            }

            with st.spinner("Sending document to backend..."):
                ok, result = post_json(f"{backend_url}/ingest", payload)

            if not ok:
                st.error(f"Ingestion failed: {result}")
            else:
                status = result.get("status", "")

                if status == "ingested":
                    st.success(f"Document ingested successfully. Chunks added: {result.get('chunks_added', 0)}")
                elif status == "duplicate":
                    st.warning(result.get("reason", "Document already ingested."))
                elif status == "conflict":
                    st.warning(result.get("reason", "document_id already exists."))
                elif status == "no content":
                    st.warning("No content was extracted from the document.")
                else:
                    st.info("Received response from backend.")

                st.json(result)


# -----------------------------
# Knowledge Base Tab
# -----------------------------
with tab_docs:
    st.header("Knowledge Base")

    if st.button("Refresh Documents", use_container_width=True):
        st.rerun()

    ok, result = get_json(f"{backend_url}/documents")

    if not ok:
        st.error(f"Could not load documents: {result}")
    else:
        documents = result.get("documents", [])
        count = result.get("count", 0)

        st.subheader(f"Ingested Documents ({count})")

        if not documents:
            st.info("No documents have been ingested yet.")
        else:
            for doc in documents:
                document_id = doc.get("document_id", "unknown")
                chunks = doc.get("chunks", 0)
                source = doc.get("source")
                owner = doc.get("owner")
                preview = doc.get("preview") or "No preview available."

                with st.expander(f"{document_id} — {chunks} chunk(s)"):
                    st.markdown(f"**Document ID:** `{document_id}`")
                    st.markdown(f"**Chunks:** `{chunks}`")

                    if source:
                        st.markdown(f"**Source:** `{source}`")
                    if owner:
                        st.markdown(f"**Owner:** `{owner}`")

                    st.markdown("**Preview:**")
                    st.write(preview)

                    detail_ok, detail_result = get_json(f"{backend_url}/documents/{document_id}")

                    if detail_ok:
                        chunk_count = detail_result.get("chunk_count", 0)
                        st.markdown(f"**Chunk Details ({chunk_count})**")

                        for chunk in detail_result.get("chunks", []):
                            idx = chunk.get("chunk_index", "unknown")
                            text = chunk.get("text", "")

                            with st.expander(f"Chunk {idx}"):
                                st.write(text)
                    else:
                        st.warning(f"Could not load chunk details: {detail_result}")