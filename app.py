import streamlit as st
from typing import List, Dict, Any
import os
from pipeline import RAGPipeline

st.set_page_config(page_title="DocLens", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    html, body, [class*="css"] { font-size: 0.85rem !important; }
    p, li { font-size: 0.85rem !important; line-height: 1.4 !important; }
    h1 { font-size: 1.4rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.1rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 0.95rem !important; margin-bottom: 0.5rem !important; }
    [data-testid="stChatMessage"] { padding: 0.4rem 0.8rem !important; margin-bottom: 0.4rem !important; }
    .stButton>button { padding: 0.2rem 0.6rem !important; font-size: 0.8rem !important; }
    .stVerticalBlock { gap: 0.4rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []
if "rag_built" not in st.session_state:
    st.session_state.rag_built = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant chunks found."
    markdown_text = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        score = chunk.get("rerank_score", 0.0)
        markdown_text += f"<span style='font-size:0.75rem; color:gray;'>[{i}] Page {meta.get('page', '?')} | {meta.get('type', '?')} | score {score:+.2f}</span>\n\n"
        markdown_text += (
            f"<div style='font-size:0.8rem;'>{chunk['text']}</div>\n\n---\n\n"
        )
    return markdown_text


def stream_adapter(
    rag: RAGPipeline,
    prompt: str,
    history: List[Dict[str, str]],
    chunks_list: List[Dict[str, Any]],
):
    for event in rag.answer_stream(prompt, chat_history=history):
        if event["type"] == "chunks":
            chunks_list.extend(event["chunks"])
        elif event["type"] == "text":
            yield event["content"]
        elif event["type"] == "error":
            yield f"\n\n**Error:** {event['message']}"


with st.sidebar:
    st.title("📄 DocLens")
    st.caption("Upload a PDF and ask questions.")

    pdf_file = st.file_uploader(
        "Choose PDF", type=["pdf"], label_visibility="collapsed"
    )

    if pdf_file is None:
        if st.session_state.rag_built:
            st.session_state.rag_built = False
            st.session_state.current_file_name = None
    else:
        if st.session_state.current_file_name != pdf_file.name:
            st.session_state.rag_built = False
            st.session_state.current_file_name = pdf_file.name

    build_disabled = st.session_state.rag_built or pdf_file is None

    if st.button(
        "Build Database",
        type="primary",
        use_container_width=True,
        disabled=build_disabled,
    ):
        if pdf_file is not None:
            with st.status("Building database...", expanded=True) as status:
                rag = get_rag_pipeline()

                st.write("Processing PDF...")
                temp_path = f"temp_{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                try:
                    count = rag.build(temp_path)
                    st.session_state.rag_built = True
                    st.session_state.messages = []
                    st.session_state.latest_chunks = []
                    status.update(
                        label=f"Ready! {count} chunks.",
                        state="complete",
                        expanded=False,
                    )
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    st.rerun()
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    status.update(label=f"Error: {e}", state="error", expanded=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_chunks = []
        st.rerun()

    st.divider()
    if st.session_state.rag_built:
        st.success("🟢 Document loaded and ready.")
    else:
        st.info("⚪ Waiting for document...")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Chat")

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.rag_built:
            st.warning("Please build the database first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    rag = get_rag_pipeline()
                    current_chunks: List[Dict[str, Any]] = []
                    history = st.session_state.messages[:-1]

                    response = st.write_stream(
                        stream_adapter(rag, prompt, history, current_chunks)
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.latest_chunks = current_chunks
                    st.rerun()

with col2:
    st.subheader("🔎 Context")
    context_container = st.container(height=500)
    with context_container:
        if st.session_state.latest_chunks:
            st.markdown(
                format_chunks(st.session_state.latest_chunks), unsafe_allow_html=True
            )
        else:
            st.caption("Retrieved chunks will appear here.")
