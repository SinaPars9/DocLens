import os
import json
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
import glob
from pipeline import RAGPipeline


def safe_remove(path: str) -> None:
    """Safely remove a file, ignoring Windows file-lock errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass  # File is still locked by Camelot, Windows will clean it up later


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

FEEDBACK_FILE = "feedback_log.json"


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


def save_feedback(
    question: str, claim: Dict[str, Any], verdict: str, reason: str = ""
) -> None:

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "llm_sentence": claim.get("sentence", ""),
        "status": claim.get("status", ""),
        "source_snippet": claim.get("source_snippet", ""),
        "user_verdict": verdict,
        "reason": reason,
    }

    data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    data.append(feedback_entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant chunks found."
    markdown_text = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        score = chunk.get("rerank_score", 0.0)
        chunk_type = meta.get("type", "?")

        markdown_text += f"<span style='font-size:0.75rem; color:gray;'>[{i}] Page {meta.get('page', '?')} | {chunk_type} | score {score:+.2f}</span>\n\n"

        if chunk_type == "table":
            markdown_text += f"{chunk['text']}\n\n---\n\n"
        else:
            markdown_text += (
                f"<div style='font-size:0.8rem;'>{chunk['text']}</div>\n\n---\n\n"
            )

    return markdown_text


def render_feedback_buttons(
    claim: Dict[str, Any], question: str, msg_idx: int, claim_idx: int
) -> None:

    state_key = f"feedback_state_{msg_idx}_{claim_idx}"

    if state_key not in st.session_state:
        st.session_state[state_key] = "idle"

    if st.session_state[state_key] == "idle":
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "👍 Correct",
                key=f"correct_{msg_idx}_{claim_idx}",
                use_container_width=True,
            ):
                save_feedback(question, claim, "Correct")
                st.session_state[state_key] = "done"
                st.rerun()
        with col2:
            if st.button(
                "👎 Incorrect",
                key=f"incorrect_{msg_idx}_{claim_idx}",
                use_container_width=True,
            ):
                st.session_state[state_key] = "asking_reason"
                st.rerun()

    elif st.session_state[state_key] == "asking_reason":
        reason = st.text_input(
            "Why is this incorrect? (e.g., Wrong year, missing context)",
            key=f"reason_{msg_idx}_{claim_idx}",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Submit",
                key=f"submit_{msg_idx}_{claim_idx}",
                type="primary",
                use_container_width=True,
            ):
                if reason.strip():
                    save_feedback(question, claim, "Incorrect", reason)
                    st.session_state[state_key] = "done"
                    st.rerun()
                else:
                    st.warning("Please enter a reason.")
        with col2:
            if st.button(
                "Cancel", key=f"cancel_{msg_idx}_{claim_idx}", use_container_width=True
            ):
                st.session_state[state_key] = "idle"
                st.rerun()

    elif st.session_state[state_key] == "done":
        st.success("✅ Feedback saved to Golden Dataset.")


def render_verification_ui(
    v_data: Dict[str, Any],
    msg_idx: int,
    question: str,
    standalone_q: str,
    expanded_qs: List[str],
    dev_mode: bool,
) -> None:

    if dev_mode:
        if standalone_q:
            with st.expander("🔍 Query Inspector (Dev Mode)", expanded=True):
                st.info(f"**Target Query:** {standalone_q}")
                if expanded_qs:
                    st.write("**Expanded Queries:**")
                    for q in expanded_qs:
                        st.write(f"- {q}")
        return

    if not v_data or "claims" not in v_data:
        return

    claims = v_data["claims"]
    verified_count = sum(1 for c in claims if "✅" in c.get("status", ""))
    error_count = len(claims) - verified_count

    is_fully_verified = v_data.get("is_fully_verified", True)
    expander_title = (
        f"🛡️ Verify Answer & Sources ({verified_count} Verified, {error_count} Issues)"
    )

    with st.expander(expander_title, expanded=not is_fully_verified):
        if is_fully_verified:
            st.success(
                "All claims and numbers are verified against the source documents."
            )

        for idx, claim in enumerate(claims):
            status: str = claim.get("status", "")
            sentence: str = claim.get("sentence", "")
            source: str = claim.get("source_snippet", "")
            details: str = claim.get("details", "")

            if "✅" in status:
                st.success(f"**{status}** — {sentence}")
            elif "❌" in status:
                st.error(f"**{status}** — {sentence}")
            else:
                st.warning(f"**{status}** — {sentence}")

            st.markdown(f"> **Source:** *{source}*")
            st.caption(f"Reasoning: {details}")

            render_feedback_buttons(claim, question, msg_idx, idx)

            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []
if "rag_built" not in st.session_state:
    st.session_state.rag_built = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

with st.sidebar:
    st.title("📄 DocLens")
    st.caption("Upload a PDF and ask questions.")

    dev_mode = st.toggle("🛠️ Developer Mode", value=False)

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

    if "is_building" not in st.session_state:
        st.session_state.is_building = False

    if pdf_file is not None and not st.session_state.rag_built:
        st.divider()
        st.markdown("### ⚙️ Build Settings")

        if "use_llm_choice" not in st.session_state:
            st.session_state.use_llm_choice = False

        use_llm_context = st.toggle(
            "🧠 Enable LLM Contextualization",
            value=st.session_state.use_llm_choice,
            disabled=st.session_state.is_building,
            help="Toggle between Fast Mode (Free) and Accurate Mode (LLM-powered).",
        )
        st.session_state.use_llm_choice = use_llm_context

        if use_llm_context:
            st.warning(
                "**Accurate Mode (LLM):**\n"
                "- ✅ Generates a unique summary for each chunk using GPT-4o-mini.\n"
                "- ⚠️ Slower ingestion and API costs apply (cached per file)."
            )
        else:
            st.info(
                "**Fast Mode (Free):**\n"
                "- ✅ Uses the first 150 characters of each page as context.\n"
                "- ⚡ Instant and zero API cost."
            )

        st.divider()

        build_clicked = st.button(
            "Build Database",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_building,
        )

        if build_clicked and not st.session_state.is_building:
            # 🌟 قفل کردن UI
            st.session_state.is_building = True

            for old_temp in glob.glob("temp_*.pdf"):
                safe_remove(old_temp)

            with st.status("Building database...", expanded=True) as status:
                rag = get_rag_pipeline()

                if use_llm_context:
                    st.write(
                        "Processing PDF with LLM Contextualization (This may take a while)..."
                    )
                else:
                    st.write("Processing PDF in Fast Mode...")

                temp_path = f"temp_{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                try:
                    count = rag.build(temp_path, use_llm_context=use_llm_context)
                    st.session_state.rag_built = True
                    st.session_state.messages = []
                    st.session_state.latest_chunks = []
                    status.update(
                        label=f"Ready! {count} chunks.",
                        state="complete",
                        expanded=False,
                    )

                    safe_remove(temp_path)

                    st.session_state.is_building = False
                    st.rerun()
                except Exception as e:
                    safe_remove(temp_path)

                    st.session_state.is_building = False
                    status.update(label=f"Error: {e}", state="error", expanded=True)

    if st.session_state.rag_built and not st.session_state.is_building:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latest_chunks = []
            st.rerun()

    st.divider()
    if st.session_state.is_building:
        st.info("⏳ Building database... Please wait.")
    elif st.session_state.rag_built:
        st.success("🟢 Document loaded and ready.")
    else:
        st.info("⚪ Waiting for document...")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Chat")

    chat_container = st.container(height=500)
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant":
                    render_verification_ui(
                        v_data=msg.get("verification", {}),
                        msg_idx=i,
                        question=msg.get("original_question", ""),
                        standalone_q=msg.get("standalone_query", ""),
                        expanded_qs=msg.get("expanded_queries", []),
                        dev_mode=dev_mode,
                    )

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

                    standalone_q = prompt
                    expanded_qs = []
                    verification_result: Dict[str, Any] = {}

                    response_placeholder = st.empty()
                    full_response = ""

                    for event in rag.answer_stream(prompt, chat_history=history):
                        if event["type"] == "query_info":
                            standalone_q = event["standalone"]
                            expanded_qs = event["expanded"]
                        elif event["type"] == "chunks":
                            current_chunks = event["chunks"]
                        elif event["type"] == "text":
                            full_response += event["content"]
                            response_placeholder.markdown(full_response + "▌")
                        elif event["type"] == "verification":
                            verification_result = event["data"]

                    response_placeholder.markdown(full_response)

                    render_verification_ui(
                        v_data=verification_result,
                        msg_idx=len(st.session_state.messages),
                        question=prompt,
                        standalone_q=standalone_q,
                        expanded_qs=expanded_qs,
                        dev_mode=dev_mode,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "standalone_query": standalone_q,
                            "expanded_queries": expanded_qs,
                            "original_question": prompt,
                            "verification": verification_result,
                        }
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
