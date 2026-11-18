import os
import json
import glob
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
from pipeline import RAGPipeline
import uuid


def safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


st.set_page_config(page_title="DocLens", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
    /* ==========================================
       1. Global Layout & Typography
       ========================================== */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1.5rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    html, body, [class*="css"] { font-size: 0.85rem !important; font-family: 'Inter', sans-serif !important; }
    p, li { font-size: 0.85rem !important; line-height: 1.5 !important; }
    h1 { font-size: 1.4rem !important; margin-bottom: 0.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.1rem !important; margin-bottom: 0.5rem !important; font-weight: 600 !important; }
    h3 { font-size: 0.95rem !important; margin-bottom: 0.5rem !important; font-weight: 600 !important; }
    
    /* ==========================================
       2. Chat Messages & Buttons
       ========================================== */
    [data-testid="stChatMessage"] { 
        padding: 0.6rem 1rem !important; 
        margin-bottom: 0.6rem !important; 
        border-radius: 12px !important;
        border: 1px solid var(--secondary-background-color) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stButton>button { 
        padding: 0.2rem 0.6rem !important; 
        font-size: 0.8rem !important; 
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stVerticalBlock { gap: 0.4rem !important; }
    
    /* ==========================================
       3. Sidebar Controls
       ========================================== */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        border-radius: 50% !important;
        padding: 5px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
        z-index: 999999 !important;
        top: 15px !important;
        left: 15px !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: var(--text-color) !important;
    }

    /* ==========================================
       4. Expander (Capsule Card Style)
       ========================================== */
    [data-testid="stExpander"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        overflow: hidden !important; /* Keeps content strictly inside the rounded corners */
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.08) !important;
        border-color: var(--primary-color) !important;
    }

    [data-testid="stExpander"] summary {
        padding: 0.5rem 1rem !important;
        font-size: 0.8rem !important;
    }
    
    [data-testid="stExpander"] summary p {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0 1rem 1rem 1rem !important;
        border-top: 1px dashed rgba(128, 128, 128, 0.2) !important;
        color: var(--text-color) !important;
    }

    /* Text inside the capsule */
    [data-testid="stExpanderDetails"] p, 
    [data-testid="stExpanderDetails"] li, 
    [data-testid="stExpanderDetails"] td, 
    [data-testid="stExpanderDetails"] th {
        color: var(--text-color) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ==========================================
       5. Table Scrolling & Styling (The Fix)
       ========================================== */
    /* Apply horizontal scroll to the wrapper, NOT the table itself */
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {
        overflow-x: auto !important;
        width: 100% !important;
        scrollbar-width: thin !important;
    }

    /* Reset table layout so it expands naturally */
    [data-testid="stExpanderDetails"] table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 0.75rem !important; /* Smaller font for dense data */
        margin-bottom: 1rem !important;
        display: table !important; /* Ensure it behaves like a table */
    }

    /* Table Headers */
    [data-testid="stExpanderDetails"] th {
        background-color: rgba(128, 128, 128, 0.1) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        padding: 6px 10px !important;
        text-align: left !important;
        font-weight: 600 !important;
        white-space: nowrap !important; /* Prevents text wrapping */
    }

    /* Table Cells */
    [data-testid="stExpanderDetails"] td {
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        padding: 6px 10px !important;
        white-space: nowrap !important; /* Prevents text/numbers wrapping */
    }

    /* ==========================================
       6. Dev Mode Query Inspector
       ========================================== */
    [data-testid="stExpander"] .stInfo {
        font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border-left: 4px solid var(--primary-color) !important; 
        border-radius: 6px !important;
    }
    
    [data-testid="stExpander"] code {
        color: var(--primary-color) !important;
        background-color: rgba(128, 128, 128, 0.1) !important;
        padding: 2px 4px !important;
        border-radius: 4px !important;
    }

    /* ==========================================
       7. Custom Scrollbar
       ========================================== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
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
            "Why is this incorrect?", key=f"reason_{msg_idx}_{claim_idx}"
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
    decomposed_qs: List[str],
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
                if decomposed_qs:
                    st.success("**Decomposed Sub-Queries:**")
                    for q in decomposed_qs:
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
            status = claim.get("status", "")
            sentence = claim.get("sentence", "")
            source = claim.get("source_snippet", "")
            details = claim.get("details", "")

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


def on_scope_change():
    new_scope = st.session_state.search_scope_widget
    if not new_scope:
        st.session_state.search_scope_widget = st.session_state.current_file_names
        new_scope = st.session_state.current_file_names

    if new_scope != st.session_state.get(
        "last_search_scope", st.session_state.current_file_names
    ):
        st.session_state.messages = []
        st.session_state.latest_chunks = []
        st.session_state.last_search_scope = new_scope
        st.session_state.show_scope_toast = True


if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []
if "rag_built" not in st.session_state:
    st.session_state.rag_built = False
if "current_file_names" not in st.session_state:
    st.session_state.current_file_names = []
if "is_building" not in st.session_state:
    st.session_state.is_building = False
if "last_search_scope" not in st.session_state:
    st.session_state.last_search_scope = []

with st.sidebar:
    st.title("📄 DocLens")
    st.caption("Upload PDFs and ask questions.")

    dev_mode = st.toggle("🛠️ Developer Mode", value=False)

    pdf_files = st.file_uploader(
        "Choose PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if pdf_files and len(pdf_files) > 2:
        st.error(
            "⚠️ You can upload a maximum of 2 PDFs at a time. Please remove the extra files."
        )
        st.stop()

    if not pdf_files:
        if st.session_state.rag_built:
            st.session_state.rag_built = False
            st.session_state.current_file_names = []
            st.session_state.messages = []
            st.session_state.latest_chunks = []
            st.session_state.last_search_scope = []
    else:
        uploaded_names = sorted([f.name for f in pdf_files])

        if (
            st.session_state.rag_built
            and st.session_state.current_file_names != uploaded_names
        ):
            st.warning(
                "⚠️ File list has changed. To apply changes, the database must be rebuilt and chat history will be cleared."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Confirm & Rebuild", type="primary", use_container_width=True
                ):
                    st.session_state.rag_built = False
                    st.session_state.current_file_names = uploaded_names
                    st.session_state.messages = []
                    st.session_state.latest_chunks = []
                    st.session_state.last_search_scope = []
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.info(
                        "Please manually revert the file uploader to keep your current chat."
                    )
                    st.stop()
            st.stop()

        if not st.session_state.rag_built:
            st.session_state.current_file_names = uploaded_names

    if pdf_files and not st.session_state.rag_built:
        st.divider()
        st.markdown("### ⚙️ Build Settings")

        if "use_llm_choice" not in st.session_state:
            st.session_state.use_llm_choice = False

        use_llm_context = st.toggle(
            "🧠 Enable LLM Contextualization",
            key="use_llm_choice",
            disabled=st.session_state.is_building,
            help="Toggle between Fast Mode (Free) and Accurate Mode (LLM-powered).",
        )

        if use_llm_context:
            st.warning(
                "**Accurate Mode (LLM):**\n- ✅ Unique summary per chunk.\n- ⚠️ Slower and API costs apply."
            )
        else:
            st.info(
                "**Fast Mode (Free):**\n- ✅ Page headers as context.\n- ⚡ Instant and zero cost."
            )

        st.divider()

        build_clicked = st.button(
            "Build Database",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_building,
        )

        if build_clicked and not st.session_state.is_building:
            st.session_state.is_building = True

            for old_temp in glob.glob("temp_*.pdf"):
                safe_remove(old_temp)

            with st.status("Building database...", expanded=True) as status:
                rag = get_rag_pipeline()
                st.write(f"Processing {len(pdf_files)} PDF(s)...")

                temp_paths = []
                for pdf_file in pdf_files:
                    unique_id = uuid.uuid4().hex[:6]
                    temp_path = f"temp_{unique_id}_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    temp_paths.append(temp_path)

                try:
                    count = rag.build(temp_paths, use_llm_context=use_llm_context)
                    st.session_state.rag_built = True
                    st.session_state.messages = []
                    st.session_state.latest_chunks = []
                    st.session_state.last_search_scope = (
                        st.session_state.current_file_names
                    )

                    status.update(
                        label=f"Ready! {count} chunks from {len(pdf_files)} documents.",
                        state="complete",
                        expanded=False,
                    )
                    for tp in temp_paths:
                        safe_remove(tp)
                    st.session_state.is_building = False
                    st.rerun()
                except Exception as e:
                    for tp in temp_paths:
                        safe_remove(tp)
                    st.session_state.is_building = False
                    status.update(label=f"Error: {e}", state="error", expanded=True)

    selected_docs = []
    if st.session_state.rag_built and not st.session_state.is_building:
        if len(st.session_state.current_file_names) > 1:
            st.divider()
            st.markdown("### 🔎 Search Scope")

            if "search_scope_widget" not in st.session_state:
                st.session_state.search_scope_widget = (
                    st.session_state.current_file_names
                )

            st.multiselect(
                "Search in:",
                options=st.session_state.current_file_names,
                key="search_scope_widget",
                on_change=on_scope_change,
                help="Select which documents to search.",
            )

            selected_docs = st.session_state.search_scope_widget
            if not selected_docs:
                selected_docs = st.session_state.current_file_names
        else:
            selected_docs = st.session_state.current_file_names

    if st.session_state.get("show_scope_toast"):
        st.toast("⚠️ Search scope changed. Chat history cleared.", icon="🔄")
        st.session_state.show_scope_toast = False

    if st.session_state.rag_built and not st.session_state.is_building:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latest_chunks = []
            st.rerun()

    st.divider()
    if st.session_state.is_building:
        st.info("⏳ Building database... Please wait.")
    elif st.session_state.rag_built:
        st.success("🟢 Document(s) loaded and ready.")
    else:
        st.info("⚪ Waiting for document(s)...")


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
                        decomposed_qs=msg.get("decomposed_queries", []),
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
                    current_chunks = []
                    history = st.session_state.messages[:-1]

                    standalone_q = prompt
                    expanded_qs = []
                    decomposed_qs = []
                    verification_result = {}

                    response_placeholder = st.empty()
                    full_response = ""

                    for event in rag.answer_stream(
                        prompt, chat_history=history, selected_documents=selected_docs
                    ):
                        if event["type"] == "query_info":
                            standalone_q = event["standalone"]
                            expanded_qs = event["expanded"]
                            decomposed_qs = event.get("decomposed", [])
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
                        decomposed_qs=decomposed_qs,
                        dev_mode=dev_mode,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "standalone_query": standalone_q,
                            "expanded_queries": expanded_qs,
                            "decomposed_queries": decomposed_qs,
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
            for i, chunk in enumerate(st.session_state.latest_chunks, 1):
                meta = chunk.get("metadata", {})
                score = chunk.get("rerank_score", 0.0)
                chunk_type = meta.get("type", "?")
                doc_name = meta.get("source_document", "Unknown")
                page = meta.get("page", "?")

                expander_label = f"**[{i}]** {doc_name} &nbsp;|&nbsp; Page {page} &nbsp;|&nbsp; {chunk_type} *({score:+.2f})*"
                with st.expander(expander_label, expanded=(i == 1)):
                    st.markdown(chunk["text"])
        else:
            st.caption("Retrieved chunks will appear here.")
