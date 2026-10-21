import gradio as gr
from typing import List, Dict, Any
from pipeline import RAGPipeline

EXAMPLE_QUESTIONS = [
    "What is this document about?",
    "Summarize the main points.",
    "What are the key numbers or figures?",
    "Who is this document for?",
]

def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant chunks found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        score = chunk.get("rerank_score", 0.0)
        parts.append(
            f"[{i}] Page {meta.get('page', '?')} - {meta.get('type', '?')} - score {score:+.2f}\n"
            f"{'-' * 50}\n"
            f"{chunk['text']}"
        )
    return "\n\n\n".join(parts)

def build_index(pdf_file: Any, rag_state: Any) -> Any:
    if not pdf_file:
        yield (rag_state, "Please select a PDF file first.", [], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), "No document loaded.")
        return

    yield (rag_state, "Building database from PDF...", [], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False), "Extracting text, creating chunks, generating embeddings...")

    try:
        rag = RAGPipeline()
        count = rag.build(pdf_file)
        yield (rag, f"Ready - **{count} chunks** indexed", [], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), "Ask a question to see the retrieved chunks here.")
    except Exception as exc:
        yield (rag_state, f"Build failed: {exc}", [], gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), "Build failed. Check the file and try again.")

def chat(message: str, history: List[Dict[str, str]], rag: Any) -> Any:
    history = list(history or [])

    if rag is None or not rag.is_built:
        if message and message.strip():
            help_msg = "**No document loaded yet.**\n\nPlease upload a PDF file and click **Build Database** first."
            yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": help_msg}], "", "Upload a PDF and click Build Database to start."
        else:
            yield history, "", "Upload a PDF first."
        return

    if not message or not message.strip():
        yield history, "", ""
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Searching..."})
    yield history, "", "Retrieving relevant chunks..."

    try:
        current_chunks = []
        for event in rag.answer_stream(message, chat_history=history[:-2]):
            if event["type"] == "chunks":
                current_chunks = event["chunks"]
                history[-1]["content"] = "Generating answer..."
                yield history, "", format_chunks(current_chunks)
            
            elif event["type"] == "text":
                if history[-1]["content"] in ["Searching...", "Generating answer..."]:
                    history[-1]["content"] = ""
                history[-1]["content"] += event["content"]
                yield history, "", format_chunks(current_chunks)
            
            elif event["type"] == "error":
                history[-1]["content"] = event["message"]
                if "queries" in event and event["queries"]:
                    history[-1]["content"] += "\n\n" + "\n".join(f"- {q}" for q in event["queries"])
                yield history, "", "Question needs clarification."
                return

    except Exception as exc:
        err = str(exc).lower()
        if "502" in err or "provider" in err or "bad gateway" in err:
            msg = "AI provider is temporarily unavailable. Please retry in a moment."
        elif "429" in err or "rate" in err:
            msg = "Too many requests. Please wait a few seconds."
        elif "timeout" in err:
            msg = "Request timed out. Please try again."
        else:
            msg = f"Error: {exc}"

        history[-1]["content"] = msg
        yield history, "", msg

with gr.Blocks(title="DocLens") as demo:
    rag_state = gr.State(None)

    gr.Markdown("# DocLens\nAsk questions about any PDF document. Upload a file, click **Build**, then start chatting.")

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            pdf_file = gr.File(label="PDF File", file_types=[".pdf"], type="filepath", height=100)
        with gr.Column(scale=1, min_width=180):
            build_button = gr.Button("Build Database", variant="primary", size="lg")
            status = gr.Markdown("No document loaded.")

    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=480):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=560, placeholder="Upload a PDF and click **Build Database** to start.")
            with gr.Row():
                message = gr.Textbox(placeholder="Ask a question about the document...", show_label=False, lines=2, interactive=True, scale=5)
                send_button = gr.Button("Send", variant="primary", interactive=True, scale=1, size="lg")
            with gr.Row():
                gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[message], label="Try one of these")
                clear_button = gr.Button("New chat", variant="secondary", size="sm")

        with gr.Column(scale=2, min_width=380):
            gr.Markdown("### Retrieved Context")
            retrieved = gr.Textbox(label="Retrieved Context", value="Chunks will appear here.", lines=24, max_lines=24, interactive=False)

    build_button.click(fn=build_index, inputs=[pdf_file, rag_state], outputs=[rag_state, status, chatbot, message, send_button, build_button, retrieved])
    send_button.click(fn=chat, inputs=[message, chatbot, rag_state], outputs=[chatbot, message, retrieved])
    message.submit(fn=chat, inputs=[message, chatbot, rag_state], outputs=[chatbot, message, retrieved])
    clear_button.click(fn=lambda: ([], "", "Chunks will appear here after you ask a question."), outputs=[chatbot, message, retrieved])

if __name__ == "__main__":
    demo.queue().launch()