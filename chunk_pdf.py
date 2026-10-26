from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from inspect_pdf import load_pdf_documents
from config import CHUNK_OVERLAP ,CHUNK_SIZE
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP)


def load_chunks(pdf_path):
    documents = load_pdf_documents(pdf_path)
    chunks = splitter.split_documents(documents)
    return chunks
