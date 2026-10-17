
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from inspect_pdf import load_pdf_documents


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 150
)

def load_chunks(pdf_path):
    documents = load_pdf_documents(pdf_path)
    chunks = splitter.split_documents(documents)
    return chunks
    