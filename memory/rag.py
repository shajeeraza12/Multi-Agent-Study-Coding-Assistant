# memory/rag.py

import os
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(find_dotenv(usecwd=True))

NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notes")
VECTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
MODEL_DIR = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_DIR
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="notes",
            embedding_function=embeddings,
            persist_directory=VECTOR_DIR,
        )
    return _vectorstore


def add_document(file_path: str, doc_id: str) -> None:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)

    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata["doc_id"] = doc_id
        c.metadata["source"] = os.path.basename(file_path)

    vs = _get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()


def retrieve_notes(query: str, k: int = 5) -> str:
    vs = _get_vectorstore()
    try:
        docs: List = vs.similarity_search(query, k=k)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""

    if not docs:
        return ""

    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        parts.append(f"[{source}] {d.page_content.strip()}")
    return "\n\n".join(parts)
