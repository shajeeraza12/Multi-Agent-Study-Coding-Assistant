# memory/notes_memory.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# Reuse the same embedding model and a separate Chroma directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root (/app or your local root)
MODEL_DIR = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")

_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=MODEL_DIR
)

_NOTES_DIR = os.path.join("memory", "notes_db")
os.makedirs(_NOTES_DIR, exist_ok=True)

_notes_vs = Chroma(
    collection_name="agent_notes",
    embedding_function=_EMBEDDINGS,
    persist_directory=_NOTES_DIR,
)

def save_note(topic: str, note: str, note_type: str = "report"):
    """Store a short summary from an agent."""
    doc = Document(
        page_content=note,
        metadata={"topic": topic, "type": note_type},
    )
    _notes_vs.add_documents([doc])
    _notes_vs.persist()

def search_notes(query: str, k: int = 4):
    """Retrieve relevant past notes for a new query."""
    docs = _notes_vs.similarity_search(query, k=k)
    return docs