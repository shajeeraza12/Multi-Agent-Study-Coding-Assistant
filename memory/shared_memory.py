from memory.rag import retrieve_notes, add_document
from memory.notes_memory import save_note, search_notes

# Simple façade so all agents call the same interface

def add_long_term_document(path: str, doc_id: str):
    """Add a new file into the shared long‑term memory store."""
    return add_document(path, doc_id)

def save_long_term_note(topic: str, note: str, note_type: str = "note"):
    """Save an abstract or summary into long‑term memory."""
    return save_note(topic=topic, note=note, note_type=note_type)

def search_long_term_notes(query: str, k: int = 3):
    """Search summaries/notes in long‑term memory."""
    return search_notes(query, k=k)

def retrieve_long_term_context(query: str):
    """Retrieve combined context from PDFs + notes for any agent."""
    pdf_context = retrieve_notes(query)
    note_docs = search_notes(query, k=5)
    note_texts = [
        f"- [{d.metadata.get('type', 'note')}] {d.page_content}"
        for d in note_docs
    ]
    notes_block = "\n".join(note_texts) if note_texts else ""
    return pdf_context, notes_block
