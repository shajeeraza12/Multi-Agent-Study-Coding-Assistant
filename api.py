# api.py
import os
import uuid
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from graph import app as graph_app
from memory.shared_memory import add_long_term_document

load_dotenv()

api = FastAPI(
    title="Multi-Agent Study & Coding Assistant API",
    description="FastAPI backend for the LangGraph-based multi-agent research assistant.",
    version="0.1.0",
)

# Allow local dev from browser tools / other UIs
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_steps: int = 15


class ChatResponse(BaseModel):
    answer: str
    final_state: dict


@api.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    # Basic API-key check (optional)
    if not os.environ.get("LITELLM_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Backend is missing LITELLM_API_KEY or TAVILY_API_KEY.",
        )

    # Use last user message as main_task
    last_user_msg: Optional[str] = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message provided.")

    initial_state = {
        "messages": [m.model_dump() for m in req.messages],
        "intent": "",
        "main_task": last_user_msg,
        "research_findings": [],
        "draft": "",
        "critique_notes": "",
        "revision_number": 0,
        "next_step": "",
        "current_sub_task": "",
        "code_question": last_user_msg,
        "code_snippet": "",
        "code_answer": "",
        "quiz_output": "",
    }

    config = {"recursion_limit": req.max_steps}

    final_state = None
    all_states = []

    for step in graph_app.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        node_state = step[node_name]
        all_states.append(node_state)
        final_state = node_state

    # Choose best answer as in app.py
    answer_state = None
    for s in reversed(all_states):
        if isinstance(s, dict) and (
            s.get("code_answer")
            or s.get("quiz_output")
            or s.get("draft")
            or s.get("research_findings")
        ):
            answer_state = s
            break

    if isinstance(answer_state, dict):
        if answer_state.get("code_answer"):
            answer = answer_state["code_answer"]
        elif answer_state.get("quiz_output"):
            answer = answer_state["quiz_output"]
        elif answer_state.get("draft"):
            answer = answer_state["draft"]
        elif answer_state.get("research_findings"):
            answer = answer_state["research_findings"][-1]
        else:
            answer = "I have processed your request, but no answer was produced."
    else:
        answer = "I have processed your request, but no answer was produced."

    return ChatResponse(answer=answer, final_state=final_state or {})


@api.post("/upload_pdf", tags=["knowledge-base"])
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    notes_dir = os.path.join(os.path.dirname(__file__), "notes")
    os.makedirs(notes_dir, exist_ok=True)

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(notes_dir, f"{doc_id}.pdf")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    add_long_term_document(save_path, doc_id)

    return {"status": "ok", "filename": file.filename, "doc_id": doc_id}
