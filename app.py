import os
import time
import streamlit as st
from dotenv import load_dotenv
from graph import app
import uuid
from memory.rag import add_document
from memory.shared_memory import add_long_term_document

load_dotenv()

st.set_page_config(
    page_title="Study & Coding Assistant",
    page_icon="🧠",
    layout="wide",
)


def check_api_keys():
    qwen_key = os.environ.get("LITELLM_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not qwen_key or not tavily_key:
        st.error("API keys not found. Set LITELLM_API_KEY and TAVILY_API_KEY in .env.")
        return False
    return True


st.title("Multi-Agent Study & Coding Assistant")

if not check_api_keys():
    st.stop()

with st.sidebar:
    st.header("Configuration")
    max_iterations = st.slider(
        "Max graph steps",
        min_value=5,
        max_value=25,
        value=15,
        help="Maximum number of LangGraph node executions per query.",
    )
    st.header("Knowledge base")

    uploaded_file = st.file_uploader(
        "Upload PDF to add to notes",
        type=["pdf"],
    )
    if uploaded_file is not None:
        notes_dir = os.path.join(os.path.dirname(__file__), "notes")
        os.makedirs(notes_dir, exist_ok=True)
        doc_id = str(uuid.uuid4())
        save_path = os.path.join(notes_dir, f"{doc_id}.pdf")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        add_long_term_document(save_path, doc_id)
        st.success(f"Added '{uploaded_file.name}' to notes.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about the course, research, or code...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    initial_state = {
        "messages": st.session_state.messages,
        "intent": "",
        "main_task": user_input,
        "research_findings": [],
        "draft": "",
        "critique_notes": "",
        "revision_number": 0,
        "next_step": "",
        "current_sub_task": "",
        "code_question": user_input,
        "code_snippet": "",
        "code_answer": "",
        "quiz_output": "",
    }

    config = {"recursion_limit": max_iterations}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_state = None
        all_states = []  # collect all node outputs

        for step in app.stream(initial_state, config=config):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            all_states.append(node_state)
            final_state = node_state
            placeholder.markdown(f"_{node_name} working..._")

        # after the loop, search from the end for a state with an answer
        answer_state = None
        for s in reversed(all_states):
            if isinstance(s, dict) and (
                s.get("code_answer") or s.get("quiz_output") or s.get("draft") or s.get("research_findings")
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

        placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("Powered by Qwen, LangChain, LangGraph, Tavily.")
