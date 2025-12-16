from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, END

from agents import (
    create_supervisor_chain,
    create_researcher_agent,
    create_writer_chain,
    create_critique_chain,
    create_router_chain,
    create_code_helper_chain,
    create_quiz_helper_chain,
)


class ChatState(TypedDict):
    # Chat history + high‑level intent
    messages: Annotated[List[dict], operator.add]
    intent: str
    answer_mode: str

    # Research workflow state
    main_task: str
    research_findings: Annotated[List[str], operator.add]
    draft: str
    critique_notes: str
    revision_number: int
    next_step: str
    current_sub_task: str

    # Code helper state
    code_question: str
    code_snippet: str
    code_answer: str

    # Quiz/checklist state
    quiz_output: str


# Instantiate chains/agents
supervisor_chain = create_supervisor_chain()
researcher_agent = create_researcher_agent()
writer_chain = create_writer_chain()
critique_chain = create_critique_chain()
router_chain = create_router_chain()
code_helper_chain = create_code_helper_chain()
quiz_helper_chain = create_quiz_helper_chain()

def router_node(state: ChatState) -> dict:
    print("\n=== ROUTER ===")
    result = router_chain(state)
    intent = result.get("intent", "research")
    print(f"Intent: {intent}")
    return result


def supervisor_node(state: ChatState) -> dict:
    print("\n=== SUPERVISOR ===")
    decision = supervisor_chain(state)
    next_step = decision.get("next_step", "researcher")
    task_desc = decision.get("task_description", "Continue work")
    print(f"Decision: {next_step}")
    print(f"Task: {task_desc}")
    return {
        "next_step": next_step,
        "current_sub_task": task_desc,
    }


def research_node(state: ChatState) -> dict:
    print("\n=== RESEARCHER ===")
    sub_task = state.get("current_sub_task", state.get("main_task"))
    print(f"Researching: {sub_task}")
    try:
        result = researcher_agent({"input": sub_task})
        findings = result.get("output", "Research completed")
        print(f"Found: {str(findings)[:100]}...")
    except Exception as e:
        print(f"Research error: {e}")
        findings = f"Research on {sub_task} - information gathered"
    return {
        "research_findings": [findings],
    }


def write_node(state: ChatState) -> dict:
    print("\n=== WRITER ===")
    draft = writer_chain(state)
    print(f"Draft created: {len(draft)} characters")
    return {
        "draft": draft,
        "revision_number": state.get("revision_number", 0) + 1,
    }


def critique_node(state: ChatState) -> dict:
    print("\n=== CRITIQUER ===")
    critique = critique_chain(state)
    print(f"Critique: {critique[:100]}...")
    is_approved = "APPROVED" in critique.upper()
    if is_approved:
        print("Draft approved")
        return {
            "critique_notes": "APPROVED",
            "next_step": "END",
        }
    else:
        print("Revisions needed")
        return {
            "critique_notes": critique,
            "next_step": "writer",
        }


def code_node(state: ChatState) -> dict:
    print("\n=== CODE HELPER ===")
    # Make previous answer available as snippet
    if state.get("code_answer") and not state.get("code_snippet"):
        state["code_snippet"] = state["code_answer"]
    result = code_helper_chain(state)
    answer = result.get("code_answer", "")
    print(f"Code answer length: {len(answer)}")
    return {
        "code_answer": answer,
    }

def quiz_node(state: ChatState) -> dict:
    print("\n=== QUIZ HELPER ===")
    result = quiz_helper_chain(state)
    output = result.get("quiz_output", "")
    print(f"Quiz/checklist length: {len(output)}")
    return {
        "quiz_output": output,
    }

def build_graph():
    workflow = StateGraph(ChatState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", write_node)
    workflow.add_node("critiquer", critique_node)
    workflow.add_node("code_helper", code_node)
    workflow.add_node("quiz_helper", quiz_node)

    # Entry point
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda s: s.get("intent", "research"),
        {
            "research": "supervisor",
            "general": "supervisor",
            "code": "code_helper",
            "quiz": "quiz_helper",
        },
    )

    # Research workflow edges
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "critiquer")
    workflow.add_edge("critiquer", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_step", "researcher"),
        {
            "researcher": "researcher",
            "writer": "writer",
            "END": END,
        },
    )
    workflow.add_edge("code_helper", END)
    workflow.add_edge("quiz_helper", END)
    return workflow.compile()


app = build_graph()