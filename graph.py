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
    create_reviewer_agent,
    check_relevancy,
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

    # Relevancy tracking for hallucination detection
    relevancy_checks: Annotated[List[dict], operator.add]
    total_checks: int
    relevant_count: int
    irrelevant_count: int


# Instantiate chains/agents
supervisor_chain = create_supervisor_chain()
researcher_agent = create_researcher_agent()
writer_chain = create_writer_chain()
critique_chain = create_critique_chain()
router_chain = create_router_chain()
code_helper_chain = create_code_helper_chain()
quiz_helper_chain = create_quiz_helper_chain()
reviewer_agent = create_reviewer_agent()


def _review_output(state: ChatState, output: str, agent_name: str) -> dict:
    """
    Helper function to review an agent's output and update relevancy statistics.
    Returns the review result dictionary.
    """
    import time
    start_time = time.time()
    
    review_result = reviewer_agent(state, output, agent_name)
    review_duration = time.time() - start_time
    
    # Calculate updated statistics
    current_total = state.get("total_checks", 0) + 1
    current_relevant = state.get("relevant_count", 0) + (1 if review_result["is_relevant"] else 0)
    current_irrelevant = state.get("irrelevant_count", 0) + (0 if review_result["is_relevant"] else 1)
    
    # Calculate relevance score by agent type for research analysis
    agent_type_relevance = state.get("agent_type_relevance", {})
    if agent_name not in agent_type_relevance:
        agent_type_relevance[agent_name] = {"total": 0, "relevant": 0}
    agent_type_relevance[agent_name]["total"] += 1
    if review_result["is_relevant"]:
        agent_type_relevance[agent_name]["relevant"] += 1
    
    # Enhanced logging for research
    relevance_rate = (current_relevant / current_total) * 100 if current_total > 0 else 0
    print(f"[RELEVANCY STATS] Total: {current_total} | Relevant: {current_relevant} | Irrelevant: {current_irrelevant} | Rate: {relevance_rate:.1f}%")
    print(f"[REVIEW METRICS] Agent: {agent_name} | Duration: {review_duration:.2f}s | Decision: {'PASS' if review_result['is_relevant'] else 'FAIL'}")
    
    # Store enhanced review data
    enhanced_review = {
        **review_result,
        "agent_name": agent_name,
        "output_length": len(output),
        "review_duration": review_duration,
        "timestamp": time.time(),
        "workflow_step": current_total
    }
    
    return {
        "relevancy_checks": [enhanced_review],
        "total_checks": current_total,
        "relevant_count": current_relevant,
        "irrelevant_count": current_irrelevant,
        "agent_type_relevance": agent_type_relevance
    }

def router_node(state: ChatState) -> dict:
    print("\n=== ROUTER ===")
    result = router_chain(state)
    intent = result.get("intent", "research")
    print(f"Intent: {intent}")
    
    # Review the router's decision
    router_output = f"Intent: {intent}, Answer Mode: {result.get('answer_mode', 'long')}"
    review = _review_output(state, router_output, "router")
    
    return {**result, **review}


def supervisor_node(state: ChatState) -> dict:
    print("\n=== SUPERVISOR ===")
    decision = supervisor_chain(state)
    next_step = decision.get("next_step", "researcher")
    task_desc = decision.get("task_description", "Continue work")
    print(f"Decision: {next_step}")
    print(f"Task: {task_desc}")
    
    # Review the supervisor's decision
    supervisor_output = f"Next Step: {next_step}, Task: {task_desc}"
    review = _review_output(state, supervisor_output, "supervisor")
    
    return {
        "next_step": next_step,
        "current_sub_task": task_desc,
        **review
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
    
    # Review the research findings
    review = _review_output(state, findings, "researcher")
    
    return {
        "research_findings": [findings],
        **review
    }


def write_node(state: ChatState) -> dict:
    print("\n=== WRITER ===")
    draft = writer_chain(state)
    print(f"Draft created: {len(draft)} characters")
    
    # Review the draft
    review = _review_output(state, draft, "writer")
    
    return {
        "draft": draft,
        "revision_number": state.get("revision_number", 0) + 1,
        **review
    }


def critique_node(state: ChatState) -> dict:
    print("\n=== CRITIQUER ===")
    critique = critique_chain(state)
    print(f"Critique: {critique[:100]}...")
    
    # Review the critique
    review = _review_output(state, critique, "critiquer")
    
    is_approved = "APPROVED" in critique.upper()
    if is_approved:
        print("Draft approved")
        return {
            "critique_notes": "APPROVED",
            "next_step": "END",
            **review
        }
    else:
        print("Revisions needed")
        return {
            "critique_notes": critique,
            "next_step": "writer",
            **review
        }


def code_node(state: ChatState) -> dict:
    print("\n=== CODE HELPER ===")
    # Make previous answer available as snippet
    if state.get("code_answer") and not state.get("code_snippet"):
        state["code_snippet"] = state["code_answer"]
    result = code_helper_chain(state)
    answer = result.get("code_answer", "")
    print(f"Code answer length: {len(answer)}")
    
    # Review the code answer
    review = _review_output(state, answer, "code_helper")
    
    return {
        "code_answer": answer,
        **review
    }

def quiz_node(state: ChatState) -> dict:
    print("\n=== QUIZ HELPER ===")
    result = quiz_helper_chain(state)
    output = result.get("quiz_output", "")
    print(f"Quiz/checklist length: {len(output)}")
    
    # Review the quiz output
    review = _review_output(state, output, "quiz_helper")
    
    return {
        "quiz_output": output,
        **review
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