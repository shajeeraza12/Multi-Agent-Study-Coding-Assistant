from typing import TypedDict, Annotated, List, Dict, Any
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

# Agent type classification for review mode routing
ORCHESTRATION_AGENTS = {"router", "supervisor", "critiquer"}
CONTENT_AGENTS = {"researcher", "writer", "code_helper", "quiz_helper"}


class ChatState(TypedDict):
    # Chat history + high-level intent
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
    agent_type_relevance: Dict[str, Dict[str, int]]

    # Phase 1: SWE-bench benchmarking fields
    condition: str  # "A" or "B"
    relevancy_score: float  # 0.0-1.0 (from Reviewer)
    hallucination_detected: bool
    iteration_count: int  # max 3 to prevent infinite loops
    swe_instance_id: str
    latencies: Dict[str, float]  # node -> delta_t
    token_usage: Dict[str, int]  # node -> estimated tokens

    # Condition C forensic fields — populated only by code_node_c when variant='c'.
    # These flow from openhands_agent._invoke_sdk / _invoke_cloud / _invoke_stub
    # through code_node_c into the ChatState so the runner can serialize them
    # for thesis-grade evidence (the Hallucination Bridge audit trail).
    openhands_status: str
    openhands_backend: str
    openhands_step_count: int
    openhands_wall_time_s: float
    openhands_action_trace: List[Dict[str, Any]]


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
    Routes to orchestration or content review mode based on agent type.
    """
    import time
    start_time = time.time()

    # Route to correct review mode
    mode = "orchestration" if agent_name in ORCHESTRATION_AGENTS else "content"

    review_result = reviewer_agent(state, output, agent_name, mode=mode)
    review_duration = time.time() - start_time

    # Update running totals
    current_total = state.get("total_checks", 0) + 1
    current_relevant = state.get("relevant_count", 0) + (1 if review_result["is_relevant"] else 0)
    current_irrelevant = state.get("irrelevant_count", 0) + (0 if review_result["is_relevant"] else 1)

    # Safely copy and update agent_type_relevance
    agent_type_relevance = dict(state.get("agent_type_relevance") or {})
    if agent_name not in agent_type_relevance:
        agent_type_relevance[agent_name] = {"total": 0, "relevant": 0}
    agent_type_relevance[agent_name]["total"] += 1
    if review_result["is_relevant"]:
        agent_type_relevance[agent_name]["relevant"] += 1

    relevance_rate = (current_relevant / current_total) * 100 if current_total > 0 else 0
    print(f"[RELEVANCY STATS] Total: {current_total} | Relevant: {current_relevant} | "
          f"Irrelevant: {current_irrelevant} | Rate: {relevance_rate:.1f}% | Mode: {mode}")

    enhanced_review = {
        **review_result,
        "agent_name": agent_name,
        "mode": mode,
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

    # If SWE-bench instance and intent already set to "code", preserve it
    existing_intent = state.get("intent", "")
    if existing_intent == "code" and state.get("swe_instance_id"):
        print(f"Intent: {existing_intent} (preserved for SWE-bench)")
        return {
            "intent": "code",
            "answer_mode": "long",
            "condition": state.get("condition", "B"),
            "swe_instance_id": state.get("swe_instance_id"),
            "relevancy_checks": [],
            "total_checks": 0,
            "relevant_count": 0,
            "irrelevant_count": 0,
            "agent_type_relevance": {},
        }

    # Otherwise, run normal routing
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
    import time

    print("\n=== CODE HELPER ===")
    start_time = time.time()

    # Make previous answer available as snippet
    if state.get("code_answer") and not state.get("code_snippet"):
        state["code_snippet"] = state["code_answer"]

    # Track iteration for refine loops
    refine_attempt = state.get("iteration_count", 0)
    if refine_attempt > 0:
        print(f"[CODE HELPER] refine attempt {refine_attempt + 1}")

    result = code_helper_chain(state)
    answer = result.get("code_answer", "")
    duration = time.time() - start_time

    print(f"Code answer length: {len(answer)}")

    # Initialize latencies/token_usage dicts if needed
    latencies = dict(state.get("latencies", {}) or {})
    token_usage = dict(state.get("token_usage", {}) or {})

    # Estimate token usage (chars / 4 as heuristic)
    token_usage["code_helper"] = len(answer) // 4
    latencies["code_helper"] = duration

    # Condition A: Run Reviewer once (baseline - no refine loop)
    if state.get("condition") == "A":
        review_a = _review_output(state, answer, "code_helper")

        # Extract score using robust method
        code_checks_a = [c for c in review_a.get("relevancy_checks", [])
                        if c.get("agent_name") == "code_helper"]
        if code_checks_a:
            review_score = code_checks_a[-1].get("confidence", 0.0)
            is_relevant = code_checks_a[-1].get("is_relevant", True)
        else:
            review_score = 0.0
            is_relevant = True

        print(f"[CONDITION A] review_score: {review_score:.2f}, is_relevant: {is_relevant}")

        return {
            "code_answer": answer,
            "latencies": latencies,
            "token_usage": token_usage,
            "relevancy_score": review_score,  # Actual score from Reviewer
            "hallucination_detected": not is_relevant,
            "iteration_count": 0,  # No refine iterations for baseline
            **review_a
        }

    # Condition B: Run review
    review = _review_output(state, answer, "code_helper")

    # Robust extraction: Find code_helper-specific check from relevancy_checks
    code_checks = [c for c in review.get("relevancy_checks", [])
                if c.get("agent_name") == "code_helper"]
    if code_checks:
        review_score = code_checks[-1].get("confidence", 0.0)
        is_relevant = code_checks[-1].get("is_relevant", True)
    else:
        review_score = 0.0
        is_relevant = True

    hallucination_detected = not is_relevant

    # Debug logging to confirm extraction
    print(f"[DEBUG] review_score extracted: {review_score:.2f}")
    print(f"[DEBUG] is_relevant: {is_relevant}")
    print(f"[DEBUG] review keys: {list(review.keys())}")

    return {
        "code_answer": answer,
        "latencies": latencies,
        "token_usage": token_usage,
        "relevancy_score": review_score,
        "hallucination_detected": hallucination_detected,
        "iteration_count": refine_attempt + 1,
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

def route_from_reviewer(state: ChatState) -> str:
    """
    Route after code_helper + Reviewer check.
    Condition A: Go directly to END (baseline - single attempt)
    Condition B: Check relevancy_score, accept or refine.
    """
    # Condition A: Go to END after single review attempt (baseline)
    if state.get("condition") == "A":
        score = state.get("relevancy_score", 0.0)
        print(f"[CONDITION A] Final score: {score:.2f} - ending workflow")
        return "END"

    # Condition B: evaluate score
    score = state.get("relevancy_score", 0.0)
    if score >= 0.7:
        print(f"[REVIEWER] Score {score:.2f} >= 0.7, accepting output")
        return "END"

    # Refine loop (max 3 iterations)
    iterations = state.get("iteration_count", 0)
    if iterations >= 3:
        print("[REVIEWER] Max iterations reached, accepting output")
        return "END"

    print(f"[REVIEWER] Score {score:.2f} < 0.7, refining (attempt {iterations + 1}/3)")
    return "refine"


def build_graph(variant: str = "b"):
    """
    Build the LangGraph workflow.

    Args:
        variant: Graph variant to build
            "a" = Baseline (no reviewer, original code_helper)
            "b" = + Reviewer (current MAS workflow)
            "c" = + OpenHands code helper

    Returns:
        Compiled LangGraph application
    """
    # Import based on variant
    if variant == "c":
        from openhands_agent import create_openhands_chain
        _code_helper = create_openhands_chain()
    else:
        _code_helper = code_helper_chain

    workflow = StateGraph(ChatState)

    # Nodes - add based on variant
    workflow.add_node("router", router_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", write_node)
    workflow.add_node("critiquer", critique_node)
    workflow.add_node("quiz_helper", quiz_node)

    # Code helper variant-specific
    if variant == "c":
        # OpenHands node for Condition C
        def code_node_c(state):
            print("\n=== OPENHANDS (Condition C) ===")
            result = _code_helper(state)
            answer = result.get("code_answer", "")
            print(f"OpenHands answer length: {len(answer)}")

            # Run reviewer for Condition C (same as B)
            review = _review_output(state, answer, "code_helper")
            code_checks = [c for c in review.get("relevancy_checks", [])
                        if c.get("agent_name") == "code_helper"]
            if code_checks:
                review_score = code_checks[-1].get("confidence", 0.0)
                is_relevant = code_checks[-1].get("is_relevant", True)
            else:
                review_score = 0.0
                is_relevant = True

            return_dict = {
                "code_answer": answer,
                "relevancy_score": review_score,
                "hallucination_detected": not is_relevant,
                "iteration_count": 1,
                **review
            }

            # Forward OpenHands forensic metadata into the ChatState update so
            # downstream consumers (swe_bench_runner, run_benchmark) can capture
            # the action trace, step count, wall time, and backend identity.
            # Required input for the Hallucination Bridge audit.
            for k in ("openhands_status", "openhands_backend",
                      "openhands_step_count", "openhands_wall_time_s",
                      "openhands_action_trace"):
                if k in result:
                    return_dict[k] = result[k]

            return return_dict

        # Wrap code_node_c so iteration_count increments on every re-entry
        # (refine loop). The original code_node_c hardcoded iteration_count=1
        # which would cause an infinite loop under route_from_reviewer if the
        # output ever scored < 0.7. This wrapper mirrors code_node's
        # `refine_attempt + 1` semantics for behavioral parity with Condition B.
        def code_node_c_with_iter(state):
            refine_attempt = state.get("iteration_count", 0)
            if refine_attempt > 0:
                print(f"[OPENHANDS] refine attempt {refine_attempt + 1}")
            result = code_node_c(state)
            result["iteration_count"] = refine_attempt + 1
            return result

        workflow.add_node("code_helper", code_node_c_with_iter)
    else:
        # Standard code node
        workflow.add_node("code_helper", code_node)

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

    # Code helper routing - variant-specific
    if variant == "a":
        # Baseline: go directly to END (no reviewer)
        workflow.add_edge("code_helper", END)
    else:
        # B and C: use reviewer routing
        workflow.add_conditional_edges(
            "code_helper",
            route_from_reviewer,
            {
                "END": END,
                "refine": "code_helper",  # Loop back for refine
            },
        )

    workflow.add_edge("quiz_helper", END)
    return workflow.compile()


# Default app (variant B)
app = build_graph(variant="b")