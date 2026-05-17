# visualize_graph.py
"""
Generate Mermaid diagrams for all three benchmark conditions (A, B, C).

Two diagram types are produced per condition:
  1. Auto-generated  — from LangGraph's draw_mermaid() (shows graph topology only)
  2. Manually authored — includes the Reviewer Agent which lives inside _review_output()
     and is therefore invisible to LangGraph's auto-renderer.

Output files → assets/
"""
import os
from graph import build_graph


# ---------------------------------------------------------------------------
# Manually authored architecture diagrams (include Reviewer Agent)
# ---------------------------------------------------------------------------

DIAGRAM_A = """flowchart TD
    U([\"👤 User Input\"]) --> router

    subgraph GRAPH [\"Condition A — Baseline (Single Agent, No Refine Loop)\"]
        router{{\"🔀 Router\"}}
        supervisor{{\"🧠 Supervisor\"}}
        researcher[\"🔍 Researcher\"]
        writer[\"✍️ Writer\"]
        critiquer[\"🧐 Critiquer\"]
        code_helper[\"💻 Code Helper\"]
        quiz_helper[\"📝 Quiz Helper\"]
    end

    subgraph REVIEWER [\"👁️ Reviewer Agent — Single Pass (score recorded, no retry)\"]
        rv_rubric[\"5-Dimension Rubric\\nCorrectness · Edge Cases · Security · Quality · Relevance\"]
    end

    router -->|\"code\"| code_helper
    router -->|\"research / general\"| supervisor
    router -->|\"quiz\"| quiz_helper

    supervisor -->|\"researcher\"| researcher
    supervisor -->|\"writer\"| writer
    supervisor -->|\"END\"| OUT

    researcher --> supervisor
    writer --> critiquer
    critiquer -->|\"APPROVED\"| supervisor
    critiquer -->|\"revisions needed\"| writer

    code_helper --> rv_rubric
    rv_rubric -->|\"score logged → END\"| OUT([\"✅ Final Output\"])
    quiz_helper --> OUT

    style REVIEWER fill:#fce4ec,stroke:#e91e63,color:#000
    style GRAPH fill:#e8f5e9,stroke:#4caf50,color:#000
"""

DIAGRAM_B = """flowchart TD
    U([\"👤 User Input\"]) --> router

    subgraph GRAPH [\"Condition B — Reviewer Agent with Refine Loop\"]
        router{{\"🔀 Router\"}}
        supervisor{{\"🧠 Supervisor\"}}
        researcher[\"🔍 Researcher\"]
        writer[\"✍️ Writer\"]
        critiquer[\"🧐 Critiquer\"]
        code_helper[\"💻 Code Helper\"]
        quiz_helper[\"📝 Quiz Helper\"]
    end

    subgraph REVIEWER [\"👁️ Reviewer Agent — Hallucination Gate\"]
        rv_rubric[\"5-Dimension Rubric\\nCorrectness · Edge Cases · Security · Quality · Relevance\"]
        rv_gate{{\"Score ≥ 0.7?\"}}
        rv_rubric --> rv_gate
    end

    router -->|\"code\"| code_helper
    router -->|\"research / general\"| supervisor
    router -->|\"quiz\"| quiz_helper

    supervisor -->|\"researcher\"| researcher
    supervisor -->|\"writer\"| writer
    supervisor -->|\"END\"| OUT

    researcher --> supervisor
    writer --> critiquer
    critiquer -->|\"APPROVED\"| supervisor
    critiquer -->|\"revisions needed\"| writer

    code_helper --> rv_rubric
    rv_gate -->|\"Yes\"| OUT([\"✅ Final Output\"])
    rv_gate -->|\"No — iter < 3\"| code_helper
    rv_gate -->|\"Max iterations reached\"| OUT

    quiz_helper --> OUT

    style REVIEWER fill:#fff3cd,stroke:#ffc107,color:#000
    style GRAPH fill:#e8f4fd,stroke:#2196f3,color:#000
"""

DIAGRAM_C = """flowchart TD
    U([\"👤 User Input\"]) --> router

    subgraph GRAPH [\"Condition C — OpenHands Agent Supervised by Reviewer\"]
        router{{\"🔀 Router\"}}
        supervisor{{\"🧠 Supervisor\"}}
        researcher[\"🔍 Researcher\"]
        writer[\"✍️ Writer\"]
        critiquer[\"🧐 Critiquer\"]
        oh_agent[\"🤖 OpenHands Agent\\n(autonomous terminal & file actions)\"]
        quiz_helper[\"📝 Quiz Helper\"]
    end

    subgraph REVIEWER [\"👁️ Reviewer Agent — Hallucination Bridge\"]
        rv_rubric[\"5-Dimension Rubric\\nCorrectness · Edge Cases · Security · Quality · Relevance\"]
        rv_gate{{\"Score ≥ 0.7?\"}}
        rv_rubric --> rv_gate
    end

    subgraph FORENSICS [\"🔬 Forensic Metadata (ChatState)\"]
        meta[\"openhands_status\\nopenhands_backend\\nopenhands_step_count\\nopenhands_wall_time_s\\nopenhands_action_trace\"]
    end

    router -->|\"code\"| oh_agent
    router -->|\"research / general\"| supervisor
    router -->|\"quiz\"| quiz_helper

    supervisor -->|\"researcher\"| researcher
    supervisor -->|\"writer\"| writer
    supervisor -->|\"END\"| OUT

    researcher --> supervisor
    writer --> critiquer
    critiquer -->|\"APPROVED\"| supervisor
    critiquer -->|\"revisions needed\"| writer

    oh_agent -->|\"action trace captured\"| FORENSICS
    oh_agent --> rv_rubric
    rv_gate -->|\"Yes\"| OUT([\"✅ Final Output\"])
    rv_gate -->|\"No — iter < 3\"| oh_agent
    rv_gate -->|\"Max iterations reached\"| OUT

    quiz_helper --> OUT

    style REVIEWER fill:#fff3cd,stroke:#ffc107,color:#000
    style GRAPH fill:#ede7f6,stroke:#9c27b0,color:#000
    style FORENSICS fill:#e3f2fd,stroke:#1565c0,color:#000
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_graph_image():
    """
    Saves Mermaid diagrams for all three benchmark conditions to the 'assets' folder.

    Files produced:
      langgraph_condition_a.mmd  — LangGraph auto-render, Condition A topology
      langgraph_condition_b.mmd  — LangGraph auto-render, Condition B topology
      langgraph_condition_c.mmd  — LangGraph auto-render, Condition C topology (requires OpenHands)
      architecture_condition_a.mmd — Full architecture incl. Reviewer Agent (manual)
      architecture_condition_b.mmd — Full architecture incl. Reviewer Agent (manual)
      architecture_condition_c.mmd — Full architecture incl. Reviewer + Forensics (manual)
    """
    os.makedirs("assets", exist_ok=True)

    # ── 1. Auto-generated LangGraph topology diagrams ─────────────────────────
    for variant, filename, label in [
        ("a", "langgraph_condition_a.mmd", "Condition A — Baseline"),
        ("b", "langgraph_condition_b.mmd", "Condition B — Reviewer"),
        ("c", "langgraph_condition_c.mmd", "Condition C — OpenHands"),
    ]:
        try:
            compiled = build_graph(variant=variant)
            mmd_text = compiled.get_graph().draw_mermaid()
            path = os.path.join("assets", filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(mmd_text)
            print(f"✓ [Auto]   {label:40s} → {path}")
        except Exception as e:
            print(f"⚠ [Auto]   {label:40s} — skipped: {e}")

    print()

    # ── 2. Manually authored diagrams (Reviewer Agent visible) ────────────────
    manual_diagrams = [
        ("architecture_condition_a.mmd", DIAGRAM_A, "Condition A — Baseline (manual)"),
        ("architecture_condition_b.mmd", DIAGRAM_B, "Condition B — Reviewer (manual)"),
        ("architecture_condition_c.mmd", DIAGRAM_C, "Condition C — OpenHands + Reviewer (manual)"),
    ]
    for filename, content, label in manual_diagrams:
        path = os.path.join("assets", filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"✓ [Manual] {label:40s} → {path}")

    print()
    print("All diagrams saved to assets/")
    print("Paste any .mmd file at https://mermaid.live/ to render or export as PNG/SVG.")


if __name__ == "__main__":
    save_graph_image()
