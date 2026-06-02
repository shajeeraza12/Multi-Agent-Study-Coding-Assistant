# Multi-Agent Study & Coding Assistant

A **Multi-Agent System (MAS)** built with **LangChain** and **LangGraph** that uses specialised agents to assist with research, learning, coding and productivity tasks. The system includes a **lightweight Reviewer Agent** for hallucination detection on coding outputs, evaluated against SWE-bench Lite under a controlled three-condition comparison (A: baseline, B: Reviewer + refine loop, C: OpenHands action agent under Reviewer supervision).

This repository is the artefact for a Master's thesis at ITMO University: *"Development of a Lightweight Hallucination Detection Algorithm for the Reviewer Agent in a Code Generation Multi-Agent System."*

**Status (June 2026):** Phase 1–5 complete. The Reviewer Agent's contribution is statistically supported on multiple generator families (Mistral 7B × qwen3-32b judge: Δ = +0.199, p < 0.001; qwen2.5:7b × qwen3-32b: Δ = +0.105, p < 0.05). See [Phase 5 Results](#phase-5-results) below or `progress.md` for the full timeline.

---

## Phase 5 Results

Six 30-instance paired A/B cells on a fixed SWE-bench Lite subset, varying generator and judge:

![Phase 5 forest plot](docs/figures/phase5_forest_plot.png)

| Cell | Generator | Judge | A | B | Δ | t | Significant |
|---|---|---|---|---|---|---|---|
| C1 | deepseek-coder:6.7b | qwen3-32b | 0.529 | 0.551 | +0.021 | +0.48 | — |
| C2 | codellama:7b | qwen3-32b | 0.355 | 0.431 | +0.076 | +1.47 | — |
| C3 | llama3.1:8b | qwen3-32b | 0.568 | 0.661 | +0.093 | +1.72 | marginal |
| **C4** | **mistral:7b** | **qwen3-32b** | **0.511** | **0.710** | **+0.199** | **+3.85** | **p < 0.001** |
| C5 | qwen3-32b | gpt-oss-120b | 0.730 | 0.769 | +0.039 | +1.90 | marginal |
| **C6** | mistral:7b | gpt-oss-120b | 0.394 | 0.435 | +0.041 | **+2.33** | **p < 0.05** |

**Key findings:**

- The Reviewer Agent's contribution generalises across model families (DeepSeek, CodeLlama, Llama, Mistral) — not just Qwen, on which earlier phases were calibrated.
- The lift scales with *refinability* rather than raw weakness. Mistral 7B is the most refine-responsive model tested, with a 20:1 lift-to-regression ratio in C4.
- Judge family affects both absolute scoring and the lift magnitude. The same generator (qwen3-32b) judged by a same-family judge produced Δ = +0.003 (Phase 1); judged by gpt-oss-120b it produced Δ = +0.039 (C5) — a 13× change attributable purely to the judge.

Full figures in [`docs/figures/`](docs/figures/). Detailed methodology and per-cell observations in [`progress.md`](progress.md). Weekend execution log in [`phase5_weekend_plan.md`](phase5_weekend_plan.md).

---

## Table of Contents

- [Phase 5 Results](#phase-5-results)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation \& Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [ABC Benchmark (SWE-bench)](#abc-benchmark-swe-bench)
- [Reproducing Phase 5 cells](#reproducing-phase-5-cells)
- [API Documentation](#api-documentation)
- [Agent Roles](#agent-roles)
- [Hallucination Detection](#hallucination-detection)
- [Memory Management](#memory-management)

---

## Project Overview

This project implements a Multi-Agent System for research, learning, and coding assistance.

### What It Does

The assistant receives diverse user queries and intelligently routes them to specialized agents:

- **Research queries** → Research + Writer + Critiquer workflow
- **Coding questions** → Code helper with Python/C++ execution + Reviewer Agent
- **Study aids** → Quiz/checklist generation
- **General questions** → Flexible research-based answering

### Why Multiple Agents?

A single generic chatbot cannot effectively handle the specialized reasoning required for different domains. This system employs dedicated agents, each optimized for their specific task, enabling superior performance across diverse problem domains.

---

## Project Structure

```
multi-agent-study-coding-assistant/
├── agents.py                    # 8 agent implementations (including Reviewer)
├── graph.py                     # LangGraph workflow with ABC variants
├── prompts.py                   # LLM prompt templates
├── tools.py                     # Code execution utilities
├── api.py                       # FastAPI backend with endpoints
├── app.py                       # Streamlit web interface
├── main.py                      # CLI entry point
├── visualize_graph.py           # Graph visualization script
├── evaluate_swe.py              # SWE-bench evaluation script
├── openhands_agent.py           # OpenHands agent for Condition C
├── openhands_client.py          # OpenHands HTTP client
├── swe_bench_runner.py          # SWE-bench benchmark runner
├── run_benchmark.py             # ABC benchmark runner
├── memory/
│   ├── shared_memory.py         # Long-term memory management
│   ├── notes_memory.py          # Notes memory module
│   └── rag.py                   # Vector DB + PDF retrieval
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
├── dockerfile                   # Docker configuration for Condition C
├── .dockerignore                # Docker ignore file
└── README.md                    # This file
```

---

## Key Features

- **8 Specialized Agents** - Router, Supervisor, Researcher, Writer, Critiquer, Code Helper, Quiz Helper, Reviewer
- **Intelligent Routing** - Automatic intent classification
- **Reviewer Agent** - Hallucination detection and quality assessment
- **Long-term Memory** - Persistent knowledge storage via Chroma + file system
- **Web Search Integration** - Real-time information via Tavily API
- **Code Execution** - Run Python, C, and C++ code safely
- **Educational Tools** - Quiz and checklist generation
- **Safety Features** - Prompt injection protection, schema validation, error handling
- **REST API** - FastAPI backend with Swagger documentation
- **User Interface** - Streamlit web application
- **ABC Benchmark** - SWE-bench evaluation with three conditions (A/B/C)

---

## System Architecture

### Multi-Agent Workflow

The system implements a **Supervisor + Router** multi-agent pattern where:

1. **Router Agent** classifies incoming user queries by intent (research, code, quiz, or general)
2. **Intent-based Routing** directs the query to appropriate specialist agents
3. **Supervisor Agent** orchestrates complex workflows (research → write → critique)
4. **Specialist Agents** process tasks with the help of the Reviewer Agent
5. **Shared State Management** enables seamless communication via TypedDict
6. **Conditional Edges** route agents based on task progress and decisions

### ABC Conditions

| Condition | Description | Runtime |
|-----------|-------------|---------|
| A | Baseline (no reviewer) | Local (Python 3.12) |
| B | With Reviewer Agent | Local (Python 3.12) |
| C | With OpenHands Agent | Local (Python 3.12 + OpenRouter/MiniMax) |

---

## Installation & Setup

### Prerequisites

- **Python**: 3.12 (required)
- **pip** for dependency management
- **Git** for cloning the repository
- **Linux** (required for Condition C)

### Step 1: Clone the Repository

```bash
git clone https://github.com/shajeeraza12/Multi-Agent-Study-Coding-Assistant.git
cd Multi-Agent-Study-Coding-Assistant
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
.\venv\Scripts\activate  # On Windows

# Or on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables Setup

Create a `.env` file in the project root with the following configuration:

#### Option A: Using Ollama (Local)

```bash
# LLM Provider
LLM_PROVIDER=ollama

# Ollama Configuration (OpenAI-compatible API)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
MODEL_NAME=qwen2.5:14b

# Tavily Search API (for web search)
TAVILY_API_KEY=your_tavily_api_key_here
```

#### Option C: Using OpenRouter (Cloud) with MiniMax LLM (for Condition C)

```bash
# LLM Provider
LLM_PROVIDER=litellm

# OpenRouter Configuration (for cloud LLM access)
LITELLM_BASE_URL=https://openrouter.ai/v1
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=minimax/minimax-long

# Tavily Search API (for web search)
TAVILY_API_KEY=your_tavily_api_key_here
```

### Getting API Keys

**Tavily API Key**:
1. Visit [tavily.com](https://tavily.com/)
2. Sign up for a free account
3. Navigate to the API keys section in your dashboard
4. Copy your API key
5. Paste it in your `.env` file as `TAVILY_API_KEY=tvly-xxxxx`

**OpenRouter API Key**:
1. Visit [openrouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to the API keys section
4. Create a new API key
5. Add credits to your account
6. Use in `.env` as `OPENROUTER_API_KEY=sk-or-xxxxx`

**Ollama Setup** (if using local models):
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Run `ollama pull qwen2.5:14b` to download a model
3. Start Ollama server: `ollama serve`

---

## Running the Application

### 1. Streamlit Web Interface

The Streamlit app provides a user-friendly chat interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### 2. FastAPI Backend

Deploy the REST API backend:

```bash
uvicorn api:api --host 0.0.0.0 --port 8000
```

Access Swagger UI documentation at `http://localhost:8000/docs`

### 3. CLI Entry Point

Run via command line:

```bash
python main.py
```

### 4. Graph Visualization

Generate a Mermaid diagram of the agent workflow:

```bash
python visualize_graph.py
```

---

## ABC Benchmark (SWE-bench)

Run the ABC benchmark to evaluate the multi-agent system on SWE-bench coding tasks:

### Prerequisites

- **Conditions A & B**: Python 3.12 virtual environment with dependencies installed
- **Condition C**: Python 3.12 on Linux + OpenRouter API key (MiniMax LLM)

### Running Each Condition

```bash
# Activate virtual environment
.\venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac

# Condition A (Baseline - no reviewer)
python evaluate_swe.py --variant a --output results_a.json --sample

# Condition B (With Reviewer Agent - hallucination detection)
python evaluate_swe.py --variant b --output results_b.json --sample

# Condition C (With OpenHands SDK - Linux only, Python 3.12)
python evaluate_swe.py --variant c --output results_c.json --sample
```

### Full Benchmark Workflow

```bash
# Run all three conditions
python run_benchmark.py --output results_abc.json --max-instances 5
```

---

## API Documentation

### POST /chat

Send a chat message to the multi-agent system.

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "max_steps": 15
}
```

**Response**:
```json
{
  "answer": "Machine learning is...",
  "final_state": {
    "draft": "...",
    "research_findings": [...],
    "code_answer": "",
    "quiz_output": ""
  }
}
```

### POST /upload_pdf

Upload a PDF document to the knowledge base for retrieval-augmented generation.

**Request**: (multipart/form-data)
```
file: <PDF file>
```

**Response**:
```json
{
  "status": "ok",
  "filename": "document.pdf",
  "doc_id": "uuid-string"
}
```

### GET /health

Health check endpoint to verify system status.

**Response**:
```json
{
  "status": "ok"
}
```

---

## Agent Roles

### Router Agent
Classifies user intent and routes to appropriate workflow (code, research, quiz, general).

### Supervisor Agent
Orchestrates multi-agent workflow and makes strategic decisions.

### Researcher Agent
Gathers comprehensive information from web and knowledge base using Tavily search and RAG.

### Writer Agent
Synthesizes findings into coherent, well-structured responses.

### Critiquer Agent
Quality assurance and constructive feedback mechanism.

### Code Helper Agent
Assists with coding questions and provides safe code execution.

### Quiz Helper Agent
Generates educational content and learning tools.

### Reviewer Agent (Hallucination Detection)
Evaluates agent outputs for relevancy and detects hallucinations in code generation tasks.

---

## Hallucination Detection

The system includes a **Reviewer Agent** that performs hallucination detection on coding tasks:

### How It Works

1. **Code Evaluation**: The Reviewer Agent evaluates code outputs against:
   - **Correctness**: Does the code solve the specific problem?
   - **Edge Cases**: Does it handle boundary conditions?
   - **Security**: Are there vulnerabilities?
   - **Code Quality**: Is the code clean and follows best practices?
   - **Relevance**: Does it directly address the user's question?

2. **Scoring**: Each criterion is rated 1-10, then averaged and scaled to 0.0-1.0

3. **Refinement Loop**: If score < 0.7, the system refines the output (up to 3 iterations)

### ABC Condition Behavior

- **Condition A**: Baseline - runs code helper once, no reviewer
- **Condition B**: With Reviewer - scores output, accepts (≥0.7) or refines (<0.7)
- **Condition C**: With OpenHands + Reviewer - uses OpenHands for code generation

---

## Memory Management

### Short-Term Memory
Messages in ChatState for single conversation session.

### Long-Term Memory
- Vector Database (Chroma) for semantic similarity search
- File System (JSON notes + PDF documents)
- Semantic search + keyword matching

### Shared Memory (Blackboard Pattern)
Collaborative reasoning space where agents can post findings and insights.

---

**Version**: 1.2.0  
**Last Updated**: June 2026

---

## Reproducing Phase 5 cells

Each Phase 5 cell can be re-run with one command, swapping `MODEL_NAME` and `JUDGE_MODEL` for each pairing. The runner patch added in May 2026 captures the first-iteration Reviewer score as the Condition-A-equivalent, so a single B-only run yields paired A/B data. See `phase5_weekend_plan.md` for the standard per-cell launch template (pre-flight curls + 30-instance run + quick analysis).

```bash
# Cell C4 example — Mistral × qwen3-32b (the strongest cell)
export LLM_PROVIDER=ollama
export MODEL_NAME=mistral:7b-instruct
export JUDGE_MODEL=iairlab/qwen3-32b
unset OLLAMA_BASE_URL   # let .env take effect with /v1

REFINE_THRESHOLD=0.85 python swe_bench_runner.py --swe-lite --variant b \
  --instance-ids @sweep_phase4_night1.json \
  --max-instances 30 \
  --no-clone \
  --output results/phase5_C4_mistral_qwen3_$(date +%Y%m%d_%H%M%S).json \
  --resume
```

For cells using a duckduck-hosted generator (e.g. C5 qwen3-32b), set `LLM_PROVIDER=litellm` and `MODEL_NAME=iairlab/qwen3-32b`. Both sides of the call then go through the same OpenAI-compatible endpoint configured in `.env`.

Regenerate the figures and summary CSV from the result JSONs:

```bash
python3 scripts/analyse_phase5.py    # produces docs/figures/phase5_*.png and phase5_summary.json
```

(The analysis script is the same one used to populate the tables in this README and in `progress.md` §14.)
