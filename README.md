# Multi-Agent Study & Coding Assistant

A sophisticated **Multi-Agent System (MAS)** built with **LangChain** and **LangGraph** that leverages multiple specialized AI agents to assist with research, learning, coding, and productivity tasks. The system demonstrates advanced agent coordination, reasoning loops, intelligent tool integration, and includes a **Reviewer Agent** for hallucination detection on coding tasks.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation \& Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [ABC Benchmark (SWE-bench)](#abc-benchmark-swe-bench)
- [API Documentation](#api-documentation)
- [Agent Roles](#agent-roles)
- [Hallucination Detection](#hallucination-detection)
- [Memory Management](#memory-management)

---

## Project Overview

This project implements a Multi-Agent System for research, learning, and coding assistance. It fulfills Lab 2 requirements for designing and implementing a multi-agent system with LangChain & LangGraph.

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
├── tests/
│   └── test_tools.py            # Unit tests
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
| A | Baseline (no reviewer) | Local (Python) |
| B | With Reviewer Agent | Local (Python) |
| C | With OpenHands Agent | Docker |

---

## Installation & Setup

### Prerequisites

- **Python**: 3.10 or higher
- **pip** for dependency management
- **Git** for cloning the repository
- **Docker** (for Condition C only)

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

#### Option B: Using OpenRouter (Cloud)

```bash
# LLM Provider
LLM_PROVIDER=litellm

# OpenRouter Configuration (for cloud LLM access)
LITELLM_BASE_URL=https://openrouter.ai/v1
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=anthropic/claude-3-sonnet

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

## Running Tests

Run the unit tests with pytest:

```bash
pytest tests/
```

Run with verbose output:

```bash
pytest -v tests/
```

---

## ABC Benchmark (SWE-bench)

Run the ABC benchmark to evaluate the multi-agent system on SWE-bench coding tasks:

### Prerequisites

- **Conditions A & B**: Python virtual environment with dependencies installed
- **Condition C**: Docker Desktop installed and running + Ollama

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

# Condition C (With OpenHands - requires Docker)
# First start OpenHands Docker container:
docker run -d -p 3000:3000 --name openhands-server ghcr.io/all-hands-ai/openhands:latest

# Then run:
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

**Version**: 1.1.0  
**Last Updated**: April 2026