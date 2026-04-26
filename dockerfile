# Dockerfile for ABC Benchmark - Condition C (OpenHands)
# ============================================================
# This image runs Condition C with OpenHands in a Docker container.
# All other conditions (A & B) run locally on your laptop.

# Base image: OpenHands with Python 3.12 + SDK
FROM ghcr.io/openhands/agent-server:latest

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir helps reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app/

# Environment variables for Docker
# Ollama runs on host, need to connect via host.docker.internal
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
ENV OLLAMA_API_KEY=ollama
ENV LLM_PROVIDER=ollama
ENV MODEL_NAME=qwen2.5:7b

# Expose port for potential debugging
EXPOSE 3000

# Default command: Run Condition C only
# Usage: docker run mas-abc-c [--variant c] [--sample] [--output results.json]
CMD ["python", "evaluate_swe.py", "--variant", "c", "--output", "results_c.json"]