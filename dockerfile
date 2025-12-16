# Dockerfile

FROM python:3.12-slim

# System deps (optional but useful for embeddings, pypdf, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt pyproject.toml* ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Default command: start FastAPI with Uvicorn
CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "8000"]
