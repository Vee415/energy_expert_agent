FROM python:3.10-slim

WORKDIR /app

# System deps: build tools (for packages without wheels) + curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir langchain-ollama

# Pre-download embedding models (~250MB, avoids runtime delay)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading all-MiniLM-L6-v2...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Downloading ms-marco-MiniLM-L-6-v2...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models ready.')"

# Copy application code
COPY src/ ./src/
COPY app.py .

# Streamlit config (headless mode for Docker)
RUN mkdir -p ~/.streamlit && \
    printf '[server]\nheadless = true\nenableCORS = false\n' > ~/.streamlit/config.toml

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]