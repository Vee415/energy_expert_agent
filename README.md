# ENTSO-E Document Agent

RAG-powered Q&A system over European energy market reports with **3-signal uncertainty quantification** — so you know when to trust the answer and when not to.

![Demo](Demo.gif)

## Why This Project Exists

Standard RAG systems give you an answer and hope for the best. This one tells you **how confident it is** using three independent signals:

| Signal | What it measures | Weight |
|--------|-----------------|--------|
| **Retrieval confidence** | Did we find relevant chunks? (cross-encoder scores) | 40% |
| **Answer consistency** | Does the answer align with retrieved context? (semantic similarity) | 35% |
| **Self-evaluation** | Does the LLM judge its own answer as grounded? | 25% |

If overall confidence < 30%, the system **declines to answer**. Between 30-60%, it answers with a warning. Above 60%, it answers normally.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Retriever                                       │
│  Vector search (Qdrant) → Cross-encoder rerank  │
│  → Keyword boost (years, regions, metrics)       │
└────────────────────┬────────────────────────────┘
                     │ top-k chunks
                     ▼
┌─────────────────────────────────────────────────┐
│  Agent                                           │
│  Build RAG prompt → LLM (Ollama or Claude)      │
│  → Generate answer                               │
└────────────────────┬────────────────────────────┘
                     │ answer + context
                     ▼
┌─────────────────────────────────────────────────┐
│  Uncertainty Quantifier                          │
│  3 signals → weighted confidence score            │
│  → decide: answer / warn / decline               │
└─────────────────────────────────────────────────┘
```

## Key Features

- **LlamaParse extraction** — Vision-based PDF parsing that handles multi-column layouts, tables, and custom fonts (far superior to PyMuPDF for complex reports)
- **Table-aware chunking** — Markdown tables detected and kept intact, narrative text split with markdown-aware separators
- **Junk filtering** — Page headers, footers, and placeholder chunks filtered out during ingestion
- **Year-based source filtering** — Queries mentioning a year (e.g. "market report 2024") automatically filter to the matching PDF
- **Hybrid search** — Dense vector search + keyword boosting for years, regions, and energy-specific metrics
- **Cross-encoder reranking** — ms-marco-MiniLM re-scores retrieved chunks for relevance
- **3-signal uncertainty** — Not just a score, but an explainable breakdown of why the system is or isn't confident
- **Dual LLM support** — Local inference with Ollama or cloud with Claude API

## Quick Start (Docker)

The fastest way to get running — no Python installs, no dependency conflicts.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (Docker Desktop or Docker Engine)
- [Ollama](https://ollama.ai) with a pulled model (for local LLM) — or an Anthropic API key

### 1. Clone and add PDFs

```bash
git clone https://github.com/Vee415/energy_expert_agent.git
cd ai_energy_expert

# Place ENTSO-E PDF reports in data/documents/
# Download from: https://www.entsoe.eu/publications/
#   - Balancing Reports
#   - Market Reports
#   - Regional Coordination Reports
#   - Implementation Monitoring Reports
cp ~/Downloads/*.pdf data/documents/
```

### 2. Ingest documents

```bash
docker compose run ingest
```

First run builds the image (~2-3 min, downloads Python deps + embedding models). Then it parses your PDFs with LlamaParse, chunks them with table detection, embeds, and uploads to Qdrant.

### 3. Start the app

```bash
# Start Ollama first (if using local LLM)
ollama serve

# Then launch the app
docker compose up app
```

Open [http://localhost:8501](http://localhost:8501) and ask questions.

### Using Claude API instead of Ollama

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-xxxxx docker compose up app
```

No Ollama needed — queries go to Claude directly.

### Common commands

| Command | What it does |
|---------|-------------|
| `docker compose run ingest` | Ingest PDFs into Qdrant (run after adding/changing PDFs) |
| `docker compose up app` | Start the Streamlit app |
| `docker compose up` | Start app + Qdrant in background |
| `docker compose down` | Stop all containers (data preserved) |
| `docker compose down -v` | Stop + wipe Qdrant data (full reset) |
| `docker compose logs app` | View app logs |

---

## Quick Start (Manual)

For local development without Docker.

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Ollama (for local LLM) or Anthropic API key

### 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install dependencies

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env to set LLM provider and API keys
```

### 4. Add PDFs and ingest

Place ENTSO-E PDFs in `data/documents/` (download from [entsoe.eu/publications](https://www.entsoe.eu/publications/)), then:

```bash
python -m src.ingestion
```

### 5. Start the LLM

For local inference:
```bash
ollama serve
ollama pull gemma3:4b
```

For Claude API:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 6. Run

```bash
# Streamlit web UI
streamlit run app.py

# Or interactive CLI
python -m src.agent
```

## Example Queries

| Query | Confidence | Result |
|-------|-----------|--------|
| "What are mFRR and aFRR?" | 73.7% | Detailed definition from reports |
| "Energy market trends 2024" | 67.5% | Good synthesis from multiple sources |
| "Who is the CEO of Tesla?" | 10% | Correctly declined (out-of-domain) |

## Project Structure

```
src/
├── ingestion.py     # LlamaParse PDF extraction, table-aware chunking, embedding, Qdrant upload
├── retrieval.py     # Vector search, cross-encoder reranking, keyword boost, source filtering
├── agent.py          # RAG agent with Ollama/Claude support, year-based source filtering
└── uncertainty.py    # 3-signal confidence scoring
app.py                # Streamlit frontend
```

## PDF Extraction & Chunking

PDF extraction quality is the #1 factor in RAG answer quality. Standard PDF loaders (PyMuPDF, PyPDF) extract raw text streams that miss content on complex layouts — multi-column reports, sidebars, and custom fonts produce hollow chunks (titles, page numbers) instead of actual findings.

This project uses **LlamaParse** (vision-based OCR + layout understanding) which produces structured markdown output. Tables come out as proper pipe-delimited markdown, narrative text preserves heading structure, and page headers/footers are filtered out.

Chunking strategy:
- **Markdown-aware splitting** — breaks on headings (`##`, `###`) and paragraphs first
- **Table preservation** — markdown tables detected and kept as whole chunks
- **Junk filtering** — page headers (`"16 // ENTSO-E Market Report 2024"`), standalone headings, and placeholders removed
- **Chunk size 1536** with 200-char overlap — large enough to carry complete findings

## Known Limitations

Honest assessment — because shipping with known gaps is better than pretending they don't exist:

- **Semantic search struggles with specific facts** — asking "Nordic mFRR volume 2021" retrieves the right document but the LLM paraphrases instead of quoting exact values
- **Chart/figure content** — LlamaParse extracts text but doesn't describe charts (requires vision mode, more credits)
- **No BM25** — hybrid search uses keyword boosting, not proper sparse retrieval
- **No citation verification** — the system doesn't validate that claims match source text
- **Single-step retrieval** — no multi-query expansion or multi-hop reasoning

Full details in [CHALLENGES.md](CHALLENGES.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | LlamaParse (vision-based OCR + layout) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | Qdrant (Docker) |
| LLM (local) | Ollama / gemma3:4b |
| LLM (cloud) | Claude 3 Haiku |
| Framework | LangChain |
| Config | Pydantic Settings |
| Frontend | Streamlit |

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM Provider: "ollama" (default) or "anthropic"
LLM_PROVIDER=ollama

# Chunking
CHUNK_SIZE=1536
CHUNK_OVERLAP=200
RETRIEVER_TOP_K=5
```

See [.env.example](.env.example) for full options.

## License

MIT