# System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENT DOCUMENT AGENT                          │
│                    RAG-powered Q&A with Uncertainty Awareness               │
└─────────────────────────────────────────────────────────────────────────────┘

                                ┌─────────────┐
                                │   USER      │
                                │  (Browser)  │
                                └──────┬──────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Streamlit (app.py)                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Query Input  │  │ Confidence   │  │ Answer + Sources Display │  │   │
│  │  │              │  │ Visualization│  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               AGENT LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DocumentAgent (src/agent.py)                 │   │
│  │                                                                     │   │
│  │   ┌─────────────────┐    ┌─────────────────────────────────────┐   │   │
│  │   │  Query Router   │───▶│      LLM Provider (Configurable)    │   │   │
│  │   │                 │    │                                     │   │   │
│  │   └─────────────────┘    │  ┌─────────────┐   ┌─────────────┐ │   │   │
│  │                          │  │  Ollama     │   │   Claude    │ │   │   │
│  │                          │  │  (Local)    │   │    API      │ │   │   │
│  │                          │  │  gemma3:4b  │   │   (Cloud)   │ │   │   │
│  │                          │  └─────────────┘   └─────────────┘ │   │   │
│  │                          └─────────────────────────────────────┘   │   │
│  │                                      │                              │   │
│  │                                      ▼                              │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │              Prompt Template (RAG Context)                │   │   │
│  │   │                                                             │   │   │
│  │   │  "Use the following context from ENTSO-E reports to      │   │   │
│  │   │   answer the question... [retrieved chunks]"               │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Retriever (src/retrieval.py)                    │   │
│  │                                                                     │   │
│  │   ┌──────────────────┐      ┌──────────────────┐                    │   │
│  │   │  EMBEDDING       │      │  RERANKER        │                    │   │
│  │   │  all-MiniLM-L6-v2│      │  cross-encoder   │                    │   │
│  │   │  (384-dim)       │      │  ms-marco-MiniLM│                    │   │
│  │   │                  │      │                  │                    │   │
│  │   │  Query ────────▶ │─────▶│  Score chunks   │                    │   │
│  │   │  [embedding]     │      │  [cross-encoder]│                    │   │
│  │   └──────────────────┘      └──────────────────┘                    │   │
│  │              │                        │                           │   │
│  │              ▼                        ▼                           │   │
│  │   ┌──────────────────────────────────────────────────────┐        │   │
│  │   │  HYBRID BOOST (Keyword Matching)                   │        │   │
│  │   │                                                      │        │   │
│  │   │  • Year matching (+2.0 boost)                       │        │   │
│  │   │  • Region keywords (+1.5 boost)                     │        │   │
│  │   │  • Metric keywords (+1.0 boost)                     │        │   │
│  │   └──────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             VECTOR DATABASE                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           Qdrant (Docker)                           │   │
│  │                                                                     │   │
│  │   Collection: "entsoe_reports"                                     │   │
│  │   │                                                                 │   │
│  │   ├── Points: 6,223                                                │   │
│  │   ├── Vector Size: 384                                             │   │
│  │   ├── Distance: Cosine                                             │   │
│  │   └── Metadata: source, chunk_id, page, text                       │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ Point Structure:                                            │   │   │
│  │   │ {                                                         │   │   │
│  │   │   id: 0,                                                  │   │   │
│  │   │   vector: [0.12, -0.45, ...],  # 384-dim embedding         │   │   │
│  │   │   payload: {                                              │   │   │
│  │   │     text: "The FRR process comprises...",                 │   │   │
│  │   │     source: "Balancing_Report_2022.pdf",                  │   │   │
│  │   │     chunk_id: 59,                                         │   │   │
│  │   │     page: 47                                              │   │   │
│  │   │   }                                                       │   │   │
│  │   │ }                                                         │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ (Populated by)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Ingestion Pipeline (src/ingestion.py)            │   │
│  │                                                                     │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │   │
│  │   │   Load PDFs  │───▶│   Chunk      │───▶│   Create        │   │   │
│  │   │   PyMuPDF    │    │   Recursive   │    │   Embeddings    │   │   │
│  │   │              │    │   512 tokens  │    │   all-MiniLM    │   │   │
│  │   └──────────────┘    └──────────────┘    └──────────────────┘   │   │
│  │          │                   │                       │             │   │
│  │          ▼                   ▼                       ▼             │   │
│  │   8 PDFs (78 MB)    6,223 chunks (512)        Batch size: 32      │   │
│  │   1,280 pages total    Overlap: 100 tokens    CPU inference      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNCERTAINTY QUANTIFICATION                          │
│                              (Parallel Track)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   UncertaintyQuantifier                            │   │
│  │                      (src/uncertainty.py)                           │   │
│  │                                                                     │   │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │   │ Signal 1:        │  │ Signal 2:        │  │ Signal 3:        │   │   │
│  │   │ Retrieval        │  │ Answer           │  │ Self-            │   │   │
│  │   │ Confidence       │  │ Consistency      │  │ Evaluation       │   │   │
│  │   │                  │  │                  │  │                  │   │   │
│  │   │ Based on         │  │ Semantic sim     │  │ LLM judges       │   │   │
│  │   │ reranker scores  │  │ (answer vs       │  │ its own answer   │   │   │
│  │   │                  │  │ context)         │  │ quality          │   │   │
│  │   └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  │           │                    │                     │               │   │
│  │           └────────────────────┼─────────────────────┘               │   │
│  │                                ▼                                     │   │
│  │   ┌──────────────────────────────────────────────────────────────┐   │   │
│  │   │  COMBINED CONFIDENCE SCORE                                   │   │   │
│  │   │  Overall = 0.4×Retrieval + 0.35×Consistency + 0.25×SelfEval │   │   │
│  │   │                                                             │   │   │
│  │   │  Thresholds:                                                │   │   │
│  │   │    - >60%: "High confidence"                               │   │   │
│  │   │    - >30%: "Answer with warning"                           │   │   │
│  │   │    - <30%: "Decline to answer"                             │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘


## Data Flow Diagram

### Query Processing Flow

```
┌─────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  USER   │────▶│  Streamlit  │────▶│  DocumentAgent │────▶│  Retriever  │
│  Query  │     │     UI      │     │   (src/)     │     │  (search)   │
└─────────┘     └─────────────┘     └──────────────┘     └──────┬──────┘
                                                                   │
                          ┌──────────────────────────────────────────┘
                          │
                          ▼
             ┌────────────────────┐
             │  Qdrant Vector DB  │
             │  Query embedding   │
             │  Cosine similarity │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │   Return top 10      │
             │   similar chunks     │
             └─────────┬────────────┘
                       │
                       ▼
             ┌────────────────────┐
             │  Cross-encoder     │
             │  Rerank chunks     │
             └─────────┬───────────┘
                       │
                       ▼
             ┌────────────────────┐
             │  Hybrid keyword    │
             │  boost applied     │
             └─────────┬───────────┘
                       │
                       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐
│  LLM Prompt │◀────│  Build context │◀────│  Top 5      │◀────│ Reranked│
│  generation │     │  (5 chunks)  │     │  chunks     │     │ results │
└──────┬──────┘     └──────────────┘     └─────────────┘     └─────────┘
       │
       ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Ollama/    │────▶│  LLM generates   │────▶│   Format    │
│   Claude     │     │  answer          │     │   response  │
└──────────────┘     └──────────────────┘     └──────┬──────┘
                                                      │
       ┌───────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  UNCERTAINTY CHECK (parallel)                                         │
│  • Calculate retrieval confidence (from reranker scores)             │
│  • Calculate consistency (answer vs context semantic similarity)     │
│  • Get self-evaluation score (LLM rates its own answer)              │
│  • Combine to overall confidence score                                │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────┐
│  Confidence │────▶│  Decision:   │────▶│  Display     │────▶│  USER   │
│  Score      │     │  Answer?     │     │  Answer +    │     │         │
│             │     │  (30% thresh)│     │  Confidence  │     │         │
└─────────────┘     └───────────────┘     └──────────────┘     └─────────┘
```


## Component Interactions

### Component Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py (Streamlit)                      │
│                         Entry point                             │
└────────────────────────────┬────────────────────────────────────┘
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      src/agent.py                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  DocumentAgent                                            │  │
│  │  ├─► src/retrieval.py (Retriever) ──► Qdrant             │  │
│  │  └─► LLM (Ollama / Claude API)                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   src/uncertainty.py                            │
│  ├─► UncertaintyQuantifier                                     │
│  │   ├─► Uses same embedding model (all-MiniLM-L6-v2)         │
│  │   └─► Uses LLM for self-evaluation                          │
│  └─► Integrates with agent response                             │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │ populates
┌────────────────────────────┴────────────────────────────────────┐
│                   src/ingestion.py                              │
│  ├─► Load PDFs (PyMuPDF)                                       │
│  ├─► Chunk (RecursiveCharacterTextSplitter)                   │
│  ├─► Embed (sentence-transformers)                              │
│  └─► Store (Qdrant client)                                      │
└─────────────────────────────────────────────────────────────────┘
```


## Technology Stack

```
┌────────────────────────────────────────────────────────────────────┐
│                          FRONTEND                                   │
│  Streamlit 1.56 ───────────────────────────────────────────────────  │
│  • Chat interface • Confidence visualization • Source display        │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION                                  │
│  LangChain + LangGraph ───────────────────────────────────────────  │
│  • Prompt templates • Agent logic • Tool integration               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                           LLM                                       │
│  Primary: Ollama (gemma3:4b) ─────── Alternative: Claude API        │
│  Local: http://localhost:11434 ──── Cloud: api.anthropic.com        │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL + RERANK                            │
│  Embeddings: sentence-transformers/all-MiniLM-L6-v2                │
│  Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2                    │
│  Vector DB: Qdrant (localhost:6333)                                 │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                       UNCERTAINTY                                   │
│  Semantic similarity (same embedding model)                        │
│  LLM self-evaluation (same LLM as agent)                           │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE                                  │
│  Docker (Qdrant) ─── Python 3.12 ─── Virtual Environment (.venv)   │
└────────────────────────────────────────────────────────────────────┘
```


## Deployment Architecture

### Local Development
```
┌─────────────────────────────────────────────────────────┐
│                    Your Laptop                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │  │  Python  │  │  Ollama  │  │   Qdrant     │  │   │
│  │  │  3.12    │  │  (LLM)   │  │   (Vector)   │  │   │
│  │  │          │  │  :11434  │  │   :6333      │  │   │
│  │  └──────────┘  └──────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                             │
│                         ▼                             │
│                   http://localhost:8501               │
└─────────────────────────────────────────────────────────┘
```

### Production (Cloud)
```
┌─────────────────────────────────────────────────────────┐
│                    Cloud Platform                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐  ┌─────────────────────────┐   │   │
│  │  │  Streamlit  │  │      Qdrant Cluster     │   │   │
│  │  │     App     │  │   (Managed/Container)   │   │   │
│  │  │  (Docker)   │  │                         │   │   │
│  │  └──────┬──────┘  └─────────────────────────┘   │   │
│  │         │                                       │   │
│  │         │ ┌─────────────────────────────────┐   │   │
│  │         └▶│         LLM Provider              │   │   │
│  │           │  ┌─────────────┐ ┌─────────────┐│   │   │
│  │           │  │ Claude API  │ │  OpenAI     ││   │   │
│  │           │  │ (Preferred) │ │  (Backup)   ││   │   │
│  │           │  └─────────────┘ └─────────────┘│   │   │
│  │           └─────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```


## Data Model

### Qdrant Point Structure
```python
{
    "id": 1234,  # Sequential integer
    "vector": [0.123, -0.456, ...],  # 384-dimensional float array
    "payload": {
        "text": "The graph below showcases an overview of the major...",
        "source": "Balancing_Report_2022.pdf",
        "chunk_id": 59,
        "chunk_index": 12,
        "page": 47,
        "file_path": "data/documents/Balancing_Report_2022.pdf"
    }
}
```

### Agent Response Structure
```python
{
    "answer": "mFRR (manual Frequency Restoration Reserve)...",
    "sources": [
        "Balancing_Report_2022.pdf",
        "Balancing_report_2024.pdf"
    ],
    "confidence": 0.737,  # Overall 0-1 score
    "reasoning": {
        "retrieval_confidence": 0.822,
        "answer_consistency": 0.632,
        "self_evaluation": 0.750
    },
    "retrieved_context": "[1] Source: Balancing_Report_2022.pdf..."
}
```


## System Characteristics

### Performance Profile
| Metric | Value | Bottleneck |
|--------|-------|------------|
| **Ingestion time** | ~100s for 1,280 pages | Embedding generation |
| **Query latency** | 2-3 seconds | LLM inference + retrieval |
| **Memory usage** | ~4-5 GB peak | Embedding models + Qdrant |
| **Throughput** | ~0.3 queries/sec | Single-threaded, CPU-bound |
| **Collection size** | 6,223 vectors | Pre-computed |

### Scalability Limits
| Component | Current | Practical Limit |
|-----------|---------|-----------------|
| Documents | 8 PDFs | ~100 PDFs (RAM-bound) |
| Chunks | 6,223 | ~100,000 (Qdrant free tier) |
| Concurrent users | 1 | ~5 (Ollama local) |
| Query complexity | Simple | Multi-hop (not implemented) |


## Security Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SECURITY                                                   │
│  ├─ PDFs stored locally (not in git)                            │
│  ├─ Qdrant has no auth (local only)                             │
│  └─ .env file for API keys (excluded from git)                │
├──────────────────────────────────────────────────────────────────┤
│  API SECURITY                                                    │
│  ├─ Ollama: localhost only (no external access)                │
│  ├─ Claude API: HTTPS + API key auth                           │
│  └─ Qdrant: localhost only (docker network)                      │
├──────────────────────────────────────────────────────────────────┤
│  PROMPT INJECTION                                                │
│  ├─ No user input sanitization (demo system)                     │
│  └─ LLM could be manipulated (acceptable risk for portfolio)     │
└─────────────────────────────────────────────────────────────────┘
```


## Future Architecture Improvements

### Short Term (Week 4+)
```
┌─────────────────────────────────────────────────────────────────┐
│  HYBRID SEARCH ENHANCEMENTS                                    │
│  ├─ Add BM25 keyword search component                          │
│  ├─ Query expansion (generate variants)                         │
│  └─ Metadata filters in UI (year, document type)               │
├──────────────────────────────────────────────────────────────────┤
│  CACHING LAYER                                                  │
│  ├─ Redis for query embeddings                                   │
│  └─ Response cache for common questions                         │
└─────────────────────────────────────────────────────────────────┘
```

### Medium Term (Month 2+)
```
┌─────────────────────────────────────────────────────────────────┐
│  STRUCTURED DATA EXTRACTION                                    │
│  ├─ Parse tables to JSON during ingestion                      │
│  ├─ Separate collection for KPIs                                │
│  └─ SQL queries for fact lookup                                 │
├──────────────────────────────────────────────────────────────────┤
│  ADVANCED RETRIEVAL                                             │
│  ├─ Parent document retrieval (return full sections)           │
│  ├─ Multi-hop reasoning (follow references)                     │
│  └─ Query understanding (detect definition vs fact queries)  │
└─────────────────────────────────────────────────────────────────┘
```

### Long Term (Month 3+)
```
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCTION FEATURES                                           │
│  ├─ Fine-tuned domain embeddings                                │
│  ├─ Citation verification layer                                 │
│  ├─ Multi-modal (chart/figure understanding)                    │
│  └─ Feedback loop for continuous improvement                    │
└─────────────────────────────────────────────────────────────────┘
```

---

*Architecture last updated: 2026-04-07*
*System version: 1.0 (Portfolio Release)*
