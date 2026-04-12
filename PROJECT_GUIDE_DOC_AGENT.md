# Project 1: Intelligent Document Agent with Uncertainty Awareness

## One-Line Pitch
A RAG-powered Q&A agent over technical documents that answers questions, cites sources, and quantifies how confident it is in each answer.

---

## Tech Stack
| Tool | Purpose |
|---|---|
| LangChain | Agent orchestration |
| Qdrant | Vector database (local via Docker) |
| sentence-transformers | Embeddings + reranking |
| RAGAS | Evaluation framework |
| Pydantic | Structured outputs |
| Streamlit | Frontend |
| HuggingFace Spaces | Deployment (free) |
| python-dotenv | Environment management |

---

## Folder Structure
```
intelligent-doc-agent/
├── data/
│   └── documents/          # Your PDFs/docs go here
├── src/
│   ├── ingestion.py        # Load, chunk, embed documents
│   ├── retrieval.py        # Query vector DB, rerank results
│   ├── agent.py            # LangChain agent + tools
│   ├── uncertainty.py      # Confidence scoring logic
│   └── evaluation.py       # RAGAS evaluation pipeline
├── app.py                  # Streamlit frontend
├── requirements.txt
├── README.md
└── .env                    # API keys (never commit this)
```

---

## What Documents to Use
Use **ENTSO-E energy market reports** (free, public, technical).
- Download from: https://www.entsoe.eu/publications/
- Recommended: Annual Reports, Market Reports, Transparency Reports
- Why: Directly ties to your energy sector interest, makes the project domain-specific and memorable in interviews

---

## Component Breakdown

### 1. `ingestion.py` — Document Pipeline
**Responsibilities:**
- Load PDFs using LangChain document loaders
- Chunk text into segments (experiment with 512 and 1024 token sizes)
- Generate embeddings using `sentence-transformers` or OpenAI
- Store embeddings in Qdrant (local Docker instance)

**Key decisions to document:**
- Why you chose your chunk size
- Which embedding model you used and why
- How you handled overlap between chunks

---

### 2. `retrieval.py` — Query & Rerank
**Responsibilities:**
- Take user query as input
- Embed the query using same model as ingestion
- Retrieve top-k chunks from Qdrant by similarity
- Rerank results using a cross-encoder (`sentence-transformers` has one built in)
- Return ranked chunks with similarity scores attached

**Key decisions to document:**
- What value of k you chose for retrieval
- Which cross-encoder model you used
- How reranking changed results vs raw retrieval

---

### 3. `agent.py` — LangChain Agent
**Responsibilities:**
- Implement a ReAct agent using LangChain
- Define tools: document retrieval, optional calculator, optional web search
- Write a system prompt that instructs the agent to always cite sources
- Use Pydantic to enforce structured output:

```python
class AgentResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    reasoning: str
```

**Key decisions to document:**
- Why ReAct architecture
- How you structured the system prompt
- How structured output improved reliability

---

### 4. `uncertainty.py` — Confidence Scoring (Your Differentiator)
**Three uncertainty signals to combine:**

**Signal 1 — Retrieval Score Threshold**
- If top similarity score < 0.70 → flag as uncertain
- Simple but effective first signal

**Signal 2 — Answer Consistency**
- Run the same query 3 times with `temperature > 0`
- Compare answers using semantic similarity
- High variance = low confidence

**Signal 3 — Self-Evaluation**
- Prompt the LLM: *"Rate your confidence in this answer from 1-10 and explain why"*
- Parse the score and reasoning

**Final Score:**
Combine all three into a single `uncertainty_score` (0.0 to 1.0).
Map to traffic light: 🟢 High confidence | 🟡 Medium | 🔴 Low

**This is what makes your project stand out from every other RAG tutorial.**

---

### 5. `evaluation.py` — RAGAS Pipeline
**Write 20-30 test questions manually across these categories:**
- Questions with clear answers in the documents
- Questions with partial answers
- Questions the documents cannot answer at all

**RAGAS metrics to measure:**
| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer match retrieved docs |
| Answer Relevancy | Does it actually answer the question |
| Context Precision | Are retrieved chunks relevant |
| Context Recall | Did we retrieve enough information |

**Output:** A results table you can put directly in your README.

---

### 6. `app.py` — Streamlit Frontend
**Keep it simple. Show:**
- Text input box for the question
- Answer displayed clearly
- Confidence meter (🟢🟡🔴 based on uncertainty score)
- Source chunks highlighted with page numbers
- Expandable section showing raw retrieved context

---

## Build Order (Week by Week)

### Week 1 — Foundation
| Day | Task |
|---|---|
| Day 1-2 | Set up repo, install dependencies, run Qdrant locally in Docker |
| Day 3-4 | Build `ingestion.py` — get documents chunked and embedded |
| Day 5-7 | Build `retrieval.py` — basic retrieval working, tune chunk size |

### Week 2 — Agent + Uncertainty
| Day | Task |
|---|---|
| Day 1-2 | Build `agent.py` — basic Q&A working end to end |
| Day 3-4 | Build `uncertainty.py` — add all three confidence signals |
| Day 5-6 | Build `app.py` — Streamlit frontend |
| Day 7 | Deploy to HuggingFace Spaces or Streamlit Cloud |

### Week 3 — Polish + Evaluate
| Day | Task |
|---|---|
| Day 1-2 | Write 20-30 test questions, run RAGAS evaluation |
| Day 3-4 | Fix biggest failures identified by evaluation |
| Day 5 | Write README with architecture diagram and results table |
| Day 6 | Write LinkedIn post about what you learned |
| Day 7 | Submit to job applications as portfolio project |

---

## How to Prompt Your Coding Agent

**Give this context upfront before asking it to write anything:**

> "I am building a RAG agent with uncertainty quantification over energy market PDFs. The stack is LangChain, Qdrant, sentence-transformers, and RAGAS. I want clean, modular, well-documented code with type hints. Each file should have one clear responsibility. Functions should have docstrings. Start with ingestion.py"

**Rules for working with the coding agent:**
- Go **file by file** — never ask it to generate everything at once
- Review and understand each file before moving to the next
- When something breaks, try to debug it yourself first for 20 minutes before asking the agent
- Ask the agent to explain decisions, not just write code

---

## Requirements.txt (Starting Point)
```
langchain
langchain-community
langchain-openai
qdrant-client
sentence-transformers
ragas
pydantic
streamlit
fastapi
uvicorn
python-dotenv
pypdf
tiktoken
openai
```

---

## README Must Include
- [ ] Architecture diagram (draw in Excalidraw, embed as image)
- [ ] RAGAS evaluation results table with actual scores
- [ ] 3-5 example questions with answers and confidence scores
- [ ] Clear installation instructions
- [ ] Live demo link
- [ ] Brief explanation of uncertainty quantification approach
- [ ] What you would improve with more time

---

## What Makes This Stand Out
| Standard RAG project | Your project |
|---|---|
| Returns an answer | Returns answer + confidence score |
| No evaluation | RAGAS evaluation with real metrics |
| Basic retrieval | Retrieval + reranking |
| Generic domain | Energy market domain (sector relevant) |
| No structure | Pydantic structured outputs |

---

## Interview Talking Points
When asked about this project be ready to answer:

1. **"Why uncertainty quantification?"** — Because in high-stakes domains like energy markets, knowing when a system doesn't know is as important as knowing the answer. Inspired by my Master's thesis work on stochastic systems.

2. **"How did you evaluate it?"** — Used RAGAS framework, measured faithfulness, relevancy, precision and recall across 25 hand-crafted test questions covering different difficulty levels.

3. **"What would you improve?"** — Add knowledge graph layer (Neo4j) for structured entity relationships, implement active learning to improve on low-confidence answers, add streaming responses for better UX.

4. **"Why Qdrant over Pinecone?"** — Qdrant runs locally for free, open source, production-grade, and I wanted to understand the infrastructure rather than abstract it away behind a managed service.

---

## Deployment Checklist
- [ ] Remove all API keys from code (use .env)
- [ ] Add .env to .gitignore
- [ ] Test cold start (fresh clone, does it work?)
- [ ] Record a 60-second demo video
- [ ] Deploy to HuggingFace Spaces
- [ ] Add live demo link to CV and LinkedIn

---

*Built as Portfolio Project 1 — AI Engineering pivot, April 2026*
*Target roles: Junior AI Engineer, ML Engineer, Applied AI Engineer*
