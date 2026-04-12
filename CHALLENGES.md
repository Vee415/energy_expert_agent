# Project Challenges: Ideal vs Realistic

This document captures the gap between an ideal production RAG system and what was implemented in this portfolio project. It demonstrates awareness of real-world constraints and technical debt.

---

## 1. Architecture: Ideal vs Implemented

### Ideal Production System
```
┌─────────────────────────────────────────────────────────────────┐
│  HYBRID SEARCH LAYER                                            │
│  ├── Semantic: Dense embeddings (384-dim, cosine)              │
│  ├── Keyword: BM25 + exact phrase matching                      │
│  ├── Structured: SQL queries on extracted KPIs                    │
│  └── Graph: Entity relationships (region→metric→year)           │
├─────────────────────────────────────────────────────────────────┤
│  ADVANCED RETRIEVAL                                             │
│  ├── Parent document retrieval (return full sections)           │
│  ├── Multi-hop (follow references across chunks)                │
│  ├── Query expansion (generate variants)                        │
│  └── Reranking: LLM-based pointwise scoring                     │
├─────────────────────────────────────────────────────────────────┤
│  ANSWER GENERATION                                              │
│  ├── Citation verification (claim → source validation)        │
│  ├── Structured output (JSON with confidence fields)            │
│  ├── Multi-step reasoning (break complex queries)               │
│  └── Fallback to web search when confidence < threshold         │
└─────────────────────────────────────────────────────────────────┘
```

### Implemented System (Portfolio Version)
```
┌─────────────────────────────────────────────────────────────────┐
│  SEMANTIC SEARCH ONLY                                           │
│  └── Dense embeddings (all-MiniLM-L6-v2, 384-dim)               │
├─────────────────────────────────────────────────────────────────┤
│  BASIC RETRIEVAL                                                │
│  ├── 512-token chunks (may split tables)                        │
│  ├── Cross-encoder reranking (ms-marco-MiniLM)                │
│  └── Simple keyword boost (basic hybrid)                        │
├─────────────────────────────────────────────────────────────────┤
│  ANSWER GENERATION                                              │
│  ├── Single-step prompting                                      │
│  ├── Local LLM (gemma3:4b) or Claude API                      │
│  └── 3-signal uncertainty quantification                        │
└─────────────────────────────────────────────────────────────────┘
```

**Gap:** ~60% of ideal capability. Missing: structured data layer, parent documents, citation verification.

---

## 2. Data Quality: Ideal vs Reality

### Ideal Data Characteristics
| Aspect | Ideal | RAG Score |
|--------|-------|-----------|
| **Format** | Structured JSON/API | 9/10 |
| **Schema** | Normalized tables (region, year, metric, value) | 9/10 |
| **Density** | Every sentence contains facts | 8/10 |
| **Consistency** | Standardized terminology | 8/10 |
| **Granularity** | Atomic facts, easily chunkable | 8/10 |

### Actual Data Characteristics
| Aspect | Reality | RAG Score |
|--------|---------|-----------|
| **Format** | PDF reports (bureaucratic prose) | 4/10 |
| **Schema** | Unstructured narrative with embedded tables | 3/10 |
| **Density** | ~30% fluff ("This report aims to...") | 4/10 |
| **Consistency** | Evolving terminology across years | 5/10 |
| **Granularity** | Multi-page arguments, fragmented tables | 3/10 |

### Specific Data Challenges

#### Challenge 2.1: Table Fragmentation
**Ideal:**
```json
{
  "table": "balancing_energy_2021",
  "rows": [
    {"region": "Nordic", "metric": "mFRR_volume", "value": 450, "unit": "GWh"},
    {"region": "Nordic", "metric": "aFRR_volume", "value": 120, "unit": "GWh"}
  ]
}
```

**Reality (before table-aware chunking):**
```
Table 4.3.2: Yearly activated volume of balancing energy
[Text continues across 3 PDF pages]
Row 1: Nordic | mFRR | 450 GWh | ...
[Chunk boundary splits here]
...continued from previous page...
Row 2: Nordic | aFRR | 120 GWh | ...
```

**Improved (after table-aware chunking):**
- Tables are now detected during ingestion and kept as single intact chunks
- Each chunk gets `content_type: "table"` or `"narrative"` metadata
- Table chunks are marked `[TABLE]` in the LLM context so the model knows to read them as structured data
- 197 table chunks detected across 1,280 pages (6,293 total chunks)
- Tables that exceed `chunk_size` stay as one oversized chunk rather than being fragmented

**Remaining limitation:** Chart/figure data extracted from PDFs (axis labels, bar chart values) appears as garbled numeric noise in chunks. These are not real data tables and provide no useful information. A chart-noise filter (detecting sequences of short numbers without sentence structure) would be needed to address this.

#### Challenge 2.2: Temporal References
- "As shown in Figure 3.2" → Figure 3.2 is 20 pages away (different chunk)
- "The previous report highlighted..." → Refers to 2022 document (not in context)
- "Since 2021..." → Requires cross-year aggregation (not supported)

#### Challenge 2.3: Implicit Information
**Query:** "Which region has the highest mFRR volume?"

**Problem:** Answer requires:
1. Finding all mFRR volume entries
2. Comparing across regions
3. Selecting maximum

**Current system:** Cannot perform aggregation across multiple chunks.

---

## 3. Performance: Expectations vs Reality

### Retrieval Metrics

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| **Recall@5** | 0.85 | ~0.60 | Semantic search misses exact keyword matches |
| **Precision@5** | 0.90 | ~0.75 | Some irrelevant chunks slip through |
| **MRR** | 0.80 | ~0.55 | Correct chunk rarely in position 1 |
| **Latency** | <500ms | ~2-3s | Local LLM + embedding overhead |

### Answer Quality

| Query Type | Ideal Answer | Actual Answer | Grade |
|------------|--------------|---------------|-------|
| **Definition** (What is mFRR?) | Comprehensive with examples | Basic definition | B+ |
| **Trend** (2024 market trends) | Specific % changes, €/MWh values | General narrative | C+ |
| **Specific fact** (Nordic mFRR 2021 volume) | "450 GWh" with source | "I don't have this information" | F |
| **Comparison** (mFRR vs aFRR) | Structured comparison table | Paragraph description | B |
| **Out-of-domain** (Tesla CEO) | "Not in my knowledge base" | Correctly declines | A |

### Why Scores Are Lower Than Ideal

1. **Chunking strategy** (512 tokens) optimized for narrative, not tables
2. **No query understanding** — system doesn't parse "2021" as a filter
3. **No aggregation capability** — can't sum/compare across chunks
4. **LLM paraphrasing** — prefers generic summaries over exact quotes

---

## 4. Technical Limitations

### 4.1 Embedding Model
**Using:** `all-MiniLM-L6-v2` (22MB, 384-dim)

**Limitations:**
- Trained on general text, not energy domain
- No understanding of units ("GWh" ≈ "MWh" in embedding space)
- Numbers treated as noise ("2021" ≈ "2022")

**Ideal:** Domain-adapted embeddings trained on energy reports

### 4.2 Context Window
**Current:** ~2,500 tokens (5 chunks × 512 tokens)

**Problem:** Complex questions need 10-15 chunks to cover:
- 2021 Nordic data
- 2022 Nordic data  
- Comparison methodology
- Regional definitions

**Ideal:** 8K-32K context with parent document retrieval

### 4.3 LLM Capability
**Using:** `gemma3:4b` (local) or `claude-3-haiku` (API)

**Limitations:**
- 4B parameters: Limited reasoning for complex aggregation
- Instruction following: Sometimes ignores "use only context" directive
- Citation: Cannot consistently cite sources

**Ideal:** GPT-4/Claude-3-Sonnet with fine-tuned citation

### 4.4 Infrastructure
**Current:** Single machine (16GB RAM, CPU inference)

**Bottlenecks:**
- Qdrant + Ollama compete for RAM
- No GPU = slow embeddings (~2s/query)
- No caching = repeated work

**Ideal:** GPU-enabled cloud deployment with Redis caching

---

## 5. Evaluation Challenges

### The Ground Truth Problem

**Challenge:** How do you evaluate answers on 1,280 pages of unstructured text?

**Attempted approaches:**
1. **Manual QA pairs** — Created 50 question-answer pairs
   - Time: 8 hours
   - Coverage: ~4% of document content
   - Bias: Questions I thought of, not user questions

2. **LLM-as-judge** — Use GPT-4 to grade answers
   - Cost: $5-10 per evaluation run
   - Problem: LLM evaluator has same limitations as system

3. **RAGAS metrics** — Faithfulness, relevance, context precision
   - Requires: More setup, API calls
   - Status: Planned for Week 3

**Reality:** Rigorous evaluation is 30-40% of project effort.

### Metrics That Matter (But We Can't Measure)

| Metric | Why It Matters | Why We Can't Measure |
|--------|---------------|----------------------|
| **User satisfaction** | Did the user find what they needed? | No user study |
| **Hallucination rate** | % of answers with false claims | Expensive to label |
| **Query coverage** | % of answerable questions answered | No exhaustive test set |
| **Latency under load** | Performance with 10+ concurrent users | Single-user setup |

---

## 6. Future Improvements (Prioritized)

### High Impact / Low Effort
- [x] Table-aware chunking (tables kept intact, `content_type` metadata, `[TABLE]` context tags)
- [ ] Increase `top_k` from 5 to 10 in retrieval
- [ ] Add metadata filters (year, document_type) to UI
- [ ] Cache embeddings to reduce latency
- [ ] Prompt engineering for structured citations

### High Impact / High Effort
- [ ] Filter chart/figure noise from table chunks (garbled axis values from PDF charts)
- [ ] Parse tables to structured JSON, store in separate collection
- [ ] Implement parent document retrieval (return full sections)
- [ ] Fine-tune embedding model on energy domain
- [ ] Add query classification (definition vs fact vs comparison)

### Critical for Production
- [ ] Hybrid search with BM25 (keyword) component
- [ ] Citation verification layer (claim ↔ source check)
- [ ] Confidence calibration (current scores not well-calibrated)
- [ ] Human-in-the-loop feedback collection

---

## 7. Lessons Learned

### What Worked
✅ **Uncertainty quantification** — Users can see when to trust answers
✅ **Cross-encoder reranking** — Significant improvement over pure vector search
✅ **Modular architecture** — Easy to swap LLM providers
✅ **Local-first approach** — Works offline, no API costs during dev
✅ **Table-aware chunking** — Keeping tables intact prevents data loss across chunk boundaries

### What Didn't Work
❌ **Pure semantic search** — Inadequate for specific fact retrieval
❌ **Single-step prompting** — Complex queries need reasoning chains
❌ **Generic embeddings** — Domain adaptation is essential

### Surprises
🤯 **PDF parsing is harder than ML** — PyMuPDF works but loses structure  
🤯 **LLMs resist quoting** — They prefer to paraphrase (causes hallucination)  
🤯 **Self-evaluation works** — LLM can judge its own answer quality  
🤯 **Users ask bad questions** — "What about 2021?" (no context)

---

## 8. Honest Self-Assessment

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Completeness** | 7/10 | Core RAG works, missing advanced features |
| **Code quality** | 8/10 | Modular, documented, type hints |
| **Production readiness** | 4/10 | Needs evaluation, caching, monitoring |
| **Innovation** | 6/10 | Uncertainty quantification is solid |
| **Data handling** | 6/10 | Acknowledges PDF limitations |
| **Documentation** | 7/10 | This file helps |

**Overall:** Solid portfolio project demonstrating real-world RAG understanding, with clear path to production improvements.

---

## 9. Advice for Future Builders

### If Starting This Project Today

1. **Spend 2x time on data prep**
   - Extract tables to structured format first
   - Add rich metadata (year, region, metric_type)

2. **Start with hybrid search**
   - BM25 + semantic from day 1
   - Don't "discover" keyword matching later

3. **Build evaluation first**
   - Create 100 QA pairs before any ML
   - Use them to guide architecture decisions

4. **Plan for the 80% case**
   - Definition questions: Easy
   - Specific fact lookup: Hard
   - Focus on what your data actually supports

5. **Show your work**
   - Document limitations (this file)
   - Interviewers respect self-awareness

---

## 10. Conclusion

This project demonstrates a **realistic understanding of production RAG limitations**:

- ❌ Not a perfect system
- ✅ A working system with known tradeoffs
- ✅ Clear documentation of gaps
- ✅ Path to improvement

**The value is not in claiming perfection, but in demonstrating deep understanding of where the hard problems lie.**

---

*Last updated: 2026-04-09*
*Author: Documenting honest assessment of portfolio project limitations*
