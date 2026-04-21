# RAG Evaluation Report

## Summary

This report evaluates the Part 1 retrieval-augmented generation pipeline built over `module1-slides.pdf`, a slide deck covering foundational MLOps concepts such as the ML lifecycle, Git, virtual environments, dependency pinning, CI/CD, artifacts, ML versus LLM system architecture, and technical debt. The system performs document ingestion, chunking, embedding generation, FAISS indexing, semantic retrieval, and grounded answer generation using a real open-weight instruct model served locally through Ollama.

The evaluated configuration is:

- **Corpus**: `module1-slides.pdf`
- **Embedding model**: `all-MiniLM-L6-v2`
- **Vector database**: FAISS
- **Chunk size**: 500 characters
- **Chunk overlap**: 75 characters
- **Retriever top-k**: 4
- **Generator**: Ollama
- **Final model**: `mistral:7b-instruct`
- **Model class**: 7B

## Design decisions

### Chunking strategy

I used paragraph-aware chunking with overlapping windows. This was a good fit for a slide deck because the source material is short, sectioned, and concept-dense. A chunk size of 500 characters preserves enough local context to capture bullet lists and definitions, while a 75-character overlap reduces the chance that an important concept is split across adjacent chunks.

I chose this setting to balance retrieval precision and retrieval coverage. Smaller chunks would likely improve precision but risk fragmenting slide content too aggressively. Larger chunks would provide more context per retrieval hit but increase noise and reduce semantic specificity.

### Embedding model

I used `all-MiniLM-L6-v2` because it is lightweight, widely used, and appropriate for semantic search over a small academic document collection. It also keeps indexing and query-time embedding latency low.

### Indexing strategy

I used FAISS with normalized embeddings and inner-product search. For a small single-document corpus, a flat exact index is simple and appropriate. This avoids unnecessary complexity while still providing fast retrieval.

## Evaluation methodology

The evaluation uses 10 handcrafted queries aligned with the document’s core topics. For each query, I compare the retrieved page numbers against manually defined relevant pages and compute the following retrieval metrics:

- **Precision@4**
- **Recall@4**
- **Reciprocal Rank**

I also inspect the generated answer qualitatively to determine whether the answer is grounded in the retrieved context or whether it includes unsupported claims. When there is an error, I attribute it to one of three categories:

1. **Retrieval failure**: the relevant chunk or page was not retrieved
2. **Generation failure**: the retrieved context was adequate, but the answer drifted or hallucinated
3. **Mixed failure**: partial retrieval plus partial generation drift

## Quantitative results

Replace the placeholder values below with the actual values from `rag_eval_results.json`.

| # | Query | Relevant Pages | Retrieved Pages | P@4 | R@4 | RR | Grounded? | Error Type |
|---|-------|----------------|-----------------|-----|-----|----|-----------|-----------|
| 1 | What are the six stages of the ML/AI lifecycle? | 19 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 2 | Why is the ML lifecycle described as circular rather than linear? | 18, 91 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 3 | What does Git track well in ML projects and what does it not track efficiently? | 21 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 4 | Why do virtual environments matter in MLOps? | 27, 29 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 5 | What is dependency pinning and why is it important for reproducibility? | 36, 37, 39 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 6 | What types of tests are mentioned and which tests are required for Milestone 0? | 40, 41, 43, 44 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 7 | What does the sample GitHub Actions workflow do? | 52, 54, 55 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 8 | What are artifacts in ML pipelines? Give some examples. | 63, 65 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 9 | How do traditional ML systems differ from LLM systems? | 69, 70, 72, 73 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| 10 | What kinds of technical debt are mentioned in ML systems and how can they be prevented? | 74, 76 | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |

### Aggregate retrieval metrics

Fill these in from the `summary` section of `rag_eval_results.json`:

- **Average Precision@4**: `<fill in>`
- **Average Recall@4**: `<fill in>`
- **Average MRR**: `<fill in>`

## Grounding analysis

This section should distinguish whether the model’s answer was actually supported by retrieved context.

### Expected strengths

The pipeline should perform well on:
- definitional questions with clear slide text
- procedural questions tied to specific pages
- comparisons with explicit bullet points or tables

### Common failure modes to check

- relevant page retrieved, but the model adds extra explanation not present in context
- partially relevant retrieval where one needed page is missing
- vague or broad questions that require synthesizing across multiple slides
- slide extraction artifacts that reduce retrieval quality

### Example qualitative assessment format

For each interesting case, write a short note like this:

- **Query**: `<copy query>`
- **Retrieved evidence quality**: `<strong / mixed / weak>`
- **Answer grounding**: `<fully grounded / partially grounded / not grounded>`
- **Observed issue**: `<none / hallucination / missing citation / incomplete synthesis>`
- **Error attribution**: `<retrieval / generation / mixed>`

## Latency analysis

Fill these values from the evaluation output and note the hardware used.

- **Average retrieval latency**: `<fill in>` ms
- **Average generation latency**: `<fill in>` ms
- **Average end-to-end latency**: `<fill in>` ms

### Interpretation

Retrieval latency should remain relatively low because the corpus is small and FAISS exact search is efficient for this scale. Generation latency will dominate total response time, especially on CPU-only Ollama runs. This is expected and should be documented honestly in the final submission.

## Error attribution

This section should separate retrieval problems from generation problems.

### Retrieval failures

Use this section for cases where the retrieved pages were not the most relevant ones, or where an important page was missed.

- `<fill in specific query case>`
- `<fill in specific query case>`

### Generation / grounding failures

Use this section for cases where retrieval was good but the model still added unsupported details or failed to answer conservatively.

- `<fill in specific query case>`
- `<fill in specific query case>`

## Conclusion

Overall, this pipeline provides a complete Part 1 RAG implementation with a reproducible local stack and a real open-weight instruct model in the generation loop. The most important strengths are architectural simplicity, transparent retrieval evaluation, and alignment with the source-of-truth document. The most important limitations are dependence on PDF text quality, sensitivity to chunking design, and model-side grounding drift during generation.

After running the final evaluation, replace all placeholders in this report with measured results from `rag_eval_results.json` and any observed failure cases from your own runs.
