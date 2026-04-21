# IDS 568 – Milestone 6  
## Retrieval-Augmented Generation (RAG) + Agent Controller

---

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG) system** and a **multi-tool agent controller** using open-weight large language models.

The system:
- Builds a **vector-based knowledge store** from course materials
- Retrieves relevant context using **semantic search (FAISS + embeddings)**
- Generates grounded answers using a **local LLM (Mistral 7B via Ollama)**
- Extends into an **agent-based system** that performs multi-step reasoning with tool selection and trace logging

---

## Part 1: RAG Pipeline

### Objective
Build a system that retrieves relevant document context and generates grounded answers using a real LLM.

### Architecture

1. Document Ingestion  
2. Embedding (Sentence Transformers)  
3. Vector Store (FAISS)  
4. Retrieval (Top-K similarity)  
5. Generation (Ollama LLM)

---

### Run Part 1

```bash
python3 rag_pipeline.py ingest --pdf_path "module1-slides.pdf" --index_dir "rag_index"
```

---

## Part 2: Agent Controller

### Objective
Build a multi-tool agent that performs retrieval + reasoning and logs traces.

---

### Run Single Task

```bash
python3 agent_controller.py run --pdf_path "module1-slides.pdf" --task "Your question" --trace_out "agent_traces/task_1.json"
```

---

### Run Full Evaluation

```bash
python3 agent_controller.py evaluate --pdf_path "module1-slides.pdf"
```

---

## Project Structure

```
project/
├── rag_pipeline.py
├── agent_controller.py
├── rag_index/
├── agent_traces/
├── module1-slides.pdf
├── rag_evaluation_report.md
├── Agent_Report.docx
├── requirements.txt
└── README.md
```

---

## Results

- 92 pages processed  
- 93 chunks created  
- 10/10 tasks completed  
- ~15–30s latency per task  

---

## Strengths

- Grounded responses  
- Transparent reasoning traces  
- Fully local system  

---

## Limitations

- High latency (CPU inference)  
- No query refinement  
- Fixed retrieval strategy  

---

## Conclusion

The project successfully demonstrates a full RAG pipeline extended into an agent-based system with reasoning capabilities and traceability.
