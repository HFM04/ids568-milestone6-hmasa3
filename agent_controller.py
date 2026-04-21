"""
agent_controller.py

Part 2: Multi-Tool Agent with Retrieval Integration for IDS 568 Milestone 6.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:7b-instruct"
OLLAMA_URL = "http://localhost:11434"
TOP_K = 4


@dataclass
class TraceStep:
    step_type: str
    content: Dict[str, Any]
    timestamp: float


@dataclass
class AgentTrace:
    trace_id: str
    task: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_duration_ms: Optional[float] = None

    def add(self, step_type: str, content: Dict[str, Any]) -> None:
        self.steps.append(TraceStep(step_type, content, time.time()))


def load_pdf(pdf_path: str) -> List[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages: List[tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append((i + 1, text))

    return pages


def chunk_text(text: str, size: int = 500, overlap: int = 75) -> List[str]:
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


class Retriever:
    def __init__(self, pdf_path: str):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Dict[str, Any]] = []

        pages = load_pdf(pdf_path)

        chunk_id = 0
        for page_num, text in pages:
            for chunk in chunk_text(text):
                self.chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": chunk,
                    }
                )
                chunk_id += 1

        embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0], start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "distance": float(distances[0][rank - 1]),
                }
            )
        return results


def generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


class Agent:
    def __init__(self, pdf_path: str):
        self.retriever = Retriever(pdf_path)

    def run(self, task: str):
        start = time.time()
        trace = AgentTrace(trace_id=str(uuid.uuid4()), task=task)

        trace.add(
            "thought",
            {
                "message": "Task received. Retrieve evidence from the course document before answering."
            },
        )

        trace.add(
            "action",
            {
                "tool": "retriever",
                "input": task,
            },
        )

        retrieval_start = time.time()
        results = self.retriever.search(task)
        retrieval_ms = (time.time() - retrieval_start) * 1000

        trace.add(
            "observation",
            {
                "tool": "retriever",
                "latency_ms": retrieval_ms,
                "results": [
                    {
                        "rank": r["rank"],
                        "page": r["page"],
                        "chunk_id": r["chunk_id"],
                        "distance": r["distance"],
                        "preview": r["text"][:220],
                    }
                    for r in results
                ],
            },
        )

        context = "\n\n".join(
            [f"(Page {r['page']}, Chunk {r['chunk_id']}) {r['text']}" for r in results]
        )

        prompt = f"""
Answer using ONLY the context below.

Task:
{task}

Context:
{context}

Instructions:
- Use only the provided context.
- If the context is insufficient, say so clearly.
- Be professional and concise.
- Cite page numbers in parentheses when possible.

Answer:
""".strip()

        trace.add(
            "thought",
            {
                "message": "Enough evidence retrieved. Use reasoning tool to synthesize the final answer."
            },
        )

        trace.add(
            "action",
            {
                "tool": "reasoner",
                "input": {
                    "task": task,
                    "num_chunks": len(results),
                },
            },
        )

        generation_start = time.time()
        answer = generate(prompt)
        generation_ms = (time.time() - generation_start) * 1000

        trace.add(
            "observation",
            {
                "tool": "reasoner",
                "latency_ms": generation_ms,
                "answer_preview": answer[:500],
            },
        )

        trace.final_answer = answer
        trace.total_duration_ms = (time.time() - start) * 1000

        return answer, trace


def save_trace(trace: AgentTrace, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "trace_id": trace.trace_id,
        "task": trace.task,
        "steps": [
            {
                "step_type": s.step_type,
                "content": s.content,
                "timestamp": s.timestamp,
            }
            for s in trace.steps
        ],
        "final_answer": trace.final_answer,
        "total_duration_ms": trace.total_duration_ms,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)


DEFAULT_TASKS = [
    "What are the six stages of the ML/AI lifecycle, and what output does each stage produce?",
    "Explain why the ML lifecycle is described as circular rather than linear.",
    "Compare what Git tracks well in ML projects versus what it does not efficiently track.",
    "Summarize why virtual environments matter in MLOps.",
    "Explain dependency pinning and why exact versions are recommended for production and CI.",
    "What kinds of tests are described in the course material, and which test type is required for Milestone 0?",
    "Summarize what a basic GitHub Actions CI workflow does in this course context.",
    "What are pipelines and artifacts in ML systems? Give examples of artifacts.",
    "Compare traditional ML systems with LLM systems across model source, customization, artifacts, and latency.",
    "What kinds of technical debt are mentioned, and what practices help prevent them?",
]


def run_task(args) -> None:
    agent = Agent(args.pdf_path)
    answer, trace = agent.run(args.task)

    if args.trace_out:
        save_trace(trace, args.trace_out)

    print("\nANSWER:\n")
    print(answer)
    print(f"\nTotal duration: {trace.total_duration_ms:.2f} ms")


def run_all(args) -> None:
    agent = Agent(args.pdf_path)
    os.makedirs(args.trace_dir, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    overall_start = time.time()

    for i, task in enumerate(DEFAULT_TASKS, start=1):
        print(f"\nRunning task {i}/10...")
        answer, trace = agent.run(task)

        trace_path = os.path.join(args.trace_dir, f"task_{i}.json")
        save_trace(trace, trace_path)

        summaries.append(
            {
                "task_id": i,
                "task": task,
                "trace_file": trace_path,
                "total_duration_ms": trace.total_duration_ms,
                "answer_preview": answer[:250],
            }
        )

        print(f"Saved trace -> {trace_path}")

    total_ms = (time.time() - overall_start) * 1000
    summary = {
        "num_tasks": len(DEFAULT_TASKS),
        "total_duration_ms": total_ms,
        "average_duration_ms": total_ms / len(DEFAULT_TASKS),
        "tasks": summaries,
    }

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nAll tasks completed.")
    print(f"Summary saved -> {args.summary_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--pdf_path", required=True)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--trace_out", default="")
    run_parser.set_defaults(func=run_task)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--pdf_path", required=True)
    eval_parser.add_argument("--trace_dir", default="agent_traces")
    eval_parser.add_argument("--summary_out", default="agent_evaluation_summary.json")
    eval_parser.set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()