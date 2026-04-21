"""Part 1 RAG pipeline for IDS 568 Milestone 6.

This script builds a complete retrieval-augmented generation pipeline over a PDF
source document. It uses:
- PyPDF for document ingestion
- custom chunking with overlap
- sentence-transformers for embeddings
- FAISS for vector search
- Ollama for grounded generation with a local open-weight instruct model

Example usage:
    python rag_pipeline.py ingest \
        --pdf_path "module1-slides.pdf" \
        --index_dir "rag_index"

    python rag_pipeline.py query \
        --index_dir "rag_index" \
        --question "What are the six lifecycle stages?"

    python rag_pipeline.py evaluate \
        --index_dir "rag_index" \
        --output_json "rag_eval_results.json"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "mistral:7b-instruct"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_TOP_K = 4


@dataclass
class PageDocument:
    page_number: int
    text: str
    source: str


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page_number: int
    source: str
    start_char: int
    end_char: int


@dataclass
class RetrievedChunk:
    chunk_id: str
    page_number: int
    source: str
    text: str
    score: float


@dataclass
class QueryResult:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    timings_ms: Dict[str, float]
    prompt: str


def normalize_whitespace(text: str) -> str:
    """Clean raw PDF text while preserving paragraph boundaries."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(re.sub(r"\s+", " ", line))
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def load_pdf_pages(pdf_path: str) -> List[PageDocument]:
    """Extract text from each PDF page as a separate page document."""
    reader = PdfReader(pdf_path)
    pages: List[PageDocument] = []
    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_whitespace(raw_text)
        if not text:
            continue
        pages.append(
            PageDocument(
                page_number=i,
                text=text,
                source=os.path.basename(pdf_path),
            )
        )
    return pages


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[tuple[str, int, int]]:
    """Split long text into overlapping windows using sentence and whitespace hints."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks: List[tuple[str, int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)
        end = max_end

        if end < text_length:
            window = text[start:end]
            candidates = [window.rfind("\n\n"), window.rfind(". "), window.rfind("; "), window.rfind(" ")]
            best = max(candidates)
            if best > int(chunk_size * 0.6):
                end = start + best + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))

        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_documents(
    pages: Sequence[PageDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Chunk each page while preserving page metadata."""
    chunks: List[TextChunk] = []

    for page in pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        running_index = 0
        page_chunk_idx = 0

        if not paragraphs:
            paragraphs = [page.text]

        buffer = ""
        buffer_start = 0

        for para in paragraphs:
            if not buffer:
                buffer = para
                buffer_start = running_index
            elif len(buffer) + 2 + len(para) <= chunk_size:
                buffer += "\n\n" + para
            else:
                for chunk_text, start_char, end_char in split_long_text(buffer, chunk_size, chunk_overlap):
                    chunks.append(
                        TextChunk(
                            chunk_id=f"p{page.page_number}_c{page_chunk_idx}",
                            text=chunk_text,
                            page_number=page.page_number,
                            source=page.source,
                            start_char=buffer_start + start_char,
                            end_char=buffer_start + end_char,
                        )
                    )
                    page_chunk_idx += 1
                buffer = para
                buffer_start = running_index
            running_index += len(para) + 2

        if buffer:
            for chunk_text, start_char, end_char in split_long_text(buffer, chunk_size, chunk_overlap):
                chunks.append(
                    TextChunk(
                        chunk_id=f"p{page.page_number}_c{page_chunk_idx}",
                        text=chunk_text,
                        page_number=page.page_number,
                        source=page.source,
                        start_char=buffer_start + start_char,
                        end_char=buffer_start + end_char,
                    )
                )
                page_chunk_idx += 1

    return chunks


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], batch_size=1)[0]


class FAISSVectorStore:
    def __init__(self, dimension: int):
        # Cosine similarity via normalized vectors and inner product.
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []

    def add(self, embeddings: np.ndarray, chunks: Sequence[TextChunk]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    source=chunk.source,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def save(self, index_dir: str) -> None:
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "FAISSVectorStore":
        in_dir = Path(index_dir)
        index = faiss.read_index(str(in_dir / "index.faiss"))
        with open(in_dir / "chunks.json", "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        store = cls(dimension=index.d)
        store.index = index
        store.chunks = [TextChunk(**item) for item in raw_chunks]
        return store


class OllamaGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 180,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()


class RAGPipeline:
    def __init__(
        self,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        ollama_model_name: str = DEFAULT_OLLAMA_MODEL,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.embedder = EmbeddingService(embed_model_name)
        self.generator = OllamaGenerator(
            model_name=ollama_model_name,
            base_url=ollama_base_url,
        )
        self.vector_store: FAISSVectorStore | None = None
        self.config: Dict[str, Any] = {
            "embed_model_name": embed_model_name,
            "ollama_model_name": ollama_model_name,
            "ollama_base_url": ollama_base_url,
        }

    def ingest_pdf(
        self,
        pdf_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> Dict[str, Any]:
        pages = load_pdf_pages(pdf_path)
        chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        store = FAISSVectorStore(dimension=embeddings.shape[1])
        store.add(embeddings, chunks)
        self.vector_store = store
        self.config.update(
            {
                "pdf_path": pdf_path,
                "pdf_filename": os.path.basename(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_pages": len(pages),
                "num_chunks": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]),
            }
        )
        return self.config.copy()

    def save(self, index_dir: str) -> None:
        if self.vector_store is None:
            raise ValueError("Vector store has not been created. Run ingest_pdf() first.")
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(index_dir)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "RAGPipeline":
        in_dir = Path(index_dir)
        with open(in_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        pipeline = cls(
            embed_model_name=config["embed_model_name"],
            ollama_model_name=config["ollama_model_name"],
            ollama_base_url=config.get("ollama_base_url", "http://localhost:11434"),
        )
        pipeline.vector_store = FAISSVectorStore.load(index_dir)
        pipeline.config = config
        return pipeline

    def build_prompt(self, question: str, retrieved_chunks: Sequence[RetrievedChunk]) -> str:
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Chunk {i} | Source: {chunk.source} | Page: {chunk.page_number} | Score: {chunk.score:.4f}]\n{chunk.text}"
            )
        context = "\n\n".join(context_blocks)
        return (
            "You are answering questions about a course slide deck. "
            "Use ONLY the provided context. "
            "If the context is insufficient, explicitly say so. "
            "When possible, cite the relevant page number(s) in your answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def retrieve(self, question: str, k: int = DEFAULT_TOP_K) -> tuple[List[RetrievedChunk], float]:
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        start = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        chunks = self.vector_store.search(query_embedding, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return chunks, latency_ms

    def query(self, question: str, k: int = DEFAULT_TOP_K) -> QueryResult:
        total_start = time.perf_counter()
        retrieved_chunks, retrieval_ms = self.retrieve(question, k=k)
        prompt = self.build_prompt(question, retrieved_chunks)

        generation_start = time.perf_counter()
        answer = self.generator.generate(prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return QueryResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            timings_ms={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "end_to_end_ms": total_ms,
            },
            prompt=prompt,
        )


DEFAULT_EVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What are the six stages of the ML/AI lifecycle?",
        "relevant_pages": [19],
    },
    {
        "query": "Why is the ML lifecycle described as circular rather than linear?",
        "relevant_pages": [18, 91],
    },
    {
        "query": "What does Git track well in ML projects and what does it not track efficiently?",
        "relevant_pages": [21],
    },
    {
        "query": "Why do virtual environments matter in MLOps?",
        "relevant_pages": [27, 29],
    },
    {
        "query": "What is dependency pinning and why is it important for reproducibility?",
        "relevant_pages": [36, 37, 39],
    },
    {
        "query": "What types of tests are mentioned and which tests are required for Milestone 0?",
        "relevant_pages": [40, 41, 43, 44],
    },
    {
        "query": "What does the sample GitHub Actions workflow do?",
        "relevant_pages": [52, 54, 55],
    },
    {
        "query": "What are artifacts in ML pipelines? Give some examples.",
        "relevant_pages": [63, 65],
    },
    {
        "query": "How do traditional ML systems differ from LLM systems?",
        "relevant_pages": [69, 70, 72, 73],
    },
    {
        "query": "What kinds of technical debt are mentioned in ML systems and how can they be prevented?",
        "relevant_pages": [74, 76],
    },
]


def precision_at_k(retrieved_pages: Sequence[int], relevant_pages: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(retrieved_pages[:k])
    relevant = set(relevant_pages)
    return sum(1 for page in top_k if page in relevant) / k


def recall_at_k(retrieved_pages: Sequence[int], relevant_pages: Sequence[int], k: int) -> float:
    relevant = set(relevant_pages)
    if not relevant:
        return 0.0
    top_k = list(retrieved_pages[:k])
    return len({page for page in top_k if page in relevant}) / len(relevant)


def reciprocal_rank(retrieved_pages: Sequence[int], relevant_pages: Sequence[int]) -> float:
    relevant = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_pipeline(
    pipeline: RAGPipeline,
    output_json: str | None = None,
    k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for item in DEFAULT_EVAL_QUERIES:
        query = item["query"]
        relevant_pages = item["relevant_pages"]
        result = pipeline.query(query, k=k)
        retrieved_pages = [chunk.page_number for chunk in result.retrieved_chunks]

        results.append(
            {
                "query": query,
                "relevant_pages": relevant_pages,
                "retrieved_pages": retrieved_pages,
                "precision_at_k": precision_at_k(retrieved_pages, relevant_pages, k),
                "recall_at_k": recall_at_k(retrieved_pages, relevant_pages, k),
                "reciprocal_rank": reciprocal_rank(retrieved_pages, relevant_pages),
                "answer": result.answer,
                "retrieved_chunks": [asdict(chunk) for chunk in result.retrieved_chunks],
                "timings_ms": result.timings_ms,
            }
        )

    summary = {
        "num_queries": len(results),
        "k": k,
        "avg_precision_at_k": float(np.mean([r["precision_at_k"] for r in results])),
        "avg_recall_at_k": float(np.mean([r["recall_at_k"] for r in results])),
        "avg_mrr": float(np.mean([r["reciprocal_rank"] for r in results])),
        "avg_retrieval_ms": float(np.mean([r["timings_ms"]["retrieval_ms"] for r in results])),
        "avg_generation_ms": float(np.mean([r["timings_ms"]["generation_ms"] for r in results])),
        "avg_end_to_end_ms": float(np.mean([r["timings_ms"]["end_to_end_ms"] for r in results])),
    }

    payload = {"summary": summary, "results": results}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def print_query_result(result: QueryResult) -> None:
    print("\n" + "=" * 80)
    print(f"Question: {result.question}")
    print("-" * 80)
    print("Retrieved Chunks:")
    for chunk in result.retrieved_chunks:
        preview = chunk.text[:180].replace("\n", " ")
        print(
            f"  - {chunk.chunk_id} | page={chunk.page_number} | score={chunk.score:.4f} | {preview}..."
        )
    print("-" * 80)
    print("Answer:")
    print(result.answer)
    print("-" * 80)
    print("Timings (ms):")
    print(json.dumps(result.timings_ms, indent=2))
    print("=" * 80 + "\n")


def print_eval_summary(payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    print("\nEvaluation Summary")
    print("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 80)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG pipeline for Module 1 slides.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF and build FAISS index.")
    ingest_parser.add_argument("--pdf_path", required=True, help="Path to the source PDF.")
    ingest_parser.add_argument("--index_dir", required=True, help="Directory to save the index.")
    ingest_parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    ingest_parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL)
    ingest_parser.add_argument("--ollama_base_url", default="http://localhost:11434")
    ingest_parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    ingest_parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    query_parser = subparsers.add_parser("query", help="Run a grounded query.")
    query_parser.add_argument("--index_dir", required=True, help="Directory containing saved index.")
    query_parser.add_argument("--question", required=True, help="Question to answer.")
    query_parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the pipeline on 10 queries.")
    eval_parser.add_argument("--index_dir", required=True, help="Directory containing saved index.")
    eval_parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    eval_parser.add_argument("--output_json", default="rag_eval_results.json")

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(
        embed_model_name=args.embed_model,
        ollama_model_name=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
    )
    config = pipeline.ingest_pdf(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    pipeline.save(args.index_dir)
    print("Index built successfully.")
    print(json.dumps(config, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline.load(args.index_dir)
    result = pipeline.query(args.question, k=args.k)
    print_query_result(result)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline.load(args.index_dir)
    payload = evaluate_pipeline(pipeline, output_json=args.output_json, k=args.k)
    print_eval_summary(payload)
    print(f"Saved detailed results to {args.output_json}")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            return cmd_ingest(args)
        if args.command == "query":
            return cmd_query(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not reach Ollama at http://localhost:11434. "
            "Start Ollama and ensure the required model is pulled.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
