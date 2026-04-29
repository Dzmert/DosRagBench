"""Full RAG pipeline: query -> retrieval -> generation with instrumentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from dosragbench.models.loader import LoadedModel
from dosragbench.pipeline.retriever import HNSWRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# Prompt templates
RAG_PROMPT_CHAT = """Answer the following question using only the information from the context below. If the context doesn't contain the answer, say so briefly.

Context:
{context}

Question: {query}

Answer:"""

RAG_PROMPT_BASE = """Context:
{context}

Question: {query}

Answer:"""


@dataclass
class RAGResponse:
    """Complete RAG pipeline response with timing breakdown."""

    query: str
    retrieved_docs: list
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    answer: str
    output_tokens: int
    input_tokens: int
    retrieved_adversarial_count: int = 0  # How many injected docs got retrieved

    @property
    def retrieval_latency_ms(self) -> float:
        return self.retrieval_latency_s * 1000

    @property
    def generation_latency_ms(self) -> float:
        return self.generation_latency_s * 1000


class RAGPipeline:
    """End-to-end RAG pipeline with latency instrumentation at each stage."""

    def __init__(
        self,
        retriever: HNSWRetriever,
        model: LoadedModel,
        top_k: int = 5,
        max_context_length: int = 2000,
    ):
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.max_context_length = max_context_length

    def query(self, query: str) -> RAGResponse:
        """Execute a full RAG query, returning an instrumented response."""
        # Retrieval stage
        retrieval = self.retriever.retrieve(query, top_k=self.top_k)

        # Build context
        context_parts = []
        total_len = 0
        for doc in retrieval.documents:
            doc_text = f"[Doc {doc.doc_id}]: {doc.text}"
            if total_len + len(doc_text) > self.max_context_length:
                break
            context_parts.append(doc_text)
            total_len += len(doc_text)
        context = "\n\n".join(context_parts)

        # Count adversarial docs that made it into the context
        adv_count = sum(1 for doc in retrieval.documents if doc.is_adversarial)

        # Format prompt
        if self.model.config.chat_template:
            prompt = RAG_PROMPT_CHAT.format(context=context, query=query)
        else:
            prompt = RAG_PROMPT_BASE.format(context=context, query=query)

        # Generation stage
        gen_result = self.model.generate(prompt)

        return RAGResponse(
            query=query,
            retrieved_docs=retrieval.documents,
            retrieval_latency_s=retrieval.latency_s,
            generation_latency_s=gen_result["latency_s"],
            total_latency_s=retrieval.latency_s + gen_result["latency_s"],
            answer=gen_result["text"],
            output_tokens=gen_result["output_tokens"],
            input_tokens=gen_result["input_tokens"],
            retrieved_adversarial_count=adv_count,
        )
