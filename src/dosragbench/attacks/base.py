"""Base class for DoS attacks on RAG systems."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from dosragbench.pipeline.retriever import Document
from dosragbench.utils.config import AttackConfig

logger = logging.getLogger(__name__)


class DoSAttack(ABC):
    """Abstract base class for all DoS attacks.

    Subclasses implement generate_adversarial_docs() which returns the adversarial
    documents to inject into the knowledge base for a given query.
    """

    def __init__(self, config: AttackConfig):
        self.config = config
        self.category = config.category

    @abstractmethod
    def generate_adversarial_docs(
        self,
        query: str,
        clean_docs: list[Document],
    ) -> list[Document]:
        """Generate adversarial documents to poison the KB for this query.

        Args:
            query: The target query we're trying to deny service on.
            clean_docs: The top-k clean documents that would normally be retrieved.
                       Useful for topical context (making adversarial docs look relevant).

        Returns:
            List of adversarial Document objects to inject into the KB.
        """
        ...

    def describe(self) -> str:
        """Human-readable description for logging."""
        return f"[{self.category}] {self.config.name} (n_docs={self.config.num_adversarial_docs})"
