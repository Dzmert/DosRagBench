"""Random-injection baseline for the C1 clustering ablation.

Isolates the *clustering* mechanism. RAND injects documents of the same style
and length as C1, but with random, non-query-targeted topics and no
similarity-based selection. If random junk docs do NOT evict gold while
clustered docs do, gold eviction is caused by the embedding-space clustering,
not merely by adding tens of thousands of documents to the corpus.
"""

from __future__ import annotations

import logging
import random

from dosragbench.attacks.base import DoSAttack
from dosragbench.attacks.c1_clustering import ADVERSARIAL_TEMPLATES, FILLER_WORDS
from dosragbench.pipeline.retriever import Document
from dosragbench.utils.config import AttackConfig

logger = logging.getLogger(__name__)

# Generic topics unrelated to typical NQ queries; hold text style constant with
# C1 while removing any targeting of the query.
GENERIC_TOPICS = [
    "municipal zoning regulations",
    "medieval textile production",
    "amateur radio protocols",
    "freshwater aquarium maintenance",
    "19th century canal engineering",
    "commercial poultry farming",
    "vintage typewriter restoration",
    "alpine botany field surveys",
    "container shipping logistics",
    "orchestral tuning conventions",
    "suburban stormwater drainage",
    "philatelic grading standards",
]


class RandomInjectionAttack(DoSAttack):
    """Baseline: C1-style docs with random topics, not clustered near the query."""

    def __init__(self, config: AttackConfig, **_ignored):
        super().__init__(config)
        self._rng = random.Random(config.seed)

    def _random_doc_text(self) -> str:
        template = self._rng.choice(ADVERSARIAL_TEMPLATES)
        filler = {k: self._rng.choice(v) for k, v in FILLER_WORDS.items()}
        filler["topic"] = self._rng.choice(GENERIC_TOPICS)
        try:
            return template.format(**filler)
        except KeyError:
            return f"General reference note on {filler['topic']}."

    def generate_adversarial_docs(
        self,
        query: str,
        clean_docs: list[Document],
        count: int | None = None,
    ) -> list[Document]:
        n = count if count is not None else self.config.num_adversarial_docs
        return [
            Document(
                doc_id=f"adv_RAND_{i}",
                text=self._random_doc_text(),
                is_adversarial=True,
                attack_category="RAND",
            )
            for i in range(n)
        ]
