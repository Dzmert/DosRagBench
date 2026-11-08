"""Blocker / jamming baseline (Shafran et al., arXiv:2406.05870).

Baseline for comparison, NOT one of our attacks. Threat model is stronger than
ours: it optimises against the target RAG via black-box feedback, whereas our
C1/D-family attacks are optimisation-free.

Construction:  d_blocker = d_r || d_j
  d_r  retrieval half. Black-box version uses the query text itself.
  d_j  generation half. 50 tokens optimised by black-box search to induce refusal.

Optimisation objective (paper, Section 4): maximise the similarity between the
target RAG's response and a fixed target refusal string, measured with a
sentence embedder. This is deliberately NOT the severity classifier the run is
scored with — optimising directly against the evaluation instrument would be
circular and inflate the baseline's measured success. The embedding objective is
an independent signal, so the severity classifier remains an unbiased judge.

Paper hyperparameters (Section 4):
  n = 50 tokens, initialised to '!' * 50
  batch size B = 32
  T = 1000 iterations
  early abort: d_j unchanged for 100 iterations
  early abort: the target phrase ("I don't know") appears in the RAG response
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional

from dosragbench.attacks.blocker_construct import TARGET_RESPONSES
from dosragbench.pipeline.retriever import Document

logger = logging.getLogger(__name__)


# Paper defaults.
N_TOKENS = 50
BATCH_SIZE = 32
MAX_ITERS = 1000
STALL_LIMIT = 100
INIT_TOKEN = "!"

# Neutral mutation vocabulary. The paper samples token replacements from natural
# text frequency (wikitext-103); a generic common-word list is a reasonable
# black-box stand-in. It is deliberately NOT biased toward refusal words — a
# refusal-word pool would hand the search its target and overstate how easily the
# blocker converges.
_NEUTRAL_VOCAB = (
    "the of and to in a is that for it as was with be by on not he this but are "
    "or his from they which one would all will there say who make when can more "
    "time no man out up them then she many some so these would other into has him "
    "your about day could state only new year most through back much before go good "
    "where our work first well way even want because any these give most us"
).split()


@dataclass
class BlockerConfig:
    n_tokens: int = N_TOKENS
    batch_size: int = BATCH_SIZE
    max_iters: int = MAX_ITERS
    stall_limit: int = STALL_LIMIT
    seed: int = 42
    target: str = "insufficient"  # which TARGET_RESPONSES refusal to steer toward
    embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"


class BlockerAttack:
    """Black-box optimised blocker document, one per query.

    Requires a response callable because the optimisation is closed-loop:
        response_fn(query, adv_docs) -> str   (the target RAG's answer)

    The attack embeds that answer and scores it by cosine similarity to a fixed
    target refusal string; it does NOT read the severity classifier, so the
    baseline is optimised independently of the instrument it is later judged by.
    """

    def __init__(
        self,
        config: Optional[BlockerConfig] = None,
        response_fn: Optional[Callable[[str, list[Document]], str]] = None,
        vocab: Optional[list[str]] = None,
    ):
        self.config = config or BlockerConfig()
        self.response_fn = response_fn
        self.rng = random.Random(self.config.seed)
        self.vocab = vocab or list(_NEUTRAL_VOCAB)
        if self.config.target not in TARGET_RESPONSES:
            raise ValueError(
                f"Unknown blocker target '{self.config.target}'. "
                f"Choose one of {sorted(TARGET_RESPONSES)}."
            )
        self.target = TARGET_RESPONSES[self.config.target]
        # Early-stop substring: the leading clause of the target refusal.
        self._target_phrase = self.target.split(".")[0].lower()
        self._embedder = None
        self._target_emb = None

    # ── embedding objective ──────────────────────────────────────────

    def _lazy_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            from dosragbench.pipeline.retriever import _resolve_device

            dev = _resolve_device(self.config.device)
            logger.info(f"Loading embedder for BLOCKER objective: {self.config.embedder_id} (device={dev})")
            self._embedder = SentenceTransformer(self.config.embedder_id, device=dev)
            self._target_emb = self._embedder.encode(
                [self.target], convert_to_numpy=True, normalize_embeddings=True
            )[0]
        return self._embedder

    def _similarity(self, response: str) -> float:
        emb = self._lazy_embedder().encode(
            [response], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return float(emb @ self._target_emb)

    # ── d_r: the retrieval half ──────────────────────────────────────

    def _retrieval_half(self, query: str) -> str:
        """Black-box d_r. The paper uses the query text itself, which is
        maximally similar to the query embedding by construction."""
        return query

    # ── d_j: the generation half ─────────────────────────────────────

    def _init_generation_half(self) -> list[str]:
        return [INIT_TOKEN] * self.config.n_tokens

    def _mutate(self, tokens: list[str]) -> list[str]:
        """One-position random substitution."""
        out = list(tokens)
        pos = self.rng.randrange(len(out))
        out[pos] = self.rng.choice(self.vocab)
        return out

    def _response(self, query: str, tokens: list[str]) -> str:
        doc = Document(
            doc_id="blocker_probe",
            text=f"{self._retrieval_half(query)} {' '.join(tokens)}",
            is_adversarial=True,
            attack_category="BLOCKER",
        )
        return self.response_fn(query, [doc])

    def _optimise(self, query: str) -> str:
        """Black-box search for a d_j that drives the target to refuse.

        Returns the best generation half found, maximising similarity between the
        target RAG's response and the target refusal string.
        """
        if self.response_fn is None:
            raise RuntimeError(
                "BlockerAttack needs response_fn to optimise. Pass a callable that "
                "runs the target RAG for a candidate blocker and returns its answer."
            )

        best = self._init_generation_half()
        best_resp = self._response(query, best)
        best_score = self._similarity(best_resp)
        stall = 0

        # Checked before the first batch: every candidate costs a full round-trip
        # against the live target, so don't spend a batch if d_j already refuses.
        if self._target_phrase in best_resp.lower():
            logger.debug("blocker: initial d_j already induces the target refusal")
            return " ".join(best)

        for it in range(self.config.max_iters):
            candidates = [self._mutate(best) for _ in range(self.config.batch_size)]
            improved = False
            for cand in candidates:
                resp = self._response(query, cand)
                score = self._similarity(resp)
                if score > best_score:
                    best_score, best, best_resp = score, cand, resp
                    improved = True

            # Early abort: the response now contains the target refusal.
            if self._target_phrase in best_resp.lower():
                logger.debug(f"blocker converged at iter {it}")
                break
            # Early abort: no improvement for stall_limit iterations.
            stall = 0 if improved else stall + 1
            if stall >= self.config.stall_limit:
                logger.debug(f"blocker stalled at iter {it}")
                break

        return " ".join(best)

    # ── public interface, matching the other attack classes ──────────

    def generate_adversarial_docs(self, query: str, retrieved_docs: list) -> list[Document]:
        """One blocker document per query."""
        d_j = self._optimise(query)
        text = f"{self._retrieval_half(query)} {d_j}"
        return [
            Document(
                doc_id=f"blocker_{abs(hash(query)) % 10**8}",
                text=text,
                is_adversarial=True,
                attack_category="BLOCKER",
            )
        ]
