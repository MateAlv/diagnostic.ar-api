"""rr-v1: re-score v3.0's top genes with the patient's negated terms.

score(g) = log p_v3(g) + w_neg * neg_llr(g) for the top-K genes by p_v3; the
rest keep log p_v3. The result is the softmax of the scores. Without negated
terms, or when no top gene annotates them, the input array is returned as is,
so the output is bit-identical to v3.0.

Provenance: copied from MateAlv/diagnostic.ar-training@9a573fd5bde1c2533e7a9ddcdc0ec4db59628a27
(unchanged since 0b1bae8, the commit EXP-B-003 ran from),
src/diagnostic_ar_training/rerank/model.py: the top-K selection of
`feature_table` and `FeatureTable.combine`, restricted to the negation feature.
"""

from collections.abc import Iterable, Sequence

import numpy as np

from rerank.knowledge import GeneKnowledge


class ReRanker:
    def __init__(
        self, knowledge: GeneKnowledge, genes: Sequence[str], neg_weight: float, top_k: int
    ) -> None:
        self.knowledge = knowledge
        self.genes = tuple(genes)
        self.neg_weight = neg_weight
        self.top_k = top_k

    def unknown_terms(self, terms: Iterable[str]) -> list[str]:
        """Terms the ontology does not know; they cannot move any gene."""
        return [t for t in terms if t not in self.knowledge.ontology]

    def rerank(self, probs: np.ndarray, excluded: Sequence[str] | None = None) -> np.ndarray:
        """v3.0's distribution over `genes`, re-scored with the patient's negated HPO terms."""
        if probs.shape != (len(self.genes),):
            raise ValueError(f"expected {len(self.genes)} probabilities, got shape {probs.shape}")
        if not excluded or self.neg_weight == 0:
            return probs
        excluded = tuple(excluded)
        top = np.argsort(probs)[::-1][: self.top_k]
        delta = self.neg_weight * np.array(
            [self.knowledge.neg_llr(self.genes[i], excluded) for i in top], dtype=float
        )
        if not np.any(delta):
            return probs
        scores = np.log(np.clip(probs, 1e-300, None))
        scores[top] += delta
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()

    def rerank_distribution(
        self, distribution: dict[str, float], excluded: Sequence[str] | None = None
    ) -> dict[str, float]:
        """The same, on the engine's `predict(..., return_distribution=True)["distribution"]`."""
        if tuple(distribution) != self.genes:
            raise ValueError("distribution keys must be the v3.0 genes, in the engine's order")
        probs = np.fromiter(distribution.values(), dtype=float, count=len(self.genes))
        out = self.rerank(probs, excluded)
        if out is probs:
            return distribution
        return {gene: float(p) for gene, p in zip(self.genes, out, strict=True)}
