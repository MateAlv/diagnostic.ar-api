"""What rr-v1's negation feature needs: the HPO is_a graph and the gene knowledge base.

Provenance: copied from MateAlv/diagnostic.ar-training@9a573fd5bde1c2533e7a9ddcdc0ec4db59628a27
(unchanged since 0b1bae8, the commit EXP-B-003 ran from):
- src/diagnostic_ar_training/ontology.py: `_curie`, the parent construction of
  `Ontology.from_json`, `Ontology.ancestors`;
- src/diagnostic_ar_training/rerank/features.py: `FeatureParams`,
  `GeneKnowledge.diseases`, `GeneKnowledge._propagated`,
  `GeneKnowledge.term_probability`, and the `neg_llr` term of `components`.
Only the negation path is kept: rr-v1[neg=0.5] gives sex, age and onset weight 0.
The arithmetic and its order are unchanged, so results match the training repo
bit for bit.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_OBO = "http://purl.obolibrary.org/obo/"


def _curie(value: str) -> str:
    if value.startswith(_OBO):
        value = value[len(_OBO) :]
    return value.replace("_", ":", 1) if value.startswith("HP_") else value


class Ontology:
    def __init__(self, parents: dict[str, tuple[str, ...]]) -> None:
        self._parents = parents

    @classmethod
    def from_json(cls, path: Path) -> "Ontology":
        graph = json.loads(Path(path).read_text())["graphs"][0]
        parents: dict[str, set[str]] = {}
        for node in graph["nodes"]:
            term = _curie(node["id"])
            if term.startswith("HP:"):
                parents.setdefault(term, set())
        for edge in graph["edges"]:
            if edge["pred"] != "is_a":
                continue
            child, parent = _curie(edge["sub"]), _curie(edge["obj"])
            if child.startswith("HP:") and parent.startswith("HP:"):
                parents.setdefault(child, set()).add(parent)
        return cls({k: tuple(sorted(v)) for k, v in parents.items()})

    def __contains__(self, term: str) -> bool:
        return term in self._parents

    def parents(self, term: str) -> tuple[str, ...]:
        return self._parents.get(term, ())

    @lru_cache(maxsize=65536)  # noqa: B019 -- one Ontology per process, cache lives with it
    def ancestors(self, term: str) -> frozenset[str]:
        """Strict ancestors (the term itself excluded)."""
        out: set[str] = set()
        for p in self.parents(term):
            out.add(p)
            out |= self.ancestors(p)
        return frozenset(out)


@dataclass(frozen=True)
class FeatureParams:
    neg_strength: float = 0.5
    unknown_frequency: float = 0.3


class GeneKnowledge:
    def __init__(self, kb: dict, ontology: Ontology, params: FeatureParams = FeatureParams()) -> None:
        if kb.get("schema") != 2:
            raise ValueError("features need a schema-2 KB (kb-v3 or later)")
        self.kb = kb
        self.ontology = ontology
        self.params = params
        self.vocabulary = kb.get("vocabulary", {})

    def diseases(self, gene: str) -> dict[str, float]:
        return self.kb["genes"].get(self.vocabulary.get(gene, gene), {})

    @lru_cache(maxsize=16384)  # noqa: B019 -- one instance per process
    def _propagated(self, disease_id: str) -> dict[str, float]:
        """P(t | disease) for every term the disease annotates, directly or through a descendant."""
        absent: dict[str, float] = {}
        for term, pheno in self.kb["diseases"][disease_id]["phenotypes"].items():
            f = pheno["freq"] if pheno["freq"] is not None else self.params.unknown_frequency
            for t in (term, *self.ontology.ancestors(term)):
                absent[t] = absent.get(t, 1.0) * (1.0 - f)
        return {t: 1.0 - a for t, a in absent.items()}

    def term_probability(self, gene: str, term: str) -> float:
        return sum(w * self._propagated(d).get(term, 0.0) for d, w in self.diseases(gene).items())

    def neg_llr(self, gene: str, excluded: tuple[str, ...]) -> float:
        """Sum over excluded terms t of log(1 - k * P(t | gene)); 0 without excluded terms or knowledge."""
        out = 0.0
        if not excluded or not self.diseases(gene):
            return out
        for term in excluded:
            out += math.log(1.0 - self.params.neg_strength * self.term_probability(gene, term))
        return out
