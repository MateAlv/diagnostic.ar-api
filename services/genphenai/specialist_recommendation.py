#!/usr/bin/env python3
"""Recommend specialties from model gene probabilities and PanelApp mappings."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from confidence import load_score_map, normalize_distribution


def _canonical_gene(gene: str) -> str:
    return gene.strip().upper()


SPECIALTY_CANONICAL: dict[str, str] = {
    # Neuro
    "NEUROLOGY": "Neurology",
    "NEUROLOGY AND NEURODEVELOPMENTAL DISORDERS": "Neurology",
    # Endocrine / metabolic
    "ENDOCRINOLOGY": "Endocrinology",
    "ENDOCRINE DISORDERS": "Endocrinology",
    "METABOLIC": "Metabolic",
    "METABOLIC DISORDERS": "Metabolic",
    "LIPIDS": "Metabolic",
    "MITOCHONDRIAL": "Metabolic",
    # Cardio
    "CARDIOLOGY": "Cardiology",
    "CARDIOVASCULAR DISORDERS": "Cardiology",
    # Renal / GU
    "RENAL": "Renal",
    "RENAL AND URINARY TRACT DISORDERS": "Renal",
    # GI
    "GASTROHEPATOLOGY": "Gastrohepatology",
    "GASTROENTEROLOGICAL DISORDERS": "Gastrohepatology",
    # Respiratory
    "RESPIRATORY": "Respiratory",
    "RESPIRATORY DISORDERS": "Respiratory",
    # Derm
    "DERMATOLOGY": "Dermatology",
    "DERMATOLOGICAL DISORDERS": "Dermatology",
    # Ophthalmology
    "OPHTHALMOLOGY": "Ophthalmology",
    "OPHTHALMOLOGICAL DISORDERS": "Ophthalmology",
    # Hearing / ear
    "HEARING AND EAR DISORDERS": "Audiology",
    "AUDIOLOGY": "Audiology",
    # Heme / immune
    "HAEMATOLOGY": "Haematology",
    "HAEMATOLOGICAL DISORDERS": "Haematology",
    "IMMUNOLOGY": "Immunology",
    "HAEMATOLOGICAL AND IMMUNOLOGICAL DISORDERS": "Haematology",
    # MSK / skeletal
    "MUSCULOSKELETAL": "Musculoskeletal",
    "SKELETAL DISORDERS": "Musculoskeletal",
    "RHEUMATOLOGICAL DISORDERS": "Musculoskeletal",
    # Developmental / congenital / fetal
    "DEVELOPMENTAL DISORDERS": "Developmental disorders",
    "DYSMORPHIC AND CONGENITAL ABNORMALITY SYNDROMES": "Developmental disorders",
    "FETAL (INCLUDING NIPD)": "Developmental disorders",
    "CILIOPATHIES": "Developmental disorders",
    # Growth
    "GROWTH DISORDERS": "Growth disorders",
    # Cancer / neoplasm
    "INHERITED CANCER": "Cancer",
    "CANCER PROGRAMME": "Cancer",
    "TUMOUR SYNDROMES": "Cancer",
    "CANCER SUSCEPTIBILITY": "Cancer",
    # Cross-cutting / meta
    "MULTISPECIALTY": "Multispecialty",
    "ULTRA-RARE DISORDERS": "Ultra-rare disorders",
    "UNKNOWN": "Unknown",
    "VIRAL RESEARCH": "Viral research",
    # Empty label
    "": "Unknown",
}


def canonicalize_specialty(specialty: str) -> str:
    cleaned = specialty.strip()
    return SPECIALTY_CANONICAL.get(cleaned.upper(), cleaned)


def _parse_panel_gene_list(raw_gene_list: str) -> list[str]:
    raw_gene_list = (raw_gene_list or "").strip()
    if not raw_gene_list:
        return []

    values: list[object]
    try:
        parsed = ast.literal_eval(raw_gene_list)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        values = list(parsed)
    else:
        # Fallback for malformed rows that are plain comma-separated strings.
        values = raw_gene_list.split(",")

    genes: list[str] = []
    seen: set[str] = set()
    for value in values:
        gene = _canonical_gene(str(value).strip().strip("'\""))
        if not gene:
            continue
        if gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def load_panelapp_gene_to_specialties(path: Path) -> dict[str, set[str]]:
    gene_to_specialties: defaultdict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"gene_list", "level2_category_specialist"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"PanelApp file is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            raw_specialty = (row.get("level2_category_specialist") or "").strip()
            if not raw_specialty:
                continue
            specialty = canonicalize_specialty(raw_specialty)

            genes = _parse_panel_gene_list(row.get("gene_list", ""))
            for gene in genes:
                gene_to_specialties[gene].add(specialty)

    return dict(gene_to_specialties)


def select_genes_by_cumulative_probability(
    scores: Mapping[str, float], cumulative_threshold: float
) -> list[tuple[str, float]]:
    if not (0 < cumulative_threshold <= 1):
        raise ValueError("cumulative_threshold must be in the range (0, 1].")

    normalized = normalize_distribution(
        {_canonical_gene(gene): score for gene, score in scores.items()}
    )
    sorted_genes = sorted(normalized.items(), key=lambda item: item[1], reverse=True)

    selected: list[tuple[str, float]] = []
    cumulative = 0.0
    for gene, probability in sorted_genes:
        selected.append((gene, probability))
        cumulative += probability
        if cumulative >= cumulative_threshold:
            break

    return selected


def recommend_specialties(
    scores: Mapping[str, float],
    gene_to_specialties: Mapping[str, set[str]],
    cumulative_threshold: float = 0.95,
) -> dict[str, object]:
    selected_genes = select_genes_by_cumulative_probability(
        scores, cumulative_threshold=cumulative_threshold
    )
    selected_probability_mass = sum(prob for _, prob in selected_genes)

    specialty_to_gene_set: defaultdict[str, set[str]] = defaultdict(set)
    specialty_to_probability_mass: defaultdict[str, float] = defaultdict(float)
    unmapped_genes: list[str] = []
    mapped_gene_count = 0

    for gene, probability in selected_genes:
        specialties = gene_to_specialties.get(gene, set())
        if not specialties:
            unmapped_genes.append(gene)
            continue

        mapped_gene_count += 1
        for specialty in specialties:
            specialty_to_gene_set[specialty].add(gene)
            specialty_to_probability_mass[specialty] += probability

    ranked_specialties = sorted(
        specialty_to_gene_set.items(),
        key=lambda item: (
            -len(item[1]),
            -specialty_to_probability_mass[item[0]],
            item[0],
        ),
    )

    ranking = []
    for specialty, genes in ranked_specialties:
        ranking.append(
            {
                "matched_genes_count": len(genes),
                "specialty": specialty,
                "fraction_of_selected_genes": len(genes) / len(selected_genes),
                "probability_mass": specialty_to_probability_mass[specialty],
                "genes": sorted(genes),
            }
        )

    top_frequency = ranking[0]["matched_genes_count"] if ranking else 0
    top_specialties = [
        row["specialty"]
        for row in ranking
        if row["matched_genes_count"] == top_frequency
    ]
    top_specialty = top_specialties[0] if top_specialties else None
    specialty_histogram = {
        row["specialty"]: row["matched_genes_count"] for row in ranking
    }

    return {
        "specialty_histogram": specialty_histogram,
        "cumulative_threshold": cumulative_threshold,
        "selected_gene_count": len(selected_genes),
        "selected_probability_mass": selected_probability_mass,
        "mapped_gene_count": mapped_gene_count,
        "unmapped_gene_count": len(unmapped_genes),
        "unmapped_genes": sorted(unmapped_genes),
        "top_frequency": top_frequency,
        "top_specialties": top_specialties,
        "top_specialty": top_specialty,
        "specialty_ranking": ranking,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend clinical specialties from model output scores using PanelApp "
            "gene-to-specialty mapping."
        )
    )
    parser.add_argument(
        "--genes_probabilities",
        type=Path,
        default=Path("data/genphenia_results.json"),
        help="Path to JSON mapping genes to probabilities/scores.",
    )
    parser.add_argument(
        "--panelapp",
        type=Path,
        default=Path("data/panelapp_panels.csv"),
        help="Path to PanelApp panels CSV.",
    )
    parser.add_argument(
        "--cumulative-threshold",
        type=float,
        default=0.95,
        help="Cumulative probability cutoff used to select top genes.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top specialties to print in text mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    score_map = load_score_map(args.genes_probabilities)
    gene_to_specialties = load_panelapp_gene_to_specialties(args.panelapp)
    result = recommend_specialties(
        score_map,
        gene_to_specialties,
        cumulative_threshold=args.cumulative_threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"genes_probabilities: {args.genes_probabilities}")
    print(f"panelapp: {args.panelapp}")
    print(f"cumulative_threshold: {result['cumulative_threshold']:.4f}")
    print(
        "selected_genes: "
        f"{result['selected_gene_count']} "
        f"(mass={result['selected_probability_mass']:.6f})"
    )
    print(
        f"mapped_genes: {result['mapped_gene_count']} | "
        f"unmapped_genes: {result['unmapped_gene_count']}"
    )
    print(f"top_specialty: {result['top_specialty']}")
    if result["top_specialties"]:
        print("top_specialties_tied: " + ", ".join(result["top_specialties"]))
    print("ranking:")

    for idx, row in enumerate(result["specialty_ranking"][: args.top], start=1):
        print(
            f"{idx}. {row['specialty']} | matched_genes={row['matched_genes_count']} | "
            f"mass={row['probability_mass']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
