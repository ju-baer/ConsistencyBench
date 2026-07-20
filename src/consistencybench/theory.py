"""
Formal theory: encodes the paper's formal definitions as Python objects — the
five transformation families with their logical constraints, consistency
rules, and real-world coverage statistics — plus the actual semantic-distance
difficulty metric

    delta = alpha * d_lex + beta * d_sem + gamma * d_syn

computed with a real sentence-embedding model (not a placeholder), so
difficulty labels are measurable and reproducible rather than nominal.

Direct, environment-independent port of Notebook Cell 4.
"""
from __future__ import annotations

import re

import numpy as np

FAMILY_FORMAL_SPEC = {
    "composition": {
        "name": "Composition",
        "logical_basis": "Relation composition (transitivity)",
        "formal_constraint": "A=>B AND B=>C IMPLIES A=>C",
        "consistency_rule": "If model accepts A=>B and B=>C, it must accept A=>C",
        "probe_structure": {
            "prompt_a": "establishes A=>B via a factual claim",
            "prompt_b": "queries A=>C given B=>C as context",
        },
        "scoring": "ljs",
        "real_world_coverage_pct": 18.4,
    },
    "reversal": {
        "name": "Reversal",
        "logical_basis": "Symmetric relation reversal",
        "formal_constraint": "rel(X,Y) <=> rel(Y,X) for symmetric relations",
        "consistency_rule": "Answers to rel(X,Y) and rel(Y,X) must be the same",
        "probe_structure": {
            "prompt_a": "queries rel(X,Y)",
            "prompt_b": "queries rel(Y,X)",
        },
        "scoring": "rbs_same",
        "real_world_coverage_pct": 14.2,
    },
    "complement": {
        "name": "Complement",
        "logical_basis": "Truth complement (negation)",
        "formal_constraint": "NOT(assert(P) AND assert(NOT-P))",
        "consistency_rule": "Model cannot affirm P and also affirm NOT-P",
        "probe_structure": {
            "prompt_a": "asserts proposition P",
            "prompt_b": "asks whether NOT-P is acceptable",
        },
        "scoring": "rbs_opposite",
        "real_world_coverage_pct": 28.6,
    },
    "ordering": {
        "name": "Ordering",
        "logical_basis": "Asymmetric temporal ordering",
        "formal_constraint": "before(A,B) <=> after(B,A)",
        "consistency_rule": "Answers to before(A,B) and after(B,A) must be the same",
        "probe_structure": {
            "prompt_a": "queries whether A came before B",
            "prompt_b": "queries whether B came after A",
        },
        "scoring": "rbs_same",
        "real_world_coverage_pct": 10.8,
    },
    "equivalence": {
        "name": "Equivalence",
        "logical_basis": "Semantic equivalence preservation",
        "formal_constraint": "equiv(pA,pB) IMPLIES compat(rA,rB)",
        "consistency_rule": "Logically equivalent prompts must yield compatible responses",
        "probe_structure": {
            "prompt_a": "surface form 1 of proposition phi",
            "prompt_b": "surface form 2 of phi (distance-maximized via delta)",
        },
        "scoring": "ljs",
        "real_world_coverage_pct": 11.4,
    },
}

_embed_model = None  # lazily loaded SentenceTransformer


def get_embed_model():
    """Lazily load and cache the sentence-embedding model used for d_sem."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def compute_semantic_distance(
    prompt_a: str, prompt_b: str, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2
) -> float:
    """
    delta(pA, pB) = alpha*d_lex + beta*d_sem + gamma*d_syn

    d_lex = 1 - Jaccard token overlap
    d_sem = cosine distance between sentence embeddings, in [0, 1]
    d_syn = indicator of different syntactic-depth bucket (sentence-length proxy)

    Returns delta in [0, 1] where higher = harder probe (more surface divergence).
    """
    def tokenize(s: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", s.lower()))

    ta, tb = tokenize(prompt_a), tokenize(prompt_b)
    d_lex = 1.0 - len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0

    model = get_embed_model()
    emb_a, emb_b = model.encode([prompt_a, prompt_b], normalize_embeddings=True)
    cos_sim = float(np.dot(emb_a, emb_b))
    d_sem = (1.0 - cos_sim) / 2.0  # map cosine similarity [-1,1] -> distance [0,1]

    len_a, len_b = len(prompt_a.split()), len(prompt_b.split())
    bucket = lambda n: 0 if n < 10 else 1 if n < 20 else 2
    d_syn = float(bucket(len_a) != bucket(len_b))

    delta = alpha * d_lex + beta * d_sem + gamma * d_syn
    return round(float(delta), 4)


def classify_difficulty(delta: float) -> str:
    if delta < 0.35:
        return "easy"
    elif delta < 0.65:
        return "medium"
    return "hard"


if __name__ == "__main__":
    print("Formal specification of transformation families:")
    total_coverage = 0.0
    for fam, spec in FAMILY_FORMAL_SPEC.items():
        print(f"  [{spec['name']}] ({fam}) — {spec['real_world_coverage_pct']}% real-world coverage")
        print(f"    {spec['consistency_rule']}")
        total_coverage += spec["real_world_coverage_pct"]
    print(f"\nTotal coverage: {total_coverage:.1f}% "
          f"(remaining {100 - total_coverage:.1f}%: multi-hop 9.8%, modal 4.2%, quantifier 2.6%)")

    demo_pairs = [
        ("Is Spanish more widely spoken than Portuguese?",
         "Is Portuguese less spoken than Spanish?"),
        ("Did the Industrial Revolution precede the French Revolution?",
         "Did the French Revolution follow the Industrial Revolution?"),
        ("Should hospitals always inform patients of risks?",
         "Is it acceptable for medical facilities to withhold treatment information "
         "from individuals seeking care about potential adverse effects?"),
    ]
    print("\nSemantic distance examples (real embeddings):")
    for a, b in demo_pairs:
        d = compute_semantic_distance(a, b)
        print(f"  delta={d:.3f} [{classify_difficulty(d)}]")
        print(f"    A: {a[:70]}")
        print(f"    B: {b[:70]}")
