"""Pathway-aware gene similarity index for structured reward scoring.

Uses gseapy pathway libraries (KEGG + Reactome) to build binary pathway
membership vectors per gene, enabling cosine-similarity-based set scoring
instead of substring matching.

Mechanism comparison uses sentence-transformers for semantic similarity.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PATHWAY_SETS: Optional[Dict[str, List[str]]] = None
_PATHWAY_NAMES: Optional[List[str]] = None
_GENE_TO_PATHWAY_IDX: Optional[Dict[str, List[int]]] = None
_N_PATHWAYS: int = 0

_SENTENCE_MODEL = None


def _ensure_pathway_index() -> None:
    """Lazily build the inverted gene→pathway index on first use."""
    global _PATHWAY_SETS, _PATHWAY_NAMES, _GENE_TO_PATHWAY_IDX, _N_PATHWAYS

    if _PATHWAY_NAMES is not None:
        return

    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed; pathway scoring will use fallback.")
        _PATHWAY_SETS = {}
        _PATHWAY_NAMES = []
        _GENE_TO_PATHWAY_IDX = {}
        _N_PATHWAYS = 0
        return

    combined: Dict[str, List[str]] = {}
    for lib_name in ("KEGG_2021_Human", "Reactome_2022"):
        try:
            combined.update(gp.get_library(lib_name))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", lib_name, exc)

    _PATHWAY_SETS = combined
    _PATHWAY_NAMES = sorted(combined.keys())
    _N_PATHWAYS = len(_PATHWAY_NAMES)

    inv: Dict[str, List[int]] = {}
    for idx, pw_name in enumerate(_PATHWAY_NAMES):
        for gene in combined[pw_name]:
            gene_upper = gene.upper().strip()
            inv.setdefault(gene_upper, []).append(idx)

    _GENE_TO_PATHWAY_IDX = inv
    logger.info(
        "Pathway index built: %d pathways, %d genes indexed.",
        _N_PATHWAYS, len(inv),
    )


def _ensure_sentence_model():
    """Lazily load the sentence-transformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; mechanism scoring will use fallback."
        )
        _SENTENCE_MODEL = None


def gene_vector(gene: str) -> np.ndarray:
    """L2-normalised binary pathway membership vector for *gene*."""
    _ensure_pathway_index()
    vec = np.zeros(_N_PATHWAYS, dtype=np.float32)
    indices = _GENE_TO_PATHWAY_IDX.get(gene.upper().strip(), [])
    if indices:
        vec[indices] = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
    return vec


def pathway_similarity(g1: str, g2: str) -> float:
    """Cosine similarity between two genes in pathway space."""
    v1 = gene_vector(g1)
    v2 = gene_vector(g2)
    dot = float(np.dot(v1, v2))
    return max(0.0, min(1.0, dot))


def marker_set_score(
    predicted: List[str],
    truth: List[str],
    sigma: float = 0.3,
) -> float:
    """Pathway-weighted Gaussian set similarity for marker genes.

    For each true marker, finds the best-matching predicted gene by
    pathway cosine similarity, then applies a Gaussian kernel:
        score_i = exp(-d^2 / (2 * sigma^2))   where d = 1 - sim
    Returns the mean score over all true markers.
    """
    if not truth:
        return 0.0
    if not predicted:
        return 0.0

    _ensure_pathway_index()

    if _N_PATHWAYS == 0:
        return _fallback_marker_score(predicted, truth)

    pred_vecs = [gene_vector(g) for g in predicted]
    scores: List[float] = []

    for true_gene in truth:
        tv = gene_vector(true_gene)
        best_sim = 0.0
        for pv in pred_vecs:
            sim = float(np.dot(tv, pv))
            if sim > best_sim:
                best_sim = sim
        d = 1.0 - best_sim
        scores.append(float(np.exp(-(d ** 2) / (2.0 * sigma ** 2))))

    return sum(scores) / len(scores)


def _fallback_marker_score(predicted: List[str], truth: List[str]) -> float:
    """Exact-match fallback when pathway data is unavailable."""
    pred_set = {g.upper().strip() for g in predicted}
    hits = sum(1 for g in truth if g.upper().strip() in pred_set)
    return hits / len(truth) if truth else 0.0


def mechanism_set_score(predicted: List[str], truth: List[str]) -> float:
    """Sentence-transformer semantic similarity for mechanism strings.

    For each truth mechanism, finds the best-matching predicted mechanism
    by cosine similarity and returns the mean of best matches.
    """
    if not truth:
        return 0.0
    if not predicted:
        return 0.0

    _ensure_sentence_model()

    if _SENTENCE_MODEL is None:
        return _fallback_mechanism_score(predicted, truth)

    pred_embs = _SENTENCE_MODEL.encode(predicted, convert_to_numpy=True)
    truth_embs = _SENTENCE_MODEL.encode(truth, convert_to_numpy=True)

    pred_norms = pred_embs / (
        np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-9
    )
    truth_norms = truth_embs / (
        np.linalg.norm(truth_embs, axis=1, keepdims=True) + 1e-9
    )

    sim_matrix = truth_norms @ pred_norms.T
    best_per_truth = sim_matrix.max(axis=1)
    return float(np.mean(np.clip(best_per_truth, 0.0, 1.0)))


def _fallback_mechanism_score(predicted: List[str], truth: List[str]) -> float:
    """Token-overlap fallback when sentence-transformers is unavailable."""
    scores: List[float] = []
    for t in truth:
        t_tokens = set(t.lower().split())
        best = 0.0
        for p in predicted:
            p_tokens = set(p.lower().split())
            union = t_tokens | p_tokens
            if union:
                overlap = len(t_tokens & p_tokens) / len(union)
                best = max(best, overlap)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def score_pathways(
    predicted: Dict[str, float],
    truth: Dict[str, float],
) -> float:
    """Score predicted pathway activations against ground truth.

    Uses normalised key matching with activity-level weighting.
    """
    if not truth:
        return 0.0
    if not predicted:
        return 0.0

    pred_norm = {k.lower().strip(): v for k, v in predicted.items()}
    total_weight = 0.0
    weighted_score = 0.0

    for pw, true_activity in truth.items():
        pw_key = pw.lower().strip()
        weight = true_activity
        total_weight += weight
        if pw_key in pred_norm:
            pred_activity = pred_norm[pw_key]
            diff = abs(pred_activity - true_activity)
            match_score = max(0.0, 1.0 - diff)
            weighted_score += weight * match_score

    return weighted_score / total_weight if total_weight > 0 else 0.0
