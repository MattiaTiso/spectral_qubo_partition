"""Spectral clustering methods only.

All Laplacian construction and eigenvalue/eigenvector calculations are
centralized in laplacian.py. This module contains method-level logic only.
"""
from __future__ import annotations
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from laplacian import (
    LaplacianType,
    build_laplacian,
    connected_components_labels,
    fiedler_vector,
    random_walk_eigenpairs,
    smallest_eigenpairs,
)


class CutType(str, Enum):
    MINCUT = "mincut"
    RATIOCUT = "ratiocut"
    NCUT = "ncut"


def spectral_embedding(W: np.ndarray, k: int, cut_type=CutType.NCUT):
    """Return the method-appropriate spectral embedding and eigenvalues."""
    cut_type = CutType(cut_type)
    if cut_type == CutType.NCUT:
        values, vectors = random_walk_eigenpairs(W, k)
        return values, vectors
    L, _ = build_laplacian(W, LaplacianType.UNNORMALIZED)
    values, vectors = smallest_eigenpairs(L, k)
    if cut_type == CutType.RATIOCUT:
        vectors = normalize(vectors, norm="l2", axis=1)
    return values, vectors


def spectral_cluster(W: np.ndarray, k: int, cut_type=CutType.NCUT,
                     n_init=20, random_state=42):
    """Cluster the rows of the first-k spectral embedding with k-means.

    If W already has exactly k connected components, return them directly.
    """
    component_labels, components = connected_components_labels(W)
    if components == k:
        return component_labels
    values, embedding = spectral_embedding(W, k, cut_type)
    del values
    return KMeans(n_clusters=k, n_init=n_init,
                  random_state=random_state).fit_predict(embedding)


def spectral_bisection(W: np.ndarray, balance="median", tolerance=1e-8):
    """Connected-graph bisection using the Fiedler vector.

    If W already has exactly two components, return those components directly.
    """
    labels, components = connected_components_labels(W, tolerance)
    if components == 2:
        return labels
    if components != 1:
        raise ValueError("The Fiedler bisection requires a connected graph or two components.")
    L, _ = build_laplacian(W, LaplacianType.UNNORMALIZED)
    _, vector, _ = fiedler_vector(L, tolerance)
    if balance == "median":
        order = np.argsort(vector, kind="stable")
        labels = np.ones(len(vector), dtype=np.int32)
        labels[order[:len(order)//2]] = 0
        return labels
    if balance == "sign":
        return (vector >= 0).astype(np.int32)
    raise ValueError("balance must be median or sign.")
