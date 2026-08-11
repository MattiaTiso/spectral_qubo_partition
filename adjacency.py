"""
adjacency.py
============
Build up adjacency matrix W starting from a QUBO matrix Q
and (optionally) from a current solution x*.

Three types supported
-------------------
- structural_trivial  : W_ij = 1 if Q_ij != 0, 0 otherwise
- structural_weighted : W_ij = |Q_ij|
- solution_sensitive  : W_ij = (-1)^(x_i + x_j) * Q_ij   [può essere negativa]

The matrix W is made symmetric by taking (W + W^T) / 2.

References
-----------
- Zhao & Tang, arXiv:2502.16212, Sec. III-A (correlation matrix Σ)
"""

from __future__ import annotations

import numpy as np
from enum import Enum


class AdjacencyType(str, Enum):
    STRUCTURAL_TRIVIAL  = "structural_trivial"
    STRUCTURAL_WEIGHTED = "structural_weighted"
    SOLUTION_SENSITIVE  = "solution_sensitive"


def symmetrize(W: np.ndarray) -> np.ndarray:
    """Return (W + W^T) / 2 to ensure symmetry."""
    return (W + W.T) / 2.0


def build_adjacency(
    Q: np.ndarray,
    adj_type: AdjacencyType | str = AdjacencyType.STRUCTURAL_WEIGHTED,
    x: np.ndarray | None = None,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Build the adjacency matrix W from the QUBO matrix Q.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        QUBO matrix (square, not necessarily symmetric).
    adj_type : AdjacencyType or str
        Type of adjacency matrix to build.
    x : np.ndarray, shape (n,), optional
        Current binary solution x* in {0,1}^n.
        Required for adj_type == 'solution_sensitive'.
    zero_diagonal : bool
        If True, sets the diagonal of W to zero (self-loops removed).

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Symmetric adjacency matrix.

    Note
    ----
    The 'solution_sensitive' variant can produce negative values:
    W_ij = (-1)^(x_i + x_j) * Q_ij.
    When using this W with standard Laplacians (which assume W >= 0),
    it is necessary to split it into positive and negative parts (multi-view as in GMC).
    """
    adj_type = AdjacencyType(adj_type)
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    assert Q.shape == (n, n), "Q must be square."

    # ---- matrice Q simmetrica per il calcolo delle correlazioni ----
    Q_sym = symmetrize(Q)

    if adj_type == AdjacencyType.STRUCTURAL_TRIVIAL:
        W = (Q_sym != 0).astype(float)

    elif adj_type == AdjacencyType.STRUCTURAL_WEIGHTED:
        W = np.abs(Q_sym)

    elif adj_type == AdjacencyType.SOLUTION_SENSITIVE:
        if x is None:
            raise ValueError(
                "'solution_sensitive' requires a solution x in {0,1}^n."
            )
        x = np.asarray(x, dtype=float)
        assert x.shape == (n,), f"x must have shape ({n},), found {x.shape}."
        # Σ_ij = (-1)^(x_i + x_j) * Q_ij  (Zhao & Tang, eq. 9)
        signs = np.outer((-1.0) ** x, (-1.0) ** x)   # (-1)^(x_i+x_j)
        W = signs * Q_sym

    else:
        raise ValueError(f"adj_type not recognized: {adj_type}")

    if zero_diagonal:
        np.fill_diagonal(W, 0.0)

    return W


def split_signed(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a matrix W (potentially with negative values) in positive part W+ and negative part W- (both non-negative).

        W = W+ - W-

    Useful for multi-view clustering on 'solution_sensitive' matrices.

    Returns
    -------
    W_pos, W_neg : np.ndarray
        W_pos = max(W, 0),  W_neg = max(-W, 0)
    """
    W_pos = np.maximum(W,  0.0)
    W_neg = np.maximum(-W, 0.0)
    return W_pos, W_neg


# ---------------------------------------------------------------------------
# Funzioni di diagnostica / utilità
# ---------------------------------------------------------------------------

def adjacency_stats(W: np.ndarray) -> dict:
    """Descriptive statistics of W (useful for debugging)."""
    return {
        "shape":      W.shape,
        "min":        float(W.min()),
        "max":        float(W.max()),
        "mean":       float(W.mean()),
        "nnz":        int(np.count_nonzero(W)),
        "density":    float(np.count_nonzero(W)) / W.size,
        "symmetric":  bool(np.allclose(W, W.T)),
        "has_negative": bool((W < 0).any()),
    }


def prepare_multiview_from_qubo(
    Q: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Explicitly builds the non-negative views used by the clustering.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Global QUBO matrix.
    x : np.ndarray, shape (n,)
        Current binary solution used by the solution-sensitive view.

    Returns
    -------
    views : dict[str, np.ndarray]
        Ordered dictionary with the views ``structural_weighted``,
        ``positive_sensitive`` and ``negative_sensitive``.

    Notes
    -----
    The solution-sensitive view with sign is split as
    ``W_sensitive = W_positive - W_negative``. All returned matrices
    are symmetric, non-negative and have a zero diagonal.
    """
    Q = np.asarray(Q, dtype=float)
    x = np.asarray(x, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if x.shape != (Q.shape[0],):
        raise ValueError(
            f"x must have shape ({Q.shape[0]},), found {x.shape}."
        )

    W_structural = build_adjacency(
        Q, adj_type=AdjacencyType.STRUCTURAL_WEIGHTED
    )
    W_sensitive = build_adjacency(
        Q, adj_type=AdjacencyType.SOLUTION_SENSITIVE, x=x
    )
    W_positive, W_negative = split_signed(W_sensitive)

    return {
        "structural_weighted": W_structural,
        "positive_sensitive": W_positive,
        "negative_sensitive": W_negative,
    }
