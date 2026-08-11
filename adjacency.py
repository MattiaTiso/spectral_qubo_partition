"""
adjacency.py
============
Costruzione delle matrici di adiacenza W a partire da una matrice QUBO Q
e (opzionalmente) da una soluzione corrente x*.

Tre tipi supportati
-------------------
- structural_trivial  : W_ij = 1 se Q_ij != 0, 0 altrimenti
- structural_weighted : W_ij = |Q_ij|
- solution_sensitive  : W_ij = (-1)^(x_i + x_j) * Q_ij   [può essere negativa]

La matrice W è resa simmetrica prendendo (W + W^T) / 2.

Riferimenti
-----------
- Slides "Pre-Processing per QSplit" (sezione Esempi di Scelta di W, C e O)
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
    """Restituisce (W + W^T) / 2 per garantire la simmetria."""
    return (W + W.T) / 2.0


def build_adjacency(
    Q: np.ndarray,
    adj_type: AdjacencyType | str = AdjacencyType.STRUCTURAL_WEIGHTED,
    x: np.ndarray | None = None,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Costruisce la matrice di adiacenza W a partire da Q.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Matrice QUBO (quadratica, non necessariamente simmetrica).
    adj_type : AdjacencyType o str
        Tipo di matrice di adiacenza da costruire.
    x : np.ndarray, shape (n,), opzionale
        Soluzione binaria corrente x* in {0,1}^n.
        Obbligatoria per adj_type == 'solution_sensitive'.
    zero_diagonal : bool
        Se True, azzera la diagonale di W (self-loop rimossi).

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Matrice di adiacenza simmetrica.

    Note
    ----
    La variante 'solution_sensitive' può produrre valori negativi:
    W_ij = (-1)^(x_i + x_j) * Q_ij.
    Quando si usa questa W con Laplaciane standard (che assumono W >= 0)
    occorre separarla in parte positiva e negativa (multi-view come in GMC).
    """
    adj_type = AdjacencyType(adj_type)
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    assert Q.shape == (n, n), "Q deve essere quadrata."

    # ---- matrice Q simmetrica per il calcolo delle correlazioni ----
    Q_sym = symmetrize(Q)

    if adj_type == AdjacencyType.STRUCTURAL_TRIVIAL:
        W = (Q_sym != 0).astype(float)

    elif adj_type == AdjacencyType.STRUCTURAL_WEIGHTED:
        W = np.abs(Q_sym)

    elif adj_type == AdjacencyType.SOLUTION_SENSITIVE:
        if x is None:
            raise ValueError(
                "'solution_sensitive' richiede una soluzione x in {0,1}^n."
            )
        x = np.asarray(x, dtype=float)
        assert x.shape == (n,), f"x deve avere forma ({n},), trovato {x.shape}."
        # Σ_ij = (-1)^(x_i + x_j) * Q_ij  (Zhao & Tang, eq. 9)
        signs = np.outer((-1.0) ** x, (-1.0) ** x)   # (-1)^(x_i+x_j)
        W = signs * Q_sym

    else:
        raise ValueError(f"adj_type non riconosciuto: {adj_type}")

    if zero_diagonal:
        np.fill_diagonal(W, 0.0)

    return W


def split_signed(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa una matrice W (potenzialmente con valori negativi) in parte
    positiva W+ e parte negativa W- (entrambe non-negative).

        W = W+ - W-

    Utile per il clustering multi-view su matrici 'solution_sensitive'.

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
    """Statistiche descrittive di W (utile per debug)."""
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
    """Costruisce esplicitamente le viste non negative usate dal clustering.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Matrice QUBO globale.
    x : np.ndarray, shape (n,)
        Soluzione binaria corrente usata dalla vista solution-sensitive.

    Returns
    -------
    views : dict[str, np.ndarray]
        Dizionario ordinato con le viste ``structural_weighted``,
        ``positive_sensitive`` e ``negative_sensitive``.

    Notes
    -----
    La vista solution-sensitive con segno viene divisa come
    ``W_sensitive = W_positive - W_negative``. Tutte le matrici restituite
    sono simmetriche, non negative e con diagonale nulla.
    """
    Q = np.asarray(Q, dtype=float)
    x = np.asarray(x, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q deve essere una matrice quadrata.")
    if x.shape != (Q.shape[0],):
        raise ValueError(
            f"x deve avere shape ({Q.shape[0]},), trovata {x.shape}."
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
