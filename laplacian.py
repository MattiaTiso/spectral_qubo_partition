"""Central spectral linear-algebra module for the project.

This is the only module that constructs graph Laplacians or computes
spectral quantities (eigenvalues, eigenvectors, zero multiplicity, connected
components). Both CPU and optional CuPy implementations live here.
"""
from __future__ import annotations
from enum import Enum
import warnings
import numpy as np
import scipy.linalg as la

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
except ImportError:
    cp = None
    cupy_eigsh = None


class LaplacianType(str, Enum):
    UNNORMALIZED = "unnormalized"
    SYMMETRIC_NORMALIZED = "symmetric_normalized"
    RANDOM_WALK = "random_walk"


def degree_vector(W: np.ndarray) -> np.ndarray:
    """Return weighted row degrees."""
    W = np.asarray(W, dtype=float)
    return W.sum(axis=1)


def build_laplacian(W: np.ndarray, lap_type=LaplacianType.UNNORMALIZED):
    """Build an unnormalized, symmetric-normalized, or random-walk Laplacian."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W deve essere quadrata.")
    lap_type = LaplacianType(lap_type)
    d = degree_vector(W)
    n = W.shape[0]
    if lap_type == LaplacianType.UNNORMALIZED:
        L = np.diag(d) - W
    elif lap_type == LaplacianType.SYMMETRIC_NORMALIZED:
        inv_sqrt = np.zeros_like(d)
        positive = d > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(d[positive])
        L = np.eye(n) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
        L[~positive, :] = 0.0
        L[:, ~positive] = 0.0
    else:
        inv = np.zeros_like(d)
        positive = d > 0
        inv[positive] = 1.0 / d[positive]
        L = np.eye(n) - inv[:, None] * W
        L[~positive, :] = 0.0
    return L, d


def unified_graph_laplacian(U: np.ndarray) -> np.ndarray:
    """Build L_U from the symmetric part of GMC's unified graph U."""
    A = 0.5 * (np.asarray(U) + np.asarray(U).T)
    return np.diag(A.sum(axis=1)) - A


def smallest_eigenpairs(L: np.ndarray, number: int):
    """Return only the requested smallest eigenpairs of a symmetric CPU matrix."""
    L = np.asarray(L, dtype=float)
    number = min(max(int(number), 1), L.shape[0])
    return la.eigh(L, subset_by_index=(0, number - 1), check_finite=False)


def random_walk_eigenpairs(W: np.ndarray, number: int):
    """Solve Shi-Malik's symmetric generalized problem (D-W)y=lambda D y."""
    W = np.asarray(W, dtype=float)
    d = degree_vector(W)
    if np.any(d <= 0):
        raise ValueError("NCut random-walk requires strictly positive degrees.")
    D = np.diag(d)
    L = D - W
    number = min(max(int(number), 1), W.shape[0])
    return la.eigh(L, D, subset_by_index=(0, number - 1), check_finite=False)


def zero_eigenvalue_multiplicity(values: np.ndarray, tolerance=1e-8) -> int:
    """Count numerically zero eigenvalues in an already sorted spectrum subset."""
    return int(np.count_nonzero(np.asarray(values) < tolerance))


def connected_components_labels(W: np.ndarray, tolerance=1e-8):
    """Label components of the symmetric threshold graph induced by W."""
    A = 0.5 * (np.asarray(W) + np.asarray(W).T) > tolerance
    n = A.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    component = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        while stack:
            node = stack.pop()
            for neighbour in np.flatnonzero(A[node]):
                if labels[neighbour] < 0:
                    labels[neighbour] = component
                    stack.append(int(neighbour))
        component += 1
    return labels, component


def fiedler_vector(L: np.ndarray, tolerance=1e-8):
    """Return the Fiedler vector, requiring a connected graph (simple zero mode)."""
    number = min(3, L.shape[0])
    values, vectors = smallest_eigenpairs(L, number)
    if values[0] >= tolerance or values[1] <= tolerance:
        raise ValueError("Fiedler vector requires a single zero eigenvalue.")
    gap = values[2] - values[1] if len(values) > 2 else np.nan
    return values[1], vectors[:, 1], gap


def cupy_available() -> bool:
    """Return True only when CuPy and a CUDA device are usable."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def unified_graph_laplacian_gpu(U):
    """CuPy version of GMC's unified-graph Laplacian."""
    A = 0.5 * (U + U.T)
    return cp.diag(cp.sum(A, axis=1)) - A


def smallest_eigenpairs_gpu(L, number: int, *, tolerance=1e-8,
                            dense_eigh_threshold=32):
    """Use partial thick-restart Lanczos on large GPU matrices, dense fallback otherwise."""
    if cp is None:
        raise RuntimeError("CuPy not available.")
    n = int(L.shape[0])
    number = min(max(int(number), 1), n)
    use_dense = n <= dense_eigh_threshold or number >= n - 1
    if not use_dense:
        try:
            values, vectors = cupy_eigsh(
                L, k=number, which="SA", tol=tolerance,
                maxiter=max(100, 20 * n)
            )
            order = cp.argsort(values)
            return values[order], vectors[:, order]
        except Exception as error:
            warnings.warn(
                f"eigsh GPU failed ({error}); fallback to cp.linalg.eigh.",
                stacklevel=2,
            )
    
    values, vectors = cp.linalg.eigh(L)
    order = cp.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    return values[:number], vectors[:, :number]


def zero_eigenvalue_multiplicity_gpu(values, tolerance=1e-8) -> int:
    """GPU counterpart of zero_eigenvalue_multiplicity."""
    return int(cp.count_nonzero(values < tolerance).get())
