"""CuPy implementation of GMC and its binary hierarchy.

GPU spectral operations remain centralized in laplacian.py. In GPU mode,
node distance blocks stay on the device and are passed to GMCGPU without
calling the serial GMC model.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from gmc import BinaryTreeNode, BinaryHierarchicalGMC, validate_feature_views
from laplacian import (
    connected_components_labels,
    cupy_available,
    smallest_eigenpairs_gpu,
    unified_graph_laplacian_gpu,
    zero_eigenvalue_multiplicity_gpu,
    cp
)
#print("laplacian.py: cp:", cp)
try:
    import cupy as cp

   #print("GPU:", cp.cuda.Device(0).name)
    print("Devices:", cp.cuda.runtime.getDeviceCount())
except Exception:
    pass

try:
    import cupy as cp
except ImportError:
    cp = None


_EPS = 1e-12


def row_squared_distances_gpu(X):
    """Squared Euclidean distances between rows of a CuPy matrix."""
    row_norms = cp.sum(X * X, axis=1)
    distances = row_norms[:, None] + row_norms[None, :] - 2.0 * (X @ X.T)
    return cp.maximum(distances, 0.0)


def project_simplex_rows_gpu(V):
    """Project every row onto {x >= 0, sum(x) = 1}."""
    if V.shape[1] == 1:
        return cp.ones_like(V)

    sorted_values = cp.sort(V, axis=1)[:, ::-1]
    cumulative_sum = cp.cumsum(sorted_values, axis=1) - 1.0
    positions = cp.arange(1, V.shape[1] + 1, dtype=V.dtype)[None, :]
    rho = (sorted_values - cumulative_sum / positions > 0.0).sum(axis=1) - 1
    rows = cp.arange(V.shape[0])
    theta = cumulative_sum[rows, rho] / (rho + 1)
    return cp.maximum(V - theta[:, None], 0.0)


def _neighbours_gpu(E, k_nn):
    order = cp.argsort(E, axis=1)
    return order[:, :k_nn], order[:, k_nn]


def _initialize_similarity_gpu(E_input, k_nn):
    E = E_input.copy()
    n = int(E.shape[0])
    cp.fill_diagonal(E, cp.inf)

    neighbours, following = _neighbours_gpu(E, k_nn)
    rows = cp.arange(n)[:, None]
    flat_rows = cp.arange(n)
    near = E[rows, neighbours]
    next_distance = E[flat_rows, following]
    denominator = k_nn * next_distance - near.sum(axis=1)

    raw = cp.full_like(near, 1.0 / k_nn)
    safe = cp.abs(denominator) >= _EPS
    raw[safe] = (
        next_distance[safe, None] - near[safe]
    ) / denominator[safe, None]

    weights = project_simplex_rows_gpu(cp.maximum(raw, 0.0))
    similarity = cp.zeros((n, n), dtype=E.dtype)
    similarity[rows, neighbours] = weights
    return similarity


def _update_similarity_gpu(E_input, U, view_weight, k_nn):
    E = E_input.copy()
    n = int(E.shape[0])
    cp.fill_diagonal(E, cp.inf)

    neighbours, following = _neighbours_gpu(E, k_nn)
    rows = cp.arange(n)[:, None]
    flat_rows = cp.arange(n)

    e_near = E[rows, neighbours]
    u_near = U[rows, neighbours]
    e_next = E[flat_rows, following]
    u_next = U[flat_rows, following]

    numerator = (
        e_next[:, None] - e_near
        + 2.0 * view_weight * (u_near - u_next[:, None])
    )
    denominator = (
        k_nn * e_next - e_near.sum(axis=1)
        - 2.0 * k_nn * view_weight * u_next
        + 2.0 * view_weight * u_near.sum(axis=1)
    )

    raw = cp.full_like(e_near, 1.0 / k_nn)
    safe = cp.abs(denominator) >= _EPS
    raw[safe] = numerator[safe] / denominator[safe, None]

    weights = project_simplex_rows_gpu(cp.maximum(raw, 0.0))
    similarity = cp.zeros((n, n), dtype=E.dtype)
    similarity[rows, neighbours] = weights
    return similarity


def _update_weights_gpu(similarities, U):
    norms = cp.stack([cp.linalg.norm(U - S, ord="fro") for S in similarities])
    weights = 1.0 / (2.0 * cp.maximum(norms, _EPS))
    return weights / weights.sum()


def _update_u_gpu(similarities, weights, F, regularization):
    n_views = len(similarities)
    n = int(similarities[0].shape[0])

    row_norms = cp.sum(F * F, axis=1)
    spectral_distances = cp.maximum(
        row_norms[:, None] + row_norms[None, :] - 2.0 * (F @ F.T),
        0.0,
    )

    candidate = cp.zeros_like(similarities[0])
    for view_id, similarity in enumerate(similarities):
        coefficient = regularization / (
            2.0 * n_views * cp.maximum(weights[view_id], _EPS)
        )
        candidate += similarity - coefficient * spectral_distances
    candidate /= n_views

    off_diagonal = ~cp.eye(n, dtype=cp.bool_)
    projected = project_simplex_rows_gpu(
        candidate[off_diagonal].reshape(n, n - 1)
    )
    U = cp.zeros_like(candidate)
    U[off_diagonal] = projected.ravel()
    return U


class GMCGPU:
    """GPU counterpart of GMC with a compatible public result interface."""

    def __init__(
        self,
        k,
        k_nn=5,
        max_iter=50,
        lam_init=1.0,
        lam_factor=2.0,
        tol=1e-8,
        dtype="float64",
        verbose=False,
    ):
        self.k = int(k)
        self.k_nn = int(k_nn)
        self.max_iter = int(max_iter)
        self.lam = float(lam_init)
        self.lam_factor = float(lam_factor)
        self.tol = float(tol)
        self.dtype = dtype
        self.verbose = bool(verbose)

        self.labels_ = None
        self.U_ = None
        self.F_ = None
        self.weights_ = None
        self.eigenvalues_ = None
        self.history_ = []
        self.elapsed_seconds_ = None

    @property
    def gpu_dtype(self):
        if self.dtype == "float32":
            return cp.float32
        if self.dtype == "float64":
            return cp.float64
        raise ValueError("dtype deve essere 'float32' oppure 'float64'.")

    def _check_gpu(self):
        if cp is None or not cupy_available():
            raise RuntimeError("CuPy/CUDA non disponibile.")

    def fit(self, feature_views):
        self._check_gpu()
        if not feature_views:
            raise ValueError("Serve almeno una vista.")

        gpu_views = []
        n_samples = None
        for view_id, view in enumerate(feature_views):
            X = cp.ascontiguousarray(cp.asarray(view, dtype=self.gpu_dtype))
            if X.ndim != 2 or X.shape[1] == 0:
                raise ValueError(f"Vista {view_id}: attesa matrice 2D non vuota.")
            if n_samples is None:
                n_samples = int(X.shape[0])
            elif int(X.shape[0]) != n_samples:
                raise ValueError("Tutte le viste devono avere lo stesso numero di righe.")
            if not bool(cp.all(cp.isfinite(X)).item()):
                raise ValueError(f"Vista {view_id}: contiene valori non finiti.")
            gpu_views.append(X)

        return self.fit_distances([
            row_squared_distances_gpu(X) for X in gpu_views
        ])

    def fit_distances(self, distance_views):
        self._check_gpu()
        start = time.perf_counter()

        if not distance_views:
            raise ValueError("Serve almeno una matrice di distanza.")

        E_list = []
        n = None
        for view_id, distance_view in enumerate(distance_views):
            E = cp.ascontiguousarray(
                cp.asarray(distance_view, dtype=self.gpu_dtype)
            )
            if E.ndim != 2 or E.shape[0] != E.shape[1]:
                raise ValueError(f"Vista {view_id}: attesa matrice quadrata.")
            if n is None:
                n = int(E.shape[0])
            elif E.shape != (n, n):
                raise ValueError("Tutte le matrici di distanza devono avere la stessa forma.")
            if not bool(cp.all(cp.isfinite(E)).item()):
                raise ValueError(f"Vista {view_id}: contiene valori non finiti.")
            E_list.append(E)

        if n < 2:
            raise ValueError("Servono almeno due campioni.")
        if self.k > n:
            raise ValueError("k non puo superare il numero di campioni.")

        if n == 2 and self.k == 2:
            self.labels_ = np.array([0, 1], dtype=np.int32)
            self.U_ = np.array([[0.0, 1.0], [1.0, 0.0]])
            self.F_ = np.eye(2)
            self.weights_ = np.ones(len(E_list)) / len(E_list)
            self.eigenvalues_ = np.array([0.0, 2.0])
            self.history_ = []
            self.elapsed_seconds_ = time.perf_counter() - start
            return self

        k_nn = min(max(self.k_nn, 1), n - 2)
        similarities = [_initialize_similarity_gpu(E, k_nn) for E in E_list]
        weights = cp.ones(len(similarities), dtype=self.gpu_dtype)
        weights /= len(similarities)

        U = cp.zeros_like(similarities[0])
        for view_id, similarity in enumerate(similarities):
            U += weights[view_id] * similarity
        U = 0.5 * (U + U.T)

        eigenvalues, F = smallest_eigenpairs_gpu(
            unified_graph_laplacian_gpu(U),
            self.k,
            tolerance=self.tol,
        )

        regularization = self.lam
        self.history_ = []

        for iteration in range(self.max_iter):
            similarities = [
                _update_similarity_gpu(
                    E_list[view_id], U, weights[view_id], k_nn
                )
                for view_id in range(len(E_list))
            ]
            weights = _update_weights_gpu(similarities, U)
            U = _update_u_gpu(similarities, weights, F, regularization)
            U = 0.5 * (U + U.T)

            diagnostic_count = min(n, self.k + 1)
            values, vectors = smallest_eigenpairs_gpu(
                unified_graph_laplacian_gpu(U),
                diagnostic_count,
                tolerance=self.tol,
            )
            eigenvalues = values[:self.k]
            F = vectors[:, :self.k]
            components = zero_eigenvalue_multiplicity_gpu(values, self.tol)

            if components < self.k:
                regularization *= self.lam_factor
            elif components > self.k:
                regularization /= self.lam_factor

            self.history_.append({
                "iter": iteration + 1,
                "n_components": components,
                "lambda": regularization,
                "weights": cp.asnumpy(weights),
                "eigenvalues": cp.asnumpy(eigenvalues),
            })

            if self.verbose:
                print(
                    f"[GMC-GPU] iter={iteration + 1} "
                    f"components={components} lambda={regularization:.6g}"
                )
            if components == self.k:
                break

        U_cpu = cp.asnumpy(U)
        labels, found = connected_components_labels(U_cpu, self.tol)
        F_cpu = cp.asnumpy(F)

        if found != self.k:
            warnings.warn(
                f"GMCGPU: {found} componenti invece di {self.k}; fallback k-means.",
                stacklevel=2,
            )
            labels = KMeans(
                n_clusters=self.k,
                n_init=20,
                random_state=42,
            ).fit_predict(normalize(F_cpu, norm="l2", axis=1))

        cp.cuda.get_current_stream().synchronize()
        #print("GPU memory used:",cp.get_default_memory_pool().used_bytes()/1024**2,"MB")
        self.labels_ = np.asarray(labels, dtype=np.int32)
        self.U_ = U_cpu
        self.F_ = F_cpu
        self.weights_ = cp.asnumpy(weights)
        self.eigenvalues_ = cp.asnumpy(eigenvalues)
        self.lam = regularization
        self.elapsed_seconds_ = time.perf_counter() - start
        return self


class BinaryHierarchicalGMCGPU:
    """Binary hierarchy using GMCGPU in GPU mode and CPU as fallback."""

    def __init__(
        self,
        k,
        execution_mode="auto",
        gmc_k_nn=5,
        gmc_max_iter=50,
        gmc_tol=1e-8,
        max_parallel_nodes=4,
        dtype="float64",
        verbose=False,
    ):
        if k < 1 or k & (k - 1):
            raise ValueError("k deve essere una potenza di due.")

        self.k = int(k)
        self.depth_ = int(math.log2(k))
        self.execution_mode = execution_mode
        self.gmc_k_nn = int(gmc_k_nn)
        self.gmc_max_iter = int(gmc_max_iter)
        self.gmc_tol = float(gmc_tol)
        self.max_parallel_nodes = max(1, int(max_parallel_nodes))
        self.dtype = dtype
        self.verbose = bool(verbose)

        self.nodes_ = {}
        self.clusters_ = []
        self.labels_ = None
        self.level_timings_ = []
        self.distance_views_ = None

    def _resolved_mode(self):
        if self.execution_mode == "auto":
            return "parallel" if cupy_available() else "serial"
        if self.execution_mode == "parallel" and not cupy_available():
            print("CuPy/CUDA non disponibile.\n Eseguendo in modalità seriale.")
            return "serial"
        if self.execution_mode not in {"serial", "parallel"}:
            raise ValueError("execution_mode deve essere serial, parallel o auto.")
        return self.execution_mode

    def fit(self, global_feature_views):
        mode = self._resolved_mode()
        views, n = validate_feature_views(global_feature_views)

        if mode == "serial":
            reference = BinaryHierarchicalGMC(
                k=self.k,
                gmc_k_nn=self.gmc_k_nn,
                gmc_max_iter=self.gmc_max_iter,
                gmc_tol=self.gmc_tol,
                verbose=self.verbose,
            ).fit(views)
            self.nodes_ = reference.nodes_
            self.clusters_ = reference.clusters_
            self.labels_ = reference.labels_
            self.distance_views_ = reference.distance_views_
            return self

        gpu_dtype = cp.float64 if self.dtype == "float64" else cp.float32
        gpu_views = [cp.asarray(X, dtype=gpu_dtype) for X in views]
        global_distances = [row_squared_distances_gpu(X) for X in gpu_views]
        self.distance_views_ = global_distances

        self.nodes_ = {
            0: BinaryTreeNode(0, 0, np.arange(n, dtype=np.int64))
        }
        current_ids = [0]
        next_id = 1
        self.level_timings_ = []

        for level in range(self.depth_):
            level_start = time.perf_counter()
            if self.verbose:
                print(f"[Binary-GPU] level={level} nodes={len(current_ids)}")

            results = []
            for node_id in current_ids:
                node = self.nodes_[node_id]
                global_indices = node.global_idx
                if global_indices.size < 2:
                    raise RuntimeError(
                        f"Il nodo {node_id} contiene meno di due campioni."
                    )

                indices_gpu = cp.asarray(global_indices, dtype=cp.int64)
                node_distances_gpu = [
                    E[indices_gpu[:, None], indices_gpu[None, :]]
                    for E in global_distances
                ]

                model = GMCGPU(
                    k=2,
                    k_nn=self.gmc_k_nn,
                    max_iter=self.gmc_max_iter,
                    tol=self.gmc_tol,
                    dtype=self.dtype,
                    verbose=False,
                ).fit_distances(node_distances_gpu)

                left_indices = global_indices[model.labels_ == 0]
                right_indices = global_indices[model.labels_ == 1]
                if left_indices.size == 0 or right_indices.size == 0:
                    raise RuntimeError(
                        f"La bisezione GPU del nodo {node_id} ha prodotto un cluster vuoto."
                    )
                results.append((node_id, left_indices, right_indices))

            child_ids = []
            for node_id, left_indices, right_indices in results:
                node = self.nodes_[node_id]

                left = BinaryTreeNode(next_id, level + 1, left_indices, node_id)
                node.left_child_id = next_id
                self.nodes_[next_id] = left
                child_ids.append(next_id)
                next_id += 1

                right = BinaryTreeNode(next_id, level + 1, right_indices, node_id)
                node.right_child_id = next_id
                self.nodes_[next_id] = right
                child_ids.append(next_id)
                next_id += 1

            current_ids = child_ids
            cp.cuda.get_current_stream().synchronize()
            self.level_timings_.append({
                "level": level,
                "nodes": len(results),
                "wall_seconds": time.perf_counter() - level_start,
            })

        leaves = sorted(
            (node for node in self.nodes_.values() if node.left_child_id is None),
            key=lambda node: node.node_id,
        )
        self.clusters_ = [node.global_idx.copy() for node in leaves]
        self.labels_ = np.full(n, -1, dtype=np.int32)
        for cluster_id, indices in enumerate(self.clusters_):
            self.labels_[indices] = cluster_id

        if np.any(self.labels_ < 0):
            missing = np.flatnonzero(self.labels_ < 0)
            raise RuntimeError(
                f"Alcuni campioni non sono stati assegnati: {missing.tolist()}."
            )
        return self
