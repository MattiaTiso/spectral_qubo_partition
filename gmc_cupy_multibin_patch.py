"""
CuPy implementation of GMC and its binary hierarchy.

GPU spectral operations remain centralized in laplacian.py.

In GPU mode, node distance blocks stay on the device and are passed to
GMCGPU without calling the serial GMC model.

Nodes belonging to the same hierarchy level can be processed concurrently
using independent CUDA streams. The maximum number of concurrent nodes is
controlled by max_parallel_nodes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
import warnings

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from gmc_multibin import (
    BinaryTreeNode,
    BinaryHierarchicalGMC,
    validate_feature_views,
)

from laplacian import (
    connected_components_labels,
    cupy_available,
    smallest_eigenpairs_gpu,
    unified_graph_laplacian_gpu,
    zero_eigenvalue_multiplicity_gpu,
)

try:
    import cupy as cp
except ImportError:
    cp = None


_EPS = 1e-12


def row_squared_distances_gpu(X):
    """
    Computes the squared Euclidean distances between the rows
    of a CuPy matrix.

    Parameters
    ----------
    X : cupy.ndarray
        Matrix with shape (n_samples, n_features).

    Returns
    -------
    cupy.ndarray
        Squared distance matrix with shape
        (n_samples, n_samples).
    """
    row_norms = cp.sum(X * X, axis=1)

    distances = row_norms[:, None] + row_norms[None, :] - 2.0 * (X @ X.T)

    return cp.maximum(distances, 0.0)


def project_simplex_rows_gpu(V):
    """
    Projects each row of V onto the simplex:

        x >= 0
        sum(x) = 1

    Parameters
    ----------
    V : cupy.ndarray
        Matrix to project.

    Returns
    -------
    cupy.ndarray
        Matrix with each row projected onto the simplex.
    """
    if V.shape[1] == 1:
        return cp.ones_like(V)

    sorted_values = cp.sort(V, axis=1)[:, ::-1]

    cumulative_sum = cp.cumsum(sorted_values, axis=1) - 1.0

    positions = cp.arange(
        1,
        V.shape[1] + 1,
        dtype=V.dtype,
    )[None, :]

    rho = (sorted_values - cumulative_sum / positions > 0.0).sum(axis=1) - 1

    rows = cp.arange(V.shape[0])

    theta = cumulative_sum[rows, rho] / (rho + 1)

    return cp.maximum(
        V - theta[:, None],
        0.0,
    )


def _neighbours_gpu(E, k_nn):
    """
    Returns the k nearest neighbours and the next neighbour.

    The next neighbour is used in the calculation of the weights
    of the similarity matrix.
    """
    order = cp.argsort(E, axis=1)

    return (
        order[:, :k_nn],
        order[:, k_nn],
    )


def _initialize_similarity_gpu(E_input, k_nn):
    """
    Initializes the similarity matrix for a view.
    """
    E = E_input.copy()
    n = int(E.shape[0])

    cp.fill_diagonal(E, cp.inf)

    neighbours, following = _neighbours_gpu(
        E,
        k_nn,
    )

    rows = cp.arange(n)[:, None]
    flat_rows = cp.arange(n)

    near = E[rows, neighbours]
    next_distance = E[flat_rows, following]

    denominator = k_nn * next_distance - near.sum(axis=1)

    raw = cp.full_like(
        near,
        1.0 / k_nn,
    )

    safe = cp.abs(denominator) >= _EPS

    raw[safe] = (next_distance[safe, None] - near[safe]) / denominator[safe, None]

    weights = project_simplex_rows_gpu(cp.maximum(raw, 0.0))

    similarity = cp.zeros(
        (n, n),
        dtype=E.dtype,
    )

    similarity[rows, neighbours] = weights

    return similarity


def _update_similarity_gpu(
    E_input,
    U,
    view_weight,
    k_nn,
):
    """
    Updates the similarity matrix for a single view.
    """
    E = E_input.copy()
    n = int(E.shape[0])

    cp.fill_diagonal(E, cp.inf)

    neighbours, following = _neighbours_gpu(
        E,
        k_nn,
    )

    rows = cp.arange(n)[:, None]
    flat_rows = cp.arange(n)

    e_near = E[rows, neighbours]
    u_near = U[rows, neighbours]

    e_next = E[flat_rows, following]
    u_next = U[flat_rows, following]

    numerator = (
        e_next[:, None] - e_near + 2.0 * view_weight * (u_near - u_next[:, None])
    )

    denominator = (
        k_nn * e_next
        - e_near.sum(axis=1)
        - 2.0 * k_nn * view_weight * u_next
        + 2.0 * view_weight * u_near.sum(axis=1)
    )

    raw = cp.full_like(
        e_near,
        1.0 / k_nn,
    )

    safe = cp.abs(denominator) >= _EPS

    raw[safe] = numerator[safe] / denominator[safe, None]

    weights = project_simplex_rows_gpu(cp.maximum(raw, 0.0))

    similarity = cp.zeros(
        (n, n),
        dtype=E.dtype,
    )

    similarity[rows, neighbours] = weights

    return similarity


def _update_weights_gpu(similarities, U):
    """
    Updates the weights associated with the views.
    """
    norms = cp.stack(
        [
            cp.linalg.norm(
                U - similarity,
                ord="fro",
            )
            for similarity in similarities
        ]
    )

    weights = 1.0 / (2.0 * cp.maximum(norms, _EPS))

    return weights / weights.sum()


def _update_u_gpu(
    similarities,
    weights,
    F,
    regularization,
):
    """
    Updates the unified matrix U.
    """
    n_views = len(similarities)
    n = int(similarities[0].shape[0])

    row_norms = cp.sum(
        F * F,
        axis=1,
    )

    spectral_distances = cp.maximum(
        row_norms[:, None] + row_norms[None, :] - 2.0 * (F @ F.T),
        0.0,
    )

    candidate = cp.zeros_like(similarities[0])

    for view_id, similarity in enumerate(similarities):
        coefficient = regularization / (
            2.0
            * n_views
            * cp.maximum(
                weights[view_id],
                _EPS,
            )
        )

        candidate += similarity - coefficient * spectral_distances

    candidate /= n_views

    off_diagonal = ~cp.eye(
        n,
        dtype=cp.bool_,
    )

    projected = project_simplex_rows_gpu(
        candidate[off_diagonal].reshape(
            n,
            n - 1,
        )
    )

    U = cp.zeros_like(candidate)
    U[off_diagonal] = projected.ravel()

    return U


class GMCGPU:
    """
    GPU implementation of GMC.

    The public interface of the results is compatible
    with the serial GMC model.
    """

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
        """
        Returns the required CuPy dtype.
        """
        if self.dtype == "float32":
            return cp.float32

        if self.dtype == "float64":
            return cp.float64

        raise ValueError("dtype must be 'float32' or 'float64'.")

    def _check_gpu(self):
        """
        Verifies that CuPy and CUDA are available.
        """
        if cp is None or not cupy_available():
            raise RuntimeError("CuPy/CUDA not available.")

    def fit(self, feature_views):
        """
        Executes GMC starting from the original features.
        """
        self._check_gpu()

        if not feature_views:
            raise ValueError("At least one view is required.")

        gpu_views = []
        n_samples = None

        for view_id, view in enumerate(feature_views):
            X = cp.ascontiguousarray(
                cp.asarray(
                    view,
                    dtype=self.gpu_dtype,
                )
            )

            if X.ndim != 2 or X.shape[1] == 0:
                raise ValueError(f"View {view_id}: " "expected a non-empty 2D matrix.")

            if n_samples is None:
                n_samples = int(X.shape[0])

            elif int(X.shape[0]) != n_samples:
                raise ValueError(
                    "All views must have " "the same number of rows."
                )

            if not bool(cp.all(cp.isfinite(X)).item()):
                raise ValueError(f"View {view_id}: " "contains non-finite values.")

            gpu_views.append(X)

        distance_views = [row_squared_distances_gpu(X) for X in gpu_views]

        return self.fit_distances(distance_views)

    def fit_distances(self, distance_views):
        """
        Executes GMC starting from precomputed distance matrices.
        """
        self._check_gpu()

        start = time.perf_counter()

        if not distance_views:
            raise ValueError("At least one distance matrix is required.")

        E_list = []
        n = None

        for view_id, distance_view in enumerate(distance_views):
            E = cp.ascontiguousarray(
                cp.asarray(
                    distance_view,
                    dtype=self.gpu_dtype,
                )
            )

            if E.ndim != 2 or E.shape[0] != E.shape[1]:
                raise ValueError(f"View {view_id}: " "expected a square matrix.")

            if n is None:
                n = int(E.shape[0])

            elif E.shape != (n, n):
                raise ValueError(
                    "All distance matrices " "must have the same shape."
                )

            if not bool(cp.all(cp.isfinite(E)).item()):
                raise ValueError(f"View {view_id}: " "contains non-finite values.")

            E_list.append(E)

        if n < 2:
            raise ValueError("At least two samples are required.")

        if self.k > n:
            raise ValueError("k cannot exceed the number of samples.")

        if n == 2 and self.k == 2:
            self.labels_ = np.array(
                [0, 1],
                dtype=np.int32,
            )

            self.U_ = np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            )

            self.F_ = np.eye(2)

            self.weights_ = np.ones(len(E_list)) / len(E_list)

            self.eigenvalues_ = np.array([0.0, 2.0])

            self.history_ = []

            self.elapsed_seconds_ = time.perf_counter() - start

            return self

        k_nn = min(
            max(self.k_nn, 1),
            n - 2,
        )

        similarities = [
            _initialize_similarity_gpu(
                E,
                k_nn,
            )
            for E in E_list
        ]

        weights = cp.ones(
            len(similarities),
            dtype=self.gpu_dtype,
        )

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

        if regularization > 1e6:
            raise RuntimeError(
                f"Lambda = {regularization:.6e}. "
                f"iter={iteration + 1}, "
                f"components={components}, "
                f"eigenvalues={cp.asnumpy(values)}"
                )

        previous_components = None
        consecutive_components = 0

        for iteration in range(self.max_iter):
            similarities = [
                _update_similarity_gpu(
                    E_list[view_id],
                    U,
                    weights[view_id],
                    k_nn,
                )
                for view_id in range(len(E_list))
            ]

            weights = _update_weights_gpu(
                similarities,
                U,
            )

            U = _update_u_gpu(
                similarities,
                weights,
                F,
                regularization,
            )

            U = 0.5 * (U + U.T)

            diagnostic_count = min(
                n,
                self.k + 1,
            )

            values, vectors = smallest_eigenpairs_gpu(
                unified_graph_laplacian_gpu(U),
                diagnostic_count,
                tolerance=self.tol,
            )

            eigenvalues = values[: self.k]
            F = vectors[:, : self.k]

            components = zero_eigenvalue_multiplicity_gpu(
                values,
                self.tol,
            )
            if components == previous_components:
                consecutive_components += 1
            else:
                consecutive_components = 0
            previous_components = components 

            if components < self.k:
                regularization *= self.lam_factor

            #elif components > self.k:
                #regularization /= self.lam_factor
            elif components > self.k:
                regularization /= self.lam_factor

                #if components == previous_components and iteration > 0:
                    #consecutive_components += 1
                    #if consecutive_components >= 1:
                        #break
            #else:
                #stable_iterations = 0

            
            self.history_.append(
                {
                    "iter": iteration + 1,
                    "n_components": components,
                    "consecutive_components": consecutive_components,
                    "lambda": regularization,
                    "weights": cp.asnumpy(weights),
                    "eigenvalues": cp.asnumpy(eigenvalues),
                }
            )

            if self.verbose:
                print(
                    f"[GMC-GPU] "
                    f"iter={iteration + 1} "
                    f"components={components} "
                    f"consecutive={consecutive_components} "
                    f"lambda={regularization:.6g}"
                    f"eigenvalues={eigenvalues.tolist()}"
                )

            if components == self.k:
                break
            if consecutive_components >= 2 and components > self.k:
                break

        U_cpu = cp.asnumpy(U)

        labels, found = connected_components_labels(
            U_cpu,
            self.tol,
        )

        F_cpu = cp.asnumpy(F)

        if found < self.k:
            warnings.warn(
                f"GMCGPU: {found} componenti "
                f"invece di {self.k}; "
                "fallback k-means.",
                stacklevel=2,
            )

            labels = KMeans(
                n_clusters=self.k,
                n_init=20,
                random_state=42,
            ).fit_predict(
                normalize(
                    F_cpu,
                    norm="l2",
                    axis=1,
                )
            )

        # Synchronize the currently active stream.
        cp.cuda.get_current_stream().synchronize()

        self.labels_ = np.asarray(
            labels,
            dtype=np.int32,
        )

        self.U_ = U_cpu
        self.F_ = F_cpu

        self.weights_ = cp.asnumpy(weights)

        self.eigenvalues_ = cp.asnumpy(eigenvalues)

        self.lam = regularization

        self.elapsed_seconds_ = time.perf_counter() - start

        return self


class BinaryHierarchicalGMCGPU(BinaryHierarchicalGMC):
    """Size-first GPU hierarchy with speculative parallel node splitting."""

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
        super().__init__(k, gmc_k_nn, gmc_max_iter, gmc_tol, verbose)
        if execution_mode not in {"serial", "parallel", "auto"}:
            raise ValueError("execution_mode must be 'serial', 'parallel' or 'auto'.")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        self.execution_mode = execution_mode
        self.max_parallel_nodes = max(1, int(max_parallel_nodes))
        self.dtype = dtype
        self.level_timings_ = []

    def _resolved_mode(self):
        if self.execution_mode == "auto":
            return "parallel" if cupy_available() else "serial"
        if self.execution_mode == "parallel" and not cupy_available():
            raise RuntimeError("CuPy/CUDA not available.")
        return self.execution_mode

    def _fit_node_gpu(self, node_id, global_distances, device_id):
        """Process a node in an independent CUDA stream."""
        node = self.nodes_[node_id]
        global_indices = node.global_idx
        if global_indices.size < 2:
            raise RuntimeError(f"The node {node_id} contains less than two samples.")

        stream = cp.cuda.Stream(non_blocking=True)
        with cp.cuda.Device(device_id):
            with stream:
                indices_gpu = cp.asarray(global_indices, dtype=cp.int64)
                node_distances = [
                    distances[indices_gpu[:, None], indices_gpu[None, :]]
                    for distances in global_distances
                ]
                model = GMCGPU(
                    k=2,
                    k_nn=self.gmc_k_nn,
                    max_iter=self.gmc_max_iter,
                    tol=self.gmc_tol,
                    dtype=self.dtype,
                    verbose=self.verbose,
                ).fit_distances(node_distances)
                stream.synchronize()

        groups = self._component_groups(global_indices, model.labels_)
        if len(groups) < 2:
            raise RuntimeError(
                f"GMCGPU did not produce a valid partition for node {node_id}."
            )
        return node_id, groups

    def _select_speculative_batch(self, frontier, split_cache):
        """Select expandable leaves globally by decreasing cluster size.

        The first node is the exact node that the serial CPU hierarchy would
        commit next. Additional uncached nodes are returned only for speculative
        parallel evaluation. Their results do not modify the hierarchy until
        they become the largest leaf in a later iteration.
        """
        position = {node_id: pos for pos, node_id in enumerate(frontier)}
        candidates = [
            node_id
            for node_id in frontier
            if self.nodes_[node_id].global_idx.size >= 2
        ]
        candidates.sort(
            key=lambda node_id: (
                -self.nodes_[node_id].global_idx.size,
                position[node_id],
            )
        )
        if not candidates:
            return None, []

        commit_id = candidates[0]
        prefetch_ids = [
            node_id for node_id in candidates if node_id not in split_cache
        ][: self.max_parallel_nodes]
        return commit_id, prefetch_ids

    def fit(self, global_feature_views):
        mode = self._resolved_mode()
        views, n = validate_feature_views(global_feature_views)
        if self.k > n:
            raise ValueError("k cannot exceed the number of data points.")

        if mode == "serial":
            reference = BinaryHierarchicalGMC(
                self.k,
                self.gmc_k_nn,
                self.gmc_max_iter,
                self.gmc_tol,
                self.verbose,
            ).fit(views)
            self.nodes_ = reference.nodes_
            self.clusters_ = reference.clusters_
            self.labels_ = reference.labels_
            self.distance_views_ = reference.distance_views_
            return self

        gpu_dtype = cp.float64 if self.dtype == "float64" else cp.float32
        gpu_views = [
            cp.ascontiguousarray(cp.asarray(view, dtype=gpu_dtype)) for view in views
        ]
        global_distances = [row_squared_distances_gpu(view) for view in gpu_views]
        cp.cuda.get_current_stream().synchronize()
        self.distance_views_ = global_distances

        device_id = int(cp.cuda.runtime.getDevice())
        self.nodes_ = {0: BinaryTreeNode(0, 0, np.arange(n, dtype=np.int64))}
        frontier = [0]
        next_id = 1
        split_cache = {}
        self.level_timings_ = []

        while len(frontier) < self.k:
            commit_id, selected_ids = self._select_speculative_batch(
                frontier,
                split_cache,
            )
            if commit_id is None:
                break

            iteration_start = time.perf_counter()
            workers = min(self.max_parallel_nodes, len(selected_ids))
            cache_hit = commit_id in split_cache

            if self.verbose:
                sizes = [
                    int(self.nodes_[node_id].global_idx.size)
                    for node_id in selected_ids
                ]
                print(
                    f"[Hierarchy-GPU] commit={commit_id} "
                    f"prefetch_nodes={len(selected_ids)} "
                    f"workers={workers} sizes={sizes} cache_hit={cache_hit}"
                )

            results = []
            if selected_ids:
                # ThreadPoolExecutor is retained: speculative node splits are
                # evaluated concurrently in independent CUDA streams.
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="gmc-speculative",
                ) as executor:
                    results = list(
                        executor.map(
                            lambda node_id: self._fit_node_gpu(
                                node_id,
                                global_distances,
                                device_id,
                            ),
                            selected_ids,
                        )
                    )

                # Batch barrier: all speculative results are complete before
                # they are inserted into the cache.
                cp.cuda.Device(device_id).synchronize()
                for node_id, groups in results:
                    split_cache[node_id] = groups

            if commit_id not in split_cache:
                raise RuntimeError(
                    f"Missing speculative split for selected node {commit_id}."
                )

            # Only the globally largest leaf is committed. This reproduces the
            # CPU hierarchy's size-first decision while retaining GPU prefetch.
            groups = split_cache.pop(commit_id)
            child_ids, next_id = self._add_children(
                commit_id,
                groups,
                next_id,
            )
            commit_position = frontier.index(commit_id)
            frontier[commit_position : commit_position + 1] = child_ids

            # Cached results are valid only while their immutable tree node is
            # still an active leaf. Drop stale entries defensively.
            frontier_set = set(frontier)
            stale_ids = [
                node_id for node_id in split_cache if node_id not in frontier_set
            ]
            for node_id in stale_ids:
                del split_cache[node_id]

            elapsed = time.perf_counter() - iteration_start
            self.level_timings_.append(
                {
                    "level": self.nodes_[commit_id].level,
                    "nodes": len(selected_ids),
                    "parallel_workers": workers,
                    "components": [len(groups) for _, groups in results],
                    "committed_node": commit_id,
                    "committed_size": int(
                        self.nodes_[commit_id].global_idx.size
                    ),
                    "committed_components": len(groups),
                    "cache_hit": cache_hit,
                    "cache_size_after": len(split_cache),
                    "leaves_after": len(frontier),
                    "wall_seconds": elapsed,
                }
            )

            if self.verbose:
                print(
                    f"[Hierarchy-GPU] committed={commit_id} "
                    f"leaves={len(frontier)}/{self.k} "
                    f"cache={len(split_cache)} time={elapsed:.6f}s"
                )

        self.clusters_ = [
            self.nodes_[node_id].global_idx.copy() for node_id in frontier
        ]
        self.labels_ = np.full(n, -1, dtype=np.int32)
        for cluster_id, indices in enumerate(self.clusters_):
            self.labels_[indices] = cluster_id
        if np.any(self.labels_ < 0):
            missing = np.flatnonzero(self.labels_ < 0)
            raise RuntimeError(f"Unassigned samples: {missing.tolist()}.")
        return self
