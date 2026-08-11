"""CPU reference implementation of Graph-based Multi-view Clustering (GMC).

A view is a feature matrix X_v with shape (n_samples, n_features_v). Rows are
always data points. BinaryHierarchicalGMC preserves the global feature space:
a node with indices idx is equivalent to using X_v[idx, :]. For efficiency,
pairwise distances are computed once globally and each node extracts the exact
block E_v[np.ix_(idx, idx)].
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from laplacian import (
    connected_components_labels,
    smallest_eigenpairs,
    unified_graph_laplacian,
    zero_eigenvalue_multiplicity,
)

_EPS = 1e-12


def validate_feature_views(views, *, require_square=False):
    """Validate views; rows must agree, feature dimensions may differ."""
    if not views:
        raise ValueError("At least one view is required.")
    validated = []
    n_samples = None
    for view_id, view in enumerate(views):
        X = np.ascontiguousarray(np.asarray(view, dtype=np.float64))
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError(f"View {view_id}: expected non-empty 2D matrix.")
        if n_samples is None:
            n_samples = X.shape[0]
        elif X.shape[0] != n_samples:
            raise ValueError("All views must have the same number of rows.")
        if require_square and X.shape != (n_samples, n_samples):
            raise ValueError(f"View {view_id}: expected shape {(n_samples, n_samples)}.")
        if not np.isfinite(X).all():
            raise ValueError(f"View {view_id}: contains non-finite values.")
        validated.append(X)
    if n_samples is None or n_samples < 2:
        raise ValueError("At least two data points are required.")
    return validated, n_samples


def row_squared_distances(X):
    """Squared Euclidean distances between rows of a rectangular matrix."""
    # r_i = ||X[i,:]||^2. One value per data point (row).
    r = np.sum(X * X, axis=1)
    E = r[:, None] + r[None, :] - 2.0 * (X @ X.T)
    return np.maximum(E, 0.0)


def project_simplex_rows(V):
    """Project every row onto {x >= 0, sum(x)=1}.

    Sorting algorithm of Duchi et al. (ICML 2008), stated directly for the
    probability simplex by Wang and Carreira-Perpinan (arXiv:1309.1541).
    """
    if V.shape[1] == 1:
        return np.ones_like(V)
    U = np.sort(V, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1.0
    positions = np.arange(1, V.shape[1] + 1, dtype=V.dtype)[None, :]
    rho = (U - css / positions > 0.0).sum(axis=1) - 1
    theta = css[np.arange(V.shape[0]), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0.0)


def _neighbours(E, k_nn):
    order = np.argsort(E, axis=1, kind="stable")
    return order[:, :k_nn], order[:, k_nn]


def _initialize_similarity(E_input, k_nn):
    E = E_input.copy()
    n = E.shape[0]
    np.fill_diagonal(E, np.inf)
    neighbours, following = _neighbours(E, k_nn)
    rows = np.arange(n)[:, None]
    near = E[rows, neighbours]
    next_distance = E[np.arange(n), following]
    denominator = k_nn * next_distance - near.sum(axis=1)
    raw = np.full_like(near, 1.0 / k_nn)
    safe = np.abs(denominator) >= _EPS
    raw[safe] = (next_distance[safe, None] - near[safe]) / denominator[safe, None]
    weights = project_simplex_rows(np.maximum(raw, 0.0))
    S = np.zeros((n, n), dtype=E.dtype)
    S[rows, neighbours] = weights
    return S


def _update_similarity(E_input, U, view_weight, k_nn):
    E = E_input.copy()
    n = E.shape[0]
    np.fill_diagonal(E, np.inf)
    neighbours, following = _neighbours(E, k_nn)
    rows = np.arange(n)[:, None]
    e_near = E[rows, neighbours]
    u_near = U[rows, neighbours]
    e_next = E[np.arange(n), following]
    u_next = U[np.arange(n), following]
    numerator = (
        e_next[:, None] - e_near + 2.0 * view_weight * (u_near - u_next[:, None])
    )
    denominator = (
        k_nn * e_next
        - e_near.sum(axis=1)
        - 2.0 * k_nn * view_weight * u_next
        + 2.0 * view_weight * u_near.sum(axis=1)
    )
    raw = np.full_like(e_near, 1.0 / k_nn)
    safe = np.abs(denominator) >= _EPS
    raw[safe] = numerator[safe] / denominator[safe, None]
    weights = project_simplex_rows(np.maximum(raw, 0.0))
    S = np.zeros((n, n), dtype=E.dtype)
    S[rows, neighbours] = weights
    return S


def _update_weights(S_list, U):
    norms = np.asarray([np.linalg.norm(U - S, ord="fro") for S in S_list])
    weights = 1.0 / (2.0 * np.maximum(norms, _EPS))
    return weights / weights.sum()


def _update_u(S_list, weights, F, regularization):
    m = len(S_list)
    n = S_list[0].shape[0]
    r = np.sum(F * F, axis=1)
    spectral_distances = np.maximum(r[:, None] + r[None, :] - 2.0 * (F @ F.T), 0.0)
    candidate = np.zeros_like(S_list[0])
    for view_id, S in enumerate(S_list):
        coefficient = regularization / (2.0 * m * max(float(weights[view_id]), _EPS))
        candidate += S - coefficient * spectral_distances
    candidate /= m
    off_diagonal = ~np.eye(n, dtype=bool)
    projected = project_simplex_rows(candidate[off_diagonal].reshape(n, n - 1))
    U = np.zeros_like(candidate)
    U[off_diagonal] = projected.ravel()
    return U


class GMC:
    """Serial CPU GMC. ``fit`` accepts rectangular feature views."""

    def __init__(
        self,
        k,
        k_nn=5,
        max_iter=50,
        lam_init=1.0,
        lam_factor=2.0,
        tol=1e-8,
        verbose=False,
    ):
        self.k = int(k)
        self.k_nn = int(k_nn)
        self.max_iter = int(max_iter)
        self.lam = float(lam_init)
        self.lam_factor = float(lam_factor)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.labels_ = self.U_ = self.F_ = self.weights_ = self.eigenvalues_ = None
        self.history_ = []
        self.elapsed_seconds_ = None

    def fit(self, feature_views):
        views, _ = validate_feature_views(feature_views)
        return self.fit_distances([row_squared_distances(X) for X in views])

    def fit_distances(self, distance_views):
        start = time.perf_counter()
        E_list, n = validate_feature_views(distance_views, require_square=True)
        if self.k > n:
            raise ValueError("k cannot exceed the number of data points.")
        if n == 2 and self.k == 2:
            self.labels_ = np.array([0, 1], dtype=np.int32)
            self.U_ = np.array([[0.0, 1.0], [1.0, 0.0]])
            self.F_ = np.eye(2)
            self.weights_ = np.ones(len(E_list)) / len(E_list)
            self.eigenvalues_ = np.array([0.0, 2.0])
            self.elapsed_seconds_ = time.perf_counter() - start
            return self
        k_nn = min(max(self.k_nn, 1), n - 2)
        S_list = [_initialize_similarity(E, k_nn) for E in E_list]
        weights = np.ones(len(S_list)) / len(S_list)
        U = sum(weights[v] * S for v, S in enumerate(S_list))
        U = 0.5 * (U + U.T)
        eigenvalues, F = smallest_eigenpairs(unified_graph_laplacian(U), self.k)
        regularization = self.lam
        if regularization > 1e6:
            raise RuntimeError(
                f"Lambda = {regularization:.6e}. "
                f"iter={iteration + 1}, "
                f"components={components}, "
                f"eigenvalues={cp.asnumpy(values)}"
                )
        self.history_ = []
        previous_components = None
        consecutive_components = 0
        for iteration in range(self.max_iter):
            S_list = [
                _update_similarity(E_list[v], U, float(weights[v]), k_nn)
                for v in range(len(E_list))
            ]
            weights = _update_weights(S_list, U)
            U = _update_u(S_list, weights, F, regularization)
            U = 0.5 * (U + U.T)
            L = unified_graph_laplacian(U)
            diagnostic_count = min(n, self.k + 1)
            values, vectors = smallest_eigenpairs(L, diagnostic_count)
            eigenvalues = values[: self.k]
            F = vectors[:, : self.k]
            components = zero_eigenvalue_multiplicity(values, self.tol)
            if components == previous_components:
                consecutive_components += 1
            else:
                consecutive_components = 0
            previous_components = components


            if components < self.k:
                regularization *= self.lam_factor
            elif components > self.k:
                regularization /= self.lam_factor

            #elif components > self.k:
                #regularization /= self.lam_factor
                #if components == previous_components and iteration > 0:
                    #stable_iterations += 1
                    #if stable_iterations >= 1:
                        #break
            #else:
                #stable_iterations = 0
                     
            self.history_.append(
                {
                    "iter": iteration + 1,
                    "n_components": components,
                    "consecutive": consecutive_components,
                    "lambda": regularization,
                    "weights": weights.copy(),
                    "eigenvalues": eigenvalues.copy(),
                }
            )
            if self.verbose:
                print(
                    f"[GMC-CPU] iter={iteration+1} components={components} "
                    f"consecutive={consecutive_components}"
                    f"lambda={regularization:.6g}"
                    f"eigenvalues={eigenvalues.tolist()}"
                )
            if components == self.k:
                break
            if consecutive_components >=2 and components > self.k:
                break
        labels, found = connected_components_labels(U, self.tol)
        if found < self.k:
            warnings.warn(
                f"GMC: {found} componenti invece di {self.k}; fallback k-means.",
                stacklevel=2,
            )
            labels = KMeans(n_clusters=self.k, n_init=20, random_state=42).fit_predict(
                normalize(F, norm="l2", axis=1)
            )
        elif found > self.k:
            warnings.warn(f"GMC: {found} componenti invece di {self.k}; ", stacklevel=2)
        self.labels_ = labels.astype(np.int32)
        self.U_, self.F_, self.weights_, self.eigenvalues_ = U, F, weights, eigenvalues
        self.lam = regularization
        self.elapsed_seconds_ = time.perf_counter() - start
        return self


@dataclass(slots=True)
class BinaryTreeNode:
    node_id: int
    level: int
    global_idx: np.ndarray
    parent_id: int | None = None
    left_child_id: int | None = None
    right_child_id: int | None = None
    child_ids: list[int] | None = None


class BinaryHierarchicalGMC:
    """ Hierarchical GMC with binary compatibility."""

    def __init__(self, k, gmc_k_nn=5, gmc_max_iter=50, gmc_tol=1e-8, verbose=False):
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = int(k)
        self.depth_ = None
        self.gmc_k_nn = int(gmc_k_nn)
        self.gmc_max_iter = int(gmc_max_iter)
        self.gmc_tol = float(gmc_tol)
        self.verbose = bool(verbose)
        self.nodes_ = {}
        self.clusters_ = []
        self.labels_ = None
        self.distance_views_ = None

    @staticmethod
    def _component_groups(global_idx, labels):
        labels = np.asarray(labels, dtype=np.int64)
        ordered, seen = [], set()
        for label in labels.tolist():
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        return [global_idx[np.flatnonzero(labels == label)] for label in ordered]

    def _add_children(self, parent_id, groups, next_id):
        parent = self.nodes_[parent_id]
        child_ids = []
        for group in groups:
            child = BinaryTreeNode(
                node_id=next_id,
                level=parent.level + 1,
                global_idx=group,
                parent_id=parent_id,
            )
            self.nodes_[next_id] = child
            child_ids.append(next_id)
            next_id += 1
        parent.child_ids = child_ids
        parent.left_child_id = child_ids[0] if len(child_ids) > 0 else None
        parent.right_child_id = child_ids[1] if len(child_ids) > 1 else None
        return child_ids, next_id

    def _select_leaf(self, frontier):
        candidates = [
            node_id for node_id in frontier if self.nodes_[node_id].global_idx.size >= 2
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda node_id: self.nodes_[node_id].global_idx.size,
        )

    def fit(self, global_feature_views):
        views, n = validate_feature_views(global_feature_views)
        if self.k > n:
            raise ValueError("k cannot exceed the number of data points.")
        self.distance_views_ = [row_squared_distances(X) for X in views]
        self.nodes_ = {0: BinaryTreeNode(0, 0, np.arange(n, dtype=np.int64))}
        frontier = [0]
        next_id = 1
        while len(frontier) < self.k:
            node_id = self._select_leaf(frontier)
            if node_id is None:
                break
            node = self.nodes_[node_id]
            idx = node.global_idx
            node_distances = [E[np.ix_(idx, idx)] for E in self.distance_views_]
            model = GMC(
                k=2,
                k_nn=self.gmc_k_nn,
                max_iter=self.gmc_max_iter,
                tol=self.gmc_tol,
                verbose=self.verbose,
            ).fit_distances(node_distances)
            groups = self._component_groups(idx, model.labels_)
            if len(groups) < 2:
                raise RuntimeError(
                    f"GMC did not produce a valid partition for node {node_id}."
                )
            child_ids, next_id = self._add_children(node_id, groups, next_id)
            position = frontier.index(node_id)
            frontier[position : position + 1] = child_ids
            if self.verbose:
                print(
                    f"[Hierarchy-CPU] node={node_id} components={len(groups)} "
                    f"leaves={len(frontier)} target={self.k}"
                )
        self.clusters_ = [
            self.nodes_[node_id].global_idx.copy() for node_id in frontier
        ]
        self.labels_ = np.full(n, -1, dtype=np.int32)
        for cluster_id, idx in enumerate(self.clusters_):
            self.labels_[idx] = cluster_id
        if np.any(self.labels_ < 0):
            missing = np.flatnonzero(self.labels_ < 0)
            raise RuntimeError(f"Some samples have not been assigned: {missing.tolist()}.")
        return self
