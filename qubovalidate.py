from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


class QuboStd:
    """
    Class for generating a symmetric QUBO matrix with blocks.
    Adds Gaussian noise to the generated QUBO matrix.
    
    """
    def __init__(self,
        n_blocks: int,
        block_size: int,
        diagonal_value: float,
        within_block_value: float,
        between_block_value: float,
        noise_std: float = 0.2,
        random_state: int = 42,
        loc: float = 0.0,):

        self.n_blocks = n_blocks
        self.block_size = block_size
        self.diagonal_value = diagonal_value
        self.within_block_value = within_block_value
        self.between_block_value = between_block_value
        self.noise_std = noise_std
        self.random_state = random_state
        self.true_labels = np.repeat(
            np.arange(self.n_blocks, dtype=np.int32),
            self.block_size,
        )
        self.qubo_matrix = None
        self.loc = loc

    def generate(self,) -> np.ndarray:
        np.random.seed(self.random_state)
        size = self.n_blocks * self.block_size
        self.qubo_matrix = np.full((size, size), self.between_block_value)

        for block in range(self.n_blocks):
            start = block * self.block_size
            end = start + self.block_size
            self.qubo_matrix[start:end, start:end] = self.within_block_value

        np.fill_diagonal(self.qubo_matrix, self.diagonal_value)

        noise = np.random.default_rng(self.random_state).normal(loc=self.loc, scale=self.noise_std, size=self.qubo_matrix.shape)
        self.qubo_matrix += 0.5 * (noise + noise.T)  # Make the noise symmetric

        return self.qubo_matrix

    def qperm(self,) -> list[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.random_state)
        size = self.n_blocks * self.block_size
        permutation = np.random.permutation(size)
        self.qubo_matrix = self.qubo_matrix[np.ix_(permutation, permutation)]
        permuted_labels = self.true_labels[permutation]
        return self.qubo_matrix, permuted_labels, permutation
    
    def stats(self, tol: float = 1e-8) -> dict[str, float]:
        if self.qubo_matrix is None:
            raise ValueError("The QUBO matrix has not been generated. Call the 'generate()' method first.")

        stats = {
            "mean": np.mean(self.qubo_matrix),
            "std": np.std(self.qubo_matrix),
            "min": np.min(self.qubo_matrix),
            "max": np.max(self.qubo_matrix),
            "nnz": np.sum(self.qubo_matrix <= tol),
            "density": float(np.count_nonzero(self.qubo_matrix) / self.qubo_matrix.size * 100),
        }
        return stats
        
def clusters_to_labels(clusters: list[np.ndarray],n_samples: int,) -> np.ndarray:
    """
    Convert a list of clusters into a label vector.

    Example
    -------
    clusters = [
        np.array([0, 2]),
        np.array([1, 3]),
    ]

    results:
        labels = [0, 1, 0, 1]

    Parameters
    ----------
    clusters : list[np.ndarray]
        List of indices belonging to each cluster.

    n_samples : int
        Total number of samples.

    Returns
    -------
    np.ndarray
        Label vector of shape (n_samples,).

    """

    if n_samples <= 0 or n_samples != len(np.asarray(clusters).ravel()):
        raise ValueError(f"n_samples does not correspond to the number of indices in the clusters ({len(np.asarray(clusters).ravel())}).")

    if  len(clusters) == 0:
        raise ValueError("The list of clusters cannot be empty.")

    labels = np.full( shape=n_samples,fill_value=-1,dtype=np.int32,)
    

    for cluster_id, cluster_indices in enumerate(clusters):
        indices = np.asarray(cluster_indices, dtype=np.int64,).ravel()

        if indices.size == 0:
            raise ValueError(f"The cluster {cluster_id} is empty.")

        labels[indices] = cluster_id
       

    if np.any(labels == -1):
        missing_indices = np.flatnonzero(labels == -1)

        raise ValueError(
            "Some samples have not been assigned: "
            f"{missing_indices.tolist()}."
        )

    return labels


def balance_binary_clusters(
    binary_clusters: list[np.ndarray],
    n_clusters: int | None = None,
) -> list[np.ndarray]:
    """
    Concatenate binary clusters in the order received and divide the
    resulting sequence into equal parts.
   """
   
    if not binary_clusters:
        raise ValueError(
            "The list of binary clusters cannot be empty."
        )

    clusters = [np.asarray(cluster, dtype=np.int64).ravel() for cluster in binary_clusters]

    if n_clusters is None:
        n_clusters = len(clusters)

    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    concatenated_indices = np.concatenate(clusters)

    if concatenated_indices.size == 0:
        raise ValueError("The binary clusters do not contain any indices.")

    if np.unique(concatenated_indices).size != concatenated_indices.size:
        raise ValueError(
            "The binary clusters contain duplicate or overlapping indices."
        )

    n_samples = concatenated_indices.size
    if n_samples % n_clusters != 0:
            raise ValueError(
                f"Divide {n_samples} elements is not possible "
                f"in {n_clusters} equal parts."
            )
    
    # np.split divide in parti esattamente uguali.
    balanced_clusters = [cluster.copy() for cluster in np.split(concatenated_indices,n_clusters,)]
    
    return balanced_clusters

def clusters_in_original_order(
    clusters: list[np.ndarray],
    permutation: np.ndarray,
) -> list[list[int]]:
    """
    Convert the indices referred to the permuted matrix to the indices
    of the original QUBO.

    permutation[new_position] = original_index
    """
    permutation = np.asarray(
        permutation,
        dtype=np.int64,
    ).ravel()

    original_clusters = []

    for cluster in clusters:
        cluster = np.asarray(
            cluster,
            dtype=np.int64,
        ).ravel()

        original_indices = permutation[cluster]

        original_clusters.append(
            sorted(original_indices.tolist())
        )

    return original_clusters


def labels_to_clusters(
    labels: np.ndarray,
) -> list[np.ndarray]:
    """
    Convert a label vector to a list of clusters.
    """
    labels = np.asarray(labels, dtype=np.int32).ravel()

    return [np.flatnonzero(labels == cluster_id) for cluster_id in np.unique(labels)]

def print_evaluation_results(
    results: dict,
    permutation: np.ndarray,
) -> None:
    """
    Print ARI, differences and composition of clusters.
    """
    direct_ari = results["direct_ari"]
    binary_original_ari = results["binary_original_ari"]
    binary_balanced_ari = results["binary_balanced_ari"]
    binary_gpu_ari = results["binary_gpu_ari"]
    binary_gpu_balanced_ari = results["binary_gpu_balanced_ari"]

    delta_direct_balanced = results["delta_direct_balanced"]
    delta_balanced_original = results["delta_balanced_original"]
    delta_direct_gpu_balanced = results["delta_direct_gpu_balanced"]

    balanced_clusters = results["balanced_clusters"]
    balanced_gpu_clusters = results["balanced_gpu_clusters"]
    binary_clusters = results["binary_clusters"]
    binary_gpu_clusters = results["binary_gpu_clusters"]

    print("\n" + "=" * 72)
    print("RESULTS COMPARISON")
    print("=" * 72)

    print(
        "Dimension original Binary:",
        [len(cluster) for cluster in binary_clusters],
    )

    print(
        "Dimension balanced Binary ",
        [len(cluster) for cluster in balanced_clusters],
    )

    print("Dimension GPU Binary:", 
          [len(cluster) for cluster in binary_gpu_clusters] if binary_gpu_clusters is not None else "N/A")

    print(
        "Dimension balanced GPU Binary:",
        [len(cluster) for cluster in balanced_gpu_clusters] if balanced_gpu_clusters is not None else "N/A",
    )

    print("\nAdjusted Rand Index")
    print("-------------------")
    print(f"Direct GMC             : {direct_ari:.6f}")
    print(f"Original Binary        : {binary_original_ari:.6f}")
    print(f"Balanced Binary       : {binary_balanced_ari:.6f}")
    if binary_gpu_balanced_ari is not None:
        print(f"Balanced GPU Binary   : {binary_gpu_balanced_ari:.6f}")

    print("\nDifferences")
    print("----------")
    print(
        "Direct - Balanced Binary      : "
        f"{delta_direct_balanced:+.6f}"
    )
    print(
        "Balanced Binary - Original    : "
        f"{delta_balanced_original:+.6f}"
    )
    if delta_direct_gpu_balanced is not None:
        print(
            "Direct - Balanced GPU Binary  : "
            f"{delta_direct_gpu_balanced:+.6f}"
        )
    

    
    

    original_binary_indices = clusters_in_original_order(
        clusters=binary_clusters,
        permutation=permutation,
    )
    binary_gpu_indices = clusters_in_original_order(
        clusters=binary_gpu_clusters,
        permutation=permutation,
    ) if binary_gpu_clusters is not None else None

    balanced_binary_indices = clusters_in_original_order(
        clusters=balanced_clusters,
        permutation=permutation,
    )
    balanced_gpu_indices = clusters_in_original_order(
        clusters=balanced_gpu_clusters,
        permutation=permutation,
    ) if balanced_gpu_clusters is not None else None


    print("\nOriginal Cluster Binary ")
    print("------------------------")

    for cluster_id, indices in enumerate(original_binary_indices):
        print(
            f"Cluster {cluster_id}: "
            f"{indices}, size={len(indices)}"
        )

    print("\nBalanced Cluster Binary ")
    print("-------------------------")

    for cluster_id, indices in enumerate(balanced_binary_indices):
        print(
            f"Cluster {cluster_id}: "
            f"{indices}, size={len(indices)}"
        )

    print("\n Cluster GPU Binary ")
    print("------------------------")       
    if binary_gpu_indices is not None:
        for cluster_id, indices in enumerate(binary_gpu_indices):
            print(
                f"Cluster {cluster_id}: "
                f"{indices}, size={len(indices)}"
            )

    print("\nBalanced Cluster Binary GPU ")
    print("------------------------------")
    if balanced_gpu_indices is not None:
        for cluster_id, indices in enumerate(balanced_gpu_indices):
            print(
                f"Cluster {cluster_id}: "
                f"{indices}, size={len(indices)}"
            )

def evaluate_clusterings(
    true_labels: np.ndarray,
    direct_labels: np.ndarray,
    binary_labels: np.ndarray,
    binary_clusters: list[np.ndarray],
    n_clusters: int,
    binary_gpu_labels: np.ndarray | None = None,
    binary_gpu_clusters: list[np.ndarray] | None = None,
    
) -> dict:
    """
    Computing ARI and differences between Direct GMC, Original Binary and Balanced Binary serial and GPU.
    """
    true_labels = np.asarray(true_labels, dtype=np.int32).ravel()
    direct_labels = np.asarray(direct_labels, dtype=np.int32).ravel() if direct_labels is not None and len(direct_labels) > 0 else []
    binary_labels = np.asarray(binary_labels, dtype=np.int32).ravel() if binary_labels is not None and len(binary_labels) > 0 else []
    binary_gpu_labels = np.asarray(binary_gpu_labels, dtype=np.int32).ravel() if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else []


    if not (true_labels.size == direct_labels.size):
        raise ValueError(
            "Labels vectors must have the same dimension."
        )

    direct_ari = adjusted_rand_score(true_labels, direct_labels) if direct_labels is not None and len(direct_labels) > 0 else 0
    binary_original_ari = adjusted_rand_score(true_labels, binary_labels) if binary_labels is not None and len(binary_labels) > 0 else 0
    binary_gpu_ari = adjusted_rand_score(true_labels, binary_gpu_labels) if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else 0

    balanced_clusters = balance_binary_clusters(
        binary_clusters=binary_clusters,
        n_clusters=n_clusters,
    ) if binary_clusters is not None and len(binary_clusters) > 0 else []
    balanced_gpu_clusters = balance_binary_clusters(
        binary_clusters=binary_gpu_clusters,
        n_clusters=n_clusters,
    ) if binary_gpu_clusters is not None and len(binary_gpu_clusters) > 0 else []
    
    balanced_labels = clusters_to_labels(
        clusters=balanced_clusters,
        n_samples=true_labels.size,
    ) if balanced_clusters is not None and len(balanced_clusters) > 0 else []
    balanced_gpu_labels = clusters_to_labels(
        clusters=balanced_gpu_clusters,
        n_samples=true_labels.size,
    ) if balanced_gpu_clusters is not None and len(balanced_gpu_clusters) > 0 else []

    binary_balanced_ari = adjusted_rand_score(true_labels, balanced_labels) if balanced_labels is not None and len(balanced_labels) > 0 else 0
    binary_gpu_balanced_ari = adjusted_rand_score(true_labels, balanced_gpu_labels) if balanced_gpu_labels is not None and len(balanced_gpu_labels) > 0 else 0

    delta_direct_balanced = direct_ari - binary_balanced_ari if balanced_labels is not None and len(balanced_labels) > 0 else 0
    delta_direct_gpu_balanced = direct_ari - binary_gpu_balanced_ari if balanced_gpu_labels is not None and len(balanced_gpu_labels) > 0 else 0
    delta_balanced_original = binary_balanced_ari - binary_original_ari if balanced_labels is not None and len(balanced_labels) > 0 else 0

    return {
        "direct_ari": direct_ari,
        "binary_original_ari": binary_original_ari,
        "binary_balanced_ari": binary_balanced_ari,
        "binary_gpu_ari": binary_gpu_ari,
        "binary_gpu_balanced_ari": binary_gpu_balanced_ari,
        "delta_direct_balanced": delta_direct_balanced,
        "delta_balanced_original": delta_balanced_original,
        "delta_direct_gpu_balanced": delta_direct_gpu_balanced,
        "balanced_clusters": balanced_clusters,
        "balanced_gpu_clusters": balanced_gpu_clusters,
        "binary_clusters": binary_clusters,
        "binary_gpu_clusters": binary_gpu_clusters,
    }