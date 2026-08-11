from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


class QuboStd:
    """
    Classe per generare una matrice QUBO  simmetrica con blocchi.
    Aggiunge un rumore gaussiano alla matrice QUBO generata.
    
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
            raise ValueError("La matrice QUBO non è stata generata. Chiama prima il metodo 'generate()'.")

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
    Converte una lista di cluster in un vettore di etichette.

    Esempio
    -------
    clusters = [
        np.array([0, 2]),
        np.array([1, 3]),
    ]

    risultato:
        labels = [0, 1, 0, 1]

    Parameters
    ----------
    clusters : list[np.ndarray]
        Lista degli indici appartenenti a ciascun cluster.

    n_samples : int
        Numero totale di campioni.

    Returns
    -------
    np.ndarray
        Vettore di label di forma (n_samples,).

    """

    if n_samples <= 0 or n_samples != len(np.asarray(clusters).ravel()):
        raise ValueError(f"n_samples non corrisponde al numero di indici nei cluster ({len(np.asarray(clusters).ravel())}).")

    if  len(clusters) == 0:
        raise ValueError("La lista dei cluster non può essere vuota.")

    labels = np.full( shape=n_samples,fill_value=-1,dtype=np.int32,)
    

    for cluster_id, cluster_indices in enumerate(clusters):
        indices = np.asarray(cluster_indices, dtype=np.int64,).ravel()

        if indices.size == 0:
            raise ValueError(f"Il cluster {cluster_id} è vuoto.")

        labels[indices] = cluster_id
       

    if np.any(labels == -1):
        missing_indices = np.flatnonzero(labels == -1)

        raise ValueError(
            "Alcuni campioni non sono stati assegnati: "
            f"{missing_indices.tolist()}."
        )

    return labels


def balance_binary_clusters(
    binary_clusters: list[np.ndarray],
    n_clusters: int | None = None,
) -> list[np.ndarray]:
    """
    Concatena i cluster Binary nell'ordine ricevuto e divide la
    sequenza risultante in parti uguali.
   """
   
    if not binary_clusters:
        raise ValueError(
            "La lista dei cluster Binary non può essere vuota."
        )

    clusters = [np.asarray(cluster, dtype=np.int64).ravel() for cluster in binary_clusters]

    if n_clusters is None:
        n_clusters = len(clusters)

    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters deve essere un intero.")

    if n_clusters <= 0:
        raise ValueError("n_clusters deve essere positivo.")

    concatenated_indices = np.concatenate(clusters)

    if concatenated_indices.size == 0:
        raise ValueError("I cluster Binary non contengono indici.")

    if np.unique(concatenated_indices).size != concatenated_indices.size:
        raise ValueError(
            "I cluster Binary contengono indici duplicati "
            "o sovrapposti."
        )

    n_samples = concatenated_indices.size
    if n_samples % n_clusters != 0:
            raise ValueError(
                f"Non è possibile dividere {n_samples} elementi "
                f"in {n_clusters} parti uguali."
            )
    
    # np.split divide in parti esattamente uguali.
    balanced_clusters = [cluster.copy() for cluster in np.split(concatenated_indices,n_clusters,)]
    
    return balanced_clusters

def clusters_in_original_order(
    clusters: list[np.ndarray],
    permutation: np.ndarray,
) -> list[list[int]]:
    """
    Converte gli indici riferiti alla matrice permutata negli indici
    originali della QUBO.

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
    Converte un vettore di label in una lista di cluster.
    """
    labels = np.asarray(labels, dtype=np.int32).ravel()

    return [np.flatnonzero(labels == cluster_id) for cluster_id in np.unique(labels)]

def print_evaluation_results(
    results: dict,
    permutation: np.ndarray,
) -> None:
    """
    Stampa ARI, differenze e composizione dei cluster.
    """
    direct_ari = results["direct_ari"]
    binary_original_ari = results["binary_original_ari"]
    binary_balanced_ari = results["binary_balanced_ari"]
    binary_gpu_ari = results["binary_gpu_ari"]

    delta_direct_balanced = results["delta_direct_balanced"]
    delta_balanced_original = results["delta_balanced_original"]
    delta_direct_gpu_balanced = results["delta_direct_gpu_balanced"]

    balanced_clusters = results["balanced_clusters"]
    balanced_gpu_clusters = results["balanced_gpu_clusters"]
    binary_clusters = results["binary_clusters"]
    binary_gpu_clusters = results["binary_gpu_clusters"]

    print("\n" + "=" * 72)
    print("CONFRONTO DEI RISULTATI")
    print("=" * 72)

    print(
        "Dimensioni Binary originale :",
        [len(cluster) for cluster in binary_clusters],
    )

    print(
        "Dimensioni Binary bilanciato:",
        [len(cluster) for cluster in balanced_clusters],
    )

    print("Dimensioni Binary GPU:", 
          [len(cluster) for cluster in binary_gpu_clusters] if binary_gpu_clusters is not None else "N/A")

    print(
        "Dimensioni Binary GPU bilanciato:",
        [len(cluster) for cluster in balanced_gpu_clusters] if balanced_gpu_clusters is not None else "N/A",
    )

    print("\nAdjusted Rand Index")
    print("-------------------")
    print(f"GMC diretto             : {direct_ari:.6f}")
    print(f"Binary originale        : {binary_original_ari:.6f}")
    print(f"Binary bilanciato       : {binary_balanced_ari:.6f}")
    if binary_gpu_ari is not None:
        print(f"Binary GPU bilanciato   : {binary_gpu_ari:.6f}")

    print("\nDifferenze")
    print("----------")
    print(
        "Diretto - Binary bilanciato      : "
        f"{delta_direct_balanced:+.6f}"
    )
    print(
        "Binary bilanciato - originale    : "
        f"{delta_balanced_original:+.6f}"
    )
    if delta_direct_gpu_balanced is not None:
        print(
            "Diretto - Binary GPU bilanciato  : "
            f"{delta_direct_gpu_balanced:+.6f}"
        )
    

    
    

    original_binary_indices = clusters_in_original_order(
        clusters=binary_clusters,
        permutation=permutation,
    )
    binary_gpu_indices = clusters_in_original_order(
        clusters=balanced_gpu_clusters,
        permutation=permutation,
    ) if balanced_gpu_clusters is not None else None

    balanced_binary_indices = clusters_in_original_order(
        clusters=balanced_clusters,
        permutation=permutation,
    )
    balanced_gpu_indices = clusters_in_original_order(
        clusters=balanced_gpu_clusters,
        permutation=permutation,
    ) if balanced_gpu_clusters is not None else None


    print("\nCluster Binary originali")
    print("------------------------")

    for cluster_id, indices in enumerate(original_binary_indices):
        print(
            f"Cluster {cluster_id}: "
            f"{indices}, size={len(indices)}"
        )

    print("\nCluster Binary bilanciati")
    print("-------------------------")

    for cluster_id, indices in enumerate(balanced_binary_indices):
        print(
            f"Cluster {cluster_id}: "
            f"{indices}, size={len(indices)}"
        )

    print("\nCluster Binary GPU originali")
    print("------------------------")       
    if binary_gpu_indices is not None:
        for cluster_id, indices in enumerate(binary_gpu_indices):
            print(
                f"Cluster {cluster_id}: "
                f"{indices}, size={len(indices)}"
            )

    print("\nCluster Binary GPU bilanciati")
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
    Calcola ARI e differenze tra GMC diretto, Binary originale e Binary bilanciato seriale e con GPU.
    """
    true_labels = np.asarray(true_labels, dtype=np.int32).ravel()
    direct_labels = np.asarray(direct_labels, dtype=np.int32).ravel()
    binary_labels = np.asarray(binary_labels, dtype=np.int32).ravel()
    binary_gpu_labels = np.asarray(binary_gpu_labels, dtype=np.int32).ravel() if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else None


    if not (true_labels.size == direct_labels.size == binary_labels.size):
        raise ValueError(
            "I vettori di label devono avere la stessa dimensione."
        )

    direct_ari = adjusted_rand_score(true_labels, direct_labels)
    binary_original_ari = adjusted_rand_score(true_labels, binary_labels)
    binary_gpu_ari = adjusted_rand_score(true_labels, binary_gpu_labels) if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else None

    balanced_clusters = balance_binary_clusters(
        binary_clusters=binary_clusters,
        n_clusters=n_clusters,
    )
    balanced_gpu_clusters = balance_binary_clusters(
        binary_clusters=binary_gpu_clusters,
        n_clusters=n_clusters,
    ) if binary_gpu_clusters is not None and len(binary_gpu_clusters) > 0 else None
    
    balanced_labels = clusters_to_labels(
        clusters=balanced_clusters,
        n_samples=true_labels.size,
    )
    balanced_gpu_labels = clusters_to_labels(
        clusters=balanced_gpu_clusters,
        n_samples=true_labels.size,
    ) if binary_gpu_clusters is not None and len(binary_gpu_clusters) > 0 else None

    binary_balanced_ari = adjusted_rand_score(true_labels, balanced_labels)
    binary_gpu_balanced_ari = adjusted_rand_score(true_labels, balanced_gpu_labels) if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else None

    delta_direct_balanced = direct_ari - binary_balanced_ari
    delta_direct_gpu_balanced = direct_ari - binary_gpu_balanced_ari if binary_gpu_labels is not None and len(binary_gpu_labels) > 0 else None
    delta_balanced_original = binary_balanced_ari - binary_original_ari

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