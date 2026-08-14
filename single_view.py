from __future__ import annotations
from spectral import spectral_embedding, spectral_cluster, CutType
import numpy as np

def single_view_clustering(W, k: int, cut_type=CutType.NCUT):
    W = np.asarray(W, dtype=int)
    """
    Perform single-view clustering on the QUBO matrix Q using a non negative view W.

    Parameters
    ----------
    W : np.ndarray
        The non-negative view for clustering.
    k : int
        The number of clusters.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each variable.
    """
    # Perform spectral clustering on the selected view
    labels = spectral_cluster(W, k, cut_type=cut_type)
    return labels