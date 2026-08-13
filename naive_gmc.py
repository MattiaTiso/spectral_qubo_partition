from __future__ import annotations
from adjacency import split_signed 
from spectral import spectral_embedding, spectral_cluster, CutType
from qubovalidate import labels_to_clusters
from sklearn.cluster import KMeans
import numpy as np

def naive_gmc_clustering(W, k : int, cut_type = CutType.NCUT):
    W = np.asarray(W, dtype=int)
    Wp, Wn = split_signed(W)
    valp, vecp = spectral_embedding(Wp, k, cut_type)
    valn, vecn = spectral_embedding(Wn, k, cut_type)
  

    val = np.concatenate((valp,valn))
    vec = np.concatenate((vecp,vecn))
   
    ordered_couples = sorted(zip(val,vec), key=lambda x: x[0])
    val, vec = map(list, zip(*ordered_couples))
    np.asarray(vec, dtype=float)
    
    labels = KMeans(n_clusters=k, n_init=20,
                  random_state=42).fit_predict(vec)
    clusters = labels_to_clusters(labels)
    return clusters, labels



    

    