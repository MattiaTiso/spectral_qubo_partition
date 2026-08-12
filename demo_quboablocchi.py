
from __future__ import annotations

import time

import numpy as np

from adjacency import prepare_multiview_from_qubo

from gmc_multibin import GMC, BinaryHierarchicalGMC
from gmc_cupy_multibin_patch import BinaryHierarchicalGMCGPU
from qubovalidate import QuboStd, evaluate_clusterings, labels_to_clusters, clusters_to_labels, print_evaluation_results

# ---------------------------------------------------------------------
#  Demo main
# ---------------------------------------------------------------------

def main() -> None:
    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=140,
    )

    n_blocks = 16
    block_size = 128
    n_variables = n_blocks * block_size
    x = np.random.randint(0, 2, size=n_variables)
    # -------------------------------------------------------------
    # A. QUBO generation
    # -------------------------------------------------------------


    qubo=QuboStd(
        n_blocks=n_blocks,
        block_size=block_size,
        diagonal_value=10.0,
        within_block_value=5.0,
        between_block_value=4.7,
        noise_std=0.5,
        random_state=42,
        loc=0.0,
    )

    qubo.generate()

    Q, expected_labels, permutation = qubo.qperm()
    expected_clusters = labels_to_clusters(expected_labels)
    feature_views= prepare_multiview_from_qubo(Q, x=x).values()
    stats = qubo.stats()
    print(f"density: {stats['density']}\n")
    

    

    print("=" * 72)
    print("ORIGINAL QUBO MATRIX")
    print("=" * 72)

    print(f"Shape                 : {Q.shape}")
    print(f"Number of variables   : {n_variables}")
    print(f"Number of clusters     : {n_blocks}")
    print(f"Dimension of blocks: {block_size}")
    print(f"Symmetric matrix    : {np.allclose(Q, Q.T)}")
    print(f"Optimal clusters      : {expected_clusters}")


    
    #Testing models
    
    # -------------------------------------------------------------
    # B. Direct GMC 
    # -------------------------------------------------------------

    direct_model = GMC(
        k=n_blocks,
        k_nn=5,
        max_iter=50,
        lam_init=1.0,
        lam_factor=2.0,
        tol=1e-8,
        verbose=True,
    )

    direct_model.fit(feature_views)
    print("Ending Direct GMC.")

    # -------------------------------------------------------------
    # C. BinaryHierarchicalGMC
    # -------------------------------------------------------------

    binary_model = BinaryHierarchicalGMC(
        k=n_blocks,
        gmc_k_nn=5,
        gmc_max_iter=50,
        gmc_tol=1e-8,
        verbose=True,
    )

    binary_start = time.perf_counter()
    binary_model.fit(feature_views)
    binary_elapsed_seconds = time.perf_counter() - binary_start

    #-------------------------------------------------------------
    # D. BinaryHierarchicalGMCGPU
    #-------------------------------------------------------------  

    binary_gpu_labels = []
    binary_gpu_clusters = []
    binary_elapsed_seconds_gpu = 0
    gpu_available = False

    try:
        import cupy as cp
        from laplacian import cupy_available
        gpu_available = cupy_available()
        if not gpu_available:
            print("CuPy available but GPU not detected. Skipping GPU GMC execution.")
    except ImportError:
        print("CuPy not available. Skipping GPU GMC execution.")

    if gpu_available:
        print("GPU available.\n")
        binary_modelgpu = BinaryHierarchicalGMCGPU(
            k=n_blocks,
            execution_mode="parallel",
            gmc_k_nn=5,
            gmc_max_iter=50,
            gmc_tol=1e-8,
            verbose=True,
            dtype="float64",
            max_parallel_nodes=1,
        )
        try:
            if binary_modelgpu._resolved_mode() == "parallel":
                print("Execution of GPU GMC in parallel mode.")
                binary_start_gpu = time.perf_counter()
                binary_modelgpu.fit(feature_views)
                binary_elapsed_seconds_gpu = time.perf_counter() - binary_start_gpu
                binary_gpu_labels = binary_modelgpu.labels_
                binary_gpu_clusters = binary_modelgpu.clusters_
        except ImportError:
            print("GPU not available.")
            binary_elapsed_seconds_gpu = 0
            binary_gpu_labels = []
            binary_gpu_clusters = []
            
    # -------------------------------------------------------------
    # E. Results validation
    # -------------------------------------------------------------
    

    if direct_model.labels_ is None:
        raise RuntimeError(
            "The Direct GMC did not produce any labels."
        )

    if binary_model.labels_ is None:
        raise RuntimeError(
            "The Binary GMC did not produce any labels."
        )
    try:
        if cupy_available() is False:
            print("GMC GPU not executed due to missing GPU or CuPy.")
        elif binary_gpu_labels is None or len(binary_gpu_labels) == 0:
            raise RuntimeError(
                "The Binary GMC GPU did not produce any labels."
            )
    except:
        pass
    
    if len(binary_model.clusters_) != n_blocks:
        raise RuntimeError(
            "The Binary GMC did not produce the expected number "
            f"of clusters: produced {len(binary_model.clusters_)}, "
            f"expected {n_blocks}."
        )

    # -------------------------------------------------------------
    # F. Bilanciamento e calcolo degli ARI
    # -------------------------------------------------------------

    results = evaluate_clusterings(
        true_labels=expected_labels,
        direct_labels=direct_model.labels_,
        binary_labels=binary_model.labels_,
        binary_clusters=binary_model.clusters_,
        n_clusters=n_blocks,
        binary_gpu_labels=binary_gpu_labels,
        binary_gpu_clusters=binary_gpu_clusters,
    )

    # -------------------------------------------------------------
    # G. Printing execution times
    # -------------------------------------------------------------

    print("\n" + "=" * 72)
    print("EXECUTION TIMES")
    print("=" * 72)

    print(
        "Direct GMC:",
        f"{direct_model.elapsed_seconds_:.6f} seconds",
    )

    print(
        "BinaryHierarchicalGMC:",
        f"{binary_elapsed_seconds:.6f} seconds",
    )

    print(
        "BinaryHierarchicalGMCGPU:",
        f"{binary_elapsed_seconds_gpu:.6f} seconds",
    ) if binary_elapsed_seconds_gpu != 0 else "N/A"

    # -------------------------------------------------------------
    # H. Printing comparisons
    # -------------------------------------------------------------

    print_evaluation_results(
       results=results, 
       permutation=permutation)

    # -------------------------------------------------------------
    # I. Printing produced labels
    # -------------------------------------------------------------

    print("\n" + "=" * 72)
    print("PERMUTED LABELS")
    print("=" * 72)

    print("Ground truth:")
    print(expected_labels)

    print("\nDirect GMC:")
    print(direct_model.labels_)

    print("\nOriginal Binary:")
    print(binary_model.labels_)

    print("\nBalanced Binary:")
    print(clusters_to_labels(results["balanced_clusters"], n_samples=len(expected_labels)))

    # -------------------------------------------------------------
    # J. Tree structure
    # -------------------------------------------------------------

    print("\n" + "=" * 72)
    print("TREE STRUCTURE CPU")
    print("=" * 72)

    for node_id in sorted(binary_model.nodes_):
        node = binary_model.nodes_[node_id]

        original_indices = sorted(
            permutation[node.global_idx].tolist()
        )

        print(
            f"node={node.node_id:2d} | "
            f"level={node.level} | "
            f"parent={str(node.parent_id):>4s} | "
            f"left={str(node.left_child_id):>4s} | "
            f"right={str(node.right_child_id):>4s} | "
            f"size={len(node.global_idx):2d} | "
            f"indici originali={original_indices}"
        )


if __name__ == "__main__":
    main()