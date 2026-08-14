# QUBO Multi-View Spectral Graph Clustering

A research-oriented Python implementation of **Graph-based Multi-view Clustering (GMC) and classical Spectral Graph Clustering** for grouping variables in Quadratic Unconstrained Binary Optimization (QUBO) problems. The repository provides direct and hierarchical clustering on CPU, optional GPU acceleration with CuPy/CUDA, graph construction utilities, spectral clustering methods, and validation tools for synthetic block-structured QUBO instances.

> **Project status:** research prototype. The API may change, and numerical behavior should be validated on the target dataset and hardware before production use.

## Overview

A QUBO problem is written as

$$
\min_{x \in \\{0,1\\}^n} x^\top Qx,
$$

where $Q \in \mathbb{R}^{n \times n}$ describes linear and pairwise interactions among binary variables. This project converts those interactions into one or more graph views and applies multi-view graph clustering to identify groups of related variables. Such groups can be used as candidate sub-QUBOs in decomposition-based optimization workflows.

The main pipeline is:

```text
QUBO matrix Q and optional binary solution x*
                     |
                     v
       Non-negative graph/feature views
                     |
                     v
       Direct or hierarchical clustering
          |                         |
          v                         v
       CPU / NumPy             GPU / CuPy
                     |
                     v
       Cluster labels and index groups
                     |
                     v
      ARI-based validation on synthetic data
```

## Features

- Direct **Graph-based Multi-view Clustering** with automatic view weighting.
- Binary hierarchical GMC that repeatedly splits an expandable leaf until the requested number of clusters is reached.
- Optional GPU implementation using **CuPy** and CUDA.
- Parallel processing of hierarchy nodes at the same logical level using independent CUDA streams.
- Centralized graph Laplacian and eigensolver utilities for CPU and GPU.
- Spectral clustering with MinCut, RatioCut, and Normalized Cut embeddings.
- Fiedler-vector graph bisection.
- QUBO-to-adjacency conversion, including solution-sensitive signed interactions.
- Synthetic block-structured QUBO generation and evaluation with the Adjusted Rand Index (ARI).

## Repository Usage Structure

```text
Files' dependency relations:

demo_quboablocchi.py
 ├── adjacency.py
 ├── naive_gmc.py ──> spectral.py ──> laplacian.py
 ├── single_view.py ──┘               │  │
 ├── gmc_cupy_multibin_patch.py ──────┬──┘
 ├── gmc_multibin.py ─────────────────┘
 └── qubovalidate.py

Directory's structure:
.
├── adjacency.py                   # QUBO adjacency construction and multiview preparation
├── demo_quboablocchi.py           # Demo entry point / experimental scaffold
├── gmc_multibin.py                # CPU GMC and binary hierarchical GMC
├── gmc_cupy_multibin_patch.py     # CuPy/CUDA GMC and parallel GPU hierarchy
├── laplacian.py                   # Shared CPU/GPU Laplacian and spectral operations
├── qubovalidate.py                # Synthetic QUBO generation and clustering evaluation
├── naive_gmc.py                   # Clustering method based on Zhao & Tang [1]
├── single_view.py                 # Clustering method using the classical spectral method with one view
├── spectral.py                    # Spectral clustering and Fiedler bisection
├── requirements.txt               # Python dependencies
└── README.md
```

## Requirements

- Python **3.12 or newer**
- A working Python environment with the packages listed in `requirements.txt`
- For GPU execution:
  - an NVIDIA GPU;
  - a compatible CUDA installation;
  - a CuPy build compatible with the installed CUDA version.

The CPU implementation does not require CUDA. GPU availability is checked at runtime through `laplacian.cupy_available()`.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

CuPy packages are CUDA-version-specific. If GPU support is required, make sure the CuPy package declared in `requirements.txt` matches the CUDA runtime available on the machine. See the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) when adapting the environment.

## Mathematical Background

### QUBO-derived graph views

Given a QUBO matrix $Q$, `adjacency.py` supports three adjacency definitions.

#### Structural trivial adjacency

$$
W_{ij} =
\begin{cases}
1, & Q_{ij} \neq 0,\\
0, & Q_{ij} = 0.
\end{cases}
$$

This view preserves only the interaction pattern.

#### Structural weighted adjacency

$$
W_{ij} = |Q_{ij}|.
$$

This view preserves interaction magnitudes and signs before any subsequent non-negative view preparation.

#### Solution-sensitive adjacency

Given a current binary solution $x^*$,

$$
W_{ij} = (-1)^{x_i^* + x_j^*}Q_{ij}.
$$

Signed matrices can be split into non-negative positive and negative parts:

$$
W^+ = \max(W,0), \qquad W^- = \max(-W,0), \qquad W = W^+ - W^-.
$$

This construction follows the correlation-based motivation used for clustering-driven sub-QUBO extraction by Zhao and Tang [1].

### Multi-view input

Each view is represented by a feature matrix

$$
X^{(v)} \in \mathbb{R}^{n \times d_v},
$$

where rows correspond to the same $n$ samples or QUBO variables, while the feature dimension $d_v$ may differ across views. The CPU and GPU implementations compute squared Euclidean distances between rows:

$$
E_{ij}^{(v)} = \lVert X_i^{(v)} - X_j^{(v)} \rVert_2^2.
$$

Distance matrices can also be supplied directly through `fit_distances()`.

### Similarity learning and simplex projection

For each view, GMC learns a row-stochastic similarity matrix $S^{(v)}$. Rows are projected onto the probability simplex

$$
\Delta = \\{s \in \mathbb{R}^n : s_i \geq 0,\quad \sum_i s_i = 1 \\}.
$$

The implementation uses the sorting-based Euclidean projection associated with Duchi et al. [3] and stated directly for the probability simplex by Wang and Carreira-Perpiñán [4].

The view weights are inversely related to the Frobenius distance from the unified graph $U$:

$$
\alpha_v \propto \frac{1}{2\lVert U-S^{(v)}\rVert_F},
\qquad
\sum_v \alpha_v = 1.
$$

The method alternates between updating view similarities, view weights, the unified graph, and its spectral embedding. This follows the mutual-reinforcement idea of GMC, in which learned view graphs and a unified graph improve one another [2].

### Unified graph Laplacian

The learned graph is symmetrized:

$$
A = \frac{U+U^\top}{2},
$$

and the unnormalized graph Laplacian is

$$
L_U = D-A,
\qquad
D_{ii}=\sum_j A_{ij}.
$$

The multiplicity of the zero eigenvalue of $L_U$ is used as a numerical estimate of the number of connected components. The regularization parameter is adjusted until the graph approaches the requested number of components. Standard relationships among graph Laplacians, connected components, and spectral clustering are reviewed by von Luxburg [5].

### Spectral clustering

The project also exposes three cut strategies through `spectral.py`:

- `CutType.MINCUT`: unnormalized Laplacian embedding;
- `CutType.RATIOCUT`: unnormalized embedding followed by row normalization;
- `CutType.NCUT`: the generalized eigenproblem

$$
(D-W)y = \lambda Dy.
$$

The Normalized Cut formulation and its generalized eigenvalue relaxation can be equivalentely derived by using eigenvalues of symmetric normalized laplacian and row normalization as well [6], which we implement. If the graph already has exactly $k$ connected components, their labels are returned directly. Otherwise, k-means is applied to the first $k$ spectral vectors.

## Quick Start

### Direct GMC on CPU

Every view must contain the same number of rows. Different views may have different feature dimensions.

```python
import numpy as np
from gmc_multibin import GMC

rng = np.random.default_rng(42)
view_1 = rng.normal(size=(100, 8))
view_2 = rng.normal(size=(100, 5))

model = GMC(
    k=4,
    k_nn=5,
    max_iter=50,
    tol=1e-8,
    verbose=True,
).fit([view_1, view_2])

print(model.labels_)
print(model.weights_)
print(model.eigenvalues_)
print(model.elapsed_seconds_)
```

### GMC from precomputed distances

```python
from gmc_multibin import GMC, row_squared_distances

E1 = row_squared_distances(view_1)
E2 = row_squared_distances(view_2)

model = GMC(k=4).fit_distances([E1, E2])
labels = model.labels_
```

Each distance view must be a finite square matrix of shape `(n_samples, n_samples)`.

### Binary hierarchical GMC on CPU

```python
from gmc_multibin import BinaryHierarchicalGMC

hierarchy = BinaryHierarchicalGMC(
    k=4,
    gmc_k_nn=5,
    gmc_max_iter=50,
    gmc_tol=1e-8,
    verbose=True,
).fit([view_1, view_2])

print(hierarchy.labels_)
print(hierarchy.clusters_)
print(hierarchy.nodes_)
```

The hierarchy repeatedly selects the largest expandable leaf and applies a two-cluster GMC split. Global pairwise distances are computed once, then exact node-specific blocks are extracted.

### GMC on GPU

```python
from gmc_cupy_multibin_patch import GMCGPU

model = GMCGPU(
    k=4,
    k_nn=5,
    max_iter=50,
    dtype="float64",
    verbose=True,
).fit([view_1, view_2])

# Public result arrays are returned as NumPy arrays.
print(model.labels_)
print(model.weights_)
```

Use `dtype="float32"` to reduce GPU memory use and potentially improve throughput, at the cost of lower numerical precision. However, with float32, a numerical aberration has been encountered during tests compared with the CPU version, leading to not detect connected components in the unified matrix U.

### Parallel GPU hierarchy

```python
from gmc_cupy_multibin_patch import BinaryHierarchicalGMCGPU

hierarchy = BinaryHierarchicalGMCGPU(
    k=8,
    execution_mode="auto",
    max_parallel_nodes=4,
    dtype="float64",
    verbose=True,
).fit([view_1, view_2])

print(hierarchy.labels_)
print(hierarchy.clusters_)
print(hierarchy.level_timings_)
```

Execution modes:

- `"serial"`: delegates to the CPU hierarchy;
- `"parallel"`: requires CuPy and a usable CUDA device;
- `"auto"`: selects GPU parallel execution when CUDA is available, otherwise CPU execution.

At each selected hierarchy level, nodes are processed through a thread pool. Each worker creates an independent non-blocking CUDA stream, followed by a device-level synchronization barrier before child nodes are inserted.

## QUBO Workflow

### Generate a synthetic block-structured QUBO

```python
from qubovalidate import QuboStd

problem = QuboStd(
    n_blocks=4,
    block_size=25,
    diagonal_value=1.0,
    within_block_value=-1.0,
    between_block_value=0.1,
    noise_std=0.2,
    random_state=42,
)

Q = problem.generate()
Q_permuted, true_labels, permutation = problem.qperm()
```

### Build multiview data from a QUBO

```python
import numpy as np
from adjacency import prepare_multiview_from_qubo

x_star = np.zeros(Q_permuted.shape[0], dtype=np.int32)
views_by_name = prepare_multiview_from_qubo(Q_permuted, x_star)

view_names = list(views_by_name)
views = list(views_by_name.values())

print(view_names)
```

The prepared views are intended to be non-negative representations suitable for the clustering pipeline. Inspect `adjacency_stats()` during experiments to verify symmetry, density, range, and sign structure.

### Evaluate clustering quality

```python
from gmc_multibin import GMC, BinaryHierarchicalGMC
from qubovalidate import evaluate_clusterings, print_evaluation_results

n_clusters = 4

direct = GMC(k=n_clusters).fit(views)
binary = BinaryHierarchicalGMC(k=n_clusters).fit(views)

results = evaluate_clusterings(
    true_labels=true_labels,
    direct_labels=direct.labels_,
    binary_labels=binary.labels_,
    binary_clusters=binary.clusters_,
    n_clusters=n_clusters,
)

print_evaluation_results(results, permutation)
```

The main quality metric is the **Adjusted Rand Index**. An ARI of `1.0` indicates identical partitions up to label permutation; values near `0.0` indicate agreement close to chance.

## Spectral Utilities

### Spectral clustering

```python
from spectral import CutType, spectral_cluster

labels = spectral_cluster(
    W,
    k=4,
    cut_type=CutType.NCUT,
    n_init=20,
    random_state=42,
)
```

### Fiedler bisection

```python
from spectral import spectral_bisection

labels = spectral_bisection(W, balance="median")
```

`balance="median"` assigns half of the ordered Fiedler-vector entries to each side when possible. `balance="sign"` splits entries at zero. The method requires a connected graph, unless the graph already contains exactly two connected components.

## Main API

### `GMC`

Constructor parameters:

- `k`: requested number of clusters;
- `k_nn`: number of nearest neighbors used in adaptive similarity learning;
- `max_iter`: maximum number of alternating updates;
- `lam_init`: initial spectral regularization coefficient;
- `lam_factor`: multiplicative regularization update factor;
- `tol`: zero-eigenvalue and graph-connectivity tolerance;
- `verbose`: print iteration diagnostics.

Important fitted attributes:

- `labels_`: cluster assignment for every sample;
- `U_`: learned unified graph;
- `F_`: spectral embedding;
- `weights_`: learned view weights;
- `eigenvalues_`: smallest eigenvalues retained by the model;
- `history_`: per-iteration diagnostic records;
- `elapsed_seconds_`: total fit time.

If the final unified graph has fewer connected components than requested, the CPU and GPU implementations use k-means on the row-normalized spectral embedding as a fallback.

### `BinaryHierarchicalGMC`

Additional fitted attributes:

- `nodes_`: mapping from node IDs to `BinaryTreeNode` instances;
- `clusters_`: global sample indices for final leaves;
- `distance_views_`: globally precomputed distance matrices.

### `BinaryHierarchicalGMCGPU`

Additional parameters and attributes:

- `execution_mode`: `"serial"`, `"parallel"`, or `"auto"`;
- `max_parallel_nodes`: maximum number of hierarchy nodes submitted concurrently;
- `dtype`: `"float32"` or `"float64"`;
- `level_timings_`: per-level GPU timing and concurrency diagnostics.

## Input Constraints and Numerical Notes

- At least one view and at least two samples are required.
- All feature views must have the same number of rows.
- All input values must be finite.
- `k` cannot exceed the number of samples.
- Precomputed distances must be square and share the same shape.
- GMC is sensitive to graph construction, feature scaling, `k_nn`, tolerance, and regularization updates.
- The zero-eigenvalue test is numerical. Changing precision may alter the detected number of components.
- Dense $n \times n$ similarities and distances require $O(n^2)$ memory per view.
- GPU acceleration is workload- and hardware-dependent. Small matrices may be faster on CPU because of transfer, launch, and synchronization overhead.
- In `laplacian.py`, large GPU problems use a partial eigensolver when possible and fall back to dense `cupy.linalg.eigh` if necessary.
- Hierarchical splitting can produce unequal leaf sizes. The validation utilities include an optional post-hoc balancing operation, but that operation changes cluster membership and should be interpreted carefully.

## Development Notes

The current `demo_quboablocchi.py` is an experimental scaffold rather than a complete end-to-end example. The snippets in this README show the intended public APIs, but they should be adapted to the exact keys returned by `prepare_multiview_from_qubo()` and to the experiment configuration used in the repository.

Recommended additions for a production-quality release include:

- automated unit tests for CPU/GPU parity;
- reproducible benchmark configurations;
- explicit dependency version pinning;
- continuous integration for CPU paths;
- input validation tests for signed and sparse QUBO matrices;
- a complete command-line demo;


## Reproducibility

For reproducible experiments:

1. fix NumPy and k-means random seeds;
2. record package, CUDA, driver, and GPU versions;
3. store all GMC constructor parameters;
4. preserve the QUBO permutation and current solution $x^*$;
5. report whether `float32` or `float64` was used;
6. save `history_` and, for GPU hierarchies, `level_timings_`.

GPU eigensolvers and parallel execution may still exhibit small floating-point variations across platforms.

## References

1. W. Zhao and G. Tang, “Clustering-Based Sub-QUBO Extraction for Hybrid QUBO Solvers,” *arXiv:2502.16212*, 2025. 
2. H. Wang, Y. Yang, and B. Liu, “GMC: Graph-Based Multi-View Clustering,” *IEEE Transactions on Knowledge and Data Engineering*, vol. 32, no. 6, pp. 1116-1129, 2020. 
3. J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, “Efficient Projections onto the $\ell_1$-Ball for Learning in High Dimensions,” *Proceedings of the 25th International Conference on Machine Learning*, pp. 272-279, 2008. 
4. W. Wang and M. Á. Carreira-Perpiñán, “Projection onto the Probability Simplex: An Efficient Algorithm with a Simple Proof, and an Application,” *arXiv:1309.1541*, 2013. 
5. U. von Luxburg, “A Tutorial on Spectral Clustering,” *Statistics and Computing*, vol. 17, pp. 395-416, 2007. 
6. Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In T. Dietterich, S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 (pp. 849 – 856). MIT Press.

