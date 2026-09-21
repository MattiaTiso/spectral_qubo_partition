from __future__ import annotations

import argparse
import gc
import importlib
import json
import time

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sklearn.metrics import adjusted_rand_score

from adjacency import prepare_multiview_from_qubo
from qubovalidate import QuboStd


ALGORITHM_REGISTRY: dict[str, dict[str, str]] = {
    "gmc": {
        "device": "cpu",
        "strategy": "direct",
        "class_name": "GMC",
        "benchmark_module": "gmc_multibin",
        "profile_module": "gmc_multibin_cpu_profiled",
    },
    "gmc_gpu": {
        "device": "gpu",
        "strategy": "direct",
        "class_name": "GMCGPU",
        "benchmark_module": "gmc_cupy_multibin_patch",
        "profile_module": "gmc_multibin_gpu_profiled",
    },
    "hierarchical_gmc": {
        "device": "cpu",
        "strategy": "hierarchical",
        "class_name": "BinaryHierarchicalGMC",
        "benchmark_module": "gmc_multibin",
        "profile_module": "gmc_multibin_cpu_profiled",
    },
    "hierarchical_gmc_gpu": {
        "device": "gpu",
        "strategy": "hierarchical",
        "class_name": "BinaryHierarchicalGMCGPU",
        "benchmark_module": "gmc_cupy_multibin_patch",
        "profile_module": "gmc_multibin_gpu_profiled",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run GMC benchmarks or profiling with selectable "
            "CPU and GPU implementations."
        )
    )

    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="QUBO matrix dimension.",
    )

    parser.add_argument(
        "--nblocks",
        type=int,
        required=True,
        help="Number of expected QUBO blocks.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate the QUBO.",
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Benchmark repetition identifier.",
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHM_REGISTRY),
        default=["gmc", "gmc_gpu"],
        help=(
            "Algorithms to execute. Available values: "
            "gmc, gmc_gpu, hierarchical_gmc, "
            "hierarchical_gmc_gpu."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("benchmark", "profile"),
        default="benchmark",
        help=(
            "benchmark executes a warm-up followed by measured runs; "
            "profile executes a warm-up followed by one profiled run."
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Alias for --mode profile.",
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help=(
            "Number of measured runs in benchmark mode. "
            "Ignored in profile mode."
        ),
    )

    parser.add_argument(
        "--max-parallel-nodes",
        type=int,
        default=1,
        help=(
            "Maximum number of nodes processed concurrently "
            "by hierarchical GPU GMC."
        ),
    )

    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "Floating-point precision used for feature views "
            "and GPU models."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from GMC models.",
    )

    parser.add_argument(
        "--print-labels",
        action="store_true",
        help="Print labels produced by each measured run.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.profile:
        args.mode = "profile"

    if args.dim <= 0:
        raise ValueError(
            f"dim must be positive. Received: {args.dim}."
        )

    if args.nblocks <= 0:
        raise ValueError(
            f"nblocks must be positive. Received: {args.nblocks}."
        )

    if args.nblocks > args.dim:
        raise ValueError(
            "nblocks cannot exceed dim. "
            f"Received dim={args.dim}, nblocks={args.nblocks}."
        )

    if args.dim % args.nblocks != 0:
        raise ValueError(
            "dim must be divisible by nblocks. "
            f"Received dim={args.dim}, nblocks={args.nblocks}."
        )

    if args.measured_runs <= 0:
        raise ValueError(
            "measured-runs must be positive. "
            f"Received: {args.measured_runs}."
        )

    if args.max_parallel_nodes <= 0:
        raise ValueError(
            "max-parallel-nodes must be positive. "
            f"Received: {args.max_parallel_nodes}."
        )

    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError(
            "The algorithms list cannot contain duplicates."
        )


def build_dataset(
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Generate the QUBO and its multiview representation."""
    block_size = args.dim // args.nblocks

    rng = np.random.default_rng(args.seed)

    initial_solution = rng.integers(
        low=0,
        high=2,
        size=args.dim,
    )

    qubo = QuboStd(
        n_blocks=args.nblocks,
        block_size=block_size,
        diagonal_value=720.0,
        within_block_value=680.0,
        between_block_value=580.0,
        noise_std=50.0,
        random_state=args.seed,
        loc=0.0,
    )

    qubo.generate()

    qubo_matrix, expected_labels, _ = qubo.qperm()

    expected_labels = np.asarray(
        expected_labels,
        dtype=np.int32,
    )

    density = float(
        qubo.stats()["density"]
    )

    prepared_views = prepare_multiview_from_qubo(
        qubo_matrix,
        x=initial_solution,
    )

    feature_dtype = np.dtype(args.dtype)

    feature_views = [
        np.ascontiguousarray(
            view,
            dtype=feature_dtype,
        )
        for view in prepared_views.values()
    ]

    return feature_views, expected_labels, density


def load_algorithm_class(
    algorithm_name: str,
    profile: bool,
) -> tuple[type, str]:
    """Dynamically load the selected algorithm implementation."""
    specification = ALGORITHM_REGISTRY[algorithm_name]

    module_key = (
        "profile_module"
        if profile
        else "benchmark_module"
    )

    module_name = specification[module_key]
    class_name = specification["class_name"]

    module = importlib.import_module(module_name)

    try:
        model_class = getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(
            f"Module {module_name!r} does not expose "
            f"class {class_name!r} for algorithm "
            f"{algorithm_name!r}."
        ) from error

    return model_class, module_name


def ensure_gpu_available() -> Any:
    """Import CuPy and verify that a CUDA GPU is available."""
    try:
        import cupy as cp
        from laplacian import cupy_available
    except ImportError as error:
        raise RuntimeError(
            f"Unable to import GPU dependencies: {error}"
        ) from error

    if not cupy_available():
        raise RuntimeError("gpu_not_available")

    return cp


def create_model(
    algorithm_name: str,
    model_class: type,
    args: argparse.Namespace,
    verbose: bool,
) -> Any:
    """Create a fresh model for the selected algorithm."""
    specification = ALGORITHM_REGISTRY[algorithm_name]

    device = specification["device"]
    strategy = specification["strategy"]

    if strategy == "direct":
        model_arguments: dict[str, Any] = {
            "k": args.nblocks,
            "k_nn": 5,
            "max_iter": 50,
            "lam_init": 1.0,
            "lam_factor": 2.0,
            "tol": 1e-8,
            "verbose": verbose,
        }

        if device == "gpu":
            model_arguments.update(
                dtype=args.dtype,
                dense_eigen_threshold=16,
            )

        return model_class(**model_arguments)

    if strategy == "hierarchical":
        model_arguments = {
            "k": args.nblocks,
            "gmc_k_nn": 5,
            "gmc_max_iter": 50,
            "gmc_tol": 1e-8,
            "verbose": verbose,
        }

        if device == "gpu":
            model_arguments.update(
                execution_mode="parallel",
                max_parallel_nodes=args.max_parallel_nodes,
                dtype=args.dtype,
            )

        return model_class(**model_arguments)

    raise ValueError(
        f"Unsupported algorithm strategy: {strategy}"
    )


def create_range_factory(
    device: str,
    profile: bool,
) -> Callable[[str], Any]:
    """Create the NVTX range factory used during profiling."""
    if not profile:
        return lambda _: nullcontext()

    if device == "cpu":
        try:
            from nvtx import annotate
        except ImportError as error:
            raise RuntimeError(
                "CPU profiling requires the nvtx package."
            ) from error

        return annotate

    if device == "gpu":
        try:
            from cupyx.profiler import time_range
        except ImportError as error:
            raise RuntimeError(
                "GPU profiling requires "
                "cupyx.profiler.time_range."
            ) from error

        return time_range

    raise ValueError(
        f"Unsupported profiling device: {device}"
    )


def execute_once(
    algorithm_name: str,
    model_class: type,
    feature_views: list[np.ndarray],
    args: argparse.Namespace,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    """Execute one model fit and return its labels and elapsed time."""
    specification = ALGORITHM_REGISTRY[algorithm_name]
    device = specification["device"]

    cupy_module = (
        ensure_gpu_available()
        if device == "gpu"
        else None
    )

    model = create_model(
        algorithm_name=algorithm_name,
        model_class=model_class,
        args=args,
        verbose=verbose,
    )

    if cupy_module is not None:
        cupy_module.cuda.Device().synchronize()

    start_time = time.perf_counter()

    try:
        model.fit(feature_views)

        if cupy_module is not None:
            cupy_module.cuda.Device().synchronize()

    except Exception:
        if cupy_module is not None:
            try:
                cupy_module.cuda.Device().synchronize()
            except Exception:
                pass

        raise

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    labels = getattr(model, "labels_", None)

    if labels is None:
        raise RuntimeError(
            f"{algorithm_name} did not produce labels."
        )

    labels_array = np.asarray(
        labels,
        dtype=np.int32,
    )

    return labels_array, elapsed_seconds


def compute_timing_statistics(
    run_records: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Compute timing statistics for measured benchmark runs."""
    if not run_records:
        raise ValueError(
            "At least one measured run is required."
        )

    times = np.asarray(
        [
            record["elapsed_seconds"]
            for record in run_records
        ],
        dtype=np.float64,
    )

    mean_seconds = float(np.mean(times))
    median_seconds = float(np.median(times))

    if times.size > 1:
        std_seconds = float(
            np.std(
                times,
                ddof=1,
            )
        )
    else:
        std_seconds = 0.0

    if mean_seconds > 0.0:
        coefficient_of_variation = float(
            100.0 * std_seconds / mean_seconds
        )
    else:
        coefficient_of_variation = 0.0

    return {
        "count": int(times.size),
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "std_seconds": std_seconds,
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "coefficient_of_variation_percent": (
            coefficient_of_variation
        ),
    }


def run_algorithm(
    algorithm_name: str,
    feature_views: list[np.ndarray],
    expected_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run benchmark or profiling for one algorithm."""
    specification = ALGORITHM_REGISTRY[algorithm_name]

    device = specification["device"]
    strategy = specification["strategy"]
    profile = args.mode == "profile"

    algorithm_result: dict[str, Any] = {
        "name": algorithm_name,
        "device": device,
        "strategy": strategy,
        "dtype": args.dtype,
        "status": "pending",
    }

    try:
        model_class, module_name = load_algorithm_class(
            algorithm_name=algorithm_name,
            profile=profile,
        )

        algorithm_result["module"] = module_name

        range_factory = create_range_factory(
            device=device,
            profile=profile,
        )

        warmup_marker = (
            f"{algorithm_name.upper()}_"
            "RUN_00_WARMUP"
        )

        with range_factory(warmup_marker):
            warmup_labels, warmup_seconds = execute_once(
                algorithm_name=algorithm_name,
                model_class=model_class,
                feature_views=feature_views,
                args=args,
                verbose=False,
            )

        warmup_ari = float(
            adjusted_rand_score(
                expected_labels,
                warmup_labels,
            )
        )

        algorithm_result["warmup"] = {
            "elapsed_seconds": warmup_seconds,
            "adjusted_rand_score": warmup_ari,
        }

        del warmup_labels
        gc.collect()

        number_of_runs = (
            1
            if profile
            else args.measured_runs
        )

        run_records: list[dict[str, Any]] = []

        for run_number in range(
            1,
            number_of_runs + 1,
        ):
            measurement_name = (
                "PROFILED"
                if profile
                else "MEASURED"
            )

            marker_name = (
                f"{algorithm_name.upper()}_"
                f"RUN_{run_number:02d}_"
                f"{measurement_name}"
            )

            with range_factory(marker_name):
                labels, elapsed_seconds = execute_once(
                    algorithm_name=algorithm_name,
                    model_class=model_class,
                    feature_views=feature_views,
                    args=args,
                    verbose=args.verbose,
                )

            ari = float(
                adjusted_rand_score(
                    expected_labels,
                    labels,
                )
            )

            run_record = {
                "run": run_number,
                "elapsed_seconds": elapsed_seconds,
                "adjusted_rand_score": ari,
            }

            run_records.append(run_record)

            print(
                f"[{algorithm_name}] "
                f"run={run_number} "
                f"elapsed={elapsed_seconds:.6f}s "
                f"ari={ari:.8f}",
                flush=True,
            )

            if args.print_labels:
                print(
                    f"[{algorithm_name}] "
                    f"labels={labels.tolist()}",
                    flush=True,
                )

            del labels
            gc.collect()

        if profile:
            algorithm_result["profiled_run"] = (
                run_records[0]
            )
        else:
            algorithm_result["measured_runs"] = (
                run_records
            )

            algorithm_result["statistics"] = (
                compute_timing_statistics(
                    run_records
                )
            )

        algorithm_result["status"] = "success"

    except Exception as error:
        algorithm_result["status"] = "error"
        algorithm_result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return algorithm_result


def write_result(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the complete benchmark result to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
            allow_nan=False,
        )

    print(
        "RESULT_JSON="
        + json.dumps(
            summary,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )

    print(
        f"Result written to: {output_path}",
        flush=True,
    )


def main() -> None:
    """Run the selected benchmark or profiling configuration."""
    args = parse_args()
    validate_arguments(args)

    print(
        "[CONFIGURATION] "
        f"mode={args.mode} "
        f"algorithms={','.join(args.algorithms)} "
        f"dim={args.dim} "
        f"nblocks={args.nblocks} "
        f"seed={args.seed} "
        f"run_id={args.run_id} "
        f"dtype={args.dtype}",
        flush=True,
    )

    (
        feature_views,
        expected_labels,
        density,
    ) = build_dataset(args)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "benchmark": {
            "dim": args.dim,
            "nblocks": args.nblocks,
            "block_size": (
                args.dim // args.nblocks
            ),
            "k2_over_n": (
                args.nblocks ** 2
            ) / args.dim,
            "seed": args.seed,
            "run_id": args.run_id,
            "density": density,
            "dtype": args.dtype,
            "max_parallel_nodes": (
                args.max_parallel_nodes
            ),
        },
        "algorithms": [],
    }

    for algorithm_name in args.algorithms:
        print(
            "[RUN] "
            f"mode={args.mode} "
            f"algorithm={algorithm_name} "
            f"dtype={args.dtype}",
            flush=True,
        )

        algorithm_result = run_algorithm(
            algorithm_name=algorithm_name,
            feature_views=feature_views,
            expected_labels=expected_labels,
            args=args,
        )

        summary["algorithms"].append(
            algorithm_result
        )

        print(
            "[RESULT] "
            f"algorithm={algorithm_name} "
            f"status={algorithm_result['status']}",
            flush=True,
        )

    write_result(
        summary=summary,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()