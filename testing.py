import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from pathlib import Path

from utils import CVRP


# ---------------------------------------------------------------------------
# Hyperparameter net builder
# ---------------------------------------------------------------------------

def build_hyperparameter_net(
        optimizers: list[str],
        alphas: list[float],
        rhos: list[float],
        beta: float,
        n_ants: int,
        v: float,
        rho_loc: float,
        max_iterations: int,
        eval_info_interval: int,
) -> list[tuple]:
    """
    Build the full Cartesian product over the three searched axes
    (optimizer × alpha × rho) while keeping all other hyperparameters fixed.

    The returned tuples match the layout expected by CVRP.optimize():
        (optimizer, alpha, beta, rho, n_ants, v, rho_loc, max_iterations, eval_info_interval)

    Parameters
    ----------
    optimizers         : e.g. ["AS", "MINMAX", "ACS"]
    alphas             : e.g. [0.5, 1.0, 2.0]
    rhos               : e.g. [0.1, 0.3, 0.5]
    beta               : fixed pheromone-heuristic balance exponent
    n_ants             : fixed number of ants per iteration
    v                  : fixed ACS greediness parameter
    rho_loc            : fixed local evaporation rate (ACS only)
    max_iterations     : fixed iteration budget
    eval_info_interval : fixed printing interval (passed through; printing
                         is suppressed inside workers)

    Returns
    -------
    List of hyperparameter tuples, length = len(optimizers)*len(alphas)*len(rhos).

    Example
    -------
    >>> net = build_hyperparameter_net(
    ...     ["AS", "ACS"], [1.0, 2.0], [0.3, 0.5],
    ...     beta=2.0, n_ants=20, v=0.9, rho_loc=0.1,
    ...     max_iterations=200, eval_info_interval=50,
    ... )
    >>> len(net)   # 2 * 2 * 2
    8
    """
    fixed = (beta, n_ants, v, rho_loc, max_iterations, eval_info_interval)
    return [
        (optimizer, alpha, *fixed[:1], rho, *fixed[1:])
        #  ^optimizer  ^alpha  ^beta   ^rho  ^n_ants, v, rho_loc, max_iter, interval
        for optimizer, alpha, rho in itertools.product(optimizers, alphas, rhos)
    ]


# ---------------------------------------------------------------------------
# Key helper
# ---------------------------------------------------------------------------

def _result_key(problem_name: str, hyperparameters: tuple) -> str:
    """
    Unique, human-readable key for one (problem, hyperparameters) result.
    Only the three searched axes (optimizer, alpha, rho) appear in the key.

    Example: "A-n33-k5_AS_1.5_0.3"
    """
    optimizer, alpha, _beta, rho, *_rest = hyperparameters
    return f"{problem_name}_{optimizer}_{alpha}_{rho}"


# ---------------------------------------------------------------------------
# Worker — module-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

def _worker(
        problem_name: str,
        hyperparameters: tuple,
        n_experiments: int,
        gap_target: float,
) -> tuple[str, dict]:
    """Runs in a subprocess: loads the problem, runs the experiment series,
    and returns (key, stats_dict)."""
    key = _result_key(problem_name, hyperparameters)
    try:
        cvrp_obj = CVRP(problem_name)
        stats = experiment_series(cvrp_obj, hyperparameters, n_experiments, gap_target)
    except Exception as exc:
        stats = {
            "best_gap": float("nan"),
            "mean_gap": float("nan"),
            "var_gap": float("nan"),
            "mean_elapsed_time": float("nan"),
            "var_elapsed_time": float("nan"),
            "mean_conv_time": float("nan"),
            "var_conv_time": float("nan"),
            "mean_conv_iterations": float("nan"),
            "var_conv_iterations": float("nan"),
            "mean_stgn_measure": float("nan"),
            "var_stgn_measure": float("nan"),
            "error": str(exc),
        }
    return key, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test(
        problem_names: list[str],
        hyperparameter_net: list[tuple],
        n_experiments: int,
        gap_target: float = 0.05,
        save_path: str | Path = "results.json",
        max_workers: int | None = None,
) -> dict[str, dict]:
    """
    Evaluate every (problem, hyperparameters) combination in parallel.

    Parameters
    ----------
    problem_names      : List of problem file stems, e.g. ['A-n33-k5', 'B-n41-k6'].
    hyperparameter_net : List of hyperparameter tuples produced by build_hyperparameter_net().
                         Layout: (optimizer, alpha, beta, rho, n_ants, v,
                                  rho_loc, max_iterations, eval_info_interval)
    n_experiments      : Number of independent runs per (problem, hyperparameters) pair.
    gap_target         : Optimality gap threshold used to measure convergence speed.
    save_path          : Where to persist results as JSON (default: "results.json").
                         Pass None to skip saving.
    max_workers        : Worker processes (default: os.cpu_count()).

    Returns
    -------
    dict[key, stats_dict] where key = "{problem}_{optimizer}_{alpha}_{rho}"
    and stats_dict contains the same fields as experiment_series() returns,
    plus optionally "error" if that run failed.

    The same dict is also written to *save_path* as JSON so it can be
    reloaded later with load_results().
    """
    combos = list(itertools.product(problem_names, hyperparameter_net))
    total = len(combos)
    results: dict[str, dict] = {}

    print(f"Submitting {total} jobs "
          f"({len(problem_names)} problems × {len(hyperparameter_net)} hyperparameter sets, "
          f"{n_experiments} experiments each) …")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_worker, p, h, n_experiments, gap_target): (p, h)
            for p, h in combos
        }
        for done_idx, future in enumerate(as_completed(futures), start=1):
            key, stats = future.result()  # never raises — errors captured inside
            results[key] = stats
            status = f"ERROR: {stats['error']}" if "error" in stats else f"gap={stats['mean_gap']:.4f}"
            print(f"[{done_idx}/{total}] {key} | {status}")

    failed = [k for k, v in results.items() if "error" in v]
    if failed:
        print(f"\n{len(failed)} job(s) failed:")
        for k in failed:
            print(f"  {k} → {results[k]['error']}")

    if save_path is not None:
        save_results(results, save_path)

    return results


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(results: dict[str, dict], path: str | Path = "results.json") -> None:
    """Serialise the results dict to a JSON file."""
    path = Path(path)

    # json doesn't know about float('inf') or float('nan') — convert to strings
    def _sanitize(obj):
        if isinstance(obj, float):
            if obj != obj:  # nan
                return "nan"
            if obj == float("inf"):
                return "inf"
        return obj

    serializable = {
        key: {k: _sanitize(v) for k, v in stats.items()}
        for key, stats in results.items()
    }
    path.write_text(json.dumps(serializable, indent=2))
    print(f"Results saved → {path.resolve()}")


def load_results(path: str | Path = "results.json") -> dict[str, dict]:
    """Load a results dict previously saved by save_results() / test()."""
    path = Path(path)
    raw = json.loads(path.read_text())

    def _restore(v):
        if v == "nan":  return float("nan")
        if v == "inf":  return float("inf")
        return v

    return {
        key: {k: _restore(v) for k, v in stats.items()}
        for key, stats in raw.items()
    }


# ---------------------------------------------------------------------------
# Statistics collection
# ---------------------------------------------------------------------------

def experiment_series(
        cvrp_obj: CVRP,
        hyperparameters: tuple,
        n_experiments: int,
        gap_target: float,
) -> dict:
    """
    Run *n_experiments* independent optimizations and return aggregated statistics
    as a named dict.

    Keys
    ----
    best_gap, mean_gap, var_gap,
    mean_elapsed_time, var_elapsed_time,
    mean_conv_time, var_conv_time,
    mean_conv_iterations, var_conv_iterations,
    mean_stgn_measure, var_stgn_measure
    """
    # (optimizer, alpha, beta, rho, n_ants, v, rho_loc, max_iterations, eval_info_interval)
    _, _, _, _, _, _, _, max_iterations, _ = hyperparameters
    bks = cvrp_obj.minimal_cost

    gaps = []
    stgn_measures = []
    conv_iterations = []
    conv_times = []
    elapsed_times = []

    for _ in range(n_experiments):
        opt_history = cvrp_obj.optimize(hyperparameters)
        elapsed_time = opt_history.elapsed_time
        elapsed_times.append(elapsed_time)

        best_solution_history = opt_history.best_solutions_history
        gaps.append(best_solution_history[-1][0])
        stgn_measures.append(_stagnation_measure(best_solution_history, max_iterations))

        ci = _convergence_speed(best_solution_history, bks, gap_target)
        conv_iterations.append(ci)
        conv_times.append(
            (elapsed_time * ci) / max_iterations if ci != float("inf") else float("inf")
        )

    gaps = (np.array(gaps) - bks) / bks
    stgn_measures = np.array(stgn_measures)
    conv_iterations = np.array(conv_iterations, dtype=float)
    conv_times = np.array(conv_times, dtype=float)
    elapsed_times = np.array(elapsed_times)

    return {
        "best_gap": float(np.min(gaps)),
        "mean_gap": float(np.mean(gaps)),
        "var_gap": float(np.var(gaps)),
        "mean_elapsed_time": float(np.mean(elapsed_times)),
        "var_elapsed_time": float(np.var(elapsed_times)),
        "mean_conv_time": float(np.nanmean(conv_times)),
        "var_conv_time": float(np.nanvar(conv_times)),
        "mean_conv_iterations": float(np.nanmean(conv_iterations)),
        "var_conv_iterations": float(np.nanvar(conv_iterations)),
        "mean_stgn_measure": float(np.mean(stgn_measures)),
        "var_stgn_measure": float(np.var(stgn_measures)),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _stagnation_measure(best_solutions_history, max_iterations):
    summ = 0
    last = 0
    for _, _, iteration in best_solutions_history:
        if iteration > last:
            summ += (iteration - last) ** 2
            last = iteration
    return summ / max_iterations ** 2

def _convergence_speed(best_solutions_history, bks, gap_target):
    for solution_cost, _, iteration in best_solutions_history:
        if solution_cost < bks * (1 + gap_target):
            return iteration
    return float('inf')
