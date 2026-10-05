"""fastcausal-backed FGES / GFCI search used by the thesis discovery runner.

This is the replacement for Java Tetrad (``tools/TetradSearch.py``,
``resources/tetrad-current.jar``) and for the retired ``fastcda`` package.
Algorithms run through ``tetrad-port`` (C++), so no JVM is required.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

SUPPORTED_ALGORITHMS = ("fges", "gfci")


class FastcausalUnavailable(ImportError):
    """Raised when fastcausal / tetrad-port is not installed."""


def _require_fastcausal():
    try:
        from fastcausal import FastCausal
    except ImportError as exc:
        raise FastcausalUnavailable(
            "fastcausal is required for discovery. Install with: "
            "pip install 'fastcausal==0.1.11'"
        ) from exc
    return FastCausal


def lag_knowledge_from_columns(
    columns,
    lag_prefixes=("lag",),
    lag_suffixes=("_lag",),
) -> dict:
    """Build a two-tier temporal knowledge dict from column names.

    Matches the hardcoded lagdrinks / drinks tiers previously passed to
    Tetrad ``add_to_tier`` in ``tradsim_fges_obj.py``.
    """
    lag_cols = []
    current_cols = []
    for col in columns:
        name = str(col)
        is_lag = any(name.startswith(p) for p in lag_prefixes) or any(
            name.endswith(s) for s in lag_suffixes
        )
        if is_lag:
            lag_cols.append(name)
        else:
            current_cols.append(name)
    if not lag_cols:
        return {}
    return {"addtemporal": {0: lag_cols, 1: current_cols}}


def run_search(
    df: pd.DataFrame,
    algorithm: str = "fges",
    alpha: float = 0.01,
    penalty_discount: float = 1.0,
    knowledge: Optional[dict | bool] = None,
    run_sem: bool = True,
    verbose: int = 0,
) -> dict[str, Any]:
    """Run FGES or GFCI and return edges plus optional SEM summaries.

    Parameters
    ----------
    knowledge:
        ``True`` builds lag tiers from column names. A dict is passed through
        to ``FastCausal.run_search``. ``False`` / ``None`` means no prior.
    """
    algorithm = algorithm.lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got {algorithm!r}"
        )

    FastCausal = _require_fastcausal()
    fc = FastCausal(verbose=verbose)

    kdict = knowledge
    if knowledge is True:
        kdict = lag_knowledge_from_columns(df.columns)
    elif knowledge is False:
        kdict = None

    kwargs: dict[str, Any] = {"penalty_discount": penalty_discount}
    if algorithm == "gfci":
        kwargs["alpha"] = alpha

    results, graph = fc.run_search(
        df,
        algorithm=algorithm,
        knowledge=kdict or None,
        run_sem=run_sem,
        **kwargs,
    )

    edges = list(results.get("edges") or [])
    sem_summary = summarize_sem(results.get("sem_results"))

    return {
        "edges": edges,
        "nodes": list(results.get("nodes") or []),
        "algorithm": algorithm,
        "parameters": {
            "alpha": alpha if algorithm == "gfci" else None,
            "penalty_discount": penalty_discount,
        },
        "sem_results": results.get("sem_results"),
        "sem_summary": sem_summary,
        "graph": graph,
        "raw": results,
    }


def summarize_sem(sem_results: Optional[dict]) -> dict[str, Optional[float]]:
    """Mean/std of absolute SEM regression estimates; NaNs if SEM was skipped."""
    empty = {"mean_abs_estimates": None, "std_abs_estimates": None, "n_estimates": 0}
    if not sem_results:
        return empty
    estimates = sem_results.get("estimates")
    if estimates is None:
        return empty
    try:
        import numpy as np

        if "op" in estimates.columns:
            estimates = estimates[estimates["op"] == "~"]
        if "Estimate" not in estimates.columns or estimates.empty:
            return empty
        abs_est = estimates["Estimate"].abs()
        return {
            "mean_abs_estimates": float(abs_est.mean()),
            "std_abs_estimates": float(abs_est.std()) if len(abs_est) > 1 else 0.0,
            "n_estimates": int(len(abs_est)),
        }
    except Exception:
        return empty
