"""Row-wise missingness (MCAR / MAR) and cheap handling for SA3 sims.

This is the **simulation** half of thesis SA3. Cell-wise MNAR, multiple
imputation, and clinical CSVs are out of scope (see README TODO).

Row MCAR vs subsample
---------------------
``tradsim_fastcausal`` already *keeps* a random fraction of rows
(``discovery.proportions``). That is a complete-data subsample, **not**
MCAR. Row MCAR independently *blanks* entire rows at a configured rate;
the blanked rows stay in the frame as NaNs until a handling strategy
drops or imputes them.

Row MAR
-------
One or more columns stay fully observed (default: the first numeric
column). A logistic model of those covariates sets ``P(row incomplete)``.
On a missing row only the *other* columns become NaN, so the mechanism
depends on observed data (MAR, not MCAR).

The intercept is calibrated so the mean predicted probability equals
``rate``. ``mar_slope`` (default 1.5) controls dependence: a positive
slope makes larger covariate values more likely to be missing.

Handling defaults (cheap)
-------------------------
- ``complete_case`` / ``listwise``: drop any row with a NaN.
  **This is the natural handler for whole-row MCAR.**
- ``mean``: column-wise mean of observed values (default simple impute
  for **MAR**, where covariate columns stay observed).
- ``median``: column-wise median of observed values (optional extra).
- Whole-row MCAR + mean/median just reinserts the column-mean (or median)
  vector as near-duplicate rows. The default grid therefore skips impute
  strategies for MCAR unless ``mcar_impute=True``.
- ``mice`` / ``mnar``: not implemented (TODO).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

MECHANISMS = ("none", "mcar", "mar")
STRATEGIES = ("none", "complete_case", "mean", "median")
IMPUTE_STRATEGIES = ("mean", "median")
DEFAULT_MAR_SLOPE = 1.5
DEFAULT_RATES = (0.0, 0.1, 0.2, 0.4)
DEFAULT_STRATEGIES = ("complete_case", "mean")
DEFAULT_MECHANISMS = ("mcar", "mar")

# Aliases accepted in config / CLI.
_MECHANISM_ALIASES = {
    "none": "none",
    "complete": "none",
    "mcar": "mcar",
    "mar": "mar",
    "mnar": "mnar",
}
_STRATEGY_ALIASES = {
    "none": "none",
    "identity": "none",
    "complete_case": "complete_case",
    "listwise": "complete_case",
    "dropna": "complete_case",
    "mean": "mean",
    "median": "median",
    "mice": "mice",
    "multiple": "mice",
    "mice_forest": "mice",
}


def parse_str_list(raw: Any, default: Iterable[str] | None = None) -> list[str]:
    """Split a comma-string or accept a YAML list."""
    if raw is None:
        return [str(x) for x in (default or [])]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_float_list(raw: Any, default: Iterable[float] | None = None) -> list[float]:
    return [float(x) for x in parse_str_list(raw, default)]


def normalize_mechanism(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _MECHANISM_ALIASES:
        raise ValueError(
            f"Unknown missingness mechanism {name!r}. "
            f"Supported: {', '.join(MECHANISMS)} (mnar is TODO)."
        )
    return _MECHANISM_ALIASES[key]


def normalize_strategy(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _STRATEGY_ALIASES:
        raise ValueError(
            f"Unknown missingness strategy {name!r}. "
            f"Supported: {', '.join(STRATEGIES)} (mice is TODO)."
        )
    return _STRATEGY_ALIASES[key]


def expand_missingness_grid(
    mechanisms: Iterable[str] | None,
    rates: Iterable[float] | None,
    strategies: Iterable[str] | None,
    *,
    enabled: bool = True,
    mcar_impute: bool = False,
) -> list[tuple[str, float, str]]:
    """Return ``(mechanism, rate, strategy)`` cells.

    Every ``rate <= 0`` (any mechanism) collapses to a single
    ``("none", 0.0, "none")`` complete-data baseline so MCAR+MAR does not
    double the no-missingness search.

    Whole-row MCAR skips ``mean`` / ``median`` unless ``mcar_impute`` is
    true (those imputes only reinsert column-mean vectors).
    """
    if not enabled:
        return [("none", 0.0, "none")]

    mechs = [normalize_mechanism(m) for m in (mechanisms or DEFAULT_MECHANISMS)]
    rate_list = [float(r) for r in (rates if rates is not None else DEFAULT_RATES)]
    strats = [normalize_strategy(s) for s in (strategies or DEFAULT_STRATEGIES)]
    if not mechs or not rate_list or not strats:
        return [("none", 0.0, "none")]

    cells: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float, str]] = set()
    for mechanism in mechs:
        if mechanism == "mnar":
            raise NotImplementedError(
                "Cell-wise MNAR is out of scope for this SA3 simulation half (TODO)."
            )
        for rate in rate_list:
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"missingness rate must be in [0, 1], got {rate}")
            if mechanism == "none" or rate <= 0.0:
                cell = ("none", 0.0, "none")
                if cell not in seen:
                    cells.append(cell)
                    seen.add(cell)
                continue
            use_strats = list(strats)
            if mechanism == "mcar" and not mcar_impute:
                use_strats = [s for s in strats if s not in IMPUTE_STRATEGIES]
                if not use_strats:
                    raise ValueError(
                        "Whole-row MCAR + mean/median reinserts column-mean rows. "
                        "Use complete_case (the natural MCAR handler), or set "
                        "mcar_impute: true to force imputation."
                    )
            for strategy in use_strats:
                if strategy == "mice":
                    raise NotImplementedError(
                        "Multiple imputation is a SA3 TODO; use complete_case or mean."
                    )
                cell = (mechanism, rate, strategy)
                if cell not in seen:
                    cells.append(cell)
                    seen.add(cell)
    return cells or [("none", 0.0, "none")]


def group_missingness_draws(
    cells: Iterable[tuple[str, float, str]],
) -> list[tuple[str, float, list[str]]]:
    """Group strategies that share one ``(mechanism, rate)`` missingness draw."""
    grouped: list[tuple[str, float, list[str]]] = []
    index: dict[tuple[str, float], int] = {}
    for mechanism, rate, strategy in cells:
        key = (mechanism, float(rate))
        if key not in index:
            index[key] = len(grouped)
            grouped.append((mechanism, float(rate), [strategy]))
        else:
            grouped[index[key]][2].append(strategy)
    return grouped


def resolve_covariates(
    df: pd.DataFrame,
    covariates: str | Iterable[str] | None = None,
) -> list[str]:
    """Columns that stay fully observed and drive row-MAR probability."""
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric:
        raise ValueError("Row MAR requires at least one numeric column.")
    if covariates is None or covariates == "" or covariates == []:
        return [str(numeric[0])]
    names = parse_str_list(covariates)
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Unknown MAR covariates: {missing}")
    return names


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def calibrate_mar_intercept(
    scores: np.ndarray,
    rate: float,
    slope: float = DEFAULT_MAR_SLOPE,
    lo: float = -20.0,
    hi: float = 20.0,
    iters: int = 48,
) -> float:
    """Binary-search intercept so ``mean(logit^{-1}(a + slope * z)) ≈ rate``."""
    rate = float(rate)
    if rate <= 0.0:
        return float("-inf")
    if rate >= 1.0:
        return float("inf")
    z = np.asarray(scores, dtype=float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mean_p = float(_logistic(mid + slope * z).mean())
        if mean_p < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mar_scores(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    frame = df.loc[:, covariates].astype(float)
    values = frame.to_numpy(dtype=float)
    mu = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    standardized = (values - mu) / sd
    return np.nansum(standardized, axis=1)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def apply_row_mcar(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank entire rows independently with probability ``rate``."""
    rate = float(rate)
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"MCAR rate must be in [0, 1], got {rate}")
    out = df.copy()
    generator = _rng(rng)
    if rate <= 0.0:
        mask = np.zeros(len(df), dtype=bool)
    elif rate >= 1.0:
        mask = np.ones(len(df), dtype=bool)
    else:
        mask = generator.random(len(df)) < rate
    if mask.any():
        out.loc[out.index[mask], :] = np.nan
    info = {
        "mechanism": "mcar",
        "n_missing_rows": int(mask.sum()),
        "missing_rate_target": rate,
        "missing_rate_empirical": float(mask.mean()) if len(df) else 0.0,
        "observed_columns": [],
        "incomplete_columns": list(df.columns),
        "row_mask": mask,
    }
    return out, info


def apply_row_mar(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator | int | None = None,
    covariates: str | Iterable[str] | None = None,
    slope: float = DEFAULT_MAR_SLOPE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank non-covariate columns with probability from observed covariates."""
    rate = float(rate)
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"MAR rate must be in [0, 1], got {rate}")
    observed = resolve_covariates(df, covariates)
    incomplete = [col for col in df.columns if col not in observed]
    if not incomplete:
        raise ValueError(
            "Row MAR requires at least one non-covariate column to blank; "
            f"every column is a covariate: {list(observed)}"
        )
    out = df.copy()
    generator = _rng(rng)
    scores = _mar_scores(df, observed)
    intercept = calibrate_mar_intercept(scores, rate, slope=slope)
    if rate <= 0.0:
        probs = np.zeros(len(df), dtype=float)
        mask = np.zeros(len(df), dtype=bool)
    elif rate >= 1.0:
        probs = np.ones(len(df), dtype=float)
        mask = np.ones(len(df), dtype=bool)
    else:
        probs = _logistic(intercept + float(slope) * scores)
        mask = generator.random(len(df)) < probs
    if mask.any():
        out.loc[out.index[mask], incomplete] = np.nan
    info = {
        "mechanism": "mar",
        "n_missing_rows": int(mask.sum()),
        "missing_rate_target": rate,
        "missing_rate_empirical": float(mask.mean()) if len(df) else 0.0,
        "observed_columns": list(observed),
        "incomplete_columns": list(incomplete),
        "mar_slope": float(slope),
        "mar_intercept": None if not np.isfinite(intercept) else float(intercept),
        "row_mask": mask,
        "probs": probs,
        "scores": scores,
    }
    return out, info


def apply_missingness(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    rng: np.random.Generator | int | None = None,
    *,
    covariates: str | Iterable[str] | None = None,
    mar_slope: float = DEFAULT_MAR_SLOPE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch row MCAR / MAR (or a complete-data no-op)."""
    mechanism = normalize_mechanism(mechanism)
    rate = float(rate)
    if mechanism == "mnar":
        raise NotImplementedError(
            "Cell-wise MNAR is out of scope for this SA3 simulation half (TODO)."
        )
    if mechanism == "none" or rate <= 0.0:
        out = df.copy()
        label = "none" if mechanism == "none" else mechanism
        return out, {
            "mechanism": label,
            "n_missing_rows": 0,
            "missing_rate_target": max(rate, 0.0),
            "missing_rate_empirical": 0.0,
            "observed_columns": [],
            "incomplete_columns": [],
        }
    if mechanism == "mcar":
        return apply_row_mcar(df, rate, rng)
    if mechanism == "mar":
        return apply_row_mar(
            df, rate, rng, covariates=covariates, slope=mar_slope
        )
    raise ValueError(f"Unsupported missingness mechanism: {mechanism}")


def handle_missing(
    df: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a cheap missing-data handler. ``mice`` is a documented TODO."""
    strategy = normalize_strategy(strategy)
    n_nan = int(df.isna().sum().sum())
    n_in = len(df)
    if strategy == "mice":
        raise NotImplementedError(
            "Multiple imputation is a SA3 TODO; use complete_case or mean."
        )
    if strategy == "none":
        if n_nan:
            raise ValueError(
                "strategy 'none' cannot be used when the frame contains NaNs"
            )
        return df.copy(), {
            "strategy": "none",
            "n_rows_kept": n_in,
            "n_rows_dropped": 0,
            "n_imputed_cells": 0,
        }
    if strategy == "complete_case":
        out = df.dropna(axis=0, how="any")
        return out, {
            "strategy": "complete_case",
            "n_rows_kept": len(out),
            "n_rows_dropped": n_in - len(out),
            "n_imputed_cells": 0,
        }
    if strategy in ("mean", "median"):
        out = df.copy()
        for col in out.columns:
            if not out[col].isna().any():
                continue
            fill = out[col].mean() if strategy == "mean" else out[col].median()
            if pd.isna(fill):
                fill = 0.0
            out[col] = out[col].fillna(fill)
        return out, {
            "strategy": strategy,
            "n_rows_kept": len(out),
            "n_rows_dropped": 0,
            "n_imputed_cells": n_nan,
        }
    raise ValueError(f"Unsupported missingness strategy: {strategy}")
