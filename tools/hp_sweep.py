"""Hyperparameter / target-FPR selection helpers for SA4 sims.

Sweep FGES ``penalty_discount`` (and GFCI ``alpha`` when that algorithm is
used), estimate empirical oriented FPR vs picause ground truth, and pick
the setting closest to each requested target FPR.

This is the **simulation** half of thesis SA4. Clinical / real-data
calibration remains out of scope (see README TODO).
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from tools.missingness import parse_float_list

DEFAULT_PENALTY_DISCOUNTS = (1.0, 2.0, 4.0)
DEFAULT_ALPHAS = (0.01, 0.05)
DEFAULT_TARGET_FPRS = (0.05, 0.10)

HP_METRIC_COLS = (
    "oriented_fpr",
    "oriented_tpr",
    "skeleton_fpr",
    "skeleton_tpr",
    "dice_skeleton",
    "dice_directed",
    "oriented_tp",
    "oriented_fp",
    "oriented_fn",
)

HP_COLS = ("penalty_discount", "alpha")


def unique_floats(values: Iterable[float]) -> list[float]:
    """Preserve order while dropping duplicate floats."""
    seen: set[float] = set()
    out: list[float] = []
    for raw in values:
        value = float(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def expand_hp_cells(
    algorithm: str,
    penalty_discounts: Iterable[float] | None,
    alphas: Iterable[float] | None,
    *,
    enabled: bool = True,
    scalar_penalty: float = 1.0,
    scalar_alpha: float = 0.01,
) -> list[tuple[float, float]]:
    """Return ``(penalty_discount, alpha)`` cells for one algorithm.

    * **FGES** varies ``penalty_discount`` (alpha is recorded, unused by search).
    * **GFCI** varies ``alpha`` (penalty stays at the scalar default).
    * Disabled sweep returns the single scalar pair.
    """
    penalty = float(scalar_penalty)
    alpha = float(scalar_alpha)
    if not enabled:
        return [(penalty, alpha)]

    penalties = unique_floats(
        penalty_discounts if penalty_discounts is not None else [penalty]
    )
    alpha_list = unique_floats(alphas if alphas is not None else [alpha])
    if not penalties:
        penalties = [penalty]
    if not alpha_list:
        alpha_list = [alpha]

    algo = str(algorithm).lower()
    if algo == "gfci":
        return [(penalty, a) for a in alpha_list]
    return [(p, alpha) for p in penalties]


def hp_group_keys(
    frame: pd.DataFrame,
    *,
    include_hp: bool = True,
    drop_constant_missingness: bool = True,
) -> list[str]:
    """Context columns present in ``frame``, plus HP columns when requested.

    Missingness keys are omitted when the grid is complete-data only so SA4
    tables stay ``algorithm × HP × proportion``.
    """
    keys = [c for c in ("algorithm", "proportion") if c in frame.columns]
    miss = ("missing_mechanism", "missing_rate", "missing_strategy")
    if all(c in frame.columns for c in miss):
        vary = frame["missing_rate"].nunique(dropna=False) > 1 or not (
            frame["missing_mechanism"].fillna("none") == "none"
        ).all()
        if vary or not drop_constant_missingness:
            keys.extend(miss)
    if include_hp:
        keys.extend(c for c in HP_COLS if c in frame.columns)
    return keys


def summarize_hp_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery metrics per algorithm × HP × (proportion / missingness).

    ``search_ok`` failures are kept in ``n_draws`` / ``success_rate`` but
    excluded from FPR / Dice means.
    """
    if summary.empty:
        return summary.copy()

    keys = hp_group_keys(summary, include_hp=True)
    if not keys:
        return pd.DataFrame()

    work = summary.copy()
    if "search_ok" not in work.columns:
        work["search_ok"] = True

    n_draws = (
        work.groupby(keys, dropna=False)
        .size()
        .rename("n_draws")
        .reset_index()
    )
    success = (
        work.groupby(keys, dropna=False)["search_ok"]
        .mean()
        .rename("success_rate")
        .reset_index()
    )
    ok = work[work["search_ok"] != False]
    metric_cols = [c for c in HP_METRIC_COLS if c in ok.columns]
    if ok.empty or not metric_cols:
        merged = n_draws.merge(success, on=keys, how="left")
        for col in metric_cols:
            merged[col] = pd.NA
        return merged

    means = ok.groupby(keys, dropna=False)[metric_cols].mean().reset_index()
    return n_draws.merge(success, on=keys, how="left").merge(
        means, on=keys, how="left"
    )


def select_hp_for_target_fprs(
    hp_metrics: pd.DataFrame,
    target_fprs: Iterable[float],
    *,
    fpr_col: str = "oriented_fpr",
) -> pd.DataFrame:
    """Pick the HP whose mean FPR is closest to each target.

    Tie-break: higher TPR, then higher directed Dice, then the more
    conservative setting (larger FGES penalty, smaller GFCI alpha).
    """
    targets = unique_floats(target_fprs)
    if hp_metrics.empty or not targets or fpr_col not in hp_metrics.columns:
        return pd.DataFrame()

    context = hp_group_keys(hp_metrics, include_hp=False)
    rows: list[dict[str, Any]] = []
    usable = hp_metrics.dropna(subset=[fpr_col])
    if usable.empty:
        return pd.DataFrame()

    if context:
        groups = usable.groupby(context, dropna=False)
    else:
        groups = [(tuple(), usable)]

    tpr_col = "oriented_tpr" if "oriented_tpr" in usable.columns else None
    dice_col = "dice_directed" if "dice_directed" in usable.columns else None

    for ctx_key, grp in groups:
        if not isinstance(ctx_key, tuple):
            ctx_key = (ctx_key,)
        context_map = dict(zip(context, ctx_key))
        for target in targets:
            ranked = grp.copy()
            ranked["_abs_diff"] = (ranked[fpr_col] - target).abs()
            ranked["_neg_tpr"] = (
                -ranked[tpr_col] if tpr_col else 0.0
            )
            ranked["_neg_dice"] = (
                -ranked[dice_col] if dice_col else 0.0
            )
            ranked["_neg_penalty"] = (
                -ranked["penalty_discount"]
                if "penalty_discount" in ranked.columns
                else 0.0
            )
            ranked["_alpha"] = (
                ranked["alpha"] if "alpha" in ranked.columns else 0.0
            )
            ranked = ranked.sort_values(
                by=[
                    "_abs_diff",
                    "_neg_tpr",
                    "_neg_dice",
                    "_neg_penalty",
                    "_alpha",
                ],
                kind="mergesort",
            )
            best = ranked.iloc[0]
            row = {
                **context_map,
                "target_fpr": target,
                "selected_penalty_discount": best.get("penalty_discount"),
                "selected_alpha": best.get("alpha"),
                "empirical_fpr": best[fpr_col],
                "abs_fpr_error": float(best["_abs_diff"]),
            }
            if tpr_col:
                row["empirical_tpr"] = best[tpr_col]
            if dice_col:
                row["dice_directed"] = best[dice_col]
            if "dice_skeleton" in best.index:
                row["dice_skeleton"] = best["dice_skeleton"]
            if "n_draws" in best.index:
                row["n_draws"] = best["n_draws"]
            if "success_rate" in best.index:
                row["success_rate"] = best["success_rate"]
            rows.append(row)

    return pd.DataFrame(rows)


def parse_target_fprs(raw: Any, default: Iterable[float] | None = None) -> list[float]:
    return unique_floats(parse_float_list(raw, default if default is not None else []))
