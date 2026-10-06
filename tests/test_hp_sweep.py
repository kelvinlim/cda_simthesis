"""Unit tests for SA4 HP grid expansion and target-FPR selection."""

import pandas as pd
import pytest

from tools.hp_sweep import (
    expand_hp_cells,
    select_hp_for_target_fprs,
    summarize_hp_metrics,
    unique_floats,
)


def test_expand_disabled_is_scalar_pair():
    assert expand_hp_cells(
        "fges",
        [1.0, 2.0],
        [0.01, 0.05],
        enabled=False,
        scalar_penalty=1.5,
        scalar_alpha=0.02,
    ) == [(1.5, 0.02)]


def test_expand_fges_varies_penalty_only():
    cells = expand_hp_cells(
        "fges",
        [1.0, 2.0, 2.0, 4.0],
        [0.01, 0.05],
        enabled=True,
        scalar_penalty=1.0,
        scalar_alpha=0.01,
    )
    assert cells == [(1.0, 0.01), (2.0, 0.01), (4.0, 0.01)]


def test_expand_gfci_varies_alpha_only():
    cells = expand_hp_cells(
        "gfci",
        [1.0, 4.0],
        [0.01, 0.05],
        enabled=True,
        scalar_penalty=2.0,
        scalar_alpha=0.01,
    )
    assert cells == [(2.0, 0.01), (2.0, 0.05)]


def test_select_hp_closest_to_each_target():
    metrics = pd.DataFrame(
        {
            "algorithm": ["fges", "fges", "fges"],
            "proportion": [1.0, 1.0, 1.0],
            "penalty_discount": [1.0, 2.0, 4.0],
            "alpha": [0.01, 0.01, 0.01],
            "oriented_fpr": [0.12, 0.06, 0.01],
            "oriented_tpr": [0.9, 0.8, 0.5],
            "dice_directed": [0.7, 0.65, 0.4],
            "dice_skeleton": [0.8, 0.75, 0.5],
            "n_draws": [10, 10, 10],
            "success_rate": [1.0, 1.0, 1.0],
        }
    )
    selected = select_hp_for_target_fprs(metrics, [0.05, 0.10])
    by_target = selected.set_index("target_fpr")
    assert by_target.loc[0.05, "selected_penalty_discount"] == 2.0
    assert by_target.loc[0.10, "selected_penalty_discount"] == 1.0
    assert by_target.loc[0.05, "empirical_fpr"] == 0.06
    assert by_target.loc[0.10, "empirical_fpr"] == 0.12


def test_select_hp_tie_breaks_to_higher_tpr_then_conservative_penalty():
    metrics = pd.DataFrame(
        {
            "algorithm": ["fges", "fges"],
            "proportion": [1.0, 1.0],
            "penalty_discount": [1.0, 4.0],
            "alpha": [0.01, 0.01],
            "oriented_fpr": [0.05, 0.05],
            "oriented_tpr": [0.7, 0.7],
            "dice_directed": [0.5, 0.5],
        }
    )
    selected = select_hp_for_target_fprs(metrics, [0.05])
    assert len(selected) == 1
    assert selected.iloc[0]["selected_penalty_discount"] == 4.0


def test_summarize_hp_metrics_excludes_failed_search_from_fpr_mean():
    summary = pd.DataFrame(
        {
            "algorithm": ["fges", "fges", "fges"],
            "proportion": [1.0, 1.0, 1.0],
            "penalty_discount": [1.0, 1.0, 1.0],
            "alpha": [0.01, 0.01, 0.01],
            "search_ok": [True, True, False],
            "oriented_fpr": [0.10, 0.20, 0.99],
            "oriented_tpr": [0.5, 0.7, 0.0],
            "dice_skeleton": [0.4, 0.6, 0.0],
            "dice_directed": [0.3, 0.5, 0.0],
            "oriented_tp": [1, 2, 0],
            "oriented_fp": [1, 2, 9],
            "oriented_fn": [1, 0, 3],
        }
    )
    metrics = summarize_hp_metrics(summary)
    assert len(metrics) == 1
    assert metrics.iloc[0]["n_draws"] == 3
    assert metrics.iloc[0]["success_rate"] == 2 / 3
    assert metrics.iloc[0]["oriented_fpr"] == pytest.approx(0.15)


def test_unique_floats_preserves_order():
    assert unique_floats([1.0, 2.0, 1.0, 4.0]) == [1.0, 2.0, 4.0]
