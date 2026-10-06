"""Optional integration smoke: skip unless fastcausal is installed."""

import pandas as pd
import pytest

pytest.importorskip("fastcausal")
pytest.importorskip("tetrad_port")

from tradsim_fastcausal import TradSimFastcausal


def test_smoke_fges(tmp_path):
    runner = TradSimFastcausal(
        config="config.yaml",
        output_dir=str(tmp_path),
        iterations=3,
        proportions="1.0,0.8",
        algorithm="fges",
        run_sem=False,
        verbose=1,
        seed=7,
    )
    summary = runner.smoke()
    assert not summary.empty
    assert set(summary["proportion"]) <= {1.0, 0.8}
    assert (summary["algorithm"] == "fges").all()
    assert (summary["n_true_edges"] > 0).all()
    assert summary["es"].iloc[0] == 0.25
    ok = summary[summary["search_ok"] != False]
    assert not ok.empty
    assert ok["dice_skeleton"].between(0.0, 1.0).all()
    assert set(ok["missing_mechanism"]) == {"mcar", "mar"}
    assert set(ok["missing_strategy"]) == {"complete_case", "mean"}
    assert set(ok["missing_rate"]) == {0.2}
    assert (ok["n_missing_rows"] > 0).all()
    assert set(ok.loc[ok["missing_mechanism"] == "mcar", "missing_strategy"]) == {
        "complete_case"
    }
    mar = ok[ok["missing_mechanism"] == "mar"]
    assert set(mar["missing_strategy"]) == {"complete_case", "mean"}
    for _, grp in mar.groupby(["proportion", "iteration"]):
        assert grp["n_missing_rows"].nunique() == 1
    assert {"oriented_tp", "oriented_fp", "oriented_fn", "oriented_fpr"}.issubset(
        ok.columns
    )
    assert {"penalty_discount", "alpha"}.issubset(ok.columns)
    # MCAR × complete_case + MAR × {complete_case, mean}; p=1.0 once, p=0.8 × 3.
    assert len(summary[summary["proportion"] == 1.0]) == 3
    assert len(summary[summary["proportion"] == 0.8]) == 9


def test_smoke_fges_subsample_only(tmp_path):
    runner = TradSimFastcausal(
        config="config.yaml",
        output_dir=str(tmp_path),
        iterations=3,
        proportions="1.0,0.8",
        algorithm="fges",
        run_sem=False,
        missingness=False,
        verbose=1,
        seed=7,
    )
    summary = runner.smoke()
    assert set(summary["missing_mechanism"]) == {"none"}
    assert set(summary["missing_rate"]) == {0.0}
    assert len(summary[summary["proportion"] == 1.0]) == 1
    assert len(summary[summary["proportion"] == 0.8]) == 3


def test_smoke_fges_hp_sweep(tmp_path):
    runner = TradSimFastcausal(
        config="config.yaml",
        output_dir=str(tmp_path),
        iterations=2,
        proportions="1.0,0.8",
        algorithm="fges",
        run_sem=False,
        hp_sweep=True,
        penalty_discounts="1.0,2.0",
        target_fprs="0.05,0.10",
        verbose=1,
        seed=7,
    )
    summary = runner.smoke()
    assert set(summary["missing_mechanism"]) == {"none"}
    assert set(summary["penalty_discount"]) == {1.0, 2.0}
    ok = summary[summary["search_ok"] != False]
    assert not ok.empty
    assert ok["oriented_fpr"].between(0.0, 1.0).all()
    # two penalties; p=1.0 once, p=0.8 × 2
    assert len(summary[summary["proportion"] == 1.0]) == 2
    assert len(summary[summary["proportion"] == 0.8]) == 4
    metrics_path = tmp_path / "smoke" / "hp_sweep_metrics.csv"
    selected_path = tmp_path / "smoke" / "hp_selected_by_target_fpr.csv"
    assert metrics_path.exists()
    assert selected_path.exists()
    selected = pd.read_csv(selected_path)
    assert set(selected["target_fpr"]) == {0.05, 0.10}
    assert set(selected["selected_penalty_discount"]).issubset({1.0, 2.0})
