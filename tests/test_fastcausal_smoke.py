"""Optional integration smoke: skip unless fastcausal is installed."""

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
    assert {"oriented_tp", "oriented_fp", "oriented_fn"}.issubset(ok.columns)
    # 2 mechanisms × 1 rate × 2 strategies; p=1.0 once, p=0.8 three times.
    assert len(summary[summary["proportion"] == 1.0]) == 4
    assert len(summary[summary["proportion"] == 0.8]) == 12


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
