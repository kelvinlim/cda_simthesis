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
    assert summary["dice_skeleton"].between(0.0, 1.0).all()
    assert len(summary[summary["proportion"] == 1.0]) == 1
    assert len(summary[summary["proportion"] == 0.8]) == 3
