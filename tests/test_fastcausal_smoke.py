"""Optional integration smoke: skip unless fastcausal is installed."""

import pytest

pytest.importorskip("fastcausal")
pytest.importorskip("tetrad_port")

from tradsim_fastcausal import TradSimFastcausal


def test_smoke_fges(tmp_path):
    runner = TradSimFastcausal(
        config="config.yaml",
        output_dir=str(tmp_path),
        iterations=1,
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
    assert summary["dice_skeleton"].between(0.0, 1.0).all()
