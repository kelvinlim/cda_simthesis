"""Runner wiring: missingness grid + CLI flags (no discovery search)."""

from tradsim_fastcausal import TradSimFastcausal, build_parser


def test_disabled_grid_is_single_none_cell():
    runner = TradSimFastcausal(config="config.yaml", missingness=False)
    assert runner._missingness_cells() == [("none", 0.0, "none")]
    assert runner._missingness_enabled() is False


def test_enabled_uses_config_yaml_grid():
    runner = TradSimFastcausal(config="config.yaml", missingness=True)
    cells = runner._missingness_cells()
    assert ("mcar", 0.0, "none") in cells
    assert ("mar", 0.0, "none") in cells
    assert ("mcar", 0.1, "complete_case") in cells
    assert ("mar", 0.4, "mean") in cells
    assert ("mcar", 0.0, "mean") not in cells


def test_cli_missing_rates_enables_grid():
    runner = TradSimFastcausal(
        config="config.yaml",
        missing_mechanisms="mcar",
        missing_rates="0.2",
        missing_strategies="complete_case",
    )
    assert runner._missingness_enabled() is True
    assert runner._missingness_cells() == [("mcar", 0.2, "complete_case")]


def test_parser_exposes_implemented_missingness_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--cmd",
            "compute",
            "--proportions",
            "1.0",
            "--missingness",
            "--missing-mechanisms",
            "mcar,mar",
            "--missing-rates",
            "0,0.2",
            "--missing-strategies",
            "complete_case,mean",
            "--mar-covariates",
            "x_1",
        ]
    )
    assert args.missingness is True
    assert args.missing_mechanisms == "mcar,mar"
    assert args.missing_rates == "0,0.2"
    assert args.missing_strategies == "complete_case,mean"
    assert args.mar_covariates == "x_1"

    off = parser.parse_args(["--no-missingness"])
    assert off.missingness is False
