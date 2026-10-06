"""Runner wiring: SA4 HP sweep + CLI flags (no discovery search)."""

from tradsim_fastcausal import TradSimFastcausal, build_parser


def test_disabled_hp_sweep_is_scalar_cell():
    runner = TradSimFastcausal(config="config.yaml", hp_sweep=False)
    assert runner._hp_sweep_enabled() is False
    assert runner._hp_cells("fges") == [(1.0, 0.01)]
    assert runner._hp_cells("gfci") == [(1.0, 0.01)]


def test_enabled_uses_config_yaml_grid():
    runner = TradSimFastcausal(config="config.yaml", hp_sweep=True)
    assert runner._hp_cells("fges") == [(1.0, 0.01), (2.0, 0.01), (4.0, 0.01)]
    assert runner._hp_cells("gfci") == [(1.0, 0.01), (1.0, 0.05)]
    assert runner._target_fprs() == [0.05, 0.10]


def test_cli_lists_enable_sweep():
    runner = TradSimFastcausal(
        config="config.yaml",
        penalty_discounts="1,2",
        target_fprs="0.05",
    )
    assert runner._hp_sweep_enabled() is True
    assert runner._hp_cells("fges") == [(1.0, 0.01), (2.0, 0.01)]
    assert runner._target_fprs() == [0.05]


def test_gfci_cli_alphas_keep_scalar_penalty():
    runner = TradSimFastcausal(
        config="config.yaml",
        algorithm="gfci",
        penalty_discount=2.0,
        alphas="0.01,0.05",
    )
    assert runner._hp_cells("gfci") == [(2.0, 0.01), (2.0, 0.05)]
    assert runner._hp_cells("fges") == [(1.0, 0.01), (2.0, 0.01), (4.0, 0.01)]


def test_parser_exposes_implemented_hp_sweep_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--cmd",
            "compute",
            "--proportions",
            "1.0",
            "--hp-sweep",
            "--penalty-discounts",
            "1,2,4",
            "--alphas",
            "0.01,0.05",
            "--target-fprs",
            "0.05,0.10",
        ]
    )
    assert args.hp_sweep is True
    assert args.penalty_discounts == "1,2,4"
    assert args.alphas == "0.01,0.05"
    assert args.target_fprs == "0.05,0.10"

    off = parser.parse_args(["--no-hp-sweep"])
    assert off.hp_sweep is False
