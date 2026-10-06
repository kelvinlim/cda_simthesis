"""Unit tests for row MCAR / MAR generators and cheap handlers."""

import numpy as np
import pandas as pd
import pytest

from tools.missingness import (
    apply_missingness,
    apply_row_mar,
    apply_row_mcar,
    expand_missingness_grid,
    group_missingness_draws,
    handle_missing,
    parse_float_list,
    parse_str_list,
)


def _frame(n=400, cols=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, cols))
    # Shift first column so MAR has a clear high tail.
    data[:, 0] += np.linspace(-2.0, 2.0, n)
    return pd.DataFrame(data, columns=[f"x_{i + 1}" for i in range(cols)])


def test_parse_lists():
    assert parse_str_list("mcar, mar") == ["mcar", "mar"]
    assert parse_float_list("0,0.1,0.2") == [0.0, 0.1, 0.2]
    assert parse_str_list(None, ["none"]) == ["none"]


def test_expand_grid_disabled_and_rate_zero():
    assert expand_missingness_grid(None, None, None, enabled=False) == [
        ("none", 0.0, "none")
    ]
    cells = expand_missingness_grid(
        ["mcar", "mar"],
        [0.0, 0.2],
        ["complete_case", "mean"],
        enabled=True,
    )
    assert cells.count(("none", 0.0, "none")) == 1
    assert ("mcar", 0.0, "none") not in cells
    assert ("mar", 0.0, "none") not in cells
    assert ("mcar", 0.2, "complete_case") in cells
    assert ("mar", 0.2, "mean") in cells
    # Default MCAR path drops mean/median (whole-row blanks → mean-vector rows).
    assert ("mcar", 0.2, "mean") not in cells
    assert ("mcar", 0.0, "complete_case") not in cells


def test_expand_grid_mcar_impute_opt_in():
    cells = expand_missingness_grid(
        ["mcar"], [0.2], ["complete_case", "mean"], mcar_impute=True
    )
    assert ("mcar", 0.2, "mean") in cells
    assert ("mcar", 0.2, "complete_case") in cells


def test_expand_grid_mcar_mean_only_requires_opt_in():
    with pytest.raises(ValueError, match="complete_case"):
        expand_missingness_grid(["mcar"], [0.2], ["mean"], mcar_impute=False)


def test_group_missingness_draws_shares_strategies():
    grouped = group_missingness_draws(
        [
            ("mcar", 0.2, "complete_case"),
            ("mar", 0.2, "complete_case"),
            ("mar", 0.2, "mean"),
        ]
    )
    assert grouped == [
        ("mcar", 0.2, ["complete_case"]),
        ("mar", 0.2, ["complete_case", "mean"]),
    ]


def test_expand_grid_rejects_mnar_and_mice():
    with pytest.raises(NotImplementedError, match="MNAR"):
        expand_missingness_grid(["mnar"], [0.2], ["complete_case"])
    with pytest.raises(NotImplementedError, match="Multiple imputation"):
        expand_missingness_grid(["mcar"], [0.2], ["mice"])


def test_row_mcar_blanks_entire_rows_near_target_rate():
    df = _frame(n=2000)
    rng = np.random.default_rng(11)
    out, info = apply_row_mcar(df, 0.25, rng)
    assert info["n_missing_rows"] == int(info["row_mask"].sum())
    assert abs(info["missing_rate_empirical"] - 0.25) < 0.04
    mask = info["row_mask"]
    assert out.loc[mask].isna().all(axis=None)
    assert not out.loc[~mask].isna().any(axis=None)
    # MCAR: missing-row covariate mean stays close to the overall mean.
    col = df["x_1"]
    assert abs(col[mask].mean() - col.mean()) < 0.35


def test_row_mar_depends_on_observed_covariate():
    df = _frame(n=3000)
    rng = np.random.default_rng(3)
    out, info = apply_row_mar(df, 0.30, rng, covariates=["x_1"], slope=2.0)
    assert info["observed_columns"] == ["x_1"]
    assert "x_1" not in info["incomplete_columns"]
    assert abs(info["missing_rate_empirical"] - 0.30) < 0.05
    assert out["x_1"].isna().sum() == 0
    mask = info["row_mask"]
    assert out.loc[mask, info["incomplete_columns"]].isna().all(axis=None)
    assert not out.loc[~mask].isna().any(axis=None)
    # Positive slope: missing rows have larger x_1 than kept rows.
    assert df.loc[mask, "x_1"].mean() > df.loc[~mask, "x_1"].mean() + 0.4


def test_row_mar_rejects_all_columns_as_covariates():
    df = _frame(n=20, cols=3)
    with pytest.raises(ValueError, match="non-covariate"):
        apply_row_mar(df, 0.2, rng=0, covariates=list(df.columns))


def test_apply_missingness_rate_zero_is_noop():
    df = _frame(n=40)
    out, info = apply_missingness(df, "mcar", 0.0, rng=0)
    assert info["n_missing_rows"] == 0
    pd.testing.assert_frame_equal(out, df)


def test_handle_complete_case_and_mean():
    df = _frame(n=20, seed=5)
    corrupted, info = apply_row_mcar(df, 0.4, rng=1)
    dropped, drop_info = handle_missing(corrupted, "listwise")
    assert drop_info["strategy"] == "complete_case"
    assert len(dropped) == len(df) - info["n_missing_rows"]
    assert not dropped.isna().any().any()

    filled, fill_info = handle_missing(corrupted, "mean")
    assert fill_info["n_imputed_cells"] > 0
    assert len(filled) == len(df)
    assert not filled.isna().any().any()
    observed_mean = corrupted["x_2"].mean()
    missing_idx = corrupted.index[info["row_mask"]]
    assert filled.loc[missing_idx, "x_2"].nunique() == 1
    assert filled.loc[missing_idx, "x_2"].iloc[0] == pytest.approx(observed_mean)


def test_handle_mice_is_todo():
    df = _frame(n=10)
    with pytest.raises(NotImplementedError, match="Multiple imputation"):
        handle_missing(df, "mice")
