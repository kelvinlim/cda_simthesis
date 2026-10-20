"""Thin loader for fetched Boston Set A (10-variable) EMA fixtures.

Discovery on *simulations* still goes through ``tradsim_fastcausal.py`` and
picause ``.txt`` ground truth. Real Boston CSVs have no companion DAG, so this
module does not run search. A later real-data runner can call ``load_set_a_frame``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SET_A = [
    "TIB",
    "TST",
    "sleepdiary_sleeplatency",
    "sleepdiary_wakes",
    "night_awakening_time",
    "sleepdiary_naptime",
    "socialize_min",
    "people_contact",
    "alcohol_bev",
    "stress",
]

DEFAULT_DEST = Path("fixtures/causal-open-data")
PRIMARY_SET_A = "datasets/boston/boston_sub132_allsurveys_10v.csv"
N_SCALING_DIR = "datasets/boston/individual"


def fixture_root(dest: Path | str | None = None) -> Path:
    return Path(dest) if dest is not None else DEFAULT_DEST


def set_a_csv_paths(dest: Path | str | None = None) -> list[Path]:
    """Primary + alternate + n-scaling Set A CSVs if they have been fetched."""
    root = fixture_root(dest)
    boston = root / "datasets" / "boston"
    paths: list[Path] = []
    for name in (
        "boston_sub132_allsurveys_10v.csv",
        "boston_sub79_allsurveys_10v.csv",
    ):
        candidate = boston / name
        if candidate.is_file():
            paths.append(candidate)
    ladder = boston / "individual"
    if ladder.is_dir():
        paths.extend(sorted(ladder.glob("boston_sub*_n*_10v.csv")))
    return paths


def load_set_a_frame(csv_path: Path | str) -> pd.DataFrame:
    """Load one Set A CSV and check the 10 expected columns."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    missing = [col for col in SET_A if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing Set A columns: {missing}")
    extra_err = [
        col
        for col in ("PANAS_PA", "PANAS_NA", "worry_scale", "PHQ9")
        if col in frame.columns
    ]
    if extra_err:
        raise ValueError(
            f"{path} looks like Full-only classic-7/mood columns, not Set A: {extra_err}"
        )
    return frame.loc[:, SET_A]


def describe_fetched_set_a(dest: Path | str | None = None) -> pd.DataFrame:
    """Shape table for fetched Set A files (empty if fetch has not been run)."""
    rows = []
    for path in set_a_csv_paths(dest):
        frame = load_set_a_frame(path)
        rows.append(
            {
                "path": str(path),
                "n": int(len(frame)),
                "p": int(frame.shape[1]),
                "columns": ",".join(frame.columns),
            }
        )
    return pd.DataFrame(rows)
