#! /usr/bin/env python3
"""Discovery runner: FGES / GFCI via fastcausal on simdata CSVs.

Compares recovered graphs to picause ground-truth ``.txt`` files and to the
100%-sample recovered graph (legacy Dice), across subsample fractions
100% → 40% and, optionally, a row-missingness grid (MCAR then MAR).

Subsample ``proportion`` and missingness ``rate`` are different axes:
proportion randomly *keeps* complete rows; MCAR/MAR *blanks* rows that a
handler then drops or imputes.

This replaces ``tradsim_fges_obj.py`` for the simulation workflow. The older
Java Tetrad / fastcda path is deprecated and is not required here.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.fastcausal_backend import run_search
from tools.graph_metrics import compare_graphs, parse_picause_graph, skeleton_pairs
from tools.missingness import (
    apply_missingness,
    expand_missingness_grid,
    handle_missing,
    parse_float_list,
    parse_str_list,
)

__version_info__ = ("0", "3", "0")
__version__ = ".".join(__version_info__)

DEFAULT_PROPORTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
DEFAULT_ITERATIONS = 20
FULL_SAMPLE_ITERATIONS = 1


def iterations_for_proportion(
    proportion: float,
    iterations: int,
    full_sample_iters: int = FULL_SAMPLE_ITERATIONS,
) -> int:
    """Full-sample baseline is deterministic; only subsample fractions repeat."""
    if float(proportion) >= 1.0:
        return max(1, int(full_sample_iters))
    return max(1, int(iterations))
SUB_RE = re.compile(r"(?:^|_)sub-(?P<sub>\d+)", re.IGNORECASE)
ITER_RE = re.compile(r"(?:^|_)iter-(?P<iter>\d+)", re.IGNORECASE)
ES_RE = re.compile(r"(?:^|_)es-(?P<es>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_case_meta(csv_path: Path) -> dict:
    stem = csv_path.stem
    sub_match = SUB_RE.search(stem)
    iter_match = ITER_RE.search(stem)
    es_match = ES_RE.search(stem)
    sub = None
    if sub_match:
        sub = sub_match.group("sub")
    elif iter_match:
        sub = iter_match.group("iter")
    es = es_match.group("es") if es_match else None
    return {
        "case": stem,
        "subject": f"sub-{int(sub):03d}" if sub is not None else stem,
        "es": float(es) if es is not None else None,
    }


def companion_txt(csv_path: Path) -> Path:
    return csv_path.with_suffix(".txt")


def discover_sim_csvs(data_dir: Path, pattern: str = "*.csv") -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    result = numeric.copy()
    for col in result.columns:
        std = result[col].std()
        if std and std > 0:
            result[col] = (result[col] - result[col].mean()) / std
        else:
            result[col] = 0.0
    return result


def mean_abs_estimates(sem_summary: dict | None) -> tuple[float | None, float | None]:
    if not sem_summary:
        return None, None
    return sem_summary.get("mean_abs_estimates"), sem_summary.get("std_abs_estimates")


class TradSimFastcausal:
    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        yaml_cfg = {}
        if self.config.get("config"):
            yaml_cfg = load_yaml_config(self.config["config"])
        self.simulation = yaml_cfg.get("simulation", {})
        self.discovery = yaml_cfg.get("discovery", {})
        self.verbose = int(self.config.get("verbose", 1))

    def _algorithm_list(self) -> list[str]:
        requested = self.config.get("algorithm") or self.discovery.get(
            "algorithm", "fges"
        )
        requested = str(requested).lower()
        if requested in ("both", "all"):
            return ["fges", "gfci"]
        return [requested]

    def _proportions(self) -> list[float]:
        raw = self.config.get("proportions")
        if raw:
            return [float(x) for x in str(raw).split(",") if str(x).strip()]
        return list(self.discovery.get("proportions", DEFAULT_PROPORTIONS))

    def _iterations(self) -> int:
        if self.config.get("iterations") is not None:
            return int(self.config["iterations"])
        return int(self.discovery.get("iterations", DEFAULT_ITERATIONS))

    def _data_dir(self) -> Path:
        if self.config.get("data_dir"):
            return Path(self.config["data_dir"])
        return Path(self.simulation.get("data_directory", "./sim_data_v2"))

    def _output_dir(self) -> Path:
        path = Path(
            self.config.get("output_dir")
            or self.discovery.get("output_directory", "./discovery_results")
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_sem(self) -> bool:
        if self.config.get("run_sem") is not None:
            return bool(self.config["run_sem"])
        return bool(self.discovery.get("run_sem", False))

    def _alpha(self) -> float:
        return float(
            self.config.get("alpha")
            or self.discovery.get("alpha", 0.01)
        )

    def _penalty(self) -> float:
        return float(
            self.config.get("penalty_discount")
            or self.discovery.get("penalty_discount", 1.0)
        )

    def _seed(self) -> int:
        return int(self.config.get("seed") or self.discovery.get("seed", 2025))

    def _missingness_yaml(self) -> dict:
        raw = self.discovery.get("missingness") or {}
        return raw if isinstance(raw, dict) else {}

    def _missingness_enabled(self) -> bool:
        if self.config.get("missingness") is not None:
            return bool(self.config["missingness"])
        if any(
            self.config.get(key) is not None
            for key in (
                "missing_mechanisms",
                "missing_rates",
                "missing_strategies",
            )
        ):
            return True
        return bool(self._missingness_yaml().get("enabled", False))

    def _missingness_cells(self) -> list[tuple[str, float, str]]:
        yaml_m = self._missingness_yaml()
        enabled = self._missingness_enabled()
        mechanisms = parse_str_list(
            self.config.get("missing_mechanisms"),
            yaml_m.get("mechanisms") if enabled else ["none"],
        )
        rates = parse_float_list(
            self.config.get("missing_rates"),
            yaml_m.get("rates") if enabled else [0.0],
        )
        strategies = parse_str_list(
            self.config.get("missing_strategies"),
            yaml_m.get("strategies") if enabled else ["none"],
        )
        return expand_missingness_grid(
            mechanisms, rates, strategies, enabled=enabled
        )

    def _mar_covariates(self):
        if self.config.get("mar_covariates") is not None:
            return self.config["mar_covariates"]
        return self._missingness_yaml().get("mar_covariates")

    def _mar_slope(self) -> float:
        yaml_m = self._missingness_yaml()
        if self.config.get("mar_slope") is not None:
            return float(self.config["mar_slope"])
        return float(yaml_m.get("mar_slope", 1.5))

    def compute(self) -> pd.DataFrame:
        data_dir = self._data_dir()
        csvs = discover_sim_csvs(data_dir, self.config.get("glob", "*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No simulation CSVs found in {data_dir}. "
                "Generate them with: python simdata.py --cmd sim"
            )

        rows = []
        for csv_path in csvs:
            txt_path = companion_txt(csv_path)
            if not txt_path.exists():
                if self.verbose:
                    print(f"Skipping {csv_path.name}: missing ground-truth {txt_path.name}")
                continue
            true_edges = parse_picause_graph(txt_path)
            df = pd.read_csv(csv_path)
            df = df.astype({col: "float64" for col in df.columns})
            meta = parse_case_meta(csv_path)
            if self.verbose:
                print(
                    f"Case {meta['case']}: {len(df)} rows, "
                    f"{len(true_edges)} true edges, algorithms={self._algorithm_list()}"
                )
            for algorithm in self._algorithm_list():
                rows.extend(
                    self._run_proportions(
                        df=df,
                        true_edges=true_edges,
                        meta=meta,
                        algorithm=algorithm,
                    )
                )

        summary = pd.DataFrame(rows)
        out = self._output_dir() / "resampled_models_fastcausal.csv"
        summary.to_csv(out, index=False)
        if self.verbose:
            print(f"Wrote {out} ({len(summary)} rows)")
            self._print_search_success(summary)
            self._print_missingness_metrics(summary)
        return summary

    def _print_search_success(self, summary: pd.DataFrame) -> None:
        """Report search_ok mean by algorithm × proportion × es (GFCI failures)."""
        if summary.empty or "search_ok" not in summary.columns:
            return
        keys = [
            c
            for c in (
                "algorithm",
                "proportion",
                "es",
                "missing_mechanism",
                "missing_rate",
                "missing_strategy",
            )
            if c in summary.columns
        ]
        rates = (
            summary.groupby(keys, dropna=False)["search_ok"]
            .mean()
            .rename("success_rate")
            .reset_index()
        )
        print(
            "Search success rate by cell "
            "(algorithm × proportion × es × missingness):"
        )
        print(rates.to_string(index=False))

    def _print_missingness_metrics(self, summary: pd.DataFrame) -> None:
        """Mean Dice / oriented counts by mechanism × rate × strategy."""
        needed = {
            "missing_mechanism",
            "missing_rate",
            "missing_strategy",
            "dice_skeleton",
        }
        if summary.empty or not needed.issubset(summary.columns):
            return
        if (
            summary["missing_rate"].nunique() <= 1
            and (summary["missing_mechanism"] == "none").all()
        ):
            return
        ok = summary
        if "search_ok" in summary.columns:
            ok = summary[summary["search_ok"] != False]
        if ok.empty:
            return
        keys = ["missing_mechanism", "missing_rate", "missing_strategy"]
        if "algorithm" in ok.columns and ok["algorithm"].nunique() > 1:
            keys = ["algorithm", *keys]
        cols = [
            c
            for c in (
                "dice_skeleton",
                "dice_directed",
                "oriented_tp",
                "oriented_fp",
                "oriented_fn",
            )
            if c in ok.columns
        ]
        table = ok.groupby(keys, dropna=False)[cols].mean().reset_index()
        print("Discovery vs simulated truth by missingness cell:")
        print(table.to_string(index=False))

    def _run_proportions(
        self,
        df: pd.DataFrame,
        true_edges: list[str],
        meta: dict,
        algorithm: str,
    ) -> list[dict]:
        rng = np.random.default_rng(self._seed())
        rows: list[dict] = []
        full_edges: list[str] | None = None
        knowledge = bool(self.config.get("knowledge", False))
        run_sem = self._run_sem()
        missing_cells = self._missingness_cells()

        for proportion in self._proportions():
            n_iter = iterations_for_proportion(proportion, self._iterations())
            for iteration in range(n_iter):
                sampled = df.sample(
                    frac=proportion,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                for mechanism, miss_rate, strategy in missing_cells:
                    if self.verbose > 1:
                        print(
                            f"  {meta['case']} {algorithm} p={proportion} "
                            f"iter={iteration + 1}/{n_iter} "
                            f"{mechanism}@{miss_rate} {strategy}"
                        )
                    base = {
                        "case": meta["case"],
                        "subject": meta["subject"],
                        "es": meta["es"],
                        "algorithm": algorithm,
                        "proportion": proportion,
                        "iteration": iteration,
                        "n_rows_sampled": len(sampled),
                        "n_true_edges": len(skeleton_pairs(true_edges)),
                        "missing_mechanism": mechanism,
                        "missing_rate": miss_rate,
                        "missing_strategy": strategy,
                    }
                    row = self._evaluate_draw(
                        sampled=sampled,
                        algorithm=algorithm,
                        proportion=proportion,
                        mechanism=mechanism,
                        miss_rate=miss_rate,
                        strategy=strategy,
                        knowledge=knowledge,
                        run_sem=run_sem,
                        rng=rng,
                    )
                    recovered = row.pop("_recovered_edges", None)
                    if (
                        recovered is not None
                        and proportion == 1.0
                        and miss_rate == 0.0
                        and full_edges is None
                    ):
                        full_edges = list(recovered)
                    if recovered is not None and row.get("search_ok"):
                        metrics = compare_graphs(
                            true_edges,
                            recovered,
                            full_sample_edges=full_edges,
                        )
                        row.update(
                            {
                                "n_recovered_edges": metrics["n_recovered_edges"],
                                "n_recovered_directed": metrics[
                                    "n_recovered_directed"
                                ],
                                "diceCoeff": metrics["dice_skeleton"],
                                "dice_skeleton": metrics["dice_skeleton"],
                                "dice_directed": metrics["dice_directed"],
                                "diceCoeffNodes": metrics["dice_nodes"],
                                "dice_vs_full": metrics["dice_vs_full_skeleton"],
                                "oriented_tp": metrics["oriented_tp"],
                                "oriented_fp": metrics["oriented_fp"],
                                "oriented_fn": metrics["oriented_fn"],
                            }
                        )
                    rows.append({**base, **row})
        return rows

    def _evaluate_draw(
        self,
        sampled: pd.DataFrame,
        algorithm: str,
        proportion: float,
        mechanism: str,
        miss_rate: float,
        strategy: str,
        knowledge: bool,
        run_sem: bool,
        rng: np.random.Generator,
    ) -> dict:
        """Apply missingness, handle it, then run discovery on one draw."""
        try:
            corrupted, miss_info = apply_missingness(
                sampled,
                mechanism,
                miss_rate,
                rng,
                covariates=self._mar_covariates(),
                mar_slope=self._mar_slope(),
            )
            handled, handle_info = handle_missing(corrupted, strategy)
        except Exception as exc:
            return {
                "search_ok": False,
                "error": str(exc),
                "n_rows": 0,
                "n_missing_rows": None,
                "missing_rate_empirical": None,
                "n_rows_dropped": None,
                "n_imputed_cells": None,
            }

        extras = {
            "n_rows": len(handled),
            "n_missing_rows": miss_info.get("n_missing_rows"),
            "missing_rate_empirical": miss_info.get("missing_rate_empirical"),
            "n_rows_dropped": handle_info.get("n_rows_dropped"),
            "n_imputed_cells": handle_info.get("n_imputed_cells"),
        }
        if handled.empty or len(handled) < 3:
            return {
                **extras,
                "search_ok": False,
                "error": (
                    f"too few rows after {strategy} "
                    f"(n={len(handled)}) for {algorithm}"
                ),
            }

        resampled = standardize_df(handled)
        try:
            result = run_search(
                resampled,
                algorithm=algorithm,
                alpha=self._alpha(),
                penalty_discount=self._penalty(),
                knowledge=knowledge,
                run_sem=run_sem,
                verbose=max(self.verbose - 2, 0),
            )
        except Exception as exc:
            print(
                f"Warning: {algorithm} failed at p={proportion} "
                f"{mechanism}@{miss_rate} {strategy}: {exc}"
            )
            return {**extras, "search_ok": False, "error": str(exc)}

        es_mean, es_std = mean_abs_estimates(result.get("sem_summary"))
        return {
            **extras,
            "search_ok": True,
            "error": None,
            "ESMean": es_mean,
            "ESStd": es_std,
            "_recovered_edges": result["edges"],
        }

    def plot(self, summary: pd.DataFrame | None = None) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        out_dir = self._output_dir()
        if summary is None:
            csv_path = out_dir / "resampled_models_fastcausal.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"{csv_path} not found. Run --cmd compute first."
                )
            summary = pd.read_csv(csv_path)

        plt.ioff()
        if "search_ok" in summary.columns:
            summary = summary[summary["search_ok"] != False]
        if summary.empty:
            raise ValueError("Summary dataframe is empty; nothing to plot.")
        hue = "case" if summary["case"].nunique() > 1 else None
        extra_hue = "algorithm" if summary["algorithm"].nunique() > 1 else hue
        subsample = summary
        if "missing_rate" in summary.columns:
            complete = summary[summary["missing_rate"] == 0.0]
            if not complete.empty:
                subsample = complete

        def _box(data, x, y, filename, title, hue_col=None):
            plt.figure(figsize=(10, 6))
            plot_hue = hue_col if hue_col is not None else (extra_hue or hue)
            kwargs = {"x": x, "y": y, "data": data}
            if plot_hue is not None and plot_hue in data.columns:
                kwargs["hue"] = plot_hue
                kwargs["palette"] = "bright"
            else:
                kwargs["color"] = "steelblue"
            sns.boxplot(**kwargs)
            plt.title(title)
            plt.tight_layout()
            path = out_dir / filename
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            if self.verbose:
                print(f"Wrote {path}")

        _box(
            subsample,
            "proportion",
            "dice_skeleton",
            "dice_skeleton_by_proportion.png",
            "Skeleton Dice vs ground truth by subsample proportion",
        )
        _box(
            subsample,
            "proportion",
            "dice_directed",
            "dice_directed_by_proportion.png",
            "Directed Dice vs ground truth by subsample proportion",
        )
        if subsample["dice_vs_full"].notna().any():
            _box(
                subsample,
                "proportion",
                "dice_vs_full",
                "dice_vs_full_by_proportion.png",
                "Skeleton Dice vs 100% recovered graph by subsample proportion",
            )
        if subsample["ESMean"].notna().any():
            _box(
                subsample,
                "proportion",
                "ESMean",
                "esmean_by_proportion.png",
                "Mean |SEM estimate| by subsample proportion",
            )

        if "missing_rate" in summary.columns and (
            summary["missing_rate"].nunique() > 1
            or summary["missing_mechanism"].nunique() > 1
            or summary["missing_strategy"].nunique() > 1
        ):
            miss = summary.copy()
            miss["missing_cell"] = (
                miss["missing_mechanism"].astype(str)
                + " / "
                + miss["missing_strategy"].astype(str)
            )
            _box(
                miss,
                "missing_rate",
                "dice_skeleton",
                "dice_skeleton_by_missing_rate.png",
                "Skeleton Dice vs ground truth by row-missingness rate",
                hue_col="missing_cell",
            )
            _box(
                miss,
                "missing_rate",
                "dice_directed",
                "dice_directed_by_missing_rate.png",
                "Directed Dice vs ground truth by row-missingness rate",
                hue_col="missing_cell",
            )

    def smoke(self) -> pd.DataFrame:
        """Tiny end-to-end dry-run: generate one SEM, recover FGES, write CSV."""
        from picause import StructuralEquationDagModel

        out_dir = self._output_dir() / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)

        seed = self._seed()
        count = 0
        while True:
            sem = StructuralEquationDagModel(
                num_var=8, num_edges=8, seed=seed, beta=0.5
            )
            count += 1
            if not sem.test_residual_overflow():
                break
            if count > 50:
                raise RuntimeError("smoke SEM: residual overflow after 50 draws")
            seed += 1
        df = sem.generate_data(80)
        csv_path = out_dir / "sub-001_vars-8_edges-8_es-0.25.csv"
        txt_path = csv_path.with_suffix(".txt")
        df.to_csv(csv_path, index=False)
        txt_path.write_text(str(sem) + "\nIterations: 1\n", encoding="utf-8")

        self.config["data_dir"] = str(out_dir)
        self.config["iterations"] = int(self.config.get("iterations") or 2)
        self.config["proportions"] = self.config.get("proportions") or "1.0,0.8"
        self.config["algorithm"] = self.config.get("algorithm") or "fges"
        if self.config.get("run_sem") is None:
            self.config["run_sem"] = False
        self.config["output_dir"] = str(out_dir)
        self.config["glob"] = "sub-*.csv"
        # Tiny SA3 grid unless the caller explicitly disabled missingness.
        if self.config.get("missingness") is not False:
            self.config["missingness"] = True
            self.config["missing_mechanisms"] = (
                self.config.get("missing_mechanisms") or "mcar,mar"
            )
            self.config["missing_rates"] = self.config.get("missing_rates") or "0.2"
            self.config["missing_strategies"] = (
                self.config.get("missing_strategies") or "complete_case,mean"
            )

        summary = self.compute()
        if summary.empty:
            raise RuntimeError("Smoke run produced no result rows.")
        required = {
            "dice_skeleton",
            "algorithm",
            "proportion",
            "case",
            "missing_mechanism",
            "missing_rate",
            "missing_strategy",
            "oriented_tp",
        }
        missing = required - set(summary.columns)
        if missing:
            raise RuntimeError(f"Smoke CSV missing columns: {missing}")
        if self.verbose:
            print(summary.head().to_string(index=False))
            print("Smoke test OK.")
        return summary


def build_parser() -> argparse.ArgumentParser:
    description = textwrap.dedent(
        """\
        Run FGES/GFCI on simdata CSVs with fastcausal (no Java).

        Examples:
          python simdata.py --cmd sim
          python tradsim_fastcausal.py --cmd compute
          python tradsim_fastcausal.py --cmd compute --proportions 1.0 --missingness
          python tradsim_fastcausal.py --cmd plot
          python tradsim_fastcausal.py --cmd smoke
        """
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--cmd",
        default="compute",
        choices=["compute", "plot", "smoke"],
        help="compute | plot | smoke (default compute)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--algorithm",
        default=None,
        help="fges, gfci, or both (default: discovery.algorithm in config)",
    )
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="subsample repeats (default 20). Proportion 1.0 always runs once.",
    )
    parser.add_argument(
        "--proportions",
        default=None,
        help="comma-separated fractions, e.g. 1.0,0.8,0.4",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--penalty-discount", dest="penalty_discount", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument(
        "--knowledge",
        action="store_true",
        help="add lag/current temporal tiers from column names",
    )
    parser.add_argument("--run-sem", dest="run_sem", action="store_true", default=None)
    parser.add_argument("--no-sem", dest="run_sem", action="store_false")
    parser.add_argument(
        "--missingness",
        dest="missingness",
        action="store_true",
        default=None,
        help="enable the discovery.missingness grid (row MCAR / MAR)",
    )
    parser.add_argument(
        "--no-missingness",
        dest="missingness",
        action="store_false",
        help="disable missingness (subsample only)",
    )
    parser.add_argument(
        "--missing-mechanisms",
        dest="missing_mechanisms",
        default=None,
        help="comma-separated: none,mcar,mar (enables missingness)",
    )
    parser.add_argument(
        "--missing-rates",
        dest="missing_rates",
        default=None,
        help="comma-separated row-missing rates, e.g. 0,0.1,0.2,0.4",
    )
    parser.add_argument(
        "--missing-strategies",
        dest="missing_strategies",
        default=None,
        help="comma-separated: complete_case,mean (median optional)",
    )
    parser.add_argument(
        "--mar-covariates",
        dest="mar_covariates",
        default=None,
        help="fully observed MAR columns (default: first numeric column)",
    )
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = TradSimFastcausal(
        cmd=args.cmd,
        config=args.config,
        algorithm=args.algorithm,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        proportions=args.proportions,
        alpha=args.alpha,
        penalty_discount=args.penalty_discount,
        seed=args.seed,
        glob=args.glob,
        knowledge=args.knowledge,
        run_sem=args.run_sem,
        missingness=args.missingness,
        missing_mechanisms=args.missing_mechanisms,
        missing_rates=args.missing_rates,
        missing_strategies=args.missing_strategies,
        mar_covariates=args.mar_covariates,
        verbose=args.verbose,
    )
    if args.cmd == "compute":
        runner.compute()
    elif args.cmd == "plot":
        runner.plot()
    elif args.cmd == "smoke":
        runner.smoke()
    return 0


if __name__ == "__main__":
    sys.exit(main())
