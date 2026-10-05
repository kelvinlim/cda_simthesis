#! /usr/bin/env python3
"""Discovery runner: FGES / GFCI via fastcausal on simdata CSVs.

Compares recovered graphs to picause ground-truth ``.txt`` files and to the
100%-sample recovered graph (legacy Dice), across subsample fractions
100% → 40%.

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

__version_info__ = ("0", "2", "0")
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
        return summary

    def _print_search_success(self, summary: pd.DataFrame) -> None:
        """Report search_ok mean by algorithm × proportion × es (GFCI failures)."""
        if summary.empty or "search_ok" not in summary.columns:
            return
        keys = [c for c in ("algorithm", "proportion", "es") if c in summary.columns]
        rates = (
            summary.groupby(keys, dropna=False)["search_ok"]
            .mean()
            .rename("success_rate")
            .reset_index()
        )
        print("Search success rate by cell (algorithm × proportion × es):")
        print(rates.to_string(index=False))

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

        for proportion in self._proportions():
            n_iter = iterations_for_proportion(proportion, self._iterations())
            for iteration in range(n_iter):
                sampled = df.sample(
                    frac=proportion,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                resampled = standardize_df(sampled)
                if self.verbose > 1:
                    print(
                        f"  {meta['case']} {algorithm} p={proportion} "
                        f"iter={iteration + 1}/{n_iter}"
                    )
                base = {
                    "case": meta["case"],
                    "subject": meta["subject"],
                    "es": meta["es"],
                    "algorithm": algorithm,
                    "proportion": proportion,
                    "iteration": iteration,
                    "n_rows": len(resampled),
                    "n_true_edges": len(skeleton_pairs(true_edges)),
                }
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
                        f"iter={iteration}: {exc}"
                    )
                    rows.append({**base, "search_ok": False, "error": str(exc)})
                    continue

                recovered = result["edges"]
                if proportion == 1.0 and full_edges is None:
                    full_edges = list(recovered)

                metrics = compare_graphs(
                    true_edges,
                    recovered,
                    full_sample_edges=full_edges,
                )
                es_mean, es_std = mean_abs_estimates(result.get("sem_summary"))
                rows.append(
                    {
                        **base,
                        "search_ok": True,
                        "error": None,
                        "n_recovered_edges": metrics["n_recovered_edges"],
                        "n_recovered_directed": metrics["n_recovered_directed"],
                        "diceCoeff": metrics["dice_skeleton"],
                        "dice_skeleton": metrics["dice_skeleton"],
                        "dice_directed": metrics["dice_directed"],
                        "diceCoeffNodes": metrics["dice_nodes"],
                        "dice_vs_full": metrics["dice_vs_full_skeleton"],
                        "oriented_tp": metrics["oriented_tp"],
                        "oriented_fp": metrics["oriented_fp"],
                        "oriented_fn": metrics["oriented_fn"],
                        "ESMean": es_mean,
                        "ESStd": es_std,
                    }
                )
        return rows

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

        def _box(y, filename, title):
            plt.figure(figsize=(10, 6))
            plot_hue = extra_hue or hue
            kwargs = {"x": "proportion", "y": y, "data": summary}
            if plot_hue is not None:
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
            "dice_skeleton",
            "dice_skeleton_by_proportion.png",
            "Skeleton Dice vs ground truth by subsample proportion",
        )
        _box(
            "dice_directed",
            "dice_directed_by_proportion.png",
            "Directed Dice vs ground truth by subsample proportion",
        )
        if summary["dice_vs_full"].notna().any():
            _box(
                "dice_vs_full",
                "dice_vs_full_by_proportion.png",
                "Skeleton Dice vs 100% recovered graph by subsample proportion",
            )
        if summary["ESMean"].notna().any():
            _box(
                "ESMean",
                "esmean_by_proportion.png",
                "Mean |SEM estimate| by subsample proportion",
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

        summary = self.compute()
        if summary.empty:
            raise RuntimeError("Smoke run produced no result rows.")
        required = {"dice_skeleton", "algorithm", "proportion", "case"}
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
