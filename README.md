# cda_simthesis

Thesis EMA simulation + causal discovery: how **subsample size / missingness**
affects **FGES** and **GFCI** recovery versus **simulated ground-truth** graphs.

The intended loop is:

1. Generate SEM data and a known DAG (`simdata.py` + `picause.py`).
2. Run discovery on those CSVs (`tradsim_fastcausal.py`).
3. Compare recovered edges to the picause `.txt` ground truth (Dice, oriented
   counts) across subsample fractions ~100% → 40%.

Discovery uses [fastcausal](https://github.com/kelvinlim/fastcausal) and
[tetrad-port](https://github.com/kelvinlim/tetrad-port) (C++ Tetrad, **no
Java**). `fastcda` and `resources/tetrad-current.jar` are not required.

## Recommended path

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. generate CSVs + companion .txt DAGs (config.yaml → simulation)
python simdata.py --cmd sim

# 2. FGES (default) on those CSVs, subsample 100% → 40%
python tradsim_fastcausal.py --cmd compute

# 3. boxplots of Dice / optional SEM |estimate|
python tradsim_fastcausal.py --cmd plot
```

GFCI, or both algorithms:

```bash
python tradsim_fastcausal.py --cmd compute --algorithm gfci
python tradsim_fastcausal.py --cmd compute --algorithm both
```

Sanity check after install:

```bash
python -c "import fastcausal; print(fastcausal.__version__)"
```

### Smoke / dry-run (no full sim grid)

```bash
python tradsim_fastcausal.py --cmd smoke
```

or a 1-file simdata grid:

```bash
python simdata.py --config config_smoke.yaml --cmd sim
python tradsim_fastcausal.py --config config_smoke.yaml --cmd compute
python tradsim_fastcausal.py --config config_smoke.yaml --cmd plot
pytest tests/ -q
```

## Simulation (`simdata.py`)

`simdata.py` uses `picause.py` (CausalPowerAnalysis fork). Parameters live in
`config.yaml` under `simulation` (`num_samples`, `effect_sizes`,
`num_variables`, `num_edges`, `iterations`, `data_directory`, `seed`).

Each case writes a CSV plus a companion `.txt` SEM dump (ground-truth DAG) into
`simulation.data_directory` (default `./sim_data_v2`). Discovery expects that
pairing.

## Discovery (`tradsim_fastcausal.py`)

Config keys are under `discovery:` in `config.yaml`. CLI flags override them.

| CLI | Config key | Role |
|-----|------------|------|
| `--algorithm` | `algorithm` | `fges` (default), `gfci`, or `both` |
| `--iterations` | `iterations` | subsample repeats (default **20**) |
| `--proportions` | `proportions` | e.g. `1.0,0.9,...,0.4` |
| `--penalty-discount` | `penalty_discount` | SEM-BIC penalty (default `1.0`) |
| `--alpha` | `alpha` | GFCI Fisher-Z (default `0.01`) |
| `--run-sem` / `--no-sem` | `run_sem` | SEM fitting (default **off**) |
| `--data-dir` | `simulation.data_directory` | input CSVs |
| `--output-dir` | `output_directory` | results (default `./discovery_results`) |
| `--seed` | `seed` | RNG seed (default `2025`) |
| `--glob` | — | CSV glob (default `*.csv`) |
| `--knowledge` | — | lag/current temporal tiers from column names |
| `--config` | — | YAML path (default `config.yaml`) |

Implementation (not extra CLIs):

- `tools/fastcausal_backend.py` — FGES/GFCI via `FastCausal.run_search`
- `tools/graph_metrics.py` — parse picause `.txt` edges; Dice / oriented counts

### Metrics

Written to `discovery_results/resampled_models_fastcausal.csv` (and plots from
`--cmd plot`):

- **`dice_skeleton` / `diceCoeff`** — undirected edge Dice vs picause truth
- **`dice_directed`** — oriented Dice vs truth (`-->` / `o->`)
- **`oriented_tp` / `oriented_fp` / `oriented_fn`** — directed recovery counts
- **`dice_vs_full`** — legacy Dice vs the **100% recovered** graph (not truth)
- **`ESMean` / `ESStd`** — mean/std of |SEM estimates| (only if SEM is on)
- **`search_ok`** — whether that draw’s search finished

`--cmd plot` writes `dice_skeleton_by_proportion.png`,
`dice_directed_by_proportion.png`, `dice_vs_full_by_proportion.png`, and
`esmean_by_proportion.png` when SEM columns are present.

### Cost defaults

Meant to be runnable, not a 100-iteration profile:

- `discovery.iterations` is **20**. Set to **100** in config or
  `--iterations 100` for a “full” profile.
- Proportion **1.0 runs once** (baseline for `dice_vs_full`). Repeats apply
  only to proportions `< 1.0`.
- **SEM is off** (`run_sem: false`). Enable with `--run-sem` or
  `run_sem: true`.

GFCI (and occasionally FGES) can fail on a draw (`search_ok=False`, e.g.
nonpositive precision diagonal). Treat that as a **success rate by cell**:
group the CSV by algorithm × proportion × effect size and take the mean of
`search_ok`. `--cmd compute` prints this table.

## Off the critical path (legacy Java / fastcda)

These are **not** required for the workflow above:

- `tradsim_fges_obj.py` — previous FGES runner (JPype + Tetrad jar, leftover
  fastcda GFCI). It calls fastcausal when installed; prefer
  `tradsim_fastcausal.py`.
- `tools/TetradSearch.py`, `tools/simulate.py`, `tools/WrappedClKci.py`
- `resources/tetrad-current.jar` (see `resources/README.md`)

Do not add `fastcda` or JPype unless you are maintaining that fallback.
`requirements.txt` pins `fastcausal==0.1.11` and leaves those Java deps
commented out.

## Not in this tree yet (TODO)

- **SA3** — missingness + imputation **against simulated ground truth**. The
  old `tradsim_fges_obj.py --cmd impute` hook is real-data only.
- **SA4** — hyperparameter / target-FPR grids. `--penalty-discount` and
  `--alpha` exist; there is no target-FPR driver.

## Historical notes

Simulated data originally came from a fork of CausalPowerAnalysis:
https://github.com/kelvinlim/CausalPowerAnalysis (`picause.py` in this repo).

Older lab notes (`ProjectNotes.md`, `cda_data_project.md`) describe Java Tetrad,
`tradsim_fges_obj.py`, and case-selection notebooks (`try_all_files.ipynb`,
`simstandard.py`) that are not required for the current runner.
