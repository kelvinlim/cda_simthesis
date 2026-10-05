# cda_simthesis

Thesis EMA simulation + causal discovery: how **subsample size / missingness**
affects **FGES** and **GFCI** recovery versus **simulated ground-truth** graphs.

The intended loop is:

1. Generate SEM data and a known DAG (`simdata.py` + `picause.py`).
2. Run discovery on those CSVs (`tradsim_fastcausal.py`).
3. Compare recovered edges to the picause `.txt` ground truth (Dice, oriented
   counts) across subsample fractions ~100% → 40% and, optionally, a row
   missingness grid (MCAR then MAR, complete-case vs mean imputation).

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

# 4. SA3 row-missingness vs the same picause DAGs (MCAR + MAR, p=1.0)
python tradsim_fastcausal.py --cmd compute --proportions 1.0 --missingness
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
| `--missingness` / `--no-missingness` | `missingness.enabled` | row MCAR/MAR grid (default **off**) |
| `--missing-mechanisms` | `missingness.mechanisms` | `mcar,mar` (or `none`) |
| `--missing-rates` | `missingness.rates` | e.g. `0,0.1,0.2,0.4` |
| `--missing-strategies` | `missingness.strategies` | `complete_case,mean` (`median` optional) |
| `--mar-covariates` | `missingness.mar_covariates` | fully observed MAR columns (default: first numeric) |

Implementation (not extra CLIs):

- `tools/fastcausal_backend.py` — FGES/GFCI via `FastCausal.run_search`
- `tools/graph_metrics.py` — parse picause `.txt` edges; Dice / oriented counts
- `tools/missingness.py` — row MCAR / MAR generators and cheap handlers

### Metrics

Written to `discovery_results/resampled_models_fastcausal.csv` (and plots from
`--cmd plot`):

- **`dice_skeleton` / `diceCoeff`** — undirected edge Dice vs picause truth
- **`dice_directed`** — oriented Dice vs truth (`-->` / `o->`)
- **`oriented_tp` / `oriented_fp` / `oriented_fn`** — directed recovery counts
- **`dice_vs_full`** — legacy Dice vs the **100% recovered** graph (not truth)
- **`ESMean` / `ESStd`** — mean/std of |SEM estimates| (only if SEM is on)
- **`search_ok`** — whether that draw’s search finished
- **`missing_mechanism` / `missing_rate` / `missing_strategy`** — SA3 cell
- **`n_missing_rows` / `missing_rate_empirical` / `n_rows_dropped` /
  `n_imputed_cells`** — how missingness was realized and handled

`--cmd compute` also prints mean Dice / oriented TP-FP-FN grouped by
mechanism × rate × strategy (and algorithm, when more than one).

`--cmd plot` writes `dice_skeleton_by_proportion.png`,
`dice_directed_by_proportion.png`, `dice_vs_full_by_proportion.png`, and
`esmean_by_proportion.png` when SEM columns are present. When the
missingness grid varies it also writes
`dice_skeleton_by_missing_rate.png` and
`dice_directed_by_missing_rate.png`.

### Cost defaults

Meant to be runnable, not a 100-iteration profile:

- `discovery.iterations` is **20**. Set to **100** in config or
  `--iterations 100` for a “full” profile.
- Proportion **1.0 runs once** (baseline for `dice_vs_full`). Repeats apply
  only to proportions `< 1.0`.
- **SEM is off** (`run_sem: false`). Enable with `--run-sem` or
  `run_sem: true`.
- **Missingness is off**. `--cmd compute` is subsample-only until you pass
  `--missingness` or set `missingness.enabled: true`. Combine the full
  proportion list with the missingness grid only if you accept the extra
  cells; SA3 should use `--proportions 1.0`.

GFCI (and occasionally FGES) can fail on a draw (`search_ok=False`, e.g.
nonpositive precision diagonal). Treat that as a **success rate by cell**:
group the CSV by algorithm × proportion × effect size × missingness cell
and take the mean of `search_ok`. `--cmd compute` prints this table.

## SA3: row missingness (`tools/missingness.py`)

**Subsample is not MCAR.** `discovery.proportions` randomly *keeps* complete
rows. Missingness *blanks* rows at a rate; a handler then listwise-deletes
or imputes. Do not treat `proportion=0.8` as “20% MCAR”.

### Mechanisms

1. **Row MCAR** — each row is independently blanked with probability
   `missing_rate`. The whole row becomes NaN (unit nonresponse).
2. **Row MAR** — one or more columns stay fully observed (default: the first
   numeric column, picause `x_1`; override with `missingness.mar_covariates`
   / `--mar-covariates`). A logistic model of those covariates sets
   `P(row incomplete)`. On a missing row, only the *other* columns become
   NaN. The intercept is calibrated so the mean predicted probability equals
   the target rate; `missingness.mar_slope` (default `1.5`) controls how
   strongly missingness tracks the covariates (positive slope → larger
   covariate values more likely missing).

### Handling defaults (cheap)

- **`complete_case`** (`listwise` / `dropna`) — drop any row with a NaN.
- **`mean`** — column-wise mean of observed values (default simple impute).
- **`median`** — optional extra; same idea with column medians.

Rate `0` is complete data and is collapsed to one `strategy=none` cell so
the handler list is not repeated.

### How to run MCAR, then MAR

Use `--proportions 1.0` so the missingness axis is not crossed with the
subsample grid. Separate output dirs if you run the two mechanisms as
two jobs (the CSV is overwritten otherwise):

```bash
# MCAR only
python tradsim_fastcausal.py --cmd compute --proportions 1.0 \
  --missing-mechanisms mcar --missing-rates 0,0.1,0.2,0.4 \
  --missing-strategies complete_case,mean \
  --output-dir ./discovery_results/sa3_mcar

# then MAR (same picause CSVs / ground-truth DAGs)
python tradsim_fastcausal.py --cmd compute --proportions 1.0 \
  --missing-mechanisms mar --missing-rates 0,0.1,0.2,0.4 \
  --missing-strategies complete_case,mean \
  --output-dir ./discovery_results/sa3_mar
```

Or both mechanisms in one grid (uses `discovery.missingness` lists in
`config.yaml`):

```bash
python tradsim_fastcausal.py --cmd compute --proportions 1.0 --missingness
python tradsim_fastcausal.py --cmd plot
```

`--cmd smoke` and `config_smoke.yaml` enable a tiny grid (`mcar,mar` ×
rate `0.2` × `complete_case,mean`) so the path stays cheap.

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

- **SA3 leftovers** — cell-wise **MNAR**; **multiple imputation** (MICE);
  **clinical / real-data** missingness. The old
  `tradsim_fges_obj.py --cmd impute` hook remains real-data only.
- **SA4** — hyperparameter / target-FPR grids. `--penalty-discount` and
  `--alpha` exist; there is no target-FPR driver.

## Historical notes

Simulated data originally came from a fork of CausalPowerAnalysis:
https://github.com/kelvinlim/CausalPowerAnalysis (`picause.py` in this repo).

Older lab notes (`ProjectNotes.md`, `cda_data_project.md`) describe Java Tetrad,
`tradsim_fges_obj.py`, and case-selection notebooks (`try_all_files.ipynb`,
`simstandard.py`) that are not required for the current runner.
