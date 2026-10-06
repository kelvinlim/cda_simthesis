# cda_simthesis

Thesis EMA simulation + causal discovery: how **subsample size / missingness /
discovery hyperparameters** affect **FGES** and **GFCI** recovery versus
**simulated ground-truth** graphs.

The intended loop is:

1. Generate SEM data and a known DAG (`simdata.py` + `picause.py`).
2. Run discovery on those CSVs (`tradsim_fastcausal.py`).
3. Compare recovered edges to the picause `.txt` ground truth (Dice, oriented
   counts, FPR/TPR) across subsample fractions ~100% → 40% and, optionally, a
   row missingness grid (MCAR then MAR) or an SA4 hyperparameter / target-FPR
   sweep.

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

# 5. SA4 HP / target-FPR selection vs the same picause DAGs (complete data)
python tradsim_fastcausal.py --cmd compute --proportions 1.0 --hp-sweep
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
python tradsim_fastcausal.py --cmd smoke --hp-sweep
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
| `--penalty-discount` | `penalty_discount` | SEM-BIC penalty (default `1.0`; scalar when sweep is off) |
| `--alpha` | `alpha` | GFCI Fisher-Z (default `0.01`; scalar when sweep is off) |
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
| `--mar-slope` | `missingness.mar_slope` | logistic slope for row MAR (default `1.5`) |
| `--hp-sweep` / `--no-hp-sweep` | `hp_sweep.enabled` | FGES penalty / GFCI alpha grid (default **off**) |
| `--penalty-discounts` | `hp_sweep.penalty_discounts` | e.g. `1,2,4` (enables hp-sweep) |
| `--alphas` | `hp_sweep.alphas` | e.g. `0.01,0.05` (enables hp-sweep; GFCI) |
| `--target-fprs` | `hp_sweep.target_fprs` | e.g. `0.05,0.10` (enables hp-sweep) |

Implementation (not extra CLIs):

- `tools/fastcausal_backend.py` — FGES/GFCI via `FastCausal.run_search`
- `tools/graph_metrics.py` — parse picause `.txt` edges; Dice / oriented counts / FPR
- `tools/missingness.py` — row MCAR / MAR generators and cheap handlers
- `tools/hp_sweep.py` — FGES/GFCI HP grid and target-FPR selection

### Metrics

Written to `discovery_results/resampled_models_fastcausal.csv` (and plots from
`--cmd plot`):

- **`dice_skeleton` / `diceCoeff`** — undirected edge Dice vs picause truth
- **`dice_directed`** — oriented Dice vs truth (`-->` / `o->`)
- **`oriented_tp` / `oriented_fp` / `oriented_fn`** — directed recovery counts
- **`oriented_fpr` / `oriented_tpr`** — FPR = `oriented_fp / (n(n-1) - n_true_directed)`;
  TPR = `oriented_tp / n_true_directed`
- **`skeleton_fpr` / `skeleton_tpr`** — same idea on the undirected skeleton
- **`n_nodes` / `n_possible_non_edges`** — FPR denominator pieces
- **`penalty_discount` / `alpha`** — HP used for that search (scalar or sweep cell)
- **`dice_vs_full`** — legacy Dice vs the **100% recovered** graph (not truth)
- **`ESMean` / `ESStd`** — mean/std of |SEM estimates| (only if SEM is on)
- **`search_ok`** — whether that draw’s search finished
- **`missing_mechanism` / `missing_rate` / `missing_strategy`** — SA3 cell
- **`n_missing_rows` / `missing_rate_empirical` / `n_rows_dropped` /
  `n_imputed_cells`** — how missingness was realized and handled

`--cmd compute` also prints mean Dice / oriented TP-FP-FN grouped by
mechanism × rate × strategy (and algorithm, when more than one). An SA4
sweep additionally writes `hp_sweep_metrics.csv` (HP × mean FPR/TPR/Dice)
and `hp_selected_by_target_fpr.csv`, and prints both tables.

`--cmd plot` writes `dice_skeleton_by_proportion.png`,
`dice_directed_by_proportion.png`, `dice_vs_full_by_proportion.png`, and
`esmean_by_proportion.png` when SEM columns are present. When the
missingness grid varies it also writes
`dice_skeleton_by_missing_rate.png` and
`dice_directed_by_missing_rate.png`. When the HP grid varies it writes
`oriented_fpr_by_penalty_discount.png` / `dice_directed_by_penalty_discount.png`
(FGES) and `oriented_fpr_by_alpha.png` / `dice_directed_by_alpha.png` (GFCI).

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
- **HP sweep is off**. `--cmd compute` uses scalar `penalty_discount` /
  `alpha` until you pass `--hp-sweep` (or a list flag). SA4 should use
  `--proportions 1.0` and leave missingness off. A fuller profile:

  ```bash
  python tradsim_fastcausal.py --cmd compute --proportions 1.0 --hp-sweep \
    --penalty-discounts 0.5,1,2,4,8 --iterations 20 \
    --output-dir ./discovery_results/sa4_fges
  python tradsim_fastcausal.py --cmd compute --algorithm gfci --proportions 1.0 \
    --hp-sweep --alphas 0.001,0.005,0.01,0.05,0.1 \
    --output-dir ./discovery_results/sa4_gfci
  ```

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
   **Complete-case is the natural handler.** Mean/median on those rows
   just reinserts the column-mean (or median) vector as near-duplicate
   observations, so they are **not** in the default MCAR path. Set
   `missingness.mcar_impute: true` only if you want that odd baseline.
   Cell-wise MCAR (blanking some columns, not the whole row) is out of
   scope here.
2. **Row MAR** — one or more columns stay fully observed (default: the first
   numeric column, picause `x_1`; override with `missingness.mar_covariates`
   / `--mar-covariates`). A logistic model of those covariates sets
   `P(row incomplete)`. On a missing row, only the *other* columns become
   NaN. The intercept is calibrated so the mean predicted probability equals
   the target rate; `missingness.mar_slope` / `--mar-slope` (default `1.5`)
   controls how strongly missingness tracks the covariates (positive slope →
   larger covariate values more likely missing). If every column is listed
   as a covariate, the runner raises (that would no longer be MAR).

### Handling defaults (cheap)

- **`complete_case`** (`listwise` / `dropna`) — drop any row with a NaN.
  Use this for **MCAR**.
- **`mean`** — column-wise mean of observed values. Default simple impute
  for **MAR** (covariates stay observed, so filled values are not identical
  copies of a single mean-row).
- **`median`** — optional extra; same idea with column medians.

Rate `0` (any mechanism) is complete data and collapses to one
`("none", 0.0, "none")` cell so MCAR+MAR does not double the baseline
search. The shared `strategies` list still includes `mean` for MAR; MCAR
skips impute strategies unless `mcar_impute: true`.

### How to run MCAR, then MAR

Use `--proportions 1.0` so the missingness axis is not crossed with the
subsample grid. Separate output dirs if you run the two mechanisms as
two jobs (the CSV is overwritten otherwise):

```bash
# MCAR: complete-case only (mean/median on whole-row blanks is a poor baseline)
python tradsim_fastcausal.py --cmd compute --proportions 1.0 \
  --missing-mechanisms mcar --missing-rates 0,0.1,0.2,0.4 \
  --missing-strategies complete_case \
  --output-dir ./discovery_results/sa3_mcar

# then MAR (same picause CSVs / ground-truth DAGs; mean impute is useful here)
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

`--cmd smoke` and `config_smoke.yaml` enable a tiny grid (`mcar` ×
`complete_case`, `mar` × `complete_case,mean`, rate `0.2`) so the path
stays cheap.

## SA4: hyperparameter / target-FPR (`tools/hp_sweep.py`)

With picause ground truth, empirical **oriented FPR** is
`oriented_fp / (n_nodes*(n_nodes-1) - n_true_directed)` — directed false
positives over possible directed non-edges. **TPR** is
`oriented_tp / n_true_directed`. SA4 sweeps the HPs that move FPR and
picks the setting closest to each requested target.

- **FGES** sweeps `penalty_discount` (SEM-BIC). Larger penalty → fewer
  edges → typically **lower FPR**. `--alpha` is recorded but unused.
- **GFCI** sweeps `alpha` (Fisher-Z). Larger alpha → more edges →
  typically **higher FPR**. `penalty_discount` stays at the scalar default.
- **`--algorithm both`** runs the FGES penalty grid and the GFCI alpha
  grid (not a full cross).
- **Target-FPR selection** (default `0.05,0.10`): for each algorithm ×
  proportion (and missingness cell, if that grid varies), pick the HP
  whose **mean** `oriented_fpr` across cases/iterations is closest to the
  target. Ties go to higher TPR, then higher directed Dice, then the more
  conservative HP (larger penalty / smaller alpha).

SA4 defaults to **complete data + subsample**. Leave missingness off
unless you explicitly want that extra cross.

### How to run

```bash
# FGES penalty grid vs picause truth, p=1.0, targets 0.05 and 0.10
python tradsim_fastcausal.py --cmd compute --proportions 1.0 --hp-sweep \
  --output-dir ./discovery_results/sa4_fges

# GFCI alpha grid
python tradsim_fastcausal.py --cmd compute --algorithm gfci --proportions 1.0 \
  --hp-sweep --output-dir ./discovery_results/sa4_gfci

# Custom lists (any of these flags also enables the sweep)
python tradsim_fastcausal.py --cmd compute --proportions 1.0 \
  --penalty-discounts 1,2,4 --target-fprs 0.05,0.10
```

`--cmd compute` writes:

- `resampled_models_fastcausal.csv` — one row per draw (includes
  `penalty_discount`, `alpha`, `oriented_fpr`, `oriented_tpr`)
- `hp_sweep_metrics.csv` — mean FPR/TPR/Dice per HP cell
- `hp_selected_by_target_fpr.csv` — selected HP per target FPR

Optional subsample with the same HP grid:

```bash
python tradsim_fastcausal.py --cmd compute --hp-sweep --proportions 1.0,0.8 \
  --output-dir ./discovery_results/sa4_subsample
```

Dry-run (complete data, two penalties; does not stack on the SA3 smoke
grid):

```bash
python tradsim_fastcausal.py --cmd smoke --hp-sweep
```

`config_smoke.yaml` keeps `hp_sweep.enabled: false` with a two-point
list so `--config config_smoke.yaml --hp-sweep` stays cheap.

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
- **SA4 leftovers** — clinical / real-data target-FPR calibration (this
  tree only selects HPs against **simulated** picause truth).

## Historical notes

Simulated data originally came from a fork of CausalPowerAnalysis:
https://github.com/kelvinlim/CausalPowerAnalysis (`picause.py` in this repo).

Older lab notes (`ProjectNotes.md`, `cda_data_project.md`) describe Java Tetrad,
`tradsim_fges_obj.py`, and case-selection notebooks (`try_all_files.ipynb`,
`simstandard.py`) that are not required for the current runner.
