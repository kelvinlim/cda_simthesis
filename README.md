# cda_simthesis

Thesis EMA simulation + causal discovery: how missingness / subsample size
affects FGES and GFCI recovery versus simulated ground-truth graphs.

Discovery now runs through [fastcausal](https://github.com/kelvinlim/fastcausal)
and [tetrad-port](https://github.com/kelvinlim/tetrad-port) (C++ Tetrad, **no
Java**). `fastcda` and `resources/tetrad-current.jar` are not required for the
simulation workflow.

## Install

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "import fastcausal; print(fastcausal.__version__)"
```

## Generate simulated data

`simdata.py` still uses `picause.py` (CausalPowerAnalysis fork). Parameters live
in `config.yaml` under `simulation`.

```bash
python simdata.py --cmd sim
```

Each case writes a CSV plus a companion `.txt` SEM dump (ground-truth DAG) into
`simulation.data_directory` (default `./sim_data_v2`).

## Run discovery with fastcausal

```bash
# FGES on the CSVs from simdata.py, subsample 100% → 40%, Dice vs ground truth
python tradsim_fastcausal.py --cmd compute

# GFCI, or both algorithms
python tradsim_fastcausal.py --cmd compute --algorithm gfci
python tradsim_fastcausal.py --cmd compute --algorithm both

# Boxplots of Dice / SEM |estimate| vs proportion
python tradsim_fastcausal.py --cmd plot
```

Outputs go to `discovery.output_directory` (default `./discovery_results`):

- `resampled_models_fastcausal.csv`
- `dice_skeleton_by_proportion.png` (undirected edge Dice vs true DAG)
- `dice_directed_by_proportion.png` (oriented Dice vs true DAG)
- `dice_vs_full_by_proportion.png` (legacy Dice vs 100% recovered graph)
- `esmean_by_proportion.png` (if SEM fitting is enabled)

Hyperparameters default to the old runner (`penalty_discount=1.0`, GFCI
`alpha=0.01`). Override via `config.yaml` `discovery:` or CLI flags
(`--iterations`, `--proportions`, `--penalty-discount`, `--alpha`, `--run-sem`).

### Grid cost

Default discovery settings are meant to be runnable, not a 100-iteration
profile:

- `discovery.iterations` is **20** (set to **100** for a “full” profile).
- The **100% sample is run once** as the `dice_vs_full` baseline; repeated
  draws apply only to proportions `< 1.0`.
- **SEM is off** (`run_sem: false`). Turn it on with `--run-sem` or
  `run_sem: true` in config.

GFCI (and occasionally FGES) can fail on a draw (`search_ok=False`, e.g.
nonpositive precision diagonal). Treat that as a **success rate by cell** —
group `resampled_models_fastcausal.csv` by algorithm × proportion × effect
size and take the mean of `search_ok`. `--cmd compute` prints this table.

### Dry-run / smoke test

Does not need a full sim grid. Either:

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

## Deprecated Java / fastcda path

`tradsim_fges_obj.py` is the previous FGES runner (JPype + Tetrad jar, leftover
fastcda GFCI). It now calls fastcausal when installed, but the preferred CLI is
`tradsim_fastcausal.py`. Java helpers remain in the tree for reference only:

- `tools/TetradSearch.py`, `tools/simulate.py`, `tools/WrappedClKci.py`
- `resources/tetrad-current.jar`

Do not add `fastcda` or JPype unless you are maintaining that fallback.

SA3 missingness+imputation against simulated ground truth is **not** in the new
runner yet (`tradsim_fges_obj.py --cmd impute` is still the old real-data hook).

## Historical notes

Simulated data originally came from a fork of CausalPowerAnalysis:
https://github.com/kelvinlim/CausalPowerAnalysis (`picause.py` in this repo).

The lab log of case selection, GFCI jar mismatches, and cloned real-data graphs
is preserved below in abbreviated form. Those experiments used
`tradsim_gfci_obj.py` / `simstandard.py`, which are not in this snapshot.

### Selecting cases (legacy GFCI notes)

Not every CausalPower graph produced usable GFCI edges. Worked files were
identified in `try_all_files.ipynb`; a later approach cloned directed graphs
from real EMA cases (`./simstandard.py --cmd clonesim`) and ran FGES via
`tradsim_fges_obj.py --cmd compute`.
