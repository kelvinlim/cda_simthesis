# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Thesis simulation study: how subsample size, row missingness (SA3) and discovery hyperparameters / target FPR (SA4) affect **FGES** and **GFCI** recovery of a **simulated ground-truth DAG**. `README.md` is the authoritative reference for every CLI flag, config key, output column and SA3/SA4 design decision. Read it before you change runner behavior, and keep it in sync when you add flags or columns.

## Commands

```bash
pip install -r requirements.txt            # pins fastcausal==0.1.11 (no Java)

pytest tests/ -q                           # all tests (pytest.ini sets pythonpath=. and testpaths=tests)
pytest tests/test_hp_sweep.py -q           # one file
pytest tests/test_graph_metrics.py -k dice # one test by keyword
pytest -m "not network"                    # skip tests that download causal-open-data fixtures

# cheap end-to-end
python simdata.py --config config_smoke.yaml --cmd sim
python tradsim_fastcausal.py --config config_smoke.yaml --cmd compute
python tradsim_fastcausal.py --config config_smoke.yaml --cmd plot
python tradsim_fastcausal.py --cmd smoke [--hp-sweep]
```

Full runs: `simdata.py --cmd sim`, then `tradsim_fastcausal.py --cmd compute|plot` with `--algorithm fges|gfci|both`, `--missingness` (SA3) or `--hp-sweep` (SA4). SA3 and SA4 runs should use `--proportions 1.0` and a separate `--output-dir` per run, because `resampled_models_fastcausal.csv` is overwritten.

There is no linter or build step configured.

## Architecture

Two-stage pipeline joined by files on disk:

1. **Simulation**: `simdata.py` drives `picause.py` (a CausalPowerAnalysis fork). For each case it writes `<name>.csv` plus a companion `<name>.txt` SEM dump, which is the ground-truth DAG, into `simulation.data_directory`. Filenames encode metadata (`sub-NNN` / `iter-NNN`, `es-0.X`). `tradsim_fastcausal.parse_case_meta` parses these back out, so keep the naming stable.
2. **Discovery**: `tradsim_fastcausal.py` (`TradSimFastcausal`) iterates over the CSVs. It **skips any CSV without a companion `.txt`**, so it cannot score real data. For each case it builds a grid of algorithm × proportion × missingness cell × HP cell × iteration, then `_run_proportions` → `_evaluate_draw` runs the search and scores it against the truth.
   - `tools/fastcausal_backend.py`: FGES/GFCI via `FastCausal.run_search` (fastcausal → tetrad-port C++)
   - `tools/graph_metrics.py`: parses picause `.txt` edges; computes Dice (skeleton and directed), oriented TP/FP/FN, FPR/TPR
   - `tools/missingness.py`: row MCAR/MAR masks and handlers (complete_case / mean / median)
   - `tools/hp_sweep.py`: the FGES `penalty_discount` grid / GFCI `alpha` grid, and target-FPR selection

Design invariants that span several files:
- **Config precedence**: YAML `discovery:` / `simulation:` sections, overridden by CLI flags. The `_xxx()` accessor methods on `TradSimFastcausal` resolve each setting. Any HP list flag (`--penalty-discounts`, `--alphas`, `--target-fprs`) also enables the sweep.
- **Determinism**: every draw gets its own RNG from `cell_rng(seed, *parts)` (a SHA-256 of the cell key), not a shared stream. Strategies within the same (mechanism, rate) cell must share one missingness mask.
- Proportion 1.0 runs once (`FULL_SAMPLE_ITERATIONS`). Missingness rate 0 collapses to a single `("none", 0.0, "none")` cell. MCAR skips the impute strategies unless `mcar_impute: true`.
- Subsampling (keeping complete rows) is **not** MCAR. Keep the two axes separate.
- A failed search is recorded as `search_ok=False`, not raised. Results are reported as a success rate per cell.

The README lists these as not implemented yet: MNAR, multiple imputation (MICE), real-data FPR calibration, and the Boston discovery loop. SA4 selects hyperparameters only against simulated picause truth.

## Configs

- `config.yaml`: 22 variables / 27 edges → `./sim_data_v2`
- `config_boston10.yaml`: 10 variables / 12 edges → `./sim_data_boston10`. This matches the width of the real Boston EMA "Set A" (Aims 3–4; see `docs/aims3-4-boston-ema.md`).
- `*_smoke.yaml`: tiny grids for fast checks

Sim data dirs, `discovery_results/` and `fixtures/causal-open-data/` are gitignored. Fetch the Boston fixtures, pinned by commit SHA, with `python3 scripts/fetch_boston_fixtures.py`. Do not vendor them into the repo. `tools/boston_realdata.py` is only a loader stub; a real-data discovery loop does not exist yet.

## Legacy code (avoid)

`tradsim_fges_obj.py`, `tools/TetradSearch.py`, `tools/simulate.py`, `tools/WrappedClKci.py`, `tools/translate.py`, `tools/visualize.py` (both operate on Java graph objects) and `resources/tetrad-current.jar` are the deprecated JPype / Java Tetrad / fastcda path. Do not add `fastcda` or JPype dependencies, and do not route new work through these files. `ProjectNotes.md` and `cda_data_project.md` are historical notes about that path.

`reports/<date>_*/` holds committed snapshots of report outputs (a docx, plots and tables). These are deliverables, not generated scratch.
