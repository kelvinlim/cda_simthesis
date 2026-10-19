# Aims 3–4: real Boston EMA and matching 10-variable simulations

Thesis Aims 3–4 need **individual / multi-subject EMA causal discovery** on
long within-person series. Graph width (`p`) in the matching simulations must
equal the real fixture width so recovery experiments transfer.

This note is the rationale. Commands live in the root `README.md`.

## Why short+full Set A (10 variables), not full-only classic 7

Boston College COVID-19 daily surveys have two instruments:

- **Full** (`covid19`) — sleep, activity, PANAS, isolation/worry/stress, PHQ-style items.
- **Short** (`covid19_short_survey`) — sleep, activity, coronavirus status, overall stress.
  It skips most mood, isolation, worry, and PHQ items.

Classic-7 columns (`alcohol_bev`, `TIB`, `TST`, `PANAS_PA`, `PANAS_NA`,
`worry_scale`, `PHQ9`) include **mood sums that exist only on Full surveys**.
Complete-case Full series therefore top out at **~61 days/person** (subject 132).
There are almost no N≥60 classic-7 individual cohorts, and **none at N≥100**.

Set A is the 10 columns asked on **both** Full and Short surveys:

`TIB`, `TST`, `sleepdiary_sleeplatency`, `sleepdiary_wakes`,
`night_awakening_time`, `sleepdiary_naptime`, `socialize_min`,
`people_contact`, `alcohol_bev`, `stress`

Pooling short+full Set A complete days yields long within-person series:

| Cohort | Count |
|--------|-------|
| Subjects with ≥60 complete Set A days | **325** |
| Subjects with ≥100 complete Set A days | **128** |

That is why Aims 3–4 use Set A (`p = 10`) rather than classic-7 (`p = 7`).
Matching picause simulations use `num_variables: 10` (not 22) so simulated
`p` equals the real EMA fixture `p`.

## Public fixtures (do not vendor copies)

Source: [kelvinlim/causal-open-data](https://github.com/kelvinlim/causal-open-data),
**CC-BY 4.0**. Pin a commit SHA (this repo pins
`8d7c7553d7f54e1cb6497fcbbc6fa2d77f7d29d4`; see `scripts/boston_fixtures_pin.json`
and `NOTICE`). No HCP data.

```bash
python3 scripts/fetch_boston_fixtures.py
```

Writes `fixtures/causal-open-data/datasets/boston/...` (gitignored) and checks
SHA256. Paths:

| Path | n × p | Role |
|------|-------|------|
| `datasets/boston/boston_sub132_allsurveys_10v.csv` | 142 × 10 | Primary long Set A case |
| `datasets/boston/boston_sub79_allsurveys_10v.csv` | 142 × 10 | Alternate Set A case |
| `datasets/boston/individual/boston_sub*_n*_10v.csv` | 11–129 × 10 | **n-scaling ladder** (six subjects) |

The n-scaling ladder is six *different* people, one per complete-case length
bin (n = 11, 27, 35, 41, 87, 129), not prefixes of a single series. Use it for
**length scaling** on real EMA. Subject 132 remains the original long fixture;
the ≥100 ladder rung is subject 305 (129 days).

Full-only contrast (not fetched by the 10v helper; still in causal-open-data):
`boston_sub132_fullsurvey_15v.csv` is 61 × 15, illustrating the Full-only
ceiling.

## 10-variable simulations vs the 22-variable grid

| Profile | Config | `p` | edges | sim CSVs |
|---------|--------|-----|-------|----------|
| Original thesis grid | `config.yaml` | 22 | 27 | `./sim_data_v2` |
| Boston-matched width | `config_boston10.yaml` | 10 | 12 | `./sim_data_boston10` |
| Cheap 10-var dry-run | `config_boston10_smoke.yaml` | 10 | 10 | `./sim_data_boston10_smoke` |

12 edges on 10 nodes keeps average degree ≈ 2.4, matching 22 vars / 27 edges
(`2*27/22 ≈ 2.45`). Simulated columns stay picause `x_1`…`x_10`; Set A names
apply to real CSVs only.

Discovery is unchanged: `tradsim_fastcausal.py` on those sim CSVs plus companion
picause `.txt` DAGs. Java / fastcda is not required.

## Later real-data runner (stub)

A full real-data loop (no picause truth, different metrics, optional knowledge)
is out of scope here. `tools/boston_realdata.py` only loads fetched Set A CSVs.

`tradsim_fastcausal.py --cmd compute` **skips** CSVs that lack a companion
`.txt` ground-truth DAG, so pointing `--data-dir` at the Boston fixtures will
not score Dice vs truth. A later runner would load the CSVs via
`load_set_a_frame`, run FGES/GFCI, and report graphs without picause FPR/TPR.
