# Fetched fixtures (not vendored)

This directory is a **download target**. Boston Set A CSVs are **not** committed
here. Pin and checksums live in `scripts/boston_fixtures_pin.json`.

```bash
python3 scripts/fetch_boston_fixtures.py
# writes fixtures/causal-open-data/datasets/boston/...
python3 scripts/fetch_boston_fixtures.py --verify-only
```

Attribution: CC-BY 4.0 (see root `NOTICE`). Source:
https://github.com/kelvinlim/causal-open-data (pinned commit, not `main`).

No HCP data. MyConnectome files are not fetched by the Boston helper.
