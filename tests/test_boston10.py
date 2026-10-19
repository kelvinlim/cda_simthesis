"""Boston Set A pin, 10-var configs, and fetch helper (no Java)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from simdata import SimData
from tools.boston_realdata import SET_A, load_set_a_frame
from tradsim_fastcausal import TradSimFastcausal

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "scripts" / "boston_fixtures_pin.json"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_boston_fixtures",
        ROOT / "scripts" / "fetch_boston_fixtures.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fetch_boston = _load_fetch_module()


def test_pin_is_commit_not_main_and_set_a_only():
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert pin["repo"] == "kelvinlim/causal-open-data"
    assert pin["commit"] != "main"
    assert len(pin["commit"]) >= 40
    assert pin["license"].startswith("CC-BY")
    assert pin["set_a_variables"] == SET_A
    assert len(SET_A) == 10

    paths = [item["path"] for item in pin["files"]]
    joined = "\n".join(paths).lower()
    assert "hcp" not in joined
    assert "myconnectome" not in joined
    assert "jang" not in joined
    assert any(p.endswith("boston_sub132_allsurveys_10v.csv") for p in paths)
    assert any("/individual/" in p and p.endswith("_10v.csv") for p in paths)
    csv_paths = [p for p in paths if p.endswith(".csv")]
    assert len(csv_paths) == 8
    for item in pin["files"]:
        assert len(item["sha256"]) == 64
        assert set(item["sha256"]) <= set("0123456789abcdef")


def test_config_yaml_stays_22var():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["simulation"]["num_variables"] == 22
    assert cfg["simulation"]["num_edges"] == 27


def test_config_boston10_is_10var():
    cfg = yaml.safe_load((ROOT / "config_boston10.yaml").read_text(encoding="utf-8"))
    assert cfg["simulation"]["num_variables"] == 10
    assert cfg["simulation"]["num_edges"] == 12
    assert cfg["simulation"]["data_directory"] != "./sim_data_v2"
    smoke = yaml.safe_load(
        (ROOT / "config_boston10_smoke.yaml").read_text(encoding="utf-8")
    )
    assert smoke["simulation"]["num_variables"] == 10
    assert smoke["simulation"]["num_edges"] == 10
    assert smoke["smoke"]["num_variables"] == 10


def test_smoke_spec_default_stays_8var():
    runner = TradSimFastcausal(config=str(ROOT / "config.yaml"))
    spec = runner._smoke_spec()
    assert spec["num_variables"] == 8
    assert spec["num_edges"] == 8


def test_smoke_spec_boston10():
    runner = TradSimFastcausal(config=str(ROOT / "config_boston10_smoke.yaml"))
    spec = runner._smoke_spec()
    assert spec["num_variables"] == 10
    assert spec["num_edges"] == 10
    assert spec["num_samples"] == 80


def test_simdata_boston10_writes_10_columns(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "simulation": {
                    "num_samples": [40],
                    "effect_sizes": [0.25],
                    "num_variables": 10,
                    "num_edges": 10,
                    "seed": 43,
                    "data_directory": str(tmp_path / "out"),
                    "iterations": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    SimData(config=str(cfg_path)).cmd("sim")
    csvs = list((tmp_path / "out").glob("*.csv"))
    assert len(csvs) == 1
    frame = pd.read_csv(csvs[0])
    assert frame.shape == (40, 10)
    assert csvs[0].with_suffix(".txt").is_file()
    assert "vars-10_" in csvs[0].name
    assert "edges-10_" in csvs[0].name


def test_fetch_verify_and_sha_mismatch(tmp_path):
    pin = fetch_boston.load_pin(PIN_PATH)
    entries = fetch_boston.dataset_entries(pin)
    rel = entries[0]["path"]
    expected = entries[0]["sha256"]
    dest = tmp_path / "fixtures"
    good = dest / rel
    good.parent.mkdir(parents=True, exist_ok=True)
    payload = b"boston-set-a-fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    item = {"path": rel, "sha256": digest}
    good.write_bytes(payload)
    fetch_boston.verify_existing(dest, [item])

    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch_boston.verify_bytes(rel, b"nope", expected)

    with pytest.raises(SystemExit, match="missing"):
        fetch_boston.verify_existing(tmp_path / "empty", entries[:1])


def test_fetch_dry_run_lists_pin(capsys):
    code = fetch_boston.main(["--pin", str(PIN_PATH), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "boston_sub132_allsurveys_10v.csv" in out
    assert "individual/boston_sub305_n129_10v.csv" in out


def test_fetch_refuses_restricted_paths():
    with pytest.raises(ValueError, match="restricted"):
        fetch_boston.dataset_entries(
            {
                "commit": "abc1234",
                "files": [
                    {
                        "path": "datasets/hcp/something.csv",
                        "sha256": "a" * 64,
                    }
                ],
            }
        )


def test_load_set_a_frame_requires_ten_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"TIB": [1], "TST": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing Set A"):
        load_set_a_frame(path)

    good = tmp_path / "good.csv"
    pd.DataFrame({col: [0.0] for col in SET_A}).to_csv(good, index=False)
    frame = load_set_a_frame(good)
    assert list(frame.columns) == SET_A
    assert frame.shape == (1, 10)


@pytest.mark.network
def test_fetch_pinned_boston10_live(tmp_path):
    dest = tmp_path / "causal-open-data"
    code = fetch_boston.main(
        ["--pin", str(PIN_PATH), "--dest", str(dest)]
    )
    assert code == 0
    primary = dest / "datasets/boston/boston_sub132_allsurveys_10v.csv"
    assert primary.is_file()
    frame = load_set_a_frame(primary)
    assert frame.shape[1] == 10
    assert list(frame.columns) == SET_A
    ladder = list((dest / "datasets/boston/individual").glob("*_10v.csv"))
    assert len(ladder) == 6
    meta = json.loads((dest / "FETCH_META.json").read_text(encoding="utf-8"))
    assert meta["commit"] == json.loads(PIN_PATH.read_text())["commit"]
    assert fetch_boston.main(["--pin", str(PIN_PATH), "--dest", str(dest), "--verify-only"]) == 0
