from pathlib import Path

from tradsim_fastcausal import parse_case_meta


def test_parse_case_meta_does_not_confuse_edges_with_es():
    meta = parse_case_meta(Path("sub-001_vars-8_edges-8_es-0.25.csv"))
    assert meta["subject"] == "sub-001"
    assert meta["es"] == 0.25


def test_parse_case_meta_legacy_rows_iter_name():
    meta = parse_case_meta(Path("rows-100_vars-14_edges-12_es-0.1_iter-003.csv"))
    assert meta["subject"] == "sub-003"
    assert meta["es"] == 0.1
