#!/usr/bin/env python3
"""Download pinned Boston Set A (10-variable) fixtures from causal-open-data.

The upstream repo is public. This helper pins a **commit SHA** (see
``scripts/boston_fixtures_pin.json``), downloads only the Boston 10-variable
paths, and refuses to write a file whose SHA256 does not match.

It does **not** vendor copies forever, does **not** fetch MyConnectome, and
does **not** touch HCP or other restricted participant data.

Examples:

  python3 scripts/fetch_boston_fixtures.py
  python3 scripts/fetch_boston_fixtures.py --dest ./fixtures/causal-open-data
  python3 scripts/fetch_boston_fixtures.py --verify-only

A GitHub token (GITHUB_TOKEN / GH_TOKEN) is optional: public
``raw.githubusercontent.com`` URLs work without one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "cda_simthesis-fetch-boston/1.0"
DEFAULT_PIN = Path(__file__).resolve().parent / "boston_fixtures_pin.json"
DEFAULT_DEST = Path("fixtures/causal-open-data")
FORBIDDEN_SUBSTRINGS = ("hcp", "connectome", "jang")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pin(path: Path) -> dict:
    pin = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pin, dict):
        raise ValueError(f"pin file is not a JSON object: {path}")
    commit = pin.get("commit")
    if not isinstance(commit, str) or len(commit) < 7:
        raise ValueError("pin is missing a commit SHA")
    files = pin.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("pin has no files list")
    return pin


def dataset_entries(pin: dict) -> list[dict]:
    """Return pin files that must be SHA256-verified on disk."""
    entries = []
    for item in pin["files"]:
        if not isinstance(item, dict):
            raise ValueError(f"pin file entry is not an object: {item!r}")
        rel = item.get("path")
        digest = item.get("sha256")
        if not rel or not digest:
            raise ValueError(f"pin entry missing path/sha256: {item!r}")
        rel_l = str(rel).lower()
        if any(bad in rel_l for bad in FORBIDDEN_SUBSTRINGS):
            raise ValueError(f"refusing non-Boston / restricted path: {rel}")
        entries.append(item)
    return entries


def raw_url(repo: str, commit: str, rel: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def download(url: str, token: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"GET {url} failed: HTTP {exc.code} {exc.reason}") from exc


def verify_bytes(rel: str, data: bytes, expected: str) -> str:
    got = sha256_bytes(data)
    if got != expected:
        raise SystemExit(f"sha256 mismatch for {rel}: {got} != {expected}")
    return got


def fixture_dest(dest: Path, rel: str) -> Path:
    return dest / rel


def verify_existing(dest: Path, entries: list[dict]) -> list[str]:
    missing = []
    for item in entries:
        rel = item["path"]
        path = fixture_dest(dest, rel)
        if not path.is_file():
            missing.append(rel)
            continue
        verify_bytes(rel, path.read_bytes(), item["sha256"])
        print(f"ok  {rel}")
    if missing:
        raise SystemExit(f"missing {len(missing)} file(s): {missing}")
    return [item["path"] for item in entries]


def fetch_entries(
    pin: dict,
    dest: Path,
    entries: list[dict],
    *,
    token: str | None,
) -> list[str]:
    repo = pin["repo"]
    commit = pin["commit"]
    dest.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for item in entries:
        rel = item["path"]
        data = download(raw_url(repo, commit, rel), token)
        verify_bytes(rel, data, item["sha256"])
        out = fixture_dest(dest, rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        fetched.append(rel)
        print(f"ok  {rel}")
    meta = {
        "repo": repo,
        "commit": commit,
        "source_url": pin.get("source_url"),
        "license": pin.get("license"),
        "files": fetched,
    }
    (dest / "FETCH_META.json").write_text(json.dumps(meta, indent=2) + "\n")
    return fetched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin",
        type=Path,
        default=DEFAULT_PIN,
        help="JSON pin (repo, commit, files + sha256)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Directory that mirrors upstream paths (default fixtures/causal-open-data)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check SHA256 of already-downloaded files; do not download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pin commit and paths; do not download.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pin = load_pin(args.pin)
    entries = dataset_entries(pin)
    repo = pin["repo"]
    commit = pin["commit"]
    print(f"pin  {repo}@{commit}  ({len(entries)} files)", file=sys.stderr)

    if args.dry_run:
        for item in entries:
            print(f"{item['sha256']}  {item['path']}")
        return 0

    if args.verify_only:
        verify_existing(args.dest, entries)
        print(f"verified {len(entries)} files in {args.dest}")
        return 0

    token = github_token()
    if token is None:
        print(
            "note: no GITHUB_TOKEN/GH_TOKEN; public raw.githubusercontent.com works",
            file=sys.stderr,
        )
    fetched = fetch_entries(pin, args.dest, entries, token=token)
    print(f"fetched {len(fetched)} files from {repo}@{commit} -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
