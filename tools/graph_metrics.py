"""Graph comparison helpers for simulated ground-truth vs recovered edges.

Parses picause / CausalPowerAnalysis SEM ``.txt`` files written by ``simdata.py``
and computes Dice / skeleton / oriented-edge metrics used by the discovery
runner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

EDGE_OPS = ("-->", "<--", "o->", "<-o", "o-o", "---", "<->")
# picause human_readable dumps pack up to five ``src --> dest`` tokens per line
EDGE_FIND_RE = re.compile(
    r"(\S+)\s+(-->|<--|o->|<-o|o-o|---|<->)\s+(\S+)"
)
NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+\.\s+")

DIRECTED_OPS = {"-->", "o->"}
REVERSE_OPS = {"<--", "<-o"}
UNDIRECTED_OPS = {"---", "o-o", "<->"}


def dice_coefficient(set_a: set, set_b: set) -> float:
    """Sørensen–Dice coefficient; empty/empty returns 0.0."""
    intersection = len(set_a.intersection(set_b))
    denom = len(set_a) + len(set_b)
    if denom == 0:
        return 0.0
    return (2.0 * intersection) / denom


def normalize_edge_string(edge: str) -> Optional[str]:
    """Return ``src OP dest`` or None if the string is not a 3-token edge."""
    parts = str(edge).strip().split()
    if len(parts) != 3:
        return None
    src, op, dest = parts
    if op not in EDGE_OPS:
        return None
    return f"{src} {op} {dest}"


def parse_edge_strings(edges: Iterable[str]) -> list[str]:
    parsed = []
    for edge in edges:
        normalized = normalize_edge_string(edge)
        if normalized:
            parsed.append(normalized)
    return parsed


def directed_pairs(edges: Iterable[str]) -> set[tuple[str, str]]:
    """Oriented pairs. ``o->`` counts as directed; undirected types are skipped."""
    pairs: set[tuple[str, str]] = set()
    for edge in parse_edge_strings(edges):
        src, op, dest = edge.split()
        if op in DIRECTED_OPS:
            pairs.add((src, dest))
        elif op in REVERSE_OPS:
            pairs.add((dest, src))
    return pairs


def skeleton_pairs(edges: Iterable[str]) -> set[tuple[str, str]]:
    """Undirected endpoint pairs, order-normalized."""
    pairs: set[tuple[str, str]] = set()
    for edge in parse_edge_strings(edges):
        src, _op, dest = edge.split()
        pairs.add(tuple(sorted((src, dest))))
    return pairs


def node_set(edges: Iterable[str]) -> set[str]:
    nodes: set[str] = set()
    for edge in parse_edge_strings(edges):
        src, _op, dest = edge.split()
        nodes.update([src, dest])
    return nodes


def parse_picause_graph(path: str | Path) -> list[str]:
    """Extract ground-truth edges from a picause SEM ``.txt`` dump.

    ``simdata.py`` writes ``sem.__str__()``, which includes an ``Edges:``
    section of tab-indented ``x_i --> x_j`` lines.
    """
    text = Path(path).read_text(encoding="utf-8")
    return parse_graph_text(text)


def edges_from_line(line: str) -> list[str]:
    """Extract one or more ``src OP dest`` tokens from a line."""
    cleaned = NUMBERED_PREFIX_RE.sub("", line)
    return [f"{src} {op} {dest}" for src, op, dest in EDGE_FIND_RE.findall(cleaned)]


def parse_graph_text(text: str) -> list[str]:
    """Parse edges from picause SEM text or Tetrad ``Graph Edges:`` output.

    picause ``pairlist2arrowstr(..., human_readable=True)`` packs up to five
    edges on a line, separated by tabs.
    """
    edges: list[str] = []
    in_picause_edges = False
    in_tetrad_edges = False

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped == "Edges:":
            in_picause_edges = True
            in_tetrad_edges = False
            continue
        if "Graph Edges:" in stripped:
            in_tetrad_edges = True
            in_picause_edges = False
            continue
        if in_picause_edges and (
            stripped.startswith("Topological Order:")
            or stripped.startswith("Model:")
            or stripped.startswith("Iterations:")
        ):
            break
        if in_tetrad_edges and stripped.startswith("Graph Attributes:"):
            break
        if in_tetrad_edges and stripped.startswith("Graph Nodes:") and edges:
            break

        if in_picause_edges or in_tetrad_edges:
            edges.extend(edges_from_line(line))
        elif EDGE_FIND_RE.search(line):
            edges.extend(edges_from_line(line))

    return edges


def format_tetrad_graph_text(edges: Iterable[str]) -> str:
    """Render edges as Tetrad-style text for legacy ``extract_edges`` callers."""
    lines = ["Graph Edges:"]
    for i, edge in enumerate(parse_edge_strings(edges), start=1):
        lines.append(f"{i}. {edge}")
    return "\n".join(lines)


def oriented_counts(true_edges: Iterable[str], recovered_edges: Iterable[str]) -> dict[str, int]:
    """Oriented TP / FP / FN against a directed ground-truth DAG.

    True edges that appear undirected in the recovered graph count as FN
    (same convention as ``picause.oriented_confusion_matrix``).
    """
    true_dir = directed_pairs(true_edges)
    rec_dir = directed_pairs(recovered_edges)
    rec_skel = skeleton_pairs(recovered_edges)

    tp = fp = fn = 0
    matched_true: set[tuple[str, str]] = set()

    for src, dest in true_dir:
        rev = (dest, src)
        if (src, dest) in rec_dir:
            tp += 1
            matched_true.add((src, dest))
        elif rev in rec_dir:
            fp += 1
            fn += 1
            matched_true.add((src, dest))
        else:
            fn += 1

    for src, dest in rec_dir:
        if (src, dest) in true_dir:
            continue
        if (dest, src) in true_dir:
            continue
        fp += 1

    # recovered undirected true edges already counted as FN above
    _ = rec_skel
    return {
        "oriented_tp": tp,
        "oriented_fp": fp,
        "oriented_fn": fn,
    }


def compare_graphs(
    true_edges: Iterable[str],
    recovered_edges: Iterable[str],
    full_sample_edges: Optional[Iterable[str]] = None,
) -> dict:
    """Dice and oriented metrics vs ground truth (and optionally vs 100% sample)."""
    true_list = parse_edge_strings(true_edges)
    rec_list = parse_edge_strings(recovered_edges)
    true_skel = skeleton_pairs(true_list)
    rec_skel = skeleton_pairs(rec_list)
    true_dir = directed_pairs(true_list)
    rec_dir = directed_pairs(rec_list)
    oriented = oriented_counts(true_list, rec_list)

    result = {
        "n_true_edges": len(true_skel),
        "n_true_directed": len(true_dir),
        "n_recovered_edges": len(rec_skel),
        "n_recovered_directed": len(rec_dir),
        "dice_skeleton": dice_coefficient(true_skel, rec_skel),
        "dice_directed": dice_coefficient(true_dir, rec_dir),
        "dice_nodes": dice_coefficient(node_set(true_list), node_set(rec_list)),
        **oriented,
    }

    if full_sample_edges is not None:
        full_list = parse_edge_strings(full_sample_edges)
        result["dice_vs_full_skeleton"] = dice_coefficient(
            skeleton_pairs(full_list), rec_skel
        )
        result["dice_vs_full_directed"] = dice_coefficient(
            directed_pairs(full_list), rec_dir
        )
    else:
        result["dice_vs_full_skeleton"] = None
        result["dice_vs_full_directed"] = None

    return result
