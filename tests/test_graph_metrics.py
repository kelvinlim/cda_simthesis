"""Unit tests for ground-truth parsing and Dice / oriented metrics."""

from pathlib import Path

from tools.graph_metrics import (
    compare_graphs,
    dice_coefficient,
    directed_pairs,
    oriented_rates,
    parse_graph_text,
    parse_picause_graph,
    possible_directed_non_edges,
    skeleton_pairs,
)


PICAUSE_SAMPLE = """Structural Equation DAG Model
4 Vertices and 3 Edges
Seed: 1
Vertices:
['x_1', 'x_2', 'x_3', 'x_4']

Edges:
	x_1 --> x_2
	x_1 --> x_3
	x_2 --> x_4
Topological Order:
	x_1 < x_2 < x_3 < x_4
Model:
	:x_1 = (0.316)e
"""


def test_dice_empty_and_identical():
    assert dice_coefficient(set(), set()) == 0.0
    assert dice_coefficient({"a"}, {"a"}) == 1.0
    assert dice_coefficient({"a", "b"}, {"b", "c"}) == 0.5


def test_parse_picause_text():
    edges = parse_graph_text(PICAUSE_SAMPLE)
    assert edges == ["x_1 --> x_2", "x_1 --> x_3", "x_2 --> x_4"]
    assert directed_pairs(edges) == {("x_1", "x_2"), ("x_1", "x_3"), ("x_2", "x_4")}
    assert skeleton_pairs(edges) == {
        ("x_1", "x_2"),
        ("x_1", "x_3"),
        ("x_2", "x_4"),
    }


def test_parse_picause_file(tmp_path: Path):
    path = tmp_path / "graph.txt"
    path.write_text(PICAUSE_SAMPLE, encoding="utf-8")
    assert parse_picause_graph(path) == ["x_1 --> x_2", "x_1 --> x_3", "x_2 --> x_4"]


def test_parse_picause_packed_edges():
    """picause packs up to five edges per line (see pairlist2arrowstr)."""
    text = """Structural Equation DAG Model
8 Vertices and 8 Edges
Edges:
	x_8 --> x_6	x_8 --> x_4	x_2 --> x_7	x_8 --> x_5	x_3 --> x_5
	x_3 --> x_7	x_3 --> x_4	x_5 --> x_6
Topological Order:
	x_1 < x_2
"""
    edges = parse_graph_text(text)
    assert len(edges) == 8
    assert "x_8 --> x_6" in edges
    assert "x_5 --> x_6" in edges


def test_parse_tetrad_numbered_edges():
    text = """Graph Edges:
1. x_1 --> x_2
2. x_3 --- x_4
Graph Attributes:
"""
    assert parse_graph_text(text) == ["x_1 --> x_2", "x_3 --- x_4"]


def test_compare_perfect_recovery():
    true = ["x_1 --> x_2", "x_2 --> x_3"]
    metrics = compare_graphs(true, true)
    assert metrics["dice_skeleton"] == 1.0
    assert metrics["dice_directed"] == 1.0
    assert metrics["oriented_tp"] == 2
    assert metrics["oriented_fp"] == 0
    assert metrics["oriented_fn"] == 0
    assert metrics["oriented_tpr"] == 1.0
    assert metrics["oriented_fpr"] == 0.0
    assert metrics["n_nodes"] == 3
    assert metrics["n_possible_non_edges"] == 3 * 2 - 2


def test_compare_reversed_and_missing():
    true = ["x_1 --> x_2", "x_2 --> x_3"]
    recovered = ["x_2 --> x_1", "x_3 --- x_4"]
    metrics = compare_graphs(true, recovered, full_sample_edges=true)
    assert metrics["oriented_tp"] == 0
    assert metrics["oriented_fn"] == 2
    assert metrics["dice_vs_full_skeleton"] == metrics["dice_skeleton"]
    assert 0.0 <= metrics["dice_skeleton"] <= 1.0


def test_oriented_fpr_uses_possible_directed_non_edges():
    """FPR = oriented_fp / (n(n-1) - n_true_directed), including isolated nodes."""
    true = ["x_1 --> x_2", "x_2 --> x_3"]
    recovered = ["x_2 --> x_1", "x_3 --> x_4"]
    metrics = compare_graphs(true, recovered, n_nodes=5)
    # 5 nodes → 20 directed pairs; 2 true edges → 18 non-edges.
    assert possible_directed_non_edges(5, 2) == 18
    # reverse of x_1-->x_2 plus novel x_3-->x_4
    assert metrics["oriented_fp"] == 2
    assert metrics["oriented_fpr"] == 2 / 18
    assert metrics["oriented_tpr"] == 0.0
    assert metrics["n_nodes"] == 5
    rates = oriented_rates(oriented_tp=2, oriented_fp=0, n_nodes=4, n_true_directed=2)
    assert rates["oriented_tpr"] == 1.0
    assert rates["oriented_fpr"] == 0.0
    assert rates["n_possible_non_edges"] == 4 * 3 - 2
