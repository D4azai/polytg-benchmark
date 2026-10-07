import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polytg.representations.graph import (  # noqa: E402
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    smiles_to_graph,
)


def test_graph_conversion_shapes_and_bidirectional_edges() -> None:
    graph = smiles_to_graph("CCO", target=150.0, row_index=7)
    assert graph.x.shape == (3, NODE_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, EDGE_FEATURE_DIM)
    assert graph.y.item() == pytest.approx(150.0)
    assert graph.sample_id.item() == 7
    directed_edges = {tuple(edge) for edge in graph.edge_index.t().tolist()}
    assert (0, 1) in directed_edges and (1, 0) in directed_edges
