import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polytg.models.gcn import GCNRegressor  # noqa: E402
from polytg.models.gin import GINRegressor  # noqa: E402


@pytest.mark.parametrize("model_type", [GINRegressor, GCNRegressor])
def test_graph_models_produce_one_prediction_per_graph(model_type) -> None:
    from torch_geometric.data import Batch, Data

    def graph(nodes: int) -> Data:
        pairs = [[index, index + 1] for index in range(nodes - 1)]
        directed = pairs + [[end, begin] for begin, end in pairs]
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        return Data(
            x=torch.randn(nodes, 23),
            edge_index=edge_index,
            edge_attr=torch.randn(len(directed), 7),
        )

    batch = Batch.from_data_list([graph(3), graph(6)])
    model = model_type(
        node_dim=23,
        edge_dim=7,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
