"""RDKit-to-PyTorch-Geometric molecular graph conversion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rdkit import Chem
from torch_geometric.data import Data

from polytg.data.validation import parse_smiles

NODE_FEATURE_DIM = 23
EDGE_FEATURE_DIM = 7


def _one_hot(value: int, choices: int) -> list[float]:
    index = min(max(int(value), 0), choices - 1)
    return [float(position == index) for position in range(choices)]


def atom_features(atom: Chem.Atom) -> list[float]:
    hybridizations = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    )
    hybridization = atom.GetHybridization()
    hybrid_index = hybridizations.index(hybridization) if hybridization in hybridizations else 5
    return (
        [atom.GetAtomicNum() / 100.0]
        + _one_hot(atom.GetDegree(), 7)
        + [float(atom.GetFormalCharge()), float(atom.GetIsAromatic())]
        + _one_hot(hybrid_index, 6)
        + _one_hot(atom.GetTotalNumHs(), 6)
        + [float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_types = (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    bond_type = bond.GetBondType()
    type_features = [float(bond_type == candidate) for candidate in bond_types]
    return type_features + [
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str, target: float | None = None, row_index: int = -1) -> Data:
    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot convert invalid SMILES to graph: {smiles!r}")
    nodes = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_indices: list[list[int]] = []
    edges: list[list[float]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_indices.extend(([begin, end], [end, begin]))
        edges.extend((features, features))
    edge_index = (
        torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        if edge_indices
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edges, dtype=torch.float32)
        if edges
        else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    )
    data = Data(
        x=torch.tensor(nodes, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        # PyG increments attributes containing ``index`` while batching.  Keep
        # the source-row identifier under a neutral name so it remains stable.
        sample_id=torch.tensor([row_index], dtype=torch.long),
    )
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float32)
    return data


def molecules_to_graphs(
    smiles: Sequence[str], targets: Sequence[float] | None = None
) -> tuple[list[Data], list[int]]:
    if targets is not None and len(smiles) != len(targets):
        raise ValueError("SMILES and target sequences must have equal lengths")
    graphs: list[Data] = []
    failures: list[int] = []
    for index, value in enumerate(smiles):
        try:
            target = None if targets is None else float(targets[index])
            graphs.append(smiles_to_graph(value, target=target, row_index=index))
        except ValueError:
            failures.append(index)
    return graphs, failures
