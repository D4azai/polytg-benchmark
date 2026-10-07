from pathlib import Path

import pytest

from polytg.utils.config import config_hash, load_config


def test_configuration_load_and_hash_are_stable() -> None:
    config = load_config(Path("configs/baseline_rdkit.yaml"))
    assert config["data"]["smiles_column"] == "smiles"
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))


def test_ecfp_configuration_has_research_defaults() -> None:
    config = load_config(Path("configs/baseline_ecfp.yaml"))
    assert config["representation"] == {
        "name": "ecfp",
        "radius": 2,
        "n_bits": 2048,
        "include_chirality": False,
        "use_bond_types": True,
    }


def test_graph_configuration_uses_validation_early_stopping() -> None:
    config = load_config(Path("configs/gin.yaml"))
    assert config["representation"]["name"] == "molecular_graph"
    assert config["model"]["name"] == "gin"
    assert config["training"]["early_stopping_patience"] > 0
    assert config["training"]["deterministic"] is True


def test_language_model_revision_is_pinned() -> None:
    config = load_config(Path("configs/chembert.yaml"))
    revision = config["representation"]["revision"]
    assert len(revision) == 40
    assert revision != "main"


def test_default_benchmark_includes_all_primary_families() -> None:
    config = load_config(Path("configs/benchmark.yaml"))
    assert config["benchmark"]["additional_configs"] == [
        "configs/gin.yaml",
        "configs/chembert.yaml",
    ]


def test_configuration_requires_research_sections(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("experiment: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing section"):
        load_config(path)
