"""候选边全集、虚增边掩码和兼容批次的单元测试。"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.topology import (
    build_edge_universe,
    expand_undirected_mask,
    validate_edge_masks,
)
from src.gref.data import load_arrays, make_batch


def _write_minimal_dataset(directory: Path, with_universe: bool) -> None:
    directory.mkdir()
    x_obs = np.zeros((1, 2, 1, 1), dtype=np.float32)
    bank = np.zeros((1, 3, 2, 1, 1), dtype=np.float32)
    arrays = {
        "X_obs": x_obs,
        "signature_bank": bank,
        "mask": np.ones((1, 2), dtype=np.float32),
        "candidate_mask": np.ones((1, 3), dtype=np.float32),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        "edge_attr": np.ones((2, 5), dtype=np.float32),
        "edge_mask": np.ones((1, 2), dtype=np.float32),
        "group_id": np.asarray([0], dtype=np.int64),
        "y_loc": np.asarray([0], dtype=np.int64),
        "y_resist": np.asarray([1.0], dtype=np.float32),
    }
    for name, value in arrays.items():
        np.save(directory / f"{name}.npy", value)
    if with_universe:
        universe = build_edge_universe(
            arrays["edge_index"], arrays["edge_attr"],
            extra_edges=np.asarray([[0, 1]], dtype=np.int64),
        )
        for name, value in universe.items():
            np.save(directory / f"{name}.npy", np.asarray([value]) if name.startswith("edge_mask") else value)
        np.save(directory / "edge_index_universe.npy", np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64))
        np.save(directory / "edge_attr_universe.npy", np.ones((4, 5), dtype=np.float32))
        np.save(directory / "edge_pair_id.npy", np.asarray([0, 0, 1, 1], dtype=np.int64))
        np.save(directory / "edge_direction.npy", np.asarray([0, 1, 0, 1], dtype=np.int64))
        np.save(directory / "edge_type.npy", np.asarray([0, 0, 3, 3], dtype=np.int64))
        np.save(directory / "edge_attr_source.npy", np.asarray([0, 0, 2, 2], dtype=np.int64))
        np.save(directory / "edge_mask_true.npy", np.asarray([[1, 1, 0, 0]], dtype=np.float32))
        np.save(directory / "edge_mask_obs.npy", np.asarray([[1, 1, 1, 1]], dtype=np.float32))
    (directory / "meta.json").write_text(json.dumps({"n_nodes": 2}), encoding="utf-8")


def test_build_edge_universe_adds_paired_parallel_edge():
    result = build_edge_universe(
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        np.ones((2, 5), dtype=np.float32),
        extra_edges=np.asarray([[0, 1]], dtype=np.int64),
        extra_edge_attr=np.full((1, 5), 2.0, dtype=np.float32),
    )
    assert result["edge_index_universe"].shape == (4, 2)
    assert np.array_equal(result["edge_pair_id"], [0, 0, 1, 1])
    assert np.array_equal(result["edge_direction"], [0, 1, 0, 1])
    assert np.array_equal(result["edge_mask_true"], [1, 1, 0, 0])
    for name in ("edge_pair_id", "edge_direction", "edge_type", "edge_attr_source", "edge_mask_true", "edge_mask_obs"):
        assert len(result[name]) == 4
    assert np.all(result["edge_attr_universe"][2:] == 2.0)


def test_expand_and_validate_masks_require_symmetric_directions():
    pair_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    expanded = expand_undirected_mask(np.asarray([[1, 0]], dtype=np.float32), pair_id)
    assert np.array_equal(expanded, [[1, 1, 0, 0]])
    validate_edge_masks(pair_id, expanded, np.ones_like(expanded))
    with np.testing.assert_raises(ValueError):
        validate_edge_masks(pair_id, np.asarray([[1, 0, 0, 0]], dtype=np.float32), np.ones((1, 4)))


def test_old_dataset_falls_back_to_true_graph_and_universe_view(tmp_path):
    directory = tmp_path / "old"
    _write_minimal_dataset(directory, with_universe=False)
    data = load_arrays(directory)
    assert data["edge_index_universe"].shape == data["edge_index"].shape
    normalization = {"mean": np.zeros((2, 1), dtype=np.float32), "std": np.ones((2, 1), dtype=np.float32)}
    batch = make_batch(data, np.asarray([0]), normalization, torch.device("cpu"), graph_view="universe")
    assert batch["edge_index"].shape[0] == 2
    assert torch.equal(batch["edge_mask"], torch.ones((1, 2)))


def test_universe_view_uses_observation_mask_only(tmp_path):
    directory = tmp_path / "universe"
    _write_minimal_dataset(directory, with_universe=True)
    data = load_arrays(directory)
    normalization = {"mean": np.zeros((2, 1), dtype=np.float32), "std": np.ones((2, 1), dtype=np.float32)}
    batch = make_batch(data, np.asarray([0]), normalization, torch.device("cpu"), graph_view="universe")
    assert batch["edge_index"].shape[0] == 4
    assert torch.equal(batch["edge_mask"], torch.ones((1, 4)))
    assert np.array_equal(data["edge_mask_true"], [[1, 1, 0, 0]])
