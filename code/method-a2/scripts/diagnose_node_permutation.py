"""用具体数值确定 Z 与 response 在节点置换下的索引对应关系。"""

import sys
from pathlib import Path

import torch

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.model.marker_response_predictor import (  # noqa: E402
    MarkerResponsePredictor,
    build_candidate_markers,
)


def main() -> int:
    """列出置换前后每个位置的对应关系。"""
    torch.manual_seed(0)
    model = MarkerResponsePredictor(
        4, 6, 6, temporal_hidden=8, shared_dim=8, decoder_hidden=8, dropout=0.0
    ).eval()
    edges = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    edge_attr = torch.ones(6, 5)
    x = torch.randn(1, 4, 6, 6)
    markers, types = build_candidate_markers(4)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    remapped = inverse[edges]
    with torch.no_grad():
        original = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
        permuted = model(
            x[:, permutation], remapped, edge_attr,
            candidate_markers=markers[:, permutation], candidate_types=types,
        )

    def find_position(source: torch.Tensor, value: torch.Tensor, axis: int) -> list[int]:
        """找出 value 的每个切片在 source 中对应位置。"""
        positions = []
        for index in range(value.shape[axis]):
            target = value.select(axis, index)
            matched = [
                candidate for candidate in range(source.shape[axis])
                if torch.allclose(source.select(axis, candidate), target, atol=1e-5)
            ]
            positions.append(matched)
        return positions

    print("Z 各位置在原输出中的对应：", find_position(original["Z"], permuted["Z"], 1))
    print("response 各候选节点位置对应（候选 0）：",
          find_position(original["response"][:, 0], permuted["response"][:, 0], 1))
    print("permutation =", permutation.tolist(), " inverse =", inverse.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
