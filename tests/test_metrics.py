import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import f1_by_type, f1_macro_micro

torch = pytest.importorskip("torch")


def test_f1_macro_micro_perfect_prediction():
    logits = torch.tensor([[5.0, 1.0], [0.1, 2.5]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    macro, micro = f1_macro_micro(logits, target)
    assert macro == pytest.approx(1.0)
    assert micro == pytest.approx(1.0)


def test_f1_by_type_handles_mixed_results():
    logits = torch.tensor([[3.0, 0.5], [1.0, 2.0], [0.1, 0.2]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    types = ["binary", "count", "binary"]
    scores = f1_by_type(logits, target, types)
    assert set(scores.keys()) == {"binary", "count"}
    assert scores["binary"] < 1.0
    assert 0.0 <= scores["count"] <= 1.0
