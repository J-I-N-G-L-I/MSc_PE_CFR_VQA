"""F1 score utilities for macro/micro aggregation and question-type splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - handled via test skips
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - guarded by tests
        raise ImportError("torch is required for F1 metric computation")


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require_torch()
    if pred.ndim != 1 or target.ndim != 1:
        raise ValueError("pred and target must be 1-D class indices")
    if pred.numel() != target.numel():
        raise ValueError("pred and target must have the same number of elements")
    num_classes = (
        int(max(pred.max().item(), target.max().item())) + 1 if pred.numel() else 0
    )
    if num_classes == 0:
        return torch.zeros((0, 0), dtype=torch.long)
    indices = target * num_classes + pred
    conf = torch.bincount(indices, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)


def _prepare_predictions(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must be 2-D tensors")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    pred_ids = pred.argmax(dim=1)
    target_ids = target.argmax(dim=1)
    return pred_ids.cpu(), target_ids.cpu()


def f1_macro_micro(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Compute macro and micro F1 scores from logits and one-hot labels."""

    _require_torch()
    pred_ids, target_ids = _prepare_predictions(pred, target)
    if pred_ids.numel() == 0:
        return 0.0, 0.0
    conf = _confusion_matrix(pred_ids, target_ids)
    if conf.numel() == 0:
        return 0.0, 0.0
    tp = conf.diag().to(torch.float32)
    fp = conf.sum(dim=0).to(torch.float32) - tp
    fn = conf.sum(dim=1).to(torch.float32) - tp

    denom = 2 * tp + fp + fn
    class_f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    support = conf.sum(dim=1)
    macro = class_f1[support > 0].mean().item() if (support > 0).any() else 0.0

    tp_total = tp.sum()
    fp_total = fp.sum()
    fn_total = fn.sum()
    micro_denom = 2 * tp_total + fp_total + fn_total
    micro = (2 * tp_total / micro_denom).item() if micro_denom > 0 else 0.0
    return macro, micro


def f1_by_type(
    pred: torch.Tensor, target: torch.Tensor, qtypes: Sequence[str]
) -> Dict[str, float]:
    """Compute F1 scores grouped by question type."""

    _require_torch()
    pred_ids, target_ids = _prepare_predictions(pred, target)
    if pred_ids.numel() != len(qtypes):
        raise ValueError("Number of question types must match batch size")
    buckets: Dict[str, Dict[str, torch.Tensor]] = defaultdict(
        lambda: {"pred": [], "target": []}
    )
    for idx, qtype in enumerate(qtypes):
        buckets[qtype]["pred"].append(pred_ids[idx])
        buckets[qtype]["target"].append(target_ids[idx])

    results: Dict[str, float] = {}
    for qtype, tensors in buckets.items():
        pred_tensor = (
            torch.stack(tensors["pred"]) if tensors["pred"] else torch.empty(0)
        )
        target_tensor = (
            torch.stack(tensors["target"]) if tensors["target"] else torch.empty(0)
        )
        if pred_tensor.numel() == 0:
            results[qtype] = 0.0
            continue
        conf = _confusion_matrix(pred_tensor, target_tensor)
        if conf.numel() == 0:
            results[qtype] = 0.0
            continue
        tp = conf.diag().to(torch.float32)
        fp = conf.sum(dim=0).to(torch.float32) - tp
        fn = conf.sum(dim=1).to(torch.float32) - tp
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
        support = conf.sum(dim=1)
        results[qtype] = f1[support > 0].mean().item() if (support > 0).any() else 0.0
    return results
