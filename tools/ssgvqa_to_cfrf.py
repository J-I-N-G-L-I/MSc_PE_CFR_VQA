"""Utilities to bridge SSG-VQA assets to the CFRF cache layout."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - handled in tests via skip
    h5py = None

JsonDict = MutableMapping[str, object]


@dataclass
class QAEntry:
    """Lightweight container for a single question/answer pair."""

    question_id: str
    question: str
    answers: List[str]
    question_type: str


@dataclass
class FrameRecord:
    """Aggregated assets for a single (video, frame) pair."""

    split: str
    video_id: str
    frame_index: int
    image_id: str
    qa_items: List[QAEntry] = field(default_factory=list)
    scene_path: Optional[Path] = None
    roi_path: Optional[Path] = None
    global_paths: List[Path] = field(default_factory=list)
    scene_vocab: Counter = field(default_factory=Counter)
    attr_phrases: List[str] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None


# -----------------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------------


def _normalise_token(token: str) -> str:
    return token.lower().strip()


def _normalise_answer(answer: str) -> str:
    return " ".join(_normalise_token(answer).split())


def _candidate_frame_stems(index: int) -> List[str]:
    """Generate plausible filename stems for a frame index."""

    candidates: List[str] = []
    neighbours = {index}
    if index > 0:
        neighbours.add(index - 1)
    neighbours.add(index + 1)
    for value in sorted(neighbours):
        for width in (6, 5, 4, 3, 2):
            candidates.append(f"{value:0{width}d}")
        candidates.append(str(value))
    # Preserve order while deduplicating
    return list(dict.fromkeys(candidates))


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------


def _resolve_video_dir(root: Path, video_id: str) -> Optional[Path]:
    candidates = [
        video_id,
        f"{video_id}_clean",
        video_id.lower(),
        f"{video_id.lower()}_clean",
    ]
    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path
    matches = sorted(root.glob(f"{video_id}*"))
    return matches[0] if matches else None


def _find_first_dataset(handle: "h5py.File") -> np.ndarray:
    for key in handle.keys():
        data = handle[key][()]
        return np.asarray(data)
    return np.asarray(handle[()])


def _load_hdf5_matrix(path: Path) -> np.ndarray:
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required to read HDF5 features")
    with h5py.File(path, "r") as fp:
        return _find_first_dataset(fp).astype(np.float32)


def _resolve_frame_file(base: Path, index: int, suffix: str) -> Tuple[Optional[Path], Optional[str]]:
    for stem in _candidate_frame_stems(index):
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return candidate, stem
    return None, None


def _question_type_from_text(question: str) -> str:
    q_lower = question.lower().strip()
    prefixes = {
        "is": "exist",
        "are": "exist",
        "does": "exist",
        "do": "exist",
        "can": "exist",
        "has": "exist",
        "have": "exist",
        "was": "exist",
        "were": "exist",
        "where": "location",
        "which": "query",
        "what": "query",
        "how many": "count",
        "count": "count",
    }
    for prefix, label in prefixes.items():
        if q_lower.startswith(prefix):
            return label
    return "other"


def _parse_answers(answer_segment: str) -> List[str]:
    answers: List[str] = []
    for raw in answer_segment.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        answers.append(_normalise_answer(candidate))
    if not answers:
        return ["unknown"]
    # Deduplicate while preserving order
    return list(dict.fromkeys(answers))


def _parse_qa_file(path: Path, image_id: str) -> List[QAEntry]:
    qa_items: List[QAEntry] = []
    with path.open("r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp.readlines() if line.strip()]
    for local_idx, line in enumerate(lines):
        parts = [segment.strip() for segment in line.split("|") if segment.strip()]
        if not parts:
            continue
        question = parts[0]
        answer_segment = parts[1] if len(parts) > 1 else "unknown"
        answers = _parse_answers(answer_segment)
        qtype = _question_type_from_text(question)
        qa_items.append(
            QAEntry(
                question_id=f"{image_id}_q{local_idx:04d}",
                question=question,
                answers=answers,
                question_type=qtype,
            )
        )
    return qa_items


def _load_scene_graph(scene_path: Path) -> Tuple[Counter, List[str], Optional[int], Optional[int]]:
    if scene_path is None or not scene_path.exists():
        return Counter(), [], None, None
    with scene_path.open("r", encoding="utf-8") as fp:
        scene_data = json.load(fp)

    if isinstance(scene_data, Mapping) and "scenes" in scene_data:
        scenes = scene_data.get("scenes")
        if isinstance(scenes, list) and scenes:
            payload = scenes[0]
        else:
            payload = scene_data
    else:
        payload = scene_data

    vocab: Counter = Counter()
    attr_phrases: List[str] = []
    width: Optional[int] = None
    height: Optional[int] = None

    objects = payload.get("objects", [])
    if isinstance(objects, Mapping):
        objects = list(objects.values())
    for obj in objects or []:
        if not isinstance(obj, Mapping):
            continue
        bbox = obj.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = max(width or 0, int(math.ceil(float(x2))))
            height = max(height or 0, int(math.ceil(float(y2))))
        for key in ("name", "component", "type", "location"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                normalised = _normalise_token(value)
                vocab.update(normalised.split())
        attributes = obj.get("attributes")
        if isinstance(attributes, Mapping):
            attributes = list(attributes.values())
        for attr in attributes or []:
            if isinstance(attr, str) and attr:
                attr_norm = _normalise_token(attr)
                vocab.update(attr_norm.split())
                component = obj.get("component") or obj.get("name")
                if isinstance(component, str) and component:
                    phrase = f"{attr_norm} {_normalise_token(component)}".strip()
                else:
                    phrase = attr_norm
                attr_phrases.append(phrase)

    relationships = payload.get("relationships")
    if isinstance(relationships, Mapping):
        for rel_type, targets in relationships.items():
            vocab.update(_normalise_token(rel_type).split())
            if isinstance(targets, Sequence):
                for target in targets:
                    if isinstance(target, str):
                        vocab.update(_normalise_token(target).split())

    info = scene_data.get("info") if isinstance(scene_data, Mapping) else None
    if isinstance(info, Mapping):
        triplets = info.get("triplet")
        if isinstance(triplets, Sequence):
            for triplet in triplets:
                if isinstance(triplet, str):
                    vocab.update(_normalise_token(triplet).replace(",", " ").split())

    if width is None or height is None:
        width = height = None

    return vocab, attr_phrases, width, height


def _determine_dimensions(
    preferred_width: Optional[int],
    preferred_height: Optional[int],
    default_width: int,
    default_height: int,
) -> Tuple[int, int]:
    width = preferred_width or default_width
    height = preferred_height or default_height
    return max(1, int(width)), max(1, int(height))


def _load_dataset(
    qa_root: Path,
    scene_root: Path,
    roi_root: Path,
    global_root: Optional[Path],
    split_config: Mapping[str, Sequence[str]],
    default_width: int,
    default_height: int,
) -> Tuple[Dict[str, List[FrameRecord]], Counter, Dict[str, Dict[str, int]]]:
    records: Dict[str, List[FrameRecord]] = {}
    answer_vocab: Counter = Counter()
    image_meta: Dict[str, Dict[str, int]] = {}

    for split, videos in split_config.items():
        split_records: List[FrameRecord] = []
        for video_id in videos:
            qa_video_dir = _resolve_video_dir(qa_root, video_id)
            if qa_video_dir is None:
                raise FileNotFoundError(f"QA directory missing for video {video_id}")
            roi_video_dir = _resolve_video_dir(roi_root, video_id)
            if roi_video_dir is None:
                raise FileNotFoundError(f"ROI directory missing for video {video_id}")
            global_video_dir = (
                _resolve_video_dir(global_root, video_id) if global_root else None
            )

            frame_files = sorted(
                [p for p in qa_video_dir.glob("*.txt") if p.is_file()],
                key=lambda p: int(p.stem),
            )
            for qa_file in frame_files:
                nominal_index = int(qa_file.stem)
                roi_base = (
                    roi_video_dir
                    / "labels"
                    / "vqa"
                    / "img_features"
                    / "roi"
                )
                roi_path, roi_stem = _resolve_frame_file(roi_base, nominal_index, ".hdf5")
                if roi_path is None or roi_stem is None:
                    continue
                image_id = f"{video_id}_{roi_stem}"
                frame_index = int(roi_stem)

                qa_items = _parse_qa_file(qa_file, image_id)
                if not qa_items:
                    continue

                global_paths: List[Path] = []
                if global_video_dir is not None:
                    global_base = global_video_dir / "vqa" / "img_features"
                    if global_base.exists():
                        for scale_dir in sorted(global_base.iterdir()):
                            if not scale_dir.is_dir():
                                continue
                            glob_path, _ = _resolve_frame_file(
                                scale_dir, frame_index, ".hdf5"
                            )
                            if glob_path is not None:
                                global_paths.append(glob_path)

                scene_candidates = [
                    scene_root / f"{video_id}_{roi_stem}.json",
                    scene_root / f"{video_id}_{nominal_index}.json",
                ]
                scene_path = next((p for p in scene_candidates if p.exists()), None)
                scene_vocab, attr_phrases, width, height = _load_scene_graph(scene_path)
                width, height = _determine_dimensions(
                    width, height, default_width, default_height
                )
                image_meta[image_id] = {"width": width, "height": height}

                for item in qa_items:
                    for answer in item.answers:
                        answer_vocab[answer] += 1

                split_records.append(
                    FrameRecord(
                        split=split,
                        video_id=video_id,
                        frame_index=frame_index,
                        image_id=image_id,
                        qa_items=qa_items,
                        scene_path=scene_path,
                        roi_path=roi_path,
                        global_paths=global_paths,
                        scene_vocab=scene_vocab,
                        attr_phrases=attr_phrases,
                        width=width,
                        height=height,
                    )
                )
        records[split] = sorted(split_records, key=lambda r: r.image_id)
    return records, answer_vocab, image_meta


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------


def _roi_to_features(
    roi_path: Path,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = _load_hdf5_matrix(roi_path)
    if data.ndim != 2 or data.shape[1] < 18:
        raise ValueError(f"Unexpected ROI feature shape {data.shape} in {roi_path}")
    coords = data[:, 14:18].astype(np.float32)
    features = data[:, 18:].astype(np.float32)
    if not np.any(features):
        spatial = np.zeros((features.shape[0], 6), dtype=np.float32)
        return features, spatial

    max_coord = float(np.max(np.abs(coords))) if coords.size else 0.0
    if max_coord <= 1.5:  # Already normalised (center_x, center_y, width, height)
        x_c, y_c, w_box, h_box = coords.T
        x1 = np.clip(x_c - w_box / 2.0, 0.0, 1.0)
        y1 = np.clip(y_c - h_box / 2.0, 0.0, 1.0)
        x2 = np.clip(x_c + w_box / 2.0, 0.0, 1.0)
        y2 = np.clip(y_c + h_box / 2.0, 0.0, 1.0)
        spatial = np.stack([x1, y1, x2, y2, w_box, h_box], axis=1)
    else:  # Pixel coordinates (x1, y1, x2, y2)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image size {width}x{height} for ROI normalisation"
            )
        x1, y1, x2, y2 = coords.T
        widths = np.clip(x2 - x1, a_min=0.0, a_max=None)
        heights = np.clip(y2 - y1, a_min=0.0, a_max=None)
        spatial = np.stack(
            [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
                widths / width,
                heights / height,
            ],
            axis=1,
        )
    return features.astype(np.float32), np.clip(spatial.astype(np.float32), 0.0, 1.0)


def _grid_spatial(index: int, grid_size: int) -> Tuple[float, float, float, float]:
    row = index // grid_size
    col = index % grid_size
    step_x = 1.0 / grid_size
    step_y = 1.0 / grid_size
    x1 = min(1.0, col * step_x)
    y1 = min(1.0, row * step_y)
    x2 = min(1.0, x1 + step_x)
    y2 = min(1.0, y1 + step_y)
    return x1, y1, x2, y2


def _load_global_features(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    feature_blocks: List[np.ndarray] = []
    spatial_blocks: List[np.ndarray] = []
    for path in paths:
        data = _load_hdf5_matrix(path)
        if data.ndim != 2:
            raise ValueError(f"Unexpected global feature shape {data.shape} in {path}")
        num_regions = data.shape[0]
        grid_size = int(round(math.sqrt(num_regions))) if num_regions > 0 else 1
        if grid_size * grid_size != num_regions or grid_size == 0:
            grid_size = 1
        spatials = []
        for idx in range(num_regions):
            x1, y1, x2, y2 = _grid_spatial(idx, grid_size)
            spatials.append([x1, y1, x2, y2, x2 - x1, y2 - y1])
        feature_blocks.append(data.astype(np.float32))
        spatial_blocks.append(np.asarray(spatials, dtype=np.float32))
    if not feature_blocks:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(spatial_blocks, axis=0),
    )


def build_hdf5(records: Sequence[FrameRecord], out_h5: Path) -> Dict[str, int]:
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required to build HDF5 caches")

    feature_segments: List[np.ndarray] = []
    spatial_segments: List[np.ndarray] = []
    pos_boxes: List[Tuple[int, int]] = []
    image_ids: List[str] = []
    cursor = 0

    for record in records:
        if record.roi_path is None:
            continue
        roi_features, roi_spatial = _roi_to_features(
            record.roi_path, record.width or 1, record.height or 1
        )
        global_features, global_spatial = _load_global_features(record.global_paths)
        if global_features.size and roi_features.size:
            features = np.concatenate([roi_features, global_features], axis=0)
            spatials = np.concatenate([roi_spatial, global_spatial], axis=0)
        elif global_features.size:
            features = global_features
            spatials = global_spatial
        else:
            features = roi_features
            spatials = roi_spatial

        if features.size == 0:
            continue

        start = cursor
        cursor += features.shape[0]
        pos_boxes.append((start, cursor))
        feature_segments.append(features)
        spatial_segments.append(spatials)
        image_ids.append(record.image_id)

    if not feature_segments:
        raise ValueError("No features found to build HDF5 cache")

    image_features_arr = np.concatenate(feature_segments, axis=0)
    spatial_features_arr = np.concatenate(spatial_segments, axis=0)
    pos_boxes_arr = np.asarray(pos_boxes, dtype=np.int32)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as fp:
        fp.create_dataset("image_features", data=image_features_arr, dtype="float32")
        fp.create_dataset("spatial_features", data=spatial_features_arr, dtype="float32")
        fp.create_dataset("pos_boxes", data=pos_boxes_arr, dtype="int32")

    return {img_id: idx for idx, img_id in enumerate(image_ids)}


# -----------------------------------------------------------------------------
# Metadata writers
# -----------------------------------------------------------------------------


def build_image_data(image_meta: Mapping[str, Dict[str, int]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(dict(sorted(image_meta.items())), fp, indent=2)


def build_qa_entities(records: Sequence[FrameRecord], out_json: Path) -> None:
    questions: List[JsonDict] = []
    missing_scene: List[str] = []
    for record in records:
        if not record.scene_vocab:
            missing_scene.append(record.image_id)
        vocab = record.scene_vocab
        for item in record.qa_items:
            tokens = {
                _normalise_token(tok)
                for tok in item.question.replace("?", "").split()
            }
            entities = sorted(tok for tok in tokens if tok and tok in vocab)
            questions.append(
                {
                    "image_id": record.image_id,
                    "question": item.question,
                    "question_id": item.question_id,
                    "question_type": item.question_type,
                    "entities": entities,
                }
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump({"questions": questions}, fp, indent=2)

    if missing_scene:
        skip_path = out_json.with_name(out_json.stem + "_missing_scene.json")
        with skip_path.open("w", encoding="utf-8") as fp:
            json.dump(sorted(set(missing_scene)), fp, indent=2)


def build_answer_tables(
    records: Mapping[str, Sequence[FrameRecord]],
    out_dir: Path,
) -> Tuple[Dict[str, int], List[str], Dict[str, List[QAEntry]]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    answers = Counter()
    questions_by_split: Dict[str, List[QAEntry]] = {}
    for split, split_records in records.items():
        entries: List[QAEntry] = []
        for record in split_records:
            entries.extend(record.qa_items)
            for item in record.qa_items:
                for answer in item.answers:
                    answers[answer] += 1
        questions_by_split[split] = entries

    if not answers:
        raise ValueError("No answers found across splits")

    sorted_answers = sorted(answers.keys())
    ans2label = {ans: idx for idx, ans in enumerate(sorted_answers)}
    label2ans = list(sorted_answers)

    import pickle

    with (out_dir / "ans2label.pkl").open("wb") as fp:
        pickle.dump(ans2label, fp)
    with (out_dir / "label2ans.pkl").open("wb") as fp:
        pickle.dump(label2ans, fp)

    for split, items in questions_by_split.items():
        targets = []
        for record in records[split]:
            for item in record.qa_items:
                labels = [ans2label[ans] for ans in item.answers]
                scores = [1.0 for _ in labels]
                targets.append(
                    {
                        "image_id": record.image_id,
                        "question_id": item.question_id,
                        "labels": labels,
                        "scores": scores,
                    }
                )
        with (out_dir / f"{split}_target.pkl").open("wb") as fp:
            pickle.dump(targets, fp)

    return ans2label, label2ans, questions_by_split


def build_stat_attr_words(
    records: Sequence[FrameRecord],
    out_stat_json: Path,
    out_attr_json: Path,
    topk: int = 30,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    stat_words: Dict[str, str] = {}
    attr_words: Dict[str, List[str]] = {}
    stat_skip: List[str] = []
    attr_skip: List[str] = []

    for record in records:
        if record.scene_vocab:
            most_common = [word for word, _ in record.scene_vocab.most_common(topk)]
            stat_words[record.image_id] = ",".join(most_common)
        else:
            stat_skip.append(record.image_id)
        if record.attr_phrases:
            attr_words[record.image_id] = record.attr_phrases[:topk]
        else:
            attr_skip.append(record.image_id)

    out_stat_json.parent.mkdir(parents=True, exist_ok=True)
    with out_stat_json.open("w", encoding="utf-8") as fp:
        json.dump(stat_words, fp, indent=2, sort_keys=True)
    stat_skip_path = out_stat_json.with_name(
        out_stat_json.stem.replace("_stats_words", "_stats_skip_imgid") + ".json"
    )
    with stat_skip_path.open("w", encoding="utf-8") as fp:
        json.dump(sorted(set(stat_skip)), fp, indent=2)

    with out_attr_json.open("w", encoding="utf-8") as fp:
        json.dump(attr_words, fp, indent=2, sort_keys=True)
    attr_skip_path = out_attr_json.with_name(
        out_attr_json.stem.replace(
            "_attr_words_non_plural_words", "_attr_skip_imgid"
        )
        + ".json"
    )
    with attr_skip_path.open("w", encoding="utf-8") as fp:
        json.dump(sorted(set(attr_skip)), fp, indent=2)

    return stat_words, attr_words


def _write_imgid2idx(mapping: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with output_path.open("wb") as fp:
        pickle.dump(mapping, fp)


# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------


def _load_split_config(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError("Split config must be a mapping of split -> [videos]")
    split_config: Dict[str, List[str]] = {}
    for split, videos in payload.items():
        if not isinstance(videos, Sequence):
            raise ValueError(f"Split '{split}' must map to a sequence of video ids")
        split_config[str(split)] = [str(video) for video in videos]
    return split_config


def run_pipeline(args: argparse.Namespace) -> None:
    qa_root = Path(args.qa_dir)
    scene_root = Path(args.scene_graph_dir)
    roi_root = Path(args.features_dir)
    global_root = Path(args.yolo_boxes_dir) if args.yolo_boxes_dir else None
    split_config = _load_split_config(Path(args.split_config))

    records, _, image_meta = _load_dataset(
        qa_root,
        scene_root,
        roi_root,
        global_root,
        split_config,
        args.default_width,
        args.default_height,
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    build_image_data(image_meta, out_root / "image_data.json")
    ans2label, label2ans, _ = build_answer_tables(records, out_root / "cache")

    for split, split_records in records.items():
        if not split_records:
            continue
        h5_name = "ori_train.hdf5" if split == "train" else f"{split}.hdf5"
        img_mapping = build_hdf5(split_records, out_root / h5_name)
        _write_imgid2idx(img_mapping, out_root / f"{split}_imgid2idx.pkl")

        build_qa_entities(
            split_records,
            out_root / f"gqa_{split}_questions_entities.json",
        )
        build_stat_attr_words(
            split_records,
            out_root / f"{split}_{args.topk}_stats_words.json",
            out_root / f"{split}_attr_words_non_plural_words.json",
            topk=args.topk,
        )

    # Write vocab just to avoid unused variable warnings
    _ = ans2label, label2ans


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SSG-VQA assets to CFRF caches",
    )
    parser.add_argument("--qa_dir", required=True, help="Directory with QA txt trees")
    parser.add_argument(
        "--scene_graph_dir", required=True, help="Directory with per-frame scene graphs"
    )
    parser.add_argument(
        "--features_dir",
        required=True,
        help="Directory with ROI HDF5 features (per video)",
    )
    parser.add_argument(
        "--yolo_boxes_dir",
        default="",
        help="Directory with global cropped features (optional)",
    )
    parser.add_argument(
        "--split_config",
        required=True,
        help="JSON mapping split name to list of video ids",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for CFRF caches"
    )
    parser.add_argument(
        "--topk", type=int, default=30, help="Number of statistical words per image"
    )
    parser.add_argument(
        "--default_width",
        type=int,
        default=400,
        help="Fallback image width when metadata is unavailable",
    )
    parser.add_argument(
        "--default_height",
        type=int,
        default=300,
        help="Fallback image height when metadata is unavailable",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_pipeline(parse_args())