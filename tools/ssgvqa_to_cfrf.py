"""Utilities to bridge SSG-VQA assets to the CFRF cache layout."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - handled in tests via skip
    h5py = None

JsonDict = MutableMapping[str, object]


def _normalise_token(token: str) -> str:
    return token.lower().strip()


def _load_questions(source: Path) -> List[JsonDict]:
    with source.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, Mapping) and "questions" in payload:
        questions = payload["questions"]
    elif isinstance(payload, Sequence):
        questions = payload
    else:
        raise ValueError(f"Unrecognised QA format in {source}")
    result: List[JsonDict] = []
    for item in questions:
        if not isinstance(item, Mapping):
            raise ValueError(f"Question entry must be a mapping, got {type(item)!r}")
        if (
            "question_id" not in item
            or "image_id" not in item
            or "question" not in item
        ):
            raise KeyError(f"Missing keys in question entry from {source}: {item}")
        result.append(dict(item))
    return result


def _resolve_scene_path(root: Path, image_id: str) -> Path | None:
    candidates = [root / f"{image_id}.json", root / image_id / "scene.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _scene_graph_vocabulary(
    scene_graph: Mapping[str, object],
) -> Tuple[Counter, List[str]]:
    vocab: Counter = Counter()
    attr_phrases: List[str] = []
    objects = scene_graph.get("objects", [])
    if isinstance(objects, Mapping):
        objects = list(objects.values())
    for obj in objects or []:
        if not isinstance(obj, Mapping):
            continue
        name = obj.get("name")
        if isinstance(name, str) and name:
            vocab.update(_normalise_token(name).split())
        attributes = obj.get("attributes")
        if isinstance(attributes, Mapping):
            attributes = list(attributes.values())
        for attr in attributes or []:
            if isinstance(attr, str) and attr:
                attr_clean = _normalise_token(attr)
                vocab.update(attr_clean.split())
                if isinstance(name, str) and name:
                    phrase = f"{attr_clean} {_normalise_token(name)}".strip()
                else:
                    phrase = attr_clean
                attr_phrases.append(phrase)
    relations = scene_graph.get("relations", [])
    if isinstance(relations, Mapping):
        relations = list(relations.values())
    for rel in relations or []:
        if not isinstance(rel, Mapping):
            continue
        predicate = rel.get("predicate")
        if isinstance(predicate, str) and predicate:
            vocab.update(_normalise_token(predicate).split())
        for role in ("subject", "object"):
            role_val = rel.get(role)
            if isinstance(role_val, str) and role_val:
                vocab.update(_normalise_token(role_val).split())
    return vocab, attr_phrases


def build_hdf5(features_dir: str, yolo_boxes_dir: str, out_h5: str) -> Dict[str, int]:
    """Create an HDF5 cache with features, spatial data and positional boxes.

    Returns a mapping from image_id to positional index used for pos_boxes.
    """

    if h5py is None:  # pragma: no cover - environment without h5py
        raise ImportError("h5py is required to build HDF5 caches")

    feature_root = Path(features_dir)
    boxes_root = Path(yolo_boxes_dir)
    if not feature_root.exists() or not boxes_root.exists():
        raise FileNotFoundError("Feature or boxes directory does not exist")

    image_features: List[np.ndarray] = []
    spatial_features: List[np.ndarray] = []
    pos_boxes: List[Tuple[int, int]] = []
    image_ids: List[str] = []
    cursor = 0

    feature_files = sorted(
        [p for p in feature_root.glob("**/*") if p.suffix.lower() in {".npz", ".npy"}]
    )
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {feature_root}")

    for feat_path in feature_files:
        rel = feat_path.relative_to(feature_root)
        image_id = rel.with_suffix("").as_posix()
        box_path = boxes_root / rel.with_suffix(".json")
        if not box_path.exists():
            raise FileNotFoundError(f"Missing YOLO boxes for {image_id} at {box_path}")
        if feat_path.suffix.lower() == ".npz":
            with np.load(feat_path) as data:
                if "features" not in data:
                    raise KeyError(
                        f"NPZ file {feat_path} must contain a 'features' array"
                    )
                features = data["features"].astype(np.float32)
        else:
            features = np.load(feat_path).astype(np.float32)
        with box_path.open("r", encoding="utf-8") as fp:
            box_payload = json.load(fp)
        boxes = box_payload.get("boxes")
        width = box_payload.get("width")
        height = box_payload.get("height")
        if boxes is None or width is None or height is None:
            raise KeyError(
                f"Boxes file {box_path} must define 'boxes', 'width' and 'height'"
            )
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
            raise ValueError(f"Boxes in {box_path} must have shape (num_boxes, 4)")
        if features.shape[0] != boxes_arr.shape[0]:
            raise ValueError(
                f"Feature/box mismatch for {image_id}: {features.shape[0]} vs {boxes_arr.shape[0]}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size ({width}, {height}) for {image_id}")
        width = float(width)
        height = float(height)
        x1, y1, x2, y2 = boxes_arr.T
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
        ).astype(np.float32)
        spatial = np.clip(spatial, 0.0, 1.0)

        start = cursor
        cursor += features.shape[0]
        pos_boxes.append((start, cursor))
        image_features.append(features)
        spatial_features.append(spatial)
        image_ids.append(image_id)

    image_features_arr = np.concatenate(image_features, axis=0)
    spatial_features_arr = np.concatenate(spatial_features, axis=0)
    pos_boxes_arr = np.asarray(pos_boxes, dtype=np.int32)

    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as fp:
        fp.create_dataset("image_features", data=image_features_arr, dtype="float32")
        fp.create_dataset(
            "spatial_features", data=spatial_features_arr, dtype="float32"
        )
        fp.create_dataset("pos_boxes", data=pos_boxes_arr, dtype="int32")

    return {img_id: idx for idx, img_id in enumerate(image_ids)}


def build_image_data(meta_path: str, out_json: str) -> None:
    meta_source = Path(meta_path)
    if not meta_source.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_source}")
    with meta_source.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    image_dict: Dict[str, Dict[str, int]] = {}
    if isinstance(payload, Mapping):
        items = payload.items()
    elif isinstance(payload, Sequence):
        items = []
        for item in payload:
            if isinstance(item, Mapping):
                key = str(item.get("image_id") or item.get("id"))
                items.append((key, item))
    else:
        raise ValueError("Unsupported metadata format")

    for key, value in items:
        if not key:
            continue
        if not isinstance(value, Mapping):
            continue
        width = value.get("width")
        height = value.get("height")
        if width is None or height is None:
            continue
        image_dict[str(key)] = {"width": int(width), "height": int(height)}

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(image_dict, fp, indent=2, sort_keys=True)


def build_qa_entities(
    qa_dir: str, scene_graph_dir: str, out_json: str
) -> List[JsonDict]:
    qa_path = Path(qa_dir)
    if qa_path.is_dir():
        raise ValueError("qa_dir must point to a QA json file for a specific split")
    questions = _load_questions(qa_path)
    scene_root = Path(scene_graph_dir)
    results: List[JsonDict] = []
    missing_scene: set[str] = set()

    scene_cache: Dict[str, Tuple[Counter, List[str]]] = {}

    for question in sorted(questions, key=lambda x: x["question_id"]):
        image_id = str(question["image_id"])
        scene_tuple = scene_cache.get(image_id)
        if scene_tuple is None:
            scene_path = _resolve_scene_path(scene_root, image_id)
            if scene_path and scene_path.exists():
                with scene_path.open("r", encoding="utf-8") as fp:
                    scene_data = json.load(fp)
                scene_tuple = _scene_graph_vocabulary(scene_data)
            else:
                missing_scene.add(image_id)
                scene_tuple = (Counter(), [])
            scene_cache[image_id] = scene_tuple
        vocab, _ = scene_tuple
        question_tokens = [
            _normalise_token(tok)
            for tok in question["question"].replace("?", "").split()
        ]
        entities = sorted({tok for tok in question_tokens if tok and tok in vocab})
        qtype = question.get("question_type") or question.get("type") or "unknown"
        results.append(
            {
                "image_id": image_id,
                "question": question["question"],
                "question_id": question["question_id"],
                "question_type": qtype,
                "entities": entities,
            }
        )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump({"questions": results}, fp, indent=2)
    if missing_scene:
        skip_path = out_path.with_name(out_path.stem + "_missing_scene.json")
        with skip_path.open("w", encoding="utf-8") as fp:
            json.dump(sorted(missing_scene), fp, indent=2)
    return results


def _normalise_answer(answer: str) -> str:
    return " ".join(_normalise_token(answer).split())


def build_answer_tables(
    qa_dir: str, out_dir: str
) -> Tuple[Dict[str, int], List[str], Dict[str, List[JsonDict]]]:
    qa_root = Path(qa_dir)
    if not qa_root.exists():
        raise FileNotFoundError(f"QA directory not found: {qa_root}")
    if not qa_root.is_dir():
        raise ValueError("qa_dir must be a directory containing split json files")

    questions_by_split: Dict[str, List[JsonDict]] = {}
    answers: List[str] = []
    for qa_file in sorted(qa_root.glob("*.json")):
        split = qa_file.stem
        questions = _load_questions(qa_file)
        questions_by_split[split] = questions
        for question in questions:
            if "answer" not in question or question["answer"] in (None, ""):
                raise ValueError(
                    f"Missing answer for question {question['question_id']}"
                )
            answers.append(str(question["answer"]))

    if not answers:
        raise ValueError("No answers found to build answer tables")

    normalised_answers = sorted({_normalise_answer(ans) for ans in answers})
    ans2label = {ans: idx for idx, ans in enumerate(normalised_answers)}
    label2ans = list(normalised_answers)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "ans2label.pkl").open("wb") as fp:
        import pickle

        pickle.dump(ans2label, fp)
    with (out_path / "label2ans.pkl").open("wb") as fp:
        import pickle

        pickle.dump(label2ans, fp)

    for split, questions in questions_by_split.items():
        targets = []
        for question in sorted(questions, key=lambda x: x["question_id"]):
            ans_norm = _normalise_answer(str(question["answer"]))
            ans_idx = ans2label[ans_norm]
            targets.append(
                {
                    "image_id": str(question["image_id"]),
                    "question_id": question["question_id"],
                    "labels": [ans_idx],
                    "scores": [1.0],
                }
            )
        cache_dir = out_path
        with (cache_dir / f"{split}_target.pkl").open("wb") as fp:
            import pickle

            pickle.dump(targets, fp)

    return ans2label, label2ans, questions_by_split


def build_stat_attr_words(
    scene_graph_dir: str,
    out_stat_json: str,
    out_attr_json: str,
    topk: int = 30,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    scene_root = Path(scene_graph_dir)
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene graph directory not found: {scene_root}")

    stat_words: Dict[str, str] = {}
    attr_words: Dict[str, List[str]] = {}
    stat_skip: List[str] = []
    attr_skip: List[str] = []

    for scene_path in sorted(scene_root.glob("*.json")):
        image_id = scene_path.stem
        with scene_path.open("r", encoding="utf-8") as fp:
            scene_data = json.load(fp)
        vocab, attr_phrases = _scene_graph_vocabulary(scene_data)
        if vocab:
            most_common = [word for word, _ in vocab.most_common(topk)]
            stat_words[image_id] = ",".join(most_common)
        else:
            stat_skip.append(image_id)
        if attr_phrases:
            attr_words[image_id] = attr_phrases[:topk]
        else:
            attr_skip.append(image_id)

    out_stat_path = Path(out_stat_json)
    out_stat_path.parent.mkdir(parents=True, exist_ok=True)
    with out_stat_path.open("w", encoding="utf-8") as fp:
        json.dump(stat_words, fp, indent=2, sort_keys=True)
    stat_skip_path = out_stat_path.with_name(
        out_stat_path.stem.replace("_stats_words", "_stats_skip_imgid") + ".json"
    )
    with stat_skip_path.open("w", encoding="utf-8") as fp:
        json.dump(stat_skip, fp, indent=2)

    out_attr_path = Path(out_attr_json)
    with out_attr_path.open("w", encoding="utf-8") as fp:
        json.dump(attr_words, fp, indent=2, sort_keys=True)
    attr_skip_path = out_attr_path.with_name(
        out_attr_path.stem.replace("_attr_words_non_plural_words", "_attr_skip_imgid")
        + ".json"
    )
    with attr_skip_path.open("w", encoding="utf-8") as fp:
        json.dump(attr_skip, fp, indent=2)

    return stat_words, attr_words


def _write_imgid2idx(mapping: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        import pickle

        pickle.dump(mapping, fp)


def run_pipeline(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    build_image_data(args.meta_path, out_root / "image_data.json")
    ans2label, label2ans, questions_by_split = build_answer_tables(
        args.qa_dir, out_root / "cache"
    )

    for split, questions in questions_by_split.items():
        features_split = Path(args.features_dir) / split
        boxes_split = Path(args.yolo_boxes_dir) / split
        h5_name = "ori_train.hdf5" if split == "train" else f"{split}.hdf5"
        img_mapping = build_hdf5(
            str(features_split), str(boxes_split), str(out_root / h5_name)
        )
        _write_imgid2idx(img_mapping, out_root / f"{split}_imgid2idx.pkl")

        sg_split = Path(args.scene_graph_dir) / split
        build_qa_entities(
            str(Path(args.qa_dir) / f"{split}.json"),
            str(sg_split),
            str(out_root / f"gqa_{split}_questions_entities.json"),
        )
        build_stat_attr_words(
            str(sg_split),
            str(out_root / f"{split}_{args.topk}_stats_words.json"),
            str(out_root / f"{split}_attr_words_non_plural_words.json"),
            topk=args.topk,
        )

    # Ensure answer tables are serialised (already handled in build_answer_tables)
    # Additional metadata can be added here if needed.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SSG-VQA assets to CFRF caches"
    )
    parser.add_argument(
        "--qa_dir", required=True, help="Directory containing split QA json files"
    )
    parser.add_argument(
        "--scene_graph_dir", required=True, help="Directory with scene graphs per split"
    )
    parser.add_argument(
        "--features_dir",
        required=True,
        help="Directory with ROI feature npz/npy files per split",
    )
    parser.add_argument(
        "--yolo_boxes_dir",
        required=True,
        help="Directory with YOLO box json files per split",
    )
    parser.add_argument("--meta_path", required=True, help="Image metadata json file")
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for CFRF caches"
    )
    parser.add_argument(
        "--topk", type=int, default=30, help="Number of statistical words per image"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_pipeline(parse_args())
