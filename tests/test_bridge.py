import argparse
import json
import pickle
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.ssgvqa_to_cfrf import build_answer_tables, run_pipeline  # noqa: E402
from src.FFOE.dataset import Dictionary, GQAFeatureDataset  # noqa: E402


@pytest.fixture()
def sample_ssg(tmp_path):
    splits = ["train", "val"]
    features_root = tmp_path / "features"
    boxes_root = tmp_path / "boxes"
    qa_root = tmp_path / "qa"
    sg_root = tmp_path / "scene"
    meta_entries = []

    vocab_terms = set()

    for split in splits:
        for root in (features_root / split, boxes_root / split, sg_root / split):
            root.mkdir(parents=True, exist_ok=True)
        questions = []
        for idx in range(2):
            image_id = f"{split}_img{idx}"
            num_boxes = idx + 1
            features = np.full((num_boxes, 4), fill_value=idx + 1, dtype=np.float32)
            np.savez(features_root / split / f"{image_id}.npz", features=features)
            width, height = 640, 480
            boxes = []
            for box_idx in range(num_boxes):
                x1 = 10 * (box_idx + 1)
                y1 = 5 * (box_idx + 1)
                x2 = x1 + 20
                y2 = y1 + 10
                boxes.append([x1, y1, x2, y2])
            with (boxes_root / split / f"{image_id}.json").open(
                "w", encoding="utf-8"
            ) as fp:
                json.dump({"boxes": boxes, "width": width, "height": height}, fp)
            scene = {
                "objects": [
                    {"name": "Grasper", "attributes": ["metal"]},
                    {"name": "liver", "attributes": ["healthy"]},
                ],
                "relations": [
                    {"subject": "grasper", "predicate": "touching", "object": "liver"}
                ],
            }
            with (sg_root / split / f"{image_id}.json").open(
                "w", encoding="utf-8"
            ) as fp:
                json.dump(scene, fp)
            question_text = "Is the grasper touching the liver?"
            vocab_terms.update(question_text.lower().replace("?", "").split())
            answer = "yes" if idx % 2 == 0 else "no"
            vocab_terms.add(answer)
            questions.append(
                {
                    "question_id": f"{image_id}_q{idx}",
                    "image_id": image_id,
                    "question": question_text,
                    "question_type": "binary",
                    "answer": answer,
                }
            )
            meta_entries.append(
                {"image_id": image_id, "width": width, "height": height}
            )
        with (qa_root / f"{split}.json").open("w", encoding="utf-8") as fp:
            json.dump({"questions": questions}, fp)

    meta_path = tmp_path / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta_entries, fp)

    dictionary_path = tmp_path / "dictionary.pkl"
    vocab_list = sorted(vocab_terms | {"metal", "healthy", "touching"})
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    with dictionary_path.open("wb") as fp:
        pickle.dump((word2idx, vocab_list), fp)

    out_root = tmp_path / "out"
    args = argparse.Namespace(
        qa_dir=str(qa_root),
        scene_graph_dir=str(sg_root),
        features_dir=str(features_root),
        yolo_boxes_dir=str(boxes_root),
        meta_path=str(meta_path),
        out_dir=str(out_root),
        topk=2,
    )
    run_pipeline(args)
    return {
        "out_root": out_root,
        "dictionary_path": dictionary_path,
        "splits": splits,
    }


def test_bridge_outputs_shapes_and_targets(sample_ssg):
    out_root = sample_ssg["out_root"]
    cache_dir = out_root / "cache"

    with (cache_dir / "ans2label.pkl").open("rb") as fp:
        ans2label = pickle.load(fp)
    with (cache_dir / "label2ans.pkl").open("rb") as fp:
        label2ans = pickle.load(fp)

    assert len(ans2label) == len(label2ans)
    assert all(label2ans[ans2label[label]] == label for label in ans2label)

    for split in sample_ssg["splits"]:
        h5_name = "ori_train.hdf5" if split == "train" else f"{split}.hdf5"
        with h5py.File(out_root / h5_name, "r") as fp:
            image_features = fp["image_features"][:]
            spatial_features = fp["spatial_features"][:]
            pos_boxes = fp["pos_boxes"][:]
        assert image_features.dtype == np.float32
        assert spatial_features.dtype == np.float32
        assert pos_boxes.dtype == np.int32
        assert image_features.shape[0] == spatial_features.shape[0]
        assert pos_boxes.shape[0] == 2
        assert pos_boxes[0, 0] == 0
        assert pos_boxes[-1, 1] == image_features.shape[0]
        assert np.all(spatial_features >= 0.0)
        assert np.all(spatial_features <= 1.0 + 1e-6)

        with (cache_dir / f"{split}_target.pkl").open("rb") as fp:
            targets = pickle.load(fp)
        assert len(targets) == 2
        for target in targets:
            assert target["scores"] == [1.0]
            label = target["labels"][0]
            assert label2ans[label] in {"yes", "no"}

        question_path = out_root / f"gqa_{split}_questions_entities.json"
        payload = json.loads(question_path.read_text(encoding="utf-8"))
        for entry in payload["questions"]:
            assert "question_type" in entry
            assert entry["entities"]

        stat_words_path = out_root / f"{split}_2_stats_words.json"
        assert stat_words_path.exists()
        attr_words_path = out_root / f"{split}_attr_words_non_plural_words.json"
        assert attr_words_path.exists()

        imgid2idx_path = out_root / f"{split}_imgid2idx.pkl"
        with imgid2idx_path.open("rb") as fp:
            mapping = pickle.load(fp)
        assert len(mapping) == 2
        assert all(isinstance(v, int) for v in mapping.values())


def test_dataset_getitem_shapes(sample_ssg):
    out_root = sample_ssg["out_root"]
    dictionary = Dictionary.load_from_file(str(sample_ssg["dictionary_path"]))
    args = argparse.Namespace(
        max_boxes=5,
        question_len=12,
        num_stat_word=2,
        use_ope=False,
        device=torch.device("cpu"),
        topk=2,
        tiny=False,
    )
    dataset = GQAFeatureDataset(
        args, "train", dictionary, dataroot=str(out_root), adaptive=True
    )
    sample = dataset[0]
    (
        features,
        spatials,
        stat_features,
        entity,
        attr_features,
        question,
        sent,
        target,
        img_id,
        ans,
    ) = sample
    assert features.shape[0] <= args.max_boxes
    assert features.shape[1] == dataset.v_dim
    assert torch.all((spatials >= 0.0) & (spatials <= 1.0))
    assert target.shape[0] == dataset.num_ans_candidates
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert isinstance(sent, str)
    assert dataset.entries[0]["question_type"] == "binary"


def test_build_answer_tables_requires_answers(tmp_path):
    qa_root = tmp_path / "qa"
    qa_root.mkdir()
    bad_question = {
        "questions": [
            {
                "question_id": "q1",
                "image_id": "img1",
                "question": "Is there an instrument?",
            }
        ]
    }
    with (qa_root / "train.json").open("w", encoding="utf-8") as fp:
        json.dump(bad_question, fp)
    with pytest.raises(ValueError):
        build_answer_tables(str(qa_root), str(tmp_path / "cache"))
