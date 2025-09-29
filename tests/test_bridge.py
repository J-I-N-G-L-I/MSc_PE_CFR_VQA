import argparse
import json
import pickle
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.ssgvqa_to_cfrf import run_pipeline  # noqa: E402
from src.FFOE.dataset import Dictionary, GQAFeatureDataset  # noqa: E402


def _write_hdf5(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fp:
        fp.create_dataset("data", data=data)


def _scene_graph(objects, triplets=None):
    payload = {
        "scenes": [
            {
                "objects": objects,
                "relationships": {"related": ["grasper", "liver"]},
            }
        ],
    }
    if triplets:
        payload["info"] = {"triplet": triplets}
    return payload


@pytest.fixture()
def sample_ssg(tmp_path):
    qa_root = tmp_path / "qa"
    roi_root = tmp_path / "roi"
    global_root = tmp_path / "global"
    scene_root = tmp_path / "scene"
    for root in (qa_root, roi_root, global_root, scene_root):
        root.mkdir(parents=True, exist_ok=True)

    split_config = {"train": ["VID01"], "val": ["VID02"]}
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(split_config), encoding="utf-8")

    dictionary_terms = set()

    # Video 1 (pixel coordinates)
    qa_vid1 = qa_root / "VID01"
    qa_vid1.mkdir()
    qa_vid1.joinpath("0.txt").write_text(
        "\n".join(
            [
                "Which anatomical structures are present?|liver, gallbladder",
                "Is the grasper visible?|True",
            ]
        ),
        encoding="utf-8",
    )
    qa_vid1.joinpath("1.txt").write_text(
        "What instrument is operating?|grasper",
        encoding="utf-8",
    )

    roi_vid1 = roi_root / "VID01" / "labels" / "vqa" / "img_features" / "roi"
    roi_vid1.mkdir(parents=True)
    roi_frame0 = np.zeros((2, 530), dtype=np.float32)
    roi_frame0[:, 18:] = 1.0
    roi_frame0[0, 14:18] = [10, 20, 40, 90]
    roi_frame0[1, 14:18] = [50, 60, 120, 180]
    _write_hdf5(roi_vid1 / "000000.hdf5", roi_frame0)
    roi_frame1 = np.zeros((1, 530), dtype=np.float32)
    roi_frame1[:, 18:] = 2.0
    roi_frame1[0, 14:18] = [30, 30, 80, 120]
    _write_hdf5(roi_vid1 / "000001.hdf5", roi_frame1)

    global_vid1 = global_root / "VID01" / "vqa" / "img_features"
    _write_hdf5(global_vid1 / "1x1" / "000000.hdf5", np.ones((1, 512), dtype=np.float32))
    _write_hdf5(global_vid1 / "4x4" / "000001.hdf5", np.full((16, 512), 0.5, dtype=np.float32))

    scene_vid1_frame0 = _scene_graph(
        [
            {
                "component": "Liver",
                "bbox": [0, 0, 120, 180],
                "attributes": ["healthy"],
            },
            {
                "component": "Grasper",
                "bbox": [10, 10, 80, 150],
                "attributes": ["metal"],
            },
        ],
        triplets=["grasper,grasp,liver"],
    )
    scene_vid1_frame1 = _scene_graph(
        [
            {
                "component": "Clip",
                "bbox": [0, 0, 80, 120],
                "attributes": ["silver"],
            }
        ]
    )
    (scene_root / "VID01_000000.json").write_text(
        json.dumps(scene_vid1_frame0), encoding="utf-8"
    )
    (scene_root / "VID01_000001.json").write_text(
        json.dumps(scene_vid1_frame1), encoding="utf-8"
    )

    dictionary_terms.update(
        {
            "which",
            "anatomical",
            "structures",
            "liver",
            "gallbladder",
            "grasper",
            "visible",
            "true",
            "instrument",
            "operating",
        }
    )

    # Video 2 (normalised coordinates, stored under _clean)
    qa_vid2 = qa_root / "VID02_clean"
    qa_vid2.mkdir()
    qa_vid2.joinpath("2.txt").write_text(
        "How many clips are present?|2",
        encoding="utf-8",
    )

    roi_vid2 = roi_root / "VID02_clean" / "labels" / "vqa" / "img_features" / "roi"
    roi_vid2.mkdir(parents=True)
    roi_frame2 = np.zeros((3, 530), dtype=np.float32)
    roi_frame2[:, 18:] = 3.0
    roi_frame2[0, 14:18] = [0.5, 0.5, 0.4, 0.4]
    roi_frame2[1, 14:18] = [0.2, 0.3, 0.2, 0.2]
    roi_frame2[2, 14:18] = [0.8, 0.6, 0.1, 0.2]
    _write_hdf5(roi_vid2 / "000002.hdf5", roi_frame2)

    global_vid2 = global_root / "VID02_clean" / "vqa" / "img_features"
    _write_hdf5(global_vid2 / "1x1" / "000002.hdf5", np.full((1, 512), 0.3, dtype=np.float32))

    scene_vid2 = _scene_graph(
        [
            {
                "component": "Clip",
                "bbox": [0, 0, 100, 100],
                "attributes": ["silver"],
            }
        ]
    )
    (scene_root / "VID02_000002.json").write_text(
        json.dumps(scene_vid2), encoding="utf-8"
    )

    dictionary_terms.update({"how", "many", "clips", "present", "2"})

    dictionary_path = tmp_path / "dictionary.pkl"
    vocab_list = sorted(dictionary_terms)
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    with dictionary_path.open("wb") as fp:
        pickle.dump((word2idx, vocab_list), fp)

    out_root = tmp_path / "out"
    args = argparse.Namespace(
        qa_dir=str(qa_root),
        scene_graph_dir=str(scene_root),
        features_dir=str(roi_root),
        yolo_boxes_dir=str(global_root),
        split_config=str(split_path),
        out_dir=str(out_root),
        topk=2,
        default_width=200,
        default_height=200,
    )
    run_pipeline(args)

    return {
        "out_root": out_root,
        "dictionary_path": dictionary_path,
        "splits": list(split_config.keys()),
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

    image_meta = json.loads((out_root / "image_data.json").read_text(encoding="utf-8"))
    assert image_meta

    for split in sample_ssg["splits"]:
        h5_name = "ori_train.hdf5" if split == "train" else f"{split}.hdf5"
        with h5py.File(out_root / h5_name, "r") as fp:
            image_features = fp["image_features"][:]
            spatial_features = fp["spatial_features"][:]
            pos_boxes = fp["pos_boxes"][:]
        assert image_features.dtype == np.float32
        assert spatial_features.dtype == np.float32
        assert image_features.shape[0] == spatial_features.shape[0]
        assert pos_boxes.dtype == np.int32
        assert pos_boxes[0, 0] == 0
        assert pos_boxes[-1, 1] == image_features.shape[0]
        assert np.all(spatial_features >= -1e-6)
        assert np.all(spatial_features <= 1.0 + 1e-6)

        with (cache_dir / f"{split}_target.pkl").open("rb") as fp:
            targets = pickle.load(fp)
        assert targets
        for target in targets:
            assert target["scores"]
            for label in target["labels"]:
                assert 0 <= label < len(label2ans)

        question_path = out_root / f"gqa_{split}_questions_entities.json"
        payload = json.loads(question_path.read_text(encoding="utf-8"))
        for entry in payload["questions"]:
            assert "question_type" in entry
            assert "question_id" in entry
            assert entry["image_id"] in image_meta

        stat_words_path = out_root / f"{split}_2_stats_words.json"
        assert stat_words_path.exists()
        attr_words_path = out_root / f"{split}_attr_words_non_plural_words.json"
        assert attr_words_path.exists()

        imgid2idx_path = out_root / f"{split}_imgid2idx.pkl"
        with imgid2idx_path.open("rb") as fp:
            mapping = pickle.load(fp)
        assert mapping
        assert all(isinstance(v, int) for v in mapping.values())


def test_dataset_getitem_shapes(sample_ssg):
    out_root = sample_ssg["out_root"]
    dictionary = Dictionary.load_from_file(str(sample_ssg["dictionary_path"]))
    args = argparse.Namespace(
        max_boxes=10,
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
    assert torch.all((spatials >= 0.0) & (spatials <= 1.0 + 1e-6))
    assert target.shape[0] == dataset.num_ans_candidates
    assert float(target.sum()) > 0.0
    assert isinstance(sent, str)
    assert ans
    assert dataset.entries[0]["question_type"] in {"exist", "query", "other", "count"}

