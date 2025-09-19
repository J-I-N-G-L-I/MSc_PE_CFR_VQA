# (Python / DL backend: the core of this project)

- Python 3.12.3, CUDA 12.1, PyTorch 2.5.

- Prefer conda to create the environment; list dev deps in requirements.txt (create if missing).

- Formatting/lint: ruff check --fix + ruff format (optionally black + isort).

- Experiment tracking: Weights & Biases (set WANDB_API_KEY; default project names are in the training script).

- Large files (data/features/checkpoints) live in ./data and ./saved_models (do not commit).

- Start with single GPU; scale later via torchrun/DDP without changing public APIs.

# Testing instructions

## Python (project-critical)

- Test stack: pytest + pytest-cov. From repo root: pytest -q


## Required unit tests (must exist and pass):

1. SSG-VQA -> CFRF cache conversion: HDF5/JSON/PKL keys & shapes match; sample items round-trip.

2. Answer tables: ans2label.pkl / label2ans.pkl are bijective; missing answers raise errors.

3. Metrics: macro/micro-F1 match reference; compute_score_with_logits remains backward compatible.

4. Dataset reader: __getitem__ tensor shapes/types/ranges are correct (boxes in [0,1]).

5. One training step: fwd+bwd+grad clipping run, loss finite and decreases on a tiny batch.


# Project overview

- Current stack: CFRF model built on LXMERT/LXRT (text/vision/cross-modal Transformers) plus BAN (Bilinear Attention) fusion, counting module, 2D-RPE (vision), and RoPE (text) - trained on general VQA (GQA).

- Goal: Port to medical VQA (SSG-VQA) with minimal intrusion by data-format bridging to the project’s existing cache layout (HDF5/JSON/PKL). Add medical-centric metrics (F1) and ablation switches.

- Deliverables: runnable training/validation/evaluation scripts; SSG-VQA bridge script + unit tests; tables/visuals ready for publication (metrics, attention).

# Tasks (what to build/change)
1) Data bridge: SSG-VQA -> CFRF cache

Add tools/ssgvqa_to_cfrf.py. Inputs are SSG-VQA official assets (QA, ROI/coords/features, scene graphs). Outputs match the project’s cache layout.

## Required outputs & formats:

- ori_train.hdf5 / val.hdf5 (HDF5)

    1. image_features: float32, shape (ΣKi, D); concatenate Ki ROI features per image/frame i in dataset order.

    2. spatial_features: float32, shape (ΣKi, 6); [x1/W, y1/H, x2/W, y2/H, (x2-x1)/W, (y2-y1)/H].

    3. pos_boxes: int32, shape (N, 2); start (inclusive), end (exclusive) indices per image into the two arrays above.

- image_data.json:

{ "VID01_000123": { "width": 1920, "height": 1080 }, ... }


- gqa_{split}_questions_entities.json:

{
  "questions": [
    {
      "image_id": "VID01_000123",
      "question": "Is the grasper touching the liver?",
      "question_id": "VID01_000123_Q0001",
      "entities": ["grasper","liver","touching"]  // derived from question ∩ scene graph vocabulary; allow empty list
    }
  ]
}


- cache/{split}_target.pkl (array of dicts):

{"image_id": "...", "question_id": "...", "labels":[ans_id], "scores":[1.0]}


- Answer tables:

    cache/ans2label.pkl: { "yes":0, "no":1, "clip":2, ... }

    cache/label2ans.pkl: ["yes","no","clip", ...]

- Semantic priors:

    {split}_{topk}_stats_words.json: { "VID01_000123": "grasper,liver,clip,bleeding,..." }

    {split}_attr_words_non_plural_words.json: { "VID01_000123": ["metal grasper","inflated gallbladder", ...] }

    If missing scene info for an image, emit to corresponding *_skip_imgid.json.

## Suggested function signatures:

    def build_hdf5(features_dir: str, yolo_boxes_dir: str, out_h5: str) -> None: ...
    def build_image_data(meta_path: str, out_json: str) -> None: ...
    def build_qa_entities(qa_dir: str, scene_graph_dir: str, out_json: str) -> None: ...
    def build_answer_tables(qa_dir: str, out_dir: str) -> None: ...
    def build_stat_attr_words(scene_graph_dir: str, out_stat_json: str, out_attr_json: str,
                            topk:int=30) -> None: ...


## Validation:

pytest tests/test_bridge.py must pass, verifying keys, shapes, ranges (boxes in [0,1]), pos_boxes continuity, and sample back-mapping.

2) Metrics: macro/micro F1 + F1 per question type

Add src/metrics/f1.py:

from typing import Sequence, Dict, Tuple
import torch

def f1_macro_micro(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]: ...
def f1_by_type(pred: torch.Tensor, target: torch.Tensor, qtypes: Sequence[str]) -> Dict[str, float]: ...


- Inputs: pred logits [B, C]; target one-hot [B, C]. Use argmax internally.

- Wire into evaluate.py and src/FFOE/train.py::evaluate to print/log Acc, F1_macro, F1_micro, and F1_by_type.

3) Small changes in train/eval scripts

- evaluate.py: read SSG-VQA question types (if provided or derivable); compute & print F1s; keep existing Acc.

- src/FFOE/train.py::evaluate: add F1 computation and wandb.log.

- src/FFOE/train.py::train: keep printing/logging train_score (Acc); F1 can be eval-only to save time.

- No change to public model APIs.

4) Language model switch: --bert_name

- Add CLI arg --bert_name (default "bert-base-uncased") and allow biomedical variants like "biobert-base-cased-v1.1" or "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext".

- Replace all hard-coded "bert-base-uncased" constructions in language_model.py / main.py (tokenizer and model) to use args.bert_name.

5) Dataset reader reuse (no major change)

- src/FFOE/dataset.py::GQAFeatureDataset already reads our cache. If the bridge outputs match, it works for SSG-VQA.

- Note: __getitem__ truncates to max_boxes; keep pos_boxes as [start, end) to align.

# Runbook (quick start)
0) Environment

pip install -r requirements.txt

1) Generate caches (example)
python tools/ssgvqa_to_cfrf.py \
  --qa_dir /path/to/SSG-VQA/qa_txt \
  --scene_graph_dir /path/to/SSG-VQA/scene_graph \
  --features_dir /path/to/SSG-VQA/features \
  --yolo_boxes_dir /path/to/SSG-VQA/yolo_boxes \
  --out_dir ./data

2) Train (single GPU first)
python -m src.FFOE.train_entry \
  --dataset GQA \
  --model CFRF_Model \
  --batch_size 64 \
  --epochs 12 \
  --omega_q 0.1 --omega_v 0.1 \
  --gamma 2 --use_counter \
  --bert_name bert-base-uncased \
  --gpu 0


If there is no train_entry, use current entry point (e.g., python src/FFOE/train.py) with equivalent args.

3) Evaluate
python evaluate.py \
  --split val \
  --epoch 12 \
  --input saved_models/GQA \
  --output saved_models/GQA \
  --write_csv \
  --gpu 0


# Coding standards

- Do not change public APIs unless strictly necessary; keep additions backward-compatible.

- Error handling: missing fields, empty ROIs, out-of-vocab answers -> explicit error or skip with counters.

- Determinism: set torch.manual_seed / numpy.random.seed / random.seed; in eval disable cudnn.benchmark.

- Logging: record to wandb; print essentials to console; write files under saved_models/<run>/log.txt.

- Performance: vectorize; batch I/O with HDF5 slicing; avoid huge Python loops.

# Dataset mapping notes (alignment details)

- Coordinates & size: ensure x1<x2, y1<y2; normalize by width/height; last two dims of spatial_features are normalized w,h.

- pos_boxes[i] = [start, end) (left-closed, right-open); keep consistent with slicing in __getitem__.

- Semantic priors:

    - stats_words: top-K object/attribute nouns, lowercased, comma-separated.

    - attr_words: 2 - 3-gram phrases ("<attr> <obj>"), singularized (basic stemming), lowercased.

- Answer tables: cover train + val answer set; do not reuse GQA’s all_ans.json.

Metrics details (implementation hints)

- compute_score_with_logits (existing): Acc = argmax match to one-hot; keep unchanged.

- F1:

    - pred = argmax(logits, dim=1), true = argmax(target, dim=1).

    - Macro-F1: per-class F1 then mean; Micro-F1: aggregate TP/FP/FN globally.

    - by_type: pass qtypes (len B), compute micro-F1 per type.

# Common pitfalls

- HDF5 ordering and pos_boxes misalignment -> ROI segments mismatch during training.

- Inconsistent image_id across QA/scene graph/features.

- Answer not in ans2label -> training index out of range.

- Missing normalization -> boxes outside [0,1] -> assertions fail.

- --bert_name not plumbed through all tokenizer/model constructs.

# For reference to the SSGVQA dataset:
- The address of each scene graph of each frame in each video (VIDxx) will be: "D:\LJ\datasets\SSGVQA\scene_graph_ssgqa\scene_graph\VID01_0.json", the inside content will be:
{"scenes": [{"objects": [{"bbox": [20, 0, 114, 171], "component": "abdominal_wall_cavity", "type": "anatomy", "center": [67.0, 85.5], "location": "top-left"}, {"bbox": [24, 0, 400, 239], "component": "liver", "type": "anatomy", "center": [212.0, 119.5], "location": "top-mid"}, {"bbox": [269, 225, 375, 239], "component": "gut", "type": "anatomy", "center": [322.0, 232.0], "location": "bottom-right"}, {"bbox": [39, 42, 387, 239], "component": "omentum", "type": "anatomy", "center": [213.0, 140.5], "location": "bottom-mid"}, {"bbox": [83, 0, 235, 230], "component": "gallbladder", "type": "anatomy", "center": [159.0, 115.0], "location": "top-mid"}, {"bbox": [121, 4, 214, 63], "component": "grasper", "type": "instrument", "center": [167.5, 33.5], "location": "top-mid"}], "image_filename": "VID01_1", "relationships": {"above": [[5], [5], [0, 1, 3, 4, 5], [0, 5], [5], []], "below": [[2, 3], [2], [], [2], [2], [0, 1, 2, 3, 4]], "grasp": [[], [], [], [], [], [4]], "horizontal": [[4], [], [], [4], [], []], "left": [[], [0, 4, 5], [0, 1, 3, 4, 5], [0, 4, 5], [0], [0]], "right": [[1, 2, 3, 4, 5], [2], [], [2], [1, 2, 3], [1, 2, 3]], "within": [[1], [3], [1, 3], [], [1], [1, 4]]}}], "info": {"split": "new", "image_index": 0, "image_filename": ["VID01_1"], "triplet": ["grasper,grasp,gallbladder"]}}

- The address of QA pairs of each each frame in each video (VIDxx) will be:"D:\LJ\datasets\SSGVQA\ssg-qa\ssg-qa\VID01\1.txt", the 2 parts seperated by | will be the question and answer, for example:
    Which anatomical structures are present?|abdominal_wall_cavity, liver, gut, omentum, gallbladder
    Which tools are present?|grasper
    What is the grasper doing?|grasp
    Which tool is operating on gallbladder|grasper
    What anatomy is at the top-left of the frame ?|abdominal_wall_cavity|[67.0, 85.5]
    What anatomy is at the top-mid of the frame ?|liver|[212.0, 119.5]

    and the answer space is: ["0","1","10","2","3","4","5","6","7","8","9","False","True","abdominal_wall_cavity","adhesion","anatomy","aspirate","bipolar","blood_vessel","blue","brown","clip","clipper","coagulate","cut","cystic_artery","cystic_duct","cystic_pedicle","cystic_plate","dissect","fluid","gallbladder","grasp","grasper","gut","hook","instrument","irrigate","irrigator","liver","omentum","pack","peritoneum","red","retract","scissors","silver","specimen_bag","specimenbag","white","yellow"]

- The global visual feature of each each frame in each video (VIDxx) will be:"D:\LJ\datasets\SSGVQA\cropped_images\cropped_images\VID01\vqa\img_features\1x1\000001.hdf5", and the roi visual features of each each frame in each video (VIDxx) will be: "D:\LJ\datasets\SSGVQA\roi_yolo_coord\roi_yolo_coord\VID01\labels\vqa\img_features\roi\000001.hdf5"