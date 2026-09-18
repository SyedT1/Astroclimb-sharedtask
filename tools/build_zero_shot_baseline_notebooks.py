#!/usr/bin/env python3
"""Build zero-shot direct/CoT baseline notebooks for three Qwen VLMs."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


MODELS = [
    {
        "slug": "qwen3vl4b",
        "folder": "astroclimb_qwen3vl4b_zero_shot",
        "filename": "astroclimb-qwen3vl4b-zero-shot.ipynb",
        "title": "Qwen3-VL-4B-Instruct",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "qwen3_vl",
    },
    {
        "slug": "qwen3vl8b",
        "folder": "astroclimb_qwen3vl8b_zero_shot",
        "filename": "astroclimb-qwen3vl8b-zero-shot.ipynb",
        "title": "Qwen3-VL-8B-Instruct",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3_vl",
    },
    {
        "slug": "qwen25vl7b",
        "folder": "astroclimb_qwen25vl7b_zero_shot",
        "filename": "astroclimb-qwen25vl7b-zero-shot.ipynb",
        "title": "Qwen2.5-VL-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2_5_vl",
    },
]


def lines(text):
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


INSTALL = r'''
# Preserve Kaggle's CUDA/PyTorch stack and install only the versions already used
# by the project's successful Qwen3-VL notebooks.
%pip install -q --upgrade-strategy only-if-needed \
    "transformers==4.57.1" "accelerate==1.10.1" "bitsandbytes==0.47.0"
'''


CONFIG = r'''
import base64
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
csv.field_size_limit(sys.maxsize)

SEED = 42
MODEL_ID = __MODEL_ID__
MODEL_FAMILY = __MODEL_FAMILY__
MODEL_SLUG = __MODEL_SLUG__
TARGET_COLUMNS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DIGIT_TO_LABEL = dict(enumerate(TARGET_COLUMNS))
VAL_PER_CLASS = 200
EXPECTED_VALIDATION_ROWS = 800
EXPECTED_TEST_ROWS = 10000
MIN_PIXELS = 256 * 256
MAX_PIXELS = 448 * 448
MAX_TEXT_CHARS = 3000
COT_MAX_NEW_TOKENS = 96

# Both prompt modes are evaluated with the same model load and validation rows.
EVALUATION_MODES = ["direct", "short_cot"]

# Full test inference is enabled for these requested baseline runs.
RUN_TEST_INFERENCE = True
TEST_MODE = "short_cot"  # "direct" or "short_cot"
TEST_LIMIT = None
REBUILD_CACHE = False

WORK_ROOT = Path("/kaggle/working") / f"astroclimb_baseline_{MODEL_SLUG}"
IMAGE_ROOT = WORK_ROOT / "images_448"
VALIDATION_MANIFEST = WORK_ROOT / "validation_800.jsonl"
TEST_MANIFEST = WORK_ROOT / "test.jsonl"
PREDICTION_ROOT = WORK_ROOT / "predictions"
WORKER_PATH = WORK_ROOT / "infer_zero_shot_ddp.py"
PROMPT_PATH = WORK_ROOT / "prompt_config.json"
for path in (WORK_ROOT, IMAGE_ROOT, PREDICTION_ROOT):
    path.mkdir(parents=True, exist_ok=True)

def locate_csv(filename):
    preferred = [
        Path("/kaggle/input/competitions/astroclimb") / filename,
        Path("/kaggle/input/astroclimb") / filename,
        Path("data") / filename,
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    roots = [Path("/kaggle/input"), Path("data")]
    matches = [p for root in roots if root.exists() for p in root.rglob(filename)]
    matches = sorted(matches, key=lambda p: ("astroclimb" not in str(p).lower(), len(str(p))))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename}. Attach the AstroCLIMB competition data.")
    return matches[0]

TRAIN_CSV = locate_csv("train.csv")
TEST_CSV = locate_csv("test.csv")
print("torch:", torch.__version__)
print("CUDA has not been initialized:", not torch.cuda.is_initialized())
print("Model:", MODEL_ID)
print("Family:", MODEL_FAMILY)
print("Train:", TRAIN_CSV)
print("Test:", TEST_CSV)
print("Work root:", WORK_ROOT)
print("Test inference enabled:", RUN_TEST_INFERENCE)
'''


PREPROCESS = r'''
def get_label(row):
    values = [int(float(row[column])) for column in TARGET_COLUMNS]
    if sum(values) != 1:
        raise ValueError(f"Invalid one-hot label for id={row.get('id')}: {values}")
    return values.index(1)

def looks_like_image(value):
    return isinstance(value, str) and value.lstrip().startswith(
        ("iVBORw0KGgo", "/9j/", "UklGR", "R0lGOD", "data:image")
    )

def normalized_modality(row):
    kinds = ["I" if looks_like_image(row[key]) else "C" for key in ("obj_1", "obj_2")]
    return "".join(sorted(kinds))

def select_validation_ids(path, per_class=200, seed=42):
    rng = random.Random(seed)
    reservoirs = {label: [] for label in range(4)}
    seen = {label: 0 for label in range(4)}
    started = time.perf_counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "obj_1", "obj_2", *TARGET_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing train columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=1):
            label = get_label(row)
            seen[label] += 1
            bucket = reservoirs[label]
            if len(bucket) < per_class:
                bucket.append(row["id"])
            else:
                position = rng.randrange(seen[label])
                if position < per_class:
                    bucket[position] = row["id"]
            if row_number % 1000 == 0:
                print(f"Split scan {row_number} | {(time.perf_counter()-started)/60:.2f} min")
    selected = {identifier for bucket in reservoirs.values() for identifier in bucket}
    print("Class totals:", {DIGIT_TO_LABEL[k]: v for k, v in seen.items()})
    assert len(selected) == EXPECTED_VALIDATION_ROWS
    return selected

def decode_image(value):
    value = value.strip()
    if value.startswith("data:image"):
        value = value.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(value, validate=False)))
    image.load()
    return image.convert("RGB")

def resize_to_area(image, max_pixels=MAX_PIXELS):
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)

def cache_object(value):
    if not looks_like_image(value):
        return {"kind": "caption", "value": value}
    raw = value.split(",", 1)[-1] if value.startswith("data:image") else value
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    destination = IMAGE_ROOT / f"{digest}.png"
    if not destination.exists():
        image = resize_to_area(decode_image(value))
        image.save(destination, format="PNG", compress_level=3)
    return {"kind": "image", "value": str(destination)}

def cache_row(row, labeled):
    record = {
        "id": str(row["id"]),
        "obj_1": cache_object(row["obj_1"]),
        "obj_2": cache_object(row["obj_2"]),
        "modality": normalized_modality(row),
    }
    if labeled:
        record["label"] = get_label(row)
    return record

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

validation_ids = select_validation_ids(TRAIN_CSV, VAL_PER_CLASS, SEED)
if REBUILD_CACHE or not VALIDATION_MANIFEST.exists():
    validation_rows = []
    started = time.perf_counter()
    with TRAIN_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            if row["id"] in validation_ids:
                validation_rows.append(cache_row(row, labeled=True))
            if row_number % 1000 == 0:
                print(f"Manifest scan {row_number} | {(time.perf_counter()-started)/60:.2f} min")
    assert len(validation_rows) == EXPECTED_VALIDATION_ROWS
    write_jsonl(VALIDATION_MANIFEST, validation_rows)

if RUN_TEST_INFERENCE and (REBUILD_CACHE or not TEST_MANIFEST.exists()):
    test_rows = []
    with TEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            if TEST_LIMIT is not None and row_number > TEST_LIMIT:
                break
            test_rows.append(cache_row(row, labeled=False))
            if row_number % 500 == 0:
                print("Test cache:", row_number)
    write_jsonl(TEST_MANIFEST, test_rows)

validation_rows = read_jsonl(VALIDATION_MANIFEST)
print("Validation rows:", len(validation_rows))
print("Validation class counts:", np.bincount([r["label"] for r in validation_rows], minlength=4).tolist())
print("Validation modalities:", pd.Series([r["modality"] for r in validation_rows]).value_counts().to_dict())
'''


PROMPTS = r'''
CLASS_DEFINITIONS = """
0 = same_figure: the exact same scientific figure, normally a figure image and its corresponding caption.
1 = same_paper: different figures or captions from the same paper or DOI.
2 = related_papers: different papers with a direct citation or research relationship.
3 = unrelated_papers: different papers without evidence of a direct relationship.
""".strip()

DIRECT_SYSTEM_PROMPT = f"""You classify relationships between two objects from astronomy papers.

{CLASS_DEFINITIONS}

The relation is symmetric. Apply the labels hierarchically: exact same figure first, same paper second, direct relationship third, and unrelated otherwise. Shared broad subject matter alone is not evidence of related papers. Do not invent hidden metadata. Do not explain your answer. Output exactly one digit: 0, 1, 2, or 3."""

SHORT_COT_SYSTEM_PROMPT = f"""You are an expert scientific-document analyst classifying two objects from astronomy papers.

{CLASS_DEFINITIONS}

The relation is symmetric. Decide hierarchically: exact same figure, same paper, directly related papers, otherwise unrelated. Examine figure-caption correspondence, figure numbering, distinctive targets, instruments, datasets, methods, authors, terminology, and explicit citation language. Shared broad subject matter alone is insufficient. Do not invent information that is not visible.

Give no more than two short evidence sentences. End with exactly one line of the form `Label: 0`, `Label: 1`, `Label: 2`, or `Label: 3`."""

DIRECT_USER_INSTRUCTION = "Classify this pair. Output one digit only."
SHORT_COT_USER_INSTRUCTION = (
    "Briefly compare the evidence and classify the relationship. "
    "Follow the required Evidence then Label format."
)

prompt_config = {
    "direct_system": DIRECT_SYSTEM_PROMPT,
    "short_cot_system": SHORT_COT_SYSTEM_PROMPT,
    "direct_user": DIRECT_USER_INSTRUCTION,
    "short_cot_user": SHORT_COT_USER_INSTRUCTION,
}
PROMPT_PATH.write_text(json.dumps(prompt_config, ensure_ascii=False, indent=2), encoding="utf-8")

print("DIRECT SYSTEM PROMPT\n" + "=" * 80)
print(DIRECT_SYSTEM_PROMPT)
print("\nDIRECT USER INSTRUCTION\n" + "=" * 80)
print(DIRECT_USER_INSTRUCTION)
print("\nSHORT-CoT SYSTEM PROMPT\n" + "=" * 80)
print(SHORT_COT_SYSTEM_PROMPT)
print("\nSHORT-CoT USER INSTRUCTION\n" + "=" * 80)
print(SHORT_COT_USER_INSTRUCTION)
print("\nSaved exact inference prompts to:", PROMPT_PATH)
'''


WORKER_SCRIPT = r'''
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
TARGET_COLUMNS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
LABEL_PATTERN = re.compile(r"Label\s*:\s*([0-3])", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"(?<!\d)([0-3])(?!\d)")

def shorten(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...middle truncated...]\n" + text[-half:]

def object_content(number, obj, max_text_chars):
    if obj["kind"] == "image":
        with Image.open(obj["value"]) as source:
            image = source.convert("RGB")
        return [
            {"type": "text", "text": f"Object {number} is a scientific figure:"},
            {"type": "image", "image": image},
        ]
    return [{
        "type": "text",
        "text": f"Object {number} is a figure caption:\n{shorten(obj['value'], max_text_chars)}",
    }]

def build_messages(row, mode, max_text_chars, prompts):
    content = object_content(1, row["obj_1"], max_text_chars)
    content += object_content(2, row["obj_2"], max_text_chars)
    if mode == "direct":
        content.append({"type": "text", "text": prompts["direct_user"]})
        system = prompts["direct_system"]
    else:
        content.append({"type": "text", "text": prompts["short_cot_user"]})
        system = prompts["short_cot_system"]
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]

def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

def parse_cot(text):
    matches = LABEL_PATTERN.findall(text)
    return int(matches[-1]) if matches else None

def load_completed(path):
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                completed[str(record["id"])] = record
    return completed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True, choices=["qwen3_vl", "qwen2_5_vl"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--modes", nargs="+", required=True, choices=["direct", "short_cot"])
    parser.add_argument("--min-pixels", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--max-text-chars", type=int, required=True)
    parser.add_argument("--cot-max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    torch.cuda.set_device(rank)

    from transformers import AutoProcessor, BitsAndBytesConfig
    if args.family == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    rows = all_rows[rank::world_size]

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = ModelClass.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": rank},
    )
    model.eval()
    model.config.use_cache = True

    label_token_ids = []
    for digit in "0123":
        token_ids = processor.tokenizer.encode(digit, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Label {digit} is not one token: {token_ids}")
        label_token_ids.append(token_ids[0])

    @torch.inference_mode()
    def direct_predict(row):
        batch = processor.apply_chat_template(
            build_messages(row, "direct", args.max_text_chars, prompts),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        batch = move_batch(batch, model.device)
        logits = model(**batch).logits[0, -1, label_token_ids].float()
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        return int(probabilities.argmax()), probabilities.tolist(), ""

    @torch.inference_mode()
    def cot_predict(row):
        messages = build_messages(row, "short_cot", args.max_text_chars, prompts)
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        batch = move_batch(batch, model.device)
        output = model.generate(
            **batch,
            max_new_tokens=args.cot_max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated = output[0, batch["input_ids"].shape[-1]:]
        raw = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        prediction = parse_cot(raw)
        repaired = False
        if prediction is None:
            repaired = True
            repair_messages = messages + [
                {"role": "assistant", "content": [{"type": "text", "text": raw}]},
                {"role": "user", "content": [{"type": "text", "text": "Return only the final digit: 0, 1, 2, or 3."}]},
            ]
            repair_batch = processor.apply_chat_template(
                repair_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            repair_batch = move_batch(repair_batch, model.device)
            repair_logits = model(**repair_batch).logits[0, -1, label_token_ids].float()
            prediction = int(repair_logits.argmax().item())
        return prediction, None, raw, repaired

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        output_path = output_dir / f"{args.split}_{mode}_rank{rank}.jsonl"
        completed = load_completed(output_path)
        pending = [row for row in rows if str(row["id"]) not in completed]
        started = time.perf_counter()
        with output_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(pending, start=1):
                if mode == "direct":
                    prediction, probabilities, raw = direct_predict(row)
                    repaired = False
                else:
                    prediction, probabilities, raw, repaired = cot_predict(row)
                record = {
                    "id": str(row["id"]),
                    "label": row.get("label"),
                    "modality": row["modality"],
                    "mode": mode,
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "raw_response": raw,
                    "repaired": repaired,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"rank={rank} mode={mode} {index}/{len(pending)} {elapsed/index:.2f}s/row "
                        f"ETA={(elapsed/index)*(len(pending)-index)/60:.1f} min",
                        flush=True,
                    )

if __name__ == "__main__":
    main()
'''

WORKER = (
    "WORKER_SOURCE = " + repr(WORKER_SCRIPT) + "\n\n"
    "WORKER_PATH.write_text(WORKER_SOURCE, encoding='utf-8')\n"
    "print('Wrote:', WORKER_PATH)\n"
)


LAUNCH = r'''
def launch_inference(manifest, split, modes):
    command = [
        sys.executable,
        "-m", "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes", "2",
        str(WORKER_PATH),
        "--manifest", str(manifest),
        "--model", MODEL_ID,
        "--family", MODEL_FAMILY,
        "--output-dir", str(PREDICTION_ROOT),
        "--prompt-config", str(PROMPT_PATH),
        "--split", split,
        "--modes", *modes,
        "--min-pixels", str(MIN_PIXELS),
        "--max-pixels", str(MAX_PIXELS),
        "--max-text-chars", str(MAX_TEXT_CHARS),
        "--cot-max-new-tokens", str(COT_MAX_NEW_TOKENS),
    ]
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("Launching:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, env=environment)
    print(f"{split} inference completed in {(time.perf_counter()-started)/60:.2f} min")

launch_inference(VALIDATION_MANIFEST, "validation", EVALUATION_MODES)
'''


EVALUATE = r'''
def load_prediction_mode(split, mode):
    records = []
    for rank in range(2):
        path = PREDICTION_ROOT / f"{split}_{mode}_rank{rank}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(records)
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {split}/{mode}")
    return frame

all_metrics = {}
for mode in EVALUATION_MODES:
    predictions = load_prediction_mode("validation", mode)
    assert len(predictions) == EXPECTED_VALIDATION_ROWS
    y_true = predictions["label"].astype(int).to_numpy()
    y_pred = predictions["prediction"].astype(int).to_numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, labels=list(range(4)), average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    print("\n", "=" * 72)
    print("Mode:", mode)
    print(f"Macro-F1: {macro_f1:.6f}")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=TARGET_COLUMNS,
        digits=6,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print("Repairs:", int(predictions["repaired"].fillna(False).astype(bool).sum()))
    for modality in sorted(predictions["modality"].unique()):
        subset = predictions[predictions["modality"] == modality]
        present = sorted(subset["label"].astype(int).unique().tolist())
        score = f1_score(
            subset["label"].astype(int),
            subset["prediction"].astype(int),
            labels=present,
            average="macro",
            zero_division=0,
        )
        print(f"{modality}: n={len(subset)} present-class macro-F1={score:.6f}")

    all_metrics[mode] = {
        "macro_f1": float(macro_f1),
        "per_class_f1": {name: float(per_class[i]) for i, name in enumerate(TARGET_COLUMNS)},
        "confusion_matrix": matrix.tolist(),
        "repairs": int(predictions["repaired"].fillna(False).astype(bool).sum()),
    }

summary = {
    "model": MODEL_ID,
    "quantization": "bitsandbytes NF4 4-bit inference only",
    "validation_rows": EXPECTED_VALIDATION_ROWS,
    "results": all_metrics,
}
metrics_path = WORK_ROOT / "validation_metrics.json"
metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nMetrics:", metrics_path)
'''


TEST = r'''
if RUN_TEST_INFERENCE:
    if TEST_MODE not in {"direct", "short_cot"}:
        raise ValueError(f"Invalid TEST_MODE: {TEST_MODE}")
    launch_inference(TEST_MANIFEST, "test", [TEST_MODE])
    predictions = load_prediction_mode("test", TEST_MODE)
    expected = EXPECTED_TEST_ROWS if TEST_LIMIT is None else TEST_LIMIT
    assert len(predictions) == expected
    predictions = predictions.set_index("id")
    with TEST_MANIFEST.open("r", encoding="utf-8") as handle:
        ordered_ids = [str(json.loads(line)["id"]) for line in handle if line.strip()]
    predictions = predictions.loc[ordered_ids].reset_index()
    predicted = predictions["prediction"].astype(int).to_numpy()
    submission = pd.DataFrame({"id": predictions["id"]})
    for class_index, column in enumerate(TARGET_COLUMNS):
        submission[column] = (predicted == class_index).astype(int)
    assert submission["id"].is_unique
    assert (submission[TARGET_COLUMNS].sum(axis=1) == 1).all()
    submission_path = WORK_ROOT / f"submission_{TEST_MODE}.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    print("Submission:", submission_path)
    print("Prediction counts:", submission[TARGET_COLUMNS].sum().to_dict())
else:
    print("Test inference skipped. Select the model and prompt mode using validation macro-F1 first.")
'''


def build_notebook(spec):
    config = (
        CONFIG.replace("__MODEL_ID__", repr(spec["model_id"]))
        .replace("__MODEL_FAMILY__", repr(spec["family"]))
        .replace("__MODEL_SLUG__", repr(spec["slug"]))
    )
    cells = [
        markdown(
            f"""
            # AstroCLIMB baseline — {spec['title']}

            This notebook evaluates **zero-shot direct classification** and **zero-shot short
            chain-of-thought** for `{spec['model_id']}`. It performs no training and attaches no
            LoRA/QLoRA adapters. NF4 is used only to make inference fit independently on both
            Kaggle T4 GPUs.

            Both prompt modes use the permanent seed-42 balanced validation split of 800 examples,
            identical image/caption limits, deterministic decoding, and the same label definitions.
            Test inference is disabled until the model and prompt mode are selected by validation
            macro-F1.
            """
        ),
        markdown("## Install the project-compatible inference stack"),
        code(INSTALL),
        markdown("## Configuration"),
        code(config),
        markdown("## Reconstruct and cache the permanent validation split"),
        code(PREPROCESS),
        markdown("## Exact prompts used for inference"),
        code(PROMPTS),
        markdown("## Write the two-GPU inference worker"),
        code(WORKER),
        markdown("## Run direct and short-CoT validation inference"),
        code(LAUNCH),
        markdown("## Compare validation results"),
        code(EVALUATE),
        markdown("## Optional test inference for the selected prompt mode"),
        code(TEST),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    baseline_root = ROOT / "baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        folder = baseline_root / spec["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / spec["filename"]
        destination.write_text(
            json.dumps(build_notebook(spec), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
