#!/usr/bin/env python3
"""Build the three Qwen3.5-9B prompt-ablation Kaggle notebooks."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def lines(text):
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


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


VARIANTS = {
    "direct": {
        "folder": "astroclimb_qwen35_9b_direct",
        "filename": "astroclimb-qwen35-9b-direct.ipynb",
        "title": "Qwen3.5-9B direct classification",
        "description": (
            "Inference-only zero-shot classification. The model is instructed not to reason "
            "aloud and emits one class digit."
        ),
        "max_tokens": 8,
        "context": 8192,
        "needs_demos": False,
    },
    "short_cot": {
        "folder": "astroclimb_qwen35_9b_short_cot",
        "filename": "astroclimb-qwen35-9b-short-cot.ipynb",
        "title": "Qwen3.5-9B short chain-of-thought",
        "description": (
            "Inference-only zero-shot classification with at most two short evidence sentences "
            "before the final class label."
        ),
        "max_tokens": 96,
        "context": 8192,
        "needs_demos": False,
    },
}


INSTALL = r'''
# Internet must be enabled for the first run. This installs the current llama.cpp
# binary. Kaggle already supplies requests, huggingface_hub, and scikit-learn;
# avoiding upgrades preserves the working Kaggle package stack.
!curl -LsSf https://llama.app/install.sh | sh
'''


CONFIG = r'''
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
csv.field_size_limit(2**31 - 1)

SEED = 42
VARIANT = __VARIANT__
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILENAME = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
MMPROJ_FILENAME = "mmproj-F16.gguf"
MODEL_ALIAS = "qwen35-9b-astroclimb"
TARGET_COLUMNS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DIGIT_TO_LABEL = dict(enumerate(TARGET_COLUMNS))
VAL_PER_CLASS = 200
EXPECTED_VALIDATION_ROWS = 800
EXPECTED_TEST_ROWS = 10000
QUERY_MAX_PIXELS = 448 * 448
DEMO_MAX_PIXELS = 256 * 256
MAX_TEXT_CHARS = 3000
DEMO_MAX_TEXT_CHARS = 1000
CONTEXT_LENGTH = __CONTEXT__
MAX_NEW_TOKENS = __MAX_TOKENS__
NEEDS_DEMOS = __NEEDS_DEMOS__

# These are validation notebooks. Enable test inference only after selecting the
# best prompt on the fixed 800-row validation split.
RUN_TEST_INFERENCE = False
TEST_LIMIT = None
REBUILD_CACHE = False

WORK_ROOT = Path("/kaggle/working") / f"astroclimb_qwen35_9b_{VARIANT}"
IMAGE_ROOT = WORK_ROOT / "images"
VALIDATION_MANIFEST = WORK_ROOT / "validation_800.jsonl"
TEST_MANIFEST = WORK_ROOT / "test.jsonl"
DEMO_PATH = WORK_ROOT / "modality_demonstrations.json"
PREDICTION_ROOT = WORK_ROOT / "predictions"
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
print("Variant:", VARIANT)
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
    if not isinstance(value, str):
        return False
    return value.lstrip().startswith(("iVBORw0KGgo", "/9j/", "UklGR", "R0lGOD", "data:image"))

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

def resize_to_area(image, max_pixels):
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)

def cache_object(value, max_pixels=QUERY_MAX_PIXELS, prefix="query"):
    if not looks_like_image(value):
        return {"kind": "caption", "value": value}
    raw = value.split(",", 1)[-1] if value.startswith("data:image") else value
    digest = hashlib.sha256((prefix + raw).encode("utf-8")).hexdigest()
    destination = IMAGE_ROOT / f"{digest}.png"
    if not destination.exists():
        image = resize_to_area(decode_image(value), max_pixels=max_pixels)
        image.save(destination, format="PNG", compress_level=3)
    return {"kind": "image", "value": str(destination)}

def cache_row(row, labeled, demo=False):
    pixels = DEMO_MAX_PIXELS if demo else QUERY_MAX_PIXELS
    prefix = "demo" if demo else "query"
    record = {
        "id": str(row["id"]),
        "obj_1": cache_object(row["obj_1"], pixels, prefix),
        "obj_2": cache_object(row["obj_2"], pixels, prefix),
        "modality": normalized_modality(row),
    }
    if labeled:
        record["label"] = get_label(row)
    return record

def stable_demo_score(identifier):
    return hashlib.sha256(f"{SEED}:{identifier}".encode()).hexdigest()

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

validation_ids = select_validation_ids(TRAIN_CSV, VAL_PER_CLASS, SEED)

if REBUILD_CACHE or not VALIDATION_MANIFEST.exists() or (NEEDS_DEMOS and not DEMO_PATH.exists()):
    validation_rows = []
    # Three shots cannot cover four labels. CI queries use 0/1/2 because exact
    # figure matching is possible; CC and II use the only structurally valid
    # relationship labels 1/2/3.
    desired_labels = {"CI": (0, 1, 2), "CC": (1, 2, 3), "II": (1, 2, 3)}
    demo_candidates = {}
    started = time.perf_counter()
    with TRAIN_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=1):
            if row["id"] in validation_ids:
                validation_rows.append(cache_row(row, labeled=True))
            elif NEEDS_DEMOS:
                modality = normalized_modality(row)
                label = get_label(row)
                if label in desired_labels.get(modality, ()):
                    key = (modality, label)
                    score = stable_demo_score(row["id"])
                    if key not in demo_candidates or score < demo_candidates[key][0]:
                        demo_candidates[key] = (score, dict(row))
            if row_number % 1000 == 0:
                print(f"Manifest scan {row_number} | {(time.perf_counter()-started)/60:.2f} min")
    assert len(validation_rows) == EXPECTED_VALIDATION_ROWS
    write_jsonl(VALIDATION_MANIFEST, validation_rows)
    if NEEDS_DEMOS:
        demos = {}
        for modality, labels in desired_labels.items():
            missing = [(modality, label) for label in labels if (modality, label) not in demo_candidates]
            if missing:
                raise RuntimeError(f"Missing demonstration candidates: {missing}")
            demos[modality] = [
                cache_row(demo_candidates[(modality, label)][1], labeled=True, demo=True)
                for label in labels
            ]
        DEMO_PATH.write_text(json.dumps(demos, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    validation_rows = read_jsonl(VALIDATION_MANIFEST)

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
demonstrations = json.loads(DEMO_PATH.read_text(encoding="utf-8")) if NEEDS_DEMOS else {}
print("Validation rows:", len(validation_rows))
print("Validation class counts:", np.bincount([r["label"] for r in validation_rows], minlength=4).tolist())
print("Validation modalities:", pd.Series([r["modality"] for r in validation_rows]).value_counts().to_dict())
if NEEDS_DEMOS:
    print("Demonstration labels:", {m: [r["label"] for r in rows] for m, rows in demonstrations.items()})
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

COT_SYSTEM_PROMPT = f"""You are an expert scientific-document analyst classifying two objects from astronomy papers.

{CLASS_DEFINITIONS}

The relation is symmetric. Decide hierarchically: exact same figure, same paper, directly related papers, otherwise unrelated. Examine figure-caption correspondence, figure numbering, distinctive targets, instruments, datasets, methods, authors, terminology, and explicit citation language. Shared broad subject matter alone is insufficient. Do not invent information that is not visible.

Give no more than two short evidence sentences. End with exactly one line of the form `Label: 0`, `Label: 1`, `Label: 2`, or `Label: 3`."""

def shorten(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...middle truncated...]\n" + text[-half:]

def image_data_url(path):
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:image/png;base64," + payload

def object_content(number, obj, demo=False):
    if obj["kind"] == "image":
        return [
            {"type": "text", "text": f"Object {number} is a scientific figure:"},
            {"type": "image_url", "image_url": {"url": image_data_url(obj["value"])}},
        ]
    limit = DEMO_MAX_TEXT_CHARS if demo else MAX_TEXT_CHARS
    return [{"type": "text", "text": f"Object {number} is a figure caption:\n{shorten(obj['value'], limit)}"}]

def pair_content(row, instruction, demo=False):
    content = object_content(1, row["obj_1"], demo=demo)
    content += object_content(2, row["obj_2"], demo=demo)
    content.append({"type": "text", "text": instruction})
    return content

def build_messages(row):
    if VARIANT == "short_cot":
        instruction = "Briefly compare the evidence and classify the relationship. Follow the required Evidence then Label format."
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": pair_content(row, instruction)},
        ]

    messages = [{"role": "system", "content": DIRECT_SYSTEM_PROMPT}]
    if VARIANT == "modality_3shot":
        for demo in demonstrations[row["modality"]]:
            messages.append({
                "role": "user",
                "content": pair_content(demo, "Classify this demonstration. Output one digit only.", demo=True),
            })
            messages.append({"role": "assistant", "content": str(demo["label"])})
    messages.append({
        "role": "user",
        "content": pair_content(row, "Classify this query. Output one digit only."),
    })
    return messages

sample_messages = build_messages(validation_rows[0])
print("Messages in one request:", len(sample_messages))
print("Prompt variant:", VARIANT)
'''


SERVERS = r'''
from huggingface_hub import hf_hub_download

def locate_llama_server_command():
    candidates = [
        ([shutil.which("llama-server")] if shutil.which("llama-server") else None),
        ([str(Path.home() / ".local/bin/llama-server")]),
        ([str(Path.home() / ".llama/bin/llama-server")]),
        # The current llama.app installer uses one multicall executable. Its
        # server is invoked as `~/.llama-app/llama serve`.
        ([str(Path.home() / ".llama-app/llama"), "serve"]),
    ]
    for candidate in candidates:
        if candidate and Path(candidate[0]).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find llama-server or ~/.llama-app/llama. Re-run the installation cell."
    )

LLAMA_SERVER_COMMAND = locate_llama_server_command()
MODEL_PATH = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
MMPROJ_PATH = hf_hub_download(MODEL_REPO, MMPROJ_FILENAME)
print("llama server command:", " ".join(LLAMA_SERVER_COMMAND))
print("Model:", MODEL_PATH)
print("Projector:", MMPROJ_PATH)

SERVER_PORTS = [8080, 8081]
SERVER_URLS = [f"http://127.0.0.1:{port}" for port in SERVER_PORTS]
SERVER_PROCESSES = []

def wait_for_server(url, process, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            response = requests.get(url + "/health", timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy: {url}")

for gpu, (port, url) in enumerate(zip(SERVER_PORTS, SERVER_URLS)):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = WORK_ROOT / f"llama_server_gpu{gpu}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        *LLAMA_SERVER_COMMAND,
        "--model", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--alias", MODEL_ALIAS,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(CONTEXT_LENGTH),
        "--n-gpu-layers", "99",
        "--parallel", "1",
        "--jinja",
        "--no-webui",
    ]
    process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
    process._astroclimb_log_handle = log_handle
    SERVER_PROCESSES.append(process)
    wait_for_server(url, process)
    print(f"GPU {gpu} server ready at {url}")
'''


INFERENCE = r'''
LABEL_PATTERN = re.compile(r"Label\s*:\s*([0-3])", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"(?<!\d)([0-3])(?!\d)")

def parse_prediction(text):
    labels = LABEL_PATTERN.findall(text)
    if labels:
        return int(labels[-1])
    # A truncated CoT can contain figure numbers in its evidence. Never mistake
    # one of those numbers for the prediction; force the explicit repair turn.
    if VARIANT == "short_cot":
        return None
    digits = DIGIT_PATTERN.findall(text.strip())
    if digits:
        return int(digits[-1])
    return None

def request_completion(server_url, messages, max_tokens=MAX_NEW_TOKENS, retries=4):
    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
        # Qwen3.5 thinking is on by default. These experiments use either direct
        # output or explicitly bounded visible reasoning, so hidden thinking is off.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            response = requests.post(server_url + "/v1/chat/completions", json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)

def predict_one(server_url, row):
    raw = request_completion(server_url, build_messages(row))
    prediction = parse_prediction(raw)
    repaired = False
    if prediction is None:
        repaired = True
        repair_messages = build_messages(row) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Return only the final digit: 0, 1, 2, or 3."},
        ]
        repair = request_completion(server_url, repair_messages, max_tokens=8)
        prediction = parse_prediction(repair)
        raw = raw + "\n[REPAIR] " + repair
    if prediction is None:
        raise ValueError(f"Could not parse a class from response: {raw!r}")
    return prediction, raw, repaired

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

def run_shard(rank, rows, split):
    output_path = PREDICTION_ROOT / f"{split}_{VARIANT}_rank{rank}.jsonl"
    completed = load_completed(output_path)
    pending = [row for row in rows if str(row["id"]) not in completed]
    started = time.perf_counter()
    with output_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            prediction, raw, repaired = predict_one(SERVER_URLS[rank], row)
            record = {
                "id": str(row["id"]),
                "prediction": prediction,
                "label": row.get("label"),
                "modality": row["modality"],
                "raw_response": raw,
                "repaired": repaired,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[record["id"]] = record
            if index % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rank={rank} {index}/{len(pending)} {elapsed/index:.2f}s/row "
                    f"ETA={(elapsed/index)*(len(pending)-index)/60:.1f} min",
                    flush=True,
                )
    return completed

def run_rows(rows, split):
    shards = [rows[rank::2] for rank in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_shard, rank, shards[rank], split) for rank in range(2)]
        completed = {}
        for future in futures:
            completed.update(future.result())
    ordered = [completed[str(row["id"])] for row in rows]
    return ordered
'''


VALIDATE = r'''
started = time.perf_counter()
validation_predictions = run_rows(validation_rows, "validation")
elapsed = time.perf_counter() - started

y_true = np.array([int(record["label"]) for record in validation_predictions])
y_pred = np.array([int(record["prediction"]) for record in validation_predictions])
macro_f1 = f1_score(y_true, y_pred, average="macro")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(4)),
    target_names=TARGET_COLUMNS,
    digits=6,
    zero_division=0,
)
matrix = confusion_matrix(y_true, y_pred, labels=list(range(4)))

print(f"Validation time: {elapsed/60:.2f} min")
print(f"Macro-F1: {macro_f1:.6f}")
print(report)
print("Confusion matrix (rows=true, columns=predicted):")
print(matrix)
print("Repairs:", sum(bool(record["repaired"]) for record in validation_predictions))

for modality in sorted({record["modality"] for record in validation_predictions}):
    mask = np.array([record["modality"] == modality for record in validation_predictions])
    present_labels = np.unique(y_true[mask]).tolist()
    print(
        f"{modality}: n={mask.sum()} present-class macro-F1="
        f"{f1_score(y_true[mask], y_pred[mask], labels=present_labels, average='macro', zero_division=0):.6f}"
    )

metrics = {
    "variant": VARIANT,
    "model": MODEL_REPO,
    "quantization": "UD-Q4_K_XL",
    "validation_rows": len(validation_predictions),
    "macro_f1": float(macro_f1),
    "per_class_f1": {
        TARGET_COLUMNS[index]: float(f1_score(y_true == index, y_pred == index, zero_division=0))
        for index in range(4)
    },
    "confusion_matrix": matrix.tolist(),
    "repairs": sum(bool(record["repaired"]) for record in validation_predictions),
    "elapsed_minutes": elapsed / 60,
}
(WORK_ROOT / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
'''


TEST = r'''
if RUN_TEST_INFERENCE:
    test_rows = read_jsonl(TEST_MANIFEST)
    started = time.perf_counter()
    test_predictions = run_rows(test_rows, "test")
    print(f"Test inference: {(time.perf_counter()-started)/60:.2f} min")

    predicted = np.array([int(record["prediction"]) for record in test_predictions])
    submission = pd.DataFrame({"id": [record["id"] for record in test_predictions]})
    for class_index, column in enumerate(TARGET_COLUMNS):
        submission[column] = (predicted == class_index).astype(int)
    expected = EXPECTED_TEST_ROWS if TEST_LIMIT is None else TEST_LIMIT
    assert len(submission) == expected
    assert submission["id"].is_unique
    assert (submission[TARGET_COLUMNS].sum(axis=1) == 1).all()
    submission_path = WORK_ROOT / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    print("Submission:", submission_path)
    print("Prediction counts:", submission[TARGET_COLUMNS].sum().to_dict())
else:
    print("Test inference skipped. Set RUN_TEST_INFERENCE=True only for the selected prompt.")
'''


SHUTDOWN = r'''
for process in SERVER_PROCESSES:
    if process.poll() is None:
        process.terminate()
for process in SERVER_PROCESSES:
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
    process._astroclimb_log_handle.close()
print("Stopped both llama.cpp servers.")

print("Artifacts:")
print(" -", WORK_ROOT / "validation_metrics.json")
print(" -", PREDICTION_ROOT)
if RUN_TEST_INFERENCE:
    print(" -", WORK_ROOT / "submission.csv")
'''


def build_notebook(name, settings):
    config = (
        CONFIG.replace("__VARIANT__", repr(name))
        .replace("__CONTEXT__", str(settings["context"]))
        .replace("__MAX_TOKENS__", str(settings["max_tokens"]))
        .replace("__NEEDS_DEMOS__", repr(settings["needs_demos"]))
    )
    cells = [
        markdown(
            f"""
            # AstroCLIMB — {settings['title']}

            {settings['description']}

            This is an **inference-only prompt ablation** using `unsloth/Qwen3.5-9B-GGUF`
            (`UD-Q4_K_XL`). It performs no training and uses no LoRA or QLoRA adapters. Two
            independent llama.cpp servers run on Kaggle T4×2, one per GPU. Model selection is
            performed on the permanent seed-42 balanced validation split of 800 examples
            (200 per class). Test inference is disabled by default.

            Start a clean Kaggle session with **GPU T4×2** and Internet enabled. Run the three
            notebooks independently and compare validation macro-F1 and per-class F1.
            """
        ),
        markdown("## Install inference dependencies"),
        code(INSTALL),
        markdown("## Configuration"),
        code(config),
        markdown("## Reconstruct the permanent validation split and cache only required objects"),
        code(PREPROCESS),
        markdown("## Prompt construction"),
        code(PROMPTS),
        markdown("## Download the official Unsloth GGUF and launch one server per T4"),
        code(SERVERS),
        markdown("## Resumable two-GPU inference helpers"),
        code(INFERENCE),
        markdown("## Evaluate the fixed 800-row validation split"),
        code(VALIDATE),
        markdown("## Optional test inference for the winning prompt"),
        code(TEST),
        markdown("## Stop servers and list artifacts"),
        code(SHUTDOWN),
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
    for name, settings in VARIANTS.items():
        directory = ROOT / settings["folder"]
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / settings["filename"]
        destination.write_text(
            json.dumps(build_notebook(name, settings), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
