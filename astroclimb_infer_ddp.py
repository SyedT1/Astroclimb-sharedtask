import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

MIN_PIXELS = 256 * 256
MAX_PIXELS = 448 * 448
MAX_TEXT_CHARS = 3000
TARGET_COLUMNS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
SYSTEM_PROMPT = """You classify the relationship between two objects from astronomy papers.
0: The objects are the figure and caption of the same scientific figure.
1: The objects are from different figures in the same paper.
2: The objects are from different papers and one paper cites the other.
3: The objects are from unrelated papers.
The relationship is symmetric. Output only one digit: 0, 1, 2, or 3.""".strip()


def shorten_caption(text, max_chars=MAX_TEXT_CHARS):
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[...middle truncated...]\n" + text[-half:]


def object_content(number, obj):
    if obj["kind"] == "image":
        with Image.open(obj["value"]) as source:
            image = source.convert("RGB")
        return [
            {"type": "text", "text": f"Object {number} is a scientific figure:"},
            {"type": "image", "image": image},
        ]
    return [{"type": "text", "text": f"Object {number} is a figure caption:\n{shorten_caption(obj['value'])}"}]


def build_messages(row, swap=False):
    obj_1, obj_2 = row["obj_1"], row["obj_2"]
    if swap:
        obj_1, obj_2 = obj_2, obj_1
    content = object_content(1, obj_1) + object_content(2, obj_2)
    content.append({"type": "text", "text": "Classify their relationship. Reply with one digit only."})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--swap-tta", action="store_true")
    parser.add_argument("--modality-mask", action="store_true")
    args = parser.parse_args()

    # Select this rank's GPU before Transformers/torchao can probe and initialize CUDA.
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    torch.cuda.set_device(rank)

    # Imports occur in fresh accelerate workers, never in a fork of a CUDA-initialized kernel.
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    with Path(args.test_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if args.test_limit >= 0:
        rows = rows[: args.test_limit]
    rows = rows[rank::world_size]

    processor = AutoProcessor.from_pretrained(args.adapter_dir, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": rank},
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    model.config.use_cache = True
    token_ids = []
    for digit in "0123":
        ids = processor.tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {digit} is not one token: {ids}")
        token_ids.append(ids[0])

    @torch.inference_mode()
    def predict(row, swap=False):
        batch = processor.apply_chat_template(
            build_messages(row, swap=swap),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        batch = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        logits = model(**batch).logits[0, -1, token_ids].float()
        if args.modality_mask and row["modality"] in {"CC", "II"}:
            logits[0] = float("-inf")
        return torch.softmax(logits, dim=-1).cpu().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"probabilities_rank{rank}.csv"
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *[f"p_{name}" for name in TARGET_COLUMNS]])
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            probabilities = predict(row)
            if args.swap_tta:
                probabilities = 0.5 * (probabilities + predict(row, swap=True))
            writer.writerow(
                {"id": row["id"], **{f"p_{name}": float(probabilities[i]) for i, name in enumerate(TARGET_COLUMNS)}}
            )
            if index % 100 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rank={rank} {index}/{len(rows)} {elapsed/index:.3f}s/row "
                    f"ETA={(elapsed/index)*(len(rows)-index)/3600:.2f}h",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    print(f"Rank {rank} finished {len(rows)} rows in {elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
