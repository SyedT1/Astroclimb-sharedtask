import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
MIN_PIXELS = 256 * 256
MAX_PIXELS = 448 * 448
MAX_TEXT_CHARS = 3000

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
        {"role": "assistant", "content": [{"type": "text", "text": str(int(row["label"]))}]},
    ]


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, path, oversample_same_figure=False):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle]
        if oversample_same_figure:
            class_zero = [row for row in self.rows if row["label"] == 0]
            self.rows.extend(class_zero)
            self.rows.extend(class_zero)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["_swap"] = random.random() < 0.5
        return row


class LabelOnlyCollator:
    def __init__(self, processor):
        self.processor = processor
        self.label_token_ids = []
        for digit in "0123":
            ids = processor.tokenizer.encode(digit, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Label {digit} is not a single token: {ids}")
            self.label_token_ids.append(ids[0])

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError(f"Expected per-device batch 1, received {len(features)}")
        row = features[0]
        messages = build_messages(row, swap=row.get("_swap", False))
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        target_id = self.label_token_ids[int(row["label"])]
        positions = torch.where(batch["input_ids"][0] == target_id)[0]
        if not len(positions):
            raise RuntimeError("Assistant label token not found.")
        labels = torch.full_like(batch["input_ids"], -100)
        labels[0, int(positions[-1])] = target_id
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--oversample-same-figure", action="store_true")
    parser.add_argument("--save-steps", type=int, default=0)
    args = parser.parse_args()

    # Select this rank's GPU before Transformers/torchao can probe and initialize CUDA.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # These imports happen only after accelerate has created clean worker processes.
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    random.seed(SEED + local_rank)
    np.random.seed(SEED + local_rank)
    torch.manual_seed(SEED + local_rank)
    started = time.perf_counter()

    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    if local_rank == 0:
        model.print_trainable_parameters()
        print(f"Model load on rank 0: {(time.perf_counter() - started) / 60:.2f} min", flush=True)

    dataset = ManifestDataset(args.train_manifest, oversample_same_figure=args.oversample_same_figure)
    if local_rank == 0:
        print(f"Effective training rows: {len(dataset)}", flush=True)
    training_args = TrainingArguments(
        output_dir=str(Path(args.work_root) / "trainer_output"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=max(1, args.save_steps),
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=LabelOnlyCollator(processor),
    )
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started

    if trainer.is_world_process_zero():
        adapter_path = Path(args.adapter_dir)
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        metrics = dict(result.metrics)
        metrics.update(
            {
                "wall_seconds": elapsed,
                "wall_minutes": elapsed / 60,
                "optimizer_steps": int(trainer.state.global_step),
                "seconds_per_optimizer_step": elapsed / max(1, trainer.state.global_step),
                "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
                "effective_training_rows": len(dataset),
            }
        )
        with (adapter_path / "training_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
