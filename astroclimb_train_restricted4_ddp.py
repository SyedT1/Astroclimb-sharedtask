import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    def __init__(self, path, random_swap=False):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle]
        self.random_swap = random_swap

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["_swap"] = self.random_swap and random.random() < 0.5
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
        batch = self.processor.apply_chat_template(
            build_messages(row, swap=row.get("_swap", False)),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        target_id = self.label_token_ids[int(row["label"])]
        positions = torch.where(batch["input_ids"][0] == target_id)[0]
        if not len(positions):
            raise RuntimeError("Assistant label token was not found in the rendered conversation.")
        labels = torch.full_like(batch["input_ids"], -100)
        labels[0, int(positions[-1])] = target_id
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Import after rank device selection so optional CUDA probes use the correct T4.
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import f1_score
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    random.seed(SEED + local_rank)
    np.random.seed(SEED + local_rank)
    torch.manual_seed(SEED + local_rank)
    load_started = time.perf_counter()

    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "right"
    collator = LabelOnlyCollator(processor)
    label_token_ids_cpu = torch.tensor(collator.label_token_ids, dtype=torch.long)
    if local_rank == 0:
        print(f"Label token IDs: {collator.label_token_ids}", flush=True)

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
        print(f"Model load: {(time.perf_counter() - load_started) / 60:.2f} min", flush=True)

    class RestrictedFourClassTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            supervised = labels.ne(-100)
            if not supervised.any(dim=1).all():
                raise RuntimeError("Every example must contain one supervised answer token.")
            answer_positions = supervised.to(torch.int64).argmax(dim=1)
            if (answer_positions == 0).any():
                raise RuntimeError("Answer token cannot occur at position zero.")
            batch_indices = torch.arange(labels.shape[0], device=labels.device)
            vocabulary_logits = outputs.logits[batch_indices, answer_positions - 1]
            label_token_ids = label_token_ids_cpu.to(vocabulary_logits.device)
            class_logits = vocabulary_logits.index_select(-1, label_token_ids).float()
            target_token_ids = labels[batch_indices, answer_positions]
            matches = target_token_ids[:, None].eq(label_token_ids[None, :])
            if not matches.any(dim=1).all():
                raise RuntimeError("A target token is outside the restricted four-label vocabulary.")
            class_targets = matches.to(torch.int64).argmax(dim=1)
            loss = F.cross_entropy(class_logits, class_targets)
            return (loss, outputs) if return_outputs else loss

    def restrict_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        supervised = labels.ne(-100)
        answer_positions = supervised.to(torch.int64).argmax(dim=1)
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        label_token_ids = label_token_ids_cpu.to(logits.device)
        return logits[batch_indices, answer_positions - 1].index_select(-1, label_token_ids)

    def compute_metrics(prediction):
        class_logits = np.asarray(prediction.predictions)
        labels = np.asarray(prediction.label_ids)
        target_token_ids = np.array(
            [row[np.flatnonzero(row != -100)[0]] for row in labels],
            dtype=np.int64,
        )
        token_to_class = {token_id: index for index, token_id in enumerate(collator.label_token_ids)}
        targets = np.array([token_to_class[int(token_id)] for token_id in target_token_ids])
        predictions = class_logits.argmax(axis=-1)
        metrics = {"macro_f1": f1_score(targets, predictions, average="macro")}
        per_class = f1_score(targets, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0)
        metrics.update({f"f1_class_{index}": float(score) for index, score in enumerate(per_class)})
        return metrics

    class QuarterMilestoneCallback(TrainerCallback):
        """Evaluate and save at 25%, 50%, 75%, and 100% of optimizer steps."""

        def on_train_begin(self, args, state, control, **kwargs):
            self.milestones = {
                max(1, int(state.max_steps * fraction + 0.5))
                for fraction in (0.25, 0.50, 0.75, 1.00)
            }
            if state.is_world_process_zero:
                print(f"Evaluation/checkpoint milestones: {sorted(self.milestones)}", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.milestones:
                control.should_evaluate = True
                control.should_save = True
            return control

    train_dataset = ManifestDataset(args.train_manifest, random_swap=True)
    validation_dataset = ManifestDataset(args.validation_manifest, random_swap=False)
    if local_rank == 0:
        print(f"Train rows: {len(train_dataset)} | Validation rows: {len(validation_dataset)}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(Path(args.work_root) / "trainer_output"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
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
        eval_strategy="steps",
        save_strategy="steps",
        # The callback below triggers the real quarter-run events. These large
        # equal values satisfy best-model strategy validation without adding events.
        eval_steps=10000,
        save_steps=10000,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=SEED,
    )
    trainer = RestrictedFourClassTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=restrict_logits_for_metrics,
        callbacks=[QuarterMilestoneCallback()],
    )

    torch.cuda.synchronize()
    train_started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started
    final_validation = trainer.evaluate()

    if trainer.is_world_process_zero():
        adapter_path = Path(args.adapter_dir)
        adapter_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(adapter_path)
        processor.save_pretrained(adapter_path)
        metrics = dict(result.metrics)
        metrics.update({f"best_{key}": value for key, value in final_validation.items()})
        metrics.update(
            {
                "wall_seconds": elapsed,
                "wall_minutes": elapsed / 60,
                "optimizer_steps": int(trainer.state.global_step),
                "seconds_per_optimizer_step": elapsed / max(1, trainer.state.global_step),
                "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "train_rows": len(train_dataset),
                "validation_rows": len(validation_dataset),
                "loss_type": "restricted_four_class_cross_entropy",
            }
        )
        with (adapter_path / "training_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
