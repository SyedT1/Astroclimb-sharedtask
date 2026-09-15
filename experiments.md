# AstroCLIMB Experiments

## Run 0: QLoRA pipeline check

Status: implemented in [`astroclimb_run0_qwen3vl_qlora.ipynb`](astroclimb_run0_qwen3vl_qlora.ipynb).

Run 0 is a small prototype of the main language-LoRA baseline. Its purpose is to verify data loading, base64 image decoding, multimodal prompting, label-only loss, QLoRA training, constrained inference, timing, and submission generation before committing a full Kaggle session.

| Setting | Run 0 |
|---|---:|
| Model | Qwen3-VL-4B-Instruct |
| Training examples | 204 |
| Validation examples | 52 |
| Vision tower | Frozen |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Learning rate | `1e-4` |
| Training duration | 60 optimizer steps |
| Image-area budget | `448 × 448` |
| Object-order augmentation | Random swap with probability 0.5 |
| GPUs | One T4 intentionally |

Run 0 is successful when it completes without errors, the label-only loss decreases, constrained validation works, and a correctly formatted submission file can be produced. Its score is not expected to be competitive.

## Experiment 1: Tiny-set overfit test

Status: planned immediately after Run 0.

Purpose: prove that the label masking, answer-token positions, LoRA gradients, and prediction code are correct before spending hours on full training.

```yaml
examples: 32
examples_per_class: 8
evaluation_set: the same 32 examples
model: Qwen/Qwen3-VL-4B-Instruct
vision_tower: frozen
lora_rank: 8
lora_alpha: 16
learning_rate: 2.0e-4
max_steps: 150-200
image_area_budget: 448x448
random_object_swap: false
gpu: one T4
```

Disable random swapping for this diagnostic so every training example is identical across passes. Evaluate the same 32 examples after training.

Success criteria:

- Label-only loss falls from approximately random-vocabulary loss toward or below `1.0`.
- Training macro-F1 reaches at least `0.90`.
- All four classes can be produced.
- Repeated inference is deterministic.

If this test fails, inspect label masking, single-token label IDs, LoRA gradients, and the assistant prompt boundary. Do not proceed to full training.

## Experiment 2: Zero-shot versus Run 0

Status: planned after the overfit test.

Purpose: measure whether the 60-step adapter improves over the untouched VLM and establish a baseline score.

Use the same fixed set of 100-200 validation examples for both evaluations:

1. Disable the LoRA adapter and evaluate the base model.
2. Enable the 60-step Run 0 adapter and evaluate again.

Keep the prompt, resolution, label-token restriction, and object order identical. Record macro-F1, per-class F1, prediction counts, seconds per row, and peak allocated memory. Do not run zero-shot inference over the complete test set.

Decision rule: continue with QLoRA if it improves macro-F1, corrects class collapse, or clearly improves the weakest per-class F1. If both systems emit almost one class exclusively, inspect the digit logits and prompt before scaling.

## Experiment 3: Permanent validation split and preprocessing

Status: required before comparing full experiments.

Purpose: ensure every subsequent score is comparable and prevent exact-object leakage between training and validation.

Build a permanent validation manifest from the complete labeled CSV:

```yaml
validation_rows: approximately 800
target_rows_per_class: approximately 200
stratification:
  - target class
  - modality pair
grouping: normalized object hash
seed: 42
```

Requirements:

- Stream the complete label-sorted CSV; never select a contiguous row interval.
- Classify modalities as caption-caption, caption-image, or image-image.
- Hash normalized captions and decoded image bytes.
- Ensure an exact object does not occur in both training and validation.
- Save train and validation IDs to versioned manifests.
- Keep validation at its natural distribution after selection; never oversample validation.
- Decode and resize each unique training image once, then train from image paths rather than repeatedly decoding base64.

All later experiments must use this exact split. If the split definition changes, previous scores are no longer directly comparable.

## Experiment 4: Main language-LoRA baseline

Status: planned after Run 0 succeeds.

This experiment scales the proven Run 0 method to the complete labeled training set.

```yaml
model: Qwen/Qwen3-VL-4B-Instruct
training_rows: approximately 9200
validation_rows: approximately 800
vision_tower: frozen
lora_targets:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-4
epochs: 1
image_area_budget: 448x448
random_object_swap: true
same_figure_oversampling: 3x
precision: FP16
quantization: 4-bit NF4
execution: two-process DDP on two T4 GPUs
```

The vision tower remains frozen because the selected LoRA targets belong to the language attention layers. The base model is loaded in 4-bit, while the adapters train in FP16.

The `same_figure` class should be repeated approximately three times in the training manifest so that the effective class counts are close to balanced for macro-F1. The validation set must remain untouched and retain its natural distribution.

Use two independent processes with one complete quantized model replica per T4. Do not use notebook `DataParallel`; it combines examples into a larger collator batch and attempts to replicate a device-mapped quantized model. Launch the full training script with Accelerate:

```bash
accelerate launch --multi_gpu --num_processes 2 train.py
```

### Relationship to Run 0

The current notebook already tests the same core technique, but it is not the full Experiment 4.

| Setting | Run 0 notebook | Full Experiment 4 |
|---|---:|---:|
| Model | Qwen3-VL-4B | Qwen3-VL-4B |
| Vision tower | Frozen | Frozen |
| LoRA targets | `q/k/v/o_proj` | `q/k/v/o_proj` |
| LoRA rank | 8 | 16 |
| LoRA alpha | 16 | 32 |
| Learning rate | `1e-4` | `1e-4` |
| Image-area budget | 448² | 448² |
| Random object swapping | Yes | Yes |
| Training examples | 204 | Approximately 9,200 |
| Training duration | 60 steps | One complete epoch |
| GPUs | One T4 | Two T4s with DDP |
| Class balancing | 64 examples per class | Oversample `same_figure` 3× |

### Evaluation

Record the following on the fixed validation split:

- Macro-F1 and per-class F1
- Confusion matrix
- Macro-F1 for caption–caption, caption–image, and image–image pairs
- Training wall time and seconds per microbatch
- Inference seconds per row and projected time for 10,000 rows
- Peak allocated GPU memory
- Prediction counts for all four classes

Do not perform full 10,000-row test inference until the checkpoint has been selected using validation results.

## Experiment 5: Modality constraint and class calibration

Status: planned after Experiment 4; no retraining required.

Purpose: incorporate a guaranteed task rule and optimize decisions for macro-F1.

The `same_figure` class is valid only for caption-image pairs. Before softmax or argmax, mask it for the other modality combinations:

```python
if modality in {"caption-caption", "image-image"}:
    class_logits[0] = float("-inf")
```

Then tune four additive class biases using only validation probabilities:

```python
prediction = np.argmax(np.log(probabilities + 1e-12) + class_bias, axis=1)
```

Use a small coordinate or grid search to maximize validation macro-F1. Save the chosen biases with the adapter. Compare these variants:

1. Raw argmax
2. Modality mask only
3. Modality mask plus calibrated class biases

Decision rule: retain every deterministic constraint that improves or preserves macro-F1. Use calibrated biases only if the gain is stable under validation bootstrapping or across folds.

## Experiment 6: Vision LoRA

Status: planned after the language-only baseline.

Purpose: determine whether adapting visual layers improves scientific-figure matching, especially `same_figure` and image-image performance.

```yaml
model: Qwen/Qwen3-VL-4B-Instruct
training_split: identical to Experiment 4
vision_and_language_lora: true
lora_targets: all-linear
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 5.0e-5
epochs: 1
image_area_budget: 448x448
same_figure_oversampling: 3x
random_object_swap: true
execution: two-process DDP on two T4 GPUs
```

Start from the base model for a clean comparison unless a continuation run is explicitly recorded as such. Compare against Experiment 4 using the identical validation split and inference constraints.

Decision rule: keep vision LoRA only if overall macro-F1 improves by at least `0.01`, or if a meaningful `same_figure`/image-image improvement occurs without substantially damaging the other classes. Record its effect on memory and seconds per microbatch.

## Experiment 7: Image-resolution ablation

Status: planned using the better adapter strategy from Experiments 4 and 6.

Purpose: test whether small plot labels, legends, panel annotations, and astronomical identifiers benefit from additional vision tokens.

| Modality | Baseline budget | Higher-resolution budget |
|---|---:|---:|
| Caption-image | 448² total pixels | 768² total pixels |
| Image-image | 448² per image | 512² per image |
| Caption-caption | No image | No image |

Keep every other training setting fixed. First benchmark the higher resolution on 200-400 validation examples to measure memory and inference cost. If it fits, train the corresponding adapter for one epoch.

Decision rule: retain higher resolution if its macro-F1 gain justifies the measured runtime. Prefer the 4B model at useful resolution over moving prematurely to an 8B model with unreadably small figures.

If an out-of-memory error occurs, reduce two-image resolution first, then maximum text length. LoRA rank should be reduced only after activation-heavy settings have been adjusted.

## Experiment 8: Symmetry test-time augmentation

Status: planned after selecting the best trained adapter.

Purpose: enforce the task's symmetric relation at inference time.

Evaluate each validation pair in both orders and average probabilities:

```python
p_forward = predict(obj_1, obj_2)
p_reverse = predict(obj_2, obj_1)
p_final = 0.5 * (p_forward + p_reverse)
```

Record forward-only macro-F1, averaged macro-F1, the fraction of pairs whose predicted class changes when reversed, and the exact runtime multiplier.

Decision rule: use symmetry TTA for the full test set only if validation macro-F1 improves by at least `0.005` or it fixes a clear instability. It approximately doubles inference time, so a negligible gain is not worthwhile.

## Experiment 9: Citation-focused auxiliary training

Status: conditional; run only if `related_papers` and `unrelated_papers` remain the main confusion.

Purpose: teach the language model citation-related similarity using the public Hugging Face metadata without first downloading every image.

Generate an auxiliary caption-caption dataset:

```yaml
related_pairs: 25000
hard_unrelated_pairs: 25000
source: AstroCLIMB DOI and citation graph
hard_negative_rule: topically similar papers with no citation edge
stage_1_epochs: 1
stage_2: fine-tune on the Kaggle multimodal training split
```

Random unrelated papers are likely too easy. Construct hard unrelated examples using caption/title embedding similarity while confirming that the papers have different DOIs and no citation edge. Keep all validation papers out of auxiliary training when their DOI mapping is available.

Compare:

1. Experiment 4 without auxiliary training
2. Auxiliary related versus hard-unrelated training, followed by Experiment 4 fine-tuning

Decision rule: retain this stage if it improves `related_papers` F1 and overall macro-F1 without turning semantically similar unrelated examples into excessive false positives.

## Experiment 10: Checkpoint ensemble and final test inference

Status: final stage only.

Purpose: produce the competition submission from configurations already selected on validation.

Candidate ensemble members should be meaningfully different, such as the best language-only adapter and the best vision-LoRA or higher-resolution adapter. Do not ensemble weak checkpoints merely because they exist.

Average saved class probabilities, apply the modality constraint and selected class biases, then produce exactly one one-hot class per row. Test a small grid of ensemble weights on validation, for example `0.25/0.75`, `0.5/0.5`, and `0.75/0.25`.

Run final test inference as two independent shards:

- GPU/process 0 handles one 5,000-row shard.
- GPU/process 1 handles the other 5,000-row shard.
- Each process loads one complete 4-bit model and adapter.
- Merge shards by `id` and validate all 10,000 IDs exactly once.

Final submission checks:

- Columns are exactly `id`, `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`.
- IDs are unique and cover every test row.
- Every prediction value is `0` or `1`.
- Every row sums to exactly one.
- There are no missing values.
- Probability files and the exact adapter/configuration are saved for reproducibility.

## Execution order and stopping rules

Run experiments in this order:

```text
Run 0 pipeline and timing check
→ Experiment 1 tiny-set overfit
→ Experiment 2 zero-shot comparison
→ Experiment 3 permanent validation/preprocessing
→ Experiment 4 full language-LoRA baseline
→ Experiment 5 constraints and calibration
→ Experiment 6 vision LoRA
→ Experiment 7 resolution ablation
→ Experiment 8 symmetry TTA
→ Experiment 9 auxiliary citation training, only if needed
→ Experiment 10 ensemble and final inference
```

General stopping rules:

- Do not scale training until the tiny-set overfit test passes.
- Do not compare models on different validation splits.
- Do not run full test inference for exploratory checkpoints.
- Treat a macro-F1 change smaller than `0.005` as noise unless repeated across folds or seeds.
- Prefer changes of at least `0.01` before accepting a substantial runtime penalty.
- Stop a run early if it predicts almost one class exclusively and validation does not recover.
- Keep `USE_SWAP_TTA=False` until Experiment 8 demonstrates that its doubled cost is justified.

## Experiment log template

Record one row per run in a CSV or spreadsheet with these fields:

```text
run_id
date
data_manifest_version
validation_split_version
seed
model
lora_targets
lora_rank
lora_alpha
learning_rate
epochs_or_steps
image_budget_caption_image
image_budget_image_image
class_balancing
random_swap
gpu_count
training_minutes
seconds_per_microbatch
peak_gpu_gib
validation_seconds_per_row
macro_f1
f1_same_figure
f1_same_paper
f1_related_papers
f1_unrelated_papers
macro_f1_caption_caption
macro_f1_caption_image
macro_f1_image_image
prediction_counts
notes
```
