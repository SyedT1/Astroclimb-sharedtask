# AstroCLIMB Experiments

This file is the experiment ledger and execution queue. Kaggle scores are user-reported macro-F1 values. Model selection for future work must use a fixed validation split rather than repeated leaderboard feedback.

## Current result summary

| Run | Notebook | Model | Training data | Loss | LoRA coverage | Kaggle macro-F1 | Status |
|---|---|---|---:|---|---|---:|---|
| Run 0 | [`astroclimb_run0_qwen3vl_qlora.ipynb`](astroclimb_run0_qwen3vl_qlora.ipynb) | Qwen3-VL-4B | 204 rows | Full-vocabulary label-token CE | Language `q/k/v/o` attention, rank 8 | — | Pipeline check completed |
| E1 | [`astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb) | Qwen3-VL-4B | 5,000 balanced rows | Full-vocabulary label-token CE | Language `q/k/v/o` attention | 0.67856 | Completed |
| E2 | [`astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb) | Qwen3-VL-4B | 10,000 rows | Full-vocabulary label-token CE | Language `q/k/v/o` attention | 0.70751 | Completed |
| E3 | [`astroclimb-restricted4class-qwen3vl-qlora.ipynb`](astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb) | Qwen3-VL-4B | 9,200 train / 800 validation | Restricted four-class CE | Language `q/k/v/o` attention | 0.71453 | Completed |
| E4 | [`astroclimb-qwen3vl8b-language-qlora.ipynb`](astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb) | Qwen3-VL-8B | 10,000 rows | Full-vocabulary label-token CE | Language `q/k/v/o` attention | **0.73230** | Completed; current best |
| E5 | [`astroclimb-alllinear-vision-language-qlora.ipynb`](astroclimb_alllinear_vision_language_qlora/astroclimb-alllinear-vision-language-qlora.ipynb) | Qwen3-VL-4B | 10,000 rows | Full-vocabulary label-token CE | All eligible vision and language linear layers | 0.71202 | Completed |
| E6 | [`astroclimb-language-projector-qlora.ipynb`](astroclimb_language_projector_qlora/astroclimb-language-projector-qlora.ipynb) | Qwen3-VL-4B | 10,000 rows | Full-vocabulary label-token CE | Language attention plus visual merger projectors | 0.71016 | Completed |
| Retired ablation | Notebook removed | Qwen3-VL-4B | Image-containing rows only | Full-vocabulary label-token CE | Visual merger projectors only | 0.48182 | Completed, diagnosed, and retired |

## Conclusions from completed experiments

- Scaling the language-attention model from 4B to 8B produced the largest observed gain: `0.70751 → 0.73230`.
- Restricting training loss to the four valid class tokens improved the 4B result: `0.70751 → 0.71453`, although the runs also differ in epoch count and training split.
- Broad visual adaptation was not competitive with the 8B language-attention run. All-linear and language-plus-projector variants reached `0.71202` and `0.71016`.
- Vision-projector-only adaptation failed badly at `0.48182`. Caption–caption rows do not traverse those trainable modules, and the resulting predictions strongly overproduced `same_paper`. Do not repeat this configuration.
- The next experiment should combine the two strongest changes: the 8B backbone and restricted four-class loss.

## Shared implementation that has been validated

```yaml
quantization: 4-bit NF4 with double quantization
compute_dtype: FP16
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
image_area_budget: 448x448
random_object_swap: true
per_device_batch_size: 1
gradient_accumulation: 8
execution: two-process DDP on two T4 GPUs
inference: four label-token logits only
```

The common submission pipeline correctly creates 10,000 unique IDs and one-hot predictions over `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`.

## Validation policy for all new experiments

Use the same permanent split for every controlled comparison:

```yaml
training_rows: 9200
validation_rows: 800
validation_examples_per_class: 200
seed: 42
```

Requirements:

- Save the exact train and validation IDs.
- Do not tune class biases, epochs, or ensemble weights from Kaggle scores.
- Save validation probabilities in addition to hard predictions.
- Report macro-F1, per-class F1, confusion matrix, modality-level F1, prediction counts, runtime, and peak memory.
- Treat gains below `0.005` as inconclusive unless they repeat across seeds or folds.
- Do not run complete test inference until a configuration has been selected on validation.

## N1: Qwen3-VL-8B restricted-loss experiment

Status: **completed**. Attention-only LoRA peaked at validation macro-F1 `0.729164` at checkpoint 863 (1.5 epochs).

Notebook: [`astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_validation_qlora/astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb)

Purpose: combine the best backbone result with the loss that improved the 4B model.

```yaml
model: Qwen/Qwen3-VL-8B-Instruct
training_rows: 9200
validation_rows: 800
loss: restricted_four_class_cross_entropy
epochs: 2
evaluation_interval: 0.5 epoch
learning_rate: 5e-5
lora_targets: [q_proj, k_proj, v_proj, o_proj]
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
global_batch_size: 16
image_area_budget: 448x448
```

Evaluate near 0.5, 1.0, 1.5, and 2.0 epochs and restore the checkpoint with the highest validation macro-F1. Continue only if memory, runtime, and class distributions remain healthy.

Success criteria:

- Beat the comparable 4B restricted-loss validation result by at least `0.005`.
- Avoid deterioration in `related_papers` F1.
- Remain within the Kaggle T4×2 session and memory limits.

## N2: Constraints, calibration, and symmetry analysis

Status: run after N1; no retraining required.

Evaluate on saved validation probabilities:

1. Raw four-class argmax.
2. Mask `same_figure` for caption–caption and image–image pairs.
3. Global additive class biases.
4. Modality-specific additive class biases.
5. Forward/reverse probability averaging.

```python
if modality in {"CC", "II"}:
    class_logits[0] = float("-inf")

prediction = np.argmax(np.log(probabilities + 1e-12) + class_bias, axis=1)
```

Use a postprocessing option only if it improves validation macro-F1 by at least `0.005` and remains stable under bootstrap resampling. Use swap TTA only if its gain justifies approximately doubling inference cost.

## N3: Full-10K 8B restricted-loss refit

Status: **notebook locked to the N1 winner; run next**. The selected configuration is attention-only LoRA for 1.5 epochs.

Notebook: [`astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_full10k_qlora/astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb)

Retrain from the original 8B base model on all 10,000 labeled rows using the selected epoch count. Apply only validation-selected postprocessing during test inference.

Required artifacts:

- adapter and processor;
- exact training configuration and metrics;
- raw probability shards;
- merged probability file;
- validated 10,000-row submission.

The leaderboard target is to improve on the current best score of `0.73230`.

## N4: Restricted-loss language-MLP expansion

Status: **completed**. Attention + MLP peaked at `0.729019` after 2.0 epochs, only `0.000145` below attention-only while using more memory and time; it was not selected.

Notebook: [`astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_language_mlp_qlora/astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb)

Test whether additional language capacity helps distinguish `same_paper`, `related_papers`, and `unrelated_papers`:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

Use the fixed 9,200/800 split, restricted loss, rank 16, alpha 32, dropout 0.05, and the same two-epoch half-epoch checkpoint schedule as N1. Compare against attention-only LoRA with every other setting fixed. Do not prioritize more visual-only adapter experiments.

## N5: Probability ensemble

Status: final-stage experiment.

Candidate members:

- current 8B full-vocabulary model (`0.73230`);
- N1/N3 8B restricted-loss model;
- 4B restricted-loss model (`0.71453`) only if its validation errors are complementary.

Test validation-selected weights such as `0.25/0.75`, `0.50/0.50`, and `0.75/0.25`. Average probabilities before applying the modality mask and calibrated biases. Hard one-hot submissions cannot be used as a proper probability ensemble.

## N6: Citation-focused auxiliary training

Status: conditional on `related_papers` remaining the weakest class.

Generate caption pairs from the public DOI citation graph:

| Pair type | Rows |
|---|---:|
| Same paper | 20,000 |
| Related papers | 30,000 |
| Random unrelated papers | 15,000 |
| Hard unrelated papers | 15,000 |
| **Total** | **80,000** |

Use one auxiliary epoch followed by one restricted-loss epoch on the Kaggle multimodal split. Keep validation papers out of auxiliary training wherever DOI mapping is available. Retain this stage only if it improves both overall macro-F1 and `related_papers` F1.

## Lower-priority experiments

- Higher image resolution, after an inference-only memory and speed benchmark.
- Longer caption context.
- Rank 8/16/32 and dropout 0/0.05/0.10 ablations.
- Multi-seed probability ensembles.

These are lower priority because model scale and class-restricted loss have stronger evidence than vision expansion or small adapter changes.

## Execution order

```text
N1: 8B restricted loss on 9,200/800
  → N2: modality mask, calibration, and symmetry analysis
  → N3: selected 8B configuration refit on all 10K
  → submit and compare with 0.73230
  → N4: language attention + MLP restricted-loss ablation
  → N5: probability ensemble
  → N6: auxiliary citation training only if related_papers remains weak
```

## Experiment log template

Record one row per run:

```text
run_id
date
notebook
model
train_manifest
validation_manifest
seed
loss_type
epochs_or_steps
learning_rate
lora_rank
lora_alpha
lora_dropout
lora_targets
image_budget
random_swap
modality_mask
class_biases
swap_tta
gpu_count
training_minutes
inference_seconds_per_row
peak_gpu_gib
validation_macro_f1
f1_same_figure
f1_same_paper
f1_related_papers
f1_unrelated_papers
prediction_counts
kaggle_macro_f1
notes
```

## Submission checks

Before every upload, verify:

- exactly 10,000 rows;
- IDs are unique and match the test set;
- columns are exactly `id`, `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`;
- every target value is `0` or `1`;
- every row sums to exactly one;
- there are no missing values or saved DataFrame index columns.
