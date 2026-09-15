# AstroCLIMB Score-Improvement Plan

## Objective

Improve the current best Kaggle macro-F1 of **0.70751** under the Kaggle **T4×2** compute constraint, while keeping model selection independent of hidden test labels.

The main technical target is better discrimination among:

- `same_paper`;
- `related_papers`;
- `unrelated_papers`.

The existing 5K validation run already obtained approximately `0.918` F1 for `same_figure`, so experiments that primarily improve visual figure–caption matching have lower priority.

## Current baselines

| Run | Model | Training data | Epochs | Loss | Kaggle macro-F1 |
|---|---|---:|---:|---|---:|
| 5K QLoRA | Qwen3-VL-4B-Instruct | 5,000 balanced pairs | 1 | Full-vocabulary label-token CE | 0.67856 |
| Full-10K QLoRA | Qwen3-VL-4B-Instruct | 10,000 pairs | 1 | Full-vocabulary label-token CE | **0.70751** |

Common baseline configuration:

```yaml
quantization: 4-bit NF4 with double quantization
compute_dtype: FP16
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]
image_area_budget: 448x448
random_object_swap: true
execution: two-process DDP on two T4 GPUs
```

## Data and validation policy

Use one permanent split for every model comparison:

```yaml
training_rows: 9200
validation_rows: 800
validation_examples_per_class: 200
seed: 42
```

Requirements:

- Never tune against `data/solution.csv` or hidden test labels.
- Never select class biases from Kaggle leaderboard feedback.
- Use the same validation IDs for every controlled comparison.
- Cache decoded images by hash and reuse the same manifests.
- Save validation probabilities, not only hard predictions.
- Report macro-F1, per-class F1, confusion matrix, modality-level F1, prediction counts, runtime, and peak memory.
- Treat improvements below `0.005` as inconclusive unless repeated across seeds.

## Experiment R1: restricted four-class loss

Status: **notebook implemented; run next**.

Notebook: [`astroclimb-restricted4class-qwen3vl-qlora.ipynb`](astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb)

Replace full-vocabulary causal-language-model loss with cross-entropy over only the four valid digit-token logits:

```python
outputs = model(**model_inputs)
next_token_logits = outputs.logits[:, answer_position - 1]
class_logits = next_token_logits[:, label_token_ids]
loss = torch.nn.functional.cross_entropy(class_logits, target_class)
```

Configuration:

```yaml
model: Qwen/Qwen3-VL-4B-Instruct
training_rows: 9200
validation_rows: 800
epochs: 2
learning_rate: 5e-5
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]
image_area_budget: 448x448
global_batch_size: 16
```

Evaluate and save at these optimizer steps:

| Step | Approximate epoch |
|---:|---:|
| 288 | 0.5 |
| 575 | 1.0 |
| 863 | 1.5 |
| 1,150 | 2.0 |

Select the checkpoint with the highest validation macro-F1.

Decision rule:

- Continue to a full-data refit if the best restricted-loss checkpoint clearly improves the comparable validation result.
- Stop or revise the loss if class predictions collapse or `related_papers` F1 deteriorates.

## Experiment R1A: LoRA configuration ablations

Status: planned after R1 confirms that restricted four-class loss trains correctly.

LoRA configuration should be ablated on the identical 9,200/800 split, but it has lower priority than validating the new loss. Change only one factor at a time and initially train each candidate for one epoch. Continue only the winning configuration to the full two-epoch checkpoint search.

### Rank and alpha ablation

Keep `lora_alpha = 2 × rank` so adapter scaling remains comparable:

| Run | LoRA rank | LoRA alpha | Dropout | Target modules |
|---|---:|---:|---:|---|
| L1 | 8 | 16 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| L2 | **16** | **32** | **0.05** | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| L3 | 32 | 64 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

Interpretation:

- Rank 8 may regularize better with only 9,200 supervised pairs and will be cheaper.
- Rank 16 is the established baseline.
- Rank 32 provides more capacity but may overfit or offer too little gain for its cost.

Select the rank using validation macro-F1, not training loss.

### Target-module ablation

After choosing the rank, compare:

| Run | Target modules | Purpose |
|---|---|---|
| T1 | `q_proj`, `v_proj` | Small, strongly regularized attention adapter |
| T2 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Current attention-only baseline |
| T3 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Adapt language attention and MLP transformations |
| T4 | All linear layers | Maximum adaptation, potentially including visual modules |

The highest-priority target expansion is:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

This may help the semantic distinction between `related_papers` and `unrelated_papers`, but it will train more parameters than attention-only LoRA.

Treat `all-linear` as a separate vision-and-language adaptation experiment. Record exactly which modules PEFT matches; do not assume that every matched module belongs to the language model.

### Dropout ablation

Test dropout only after rank and target modules have been selected:

| Run | LoRA dropout | Motivation |
|---|---:|---|
| D1 | 0.00 | Maximum fitting capacity |
| D2 | **0.05** | Current baseline |
| D3 | 0.10 | Stronger regularization if later checkpoints overfit |

Use the checkpoint trajectory to interpret dropout:

- If validation continues improving through two epochs, prefer `0.00` or `0.05`.
- If training loss improves while validation macro-F1 declines, test `0.10`.

### Compact LoRA sweep

To control GPU time, run only this compact sequence:

| Run | Rank/alpha | Targets | Dropout | Initial epochs |
|---|---|---|---:|---:|
| A | 8/16 | `q/k/v/o_proj` | 0.05 | 1 |
| B | 16/32 | `q/k/v/o_proj` | 0.05 | 1 |
| C | 32/64 | `q/k/v/o_proj` | 0.05 | 1 |
| D | Best rank/alpha | `q/k/v/o_proj + gate/up/down_proj` | 0.05 | 1 |
| E | Best rank/alpha | All linear layers | 0.05 | 1 |

After selecting the best rank and targets, test dropout only if the validation trajectory shows a reason to do so. Train the final winning LoRA configuration for two epochs with the same half-epoch checkpoint schedule as R1.

### LoRA selection criteria

Rank candidates in this order:

1. Validation macro-F1
2. `related_papers` F1
3. Mean macro-F1 across caption–caption, caption–image, and image–image subsets
4. Stability across the half-epoch checkpoints
5. Peak GPU memory and training time

Decision rules:

- Require at least `0.005` macro-F1 improvement to replace the current LoRA configuration.
- Prefer the smaller adapter when scores differ by less than `0.005`.
- Stop a candidate if predictions collapse or the projected runtime threatens the Kaggle limit.
- Do not change rank, alpha, targets, learning rate, and dropout simultaneously.
- Do not promote a configuration based only on lower training loss.

## Experiment R2: modality constraints and calibration

Status: planned immediately after R1; no retraining required.

Evaluate these variants using R1 validation probabilities:

1. Raw four-class argmax
2. Modality mask only
3. Global additive class biases
4. Modality-specific class biases
5. Modality mask plus modality-specific biases

Apply the guaranteed structural rule:

```python
if modality in {"CC", "II"}:
    class_logits[0] = float("-inf")
```

Tune biases only on validation predictions:

```python
prediction = np.argmax(np.log(probabilities + 1e-12) + class_bias, axis=1)
```

Decision rule: retain a deterministic postprocessing variant if it improves validation macro-F1 by at least `0.005` and does not produce an implausible class distribution.

## Experiment R3: full-10K restricted-loss refit

Status: run after selecting the R1 checkpoint and R2 postprocessing.

Retrain from the original base model on all 10,000 labeled pairs using the selected epoch count.

Example when R1 selects 1.5 epochs:

```yaml
model: Qwen/Qwen3-VL-4B-Instruct
training_rows: 10000
epochs: 1.5
loss: restricted_four_class_cross_entropy
learning_rate: 5e-5
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
```

Run inference on all 10,000 test rows, apply only the R2 settings chosen on validation, and compare the Kaggle score against `0.70751`.

Required artifacts:

- final adapter;
- training configuration and metrics;
- raw probability shards;
- merged probability file;
- validated 10,000-row `submission.csv`.

## Experiment R4: Qwen3-VL-8B pilot

Status: conditional on finishing R1–R3.

Start with a small feasibility run:

```yaml
model: Qwen/Qwen3-VL-8B-Instruct
training_rows: 2000
max_optimizer_steps: 200
loss: restricted_four_class_cross_entropy
quantization: 4-bit NF4
lora_rank: 16
lora_alpha: 32
learning_rate: 5e-5
image_area_budget: 448x448
per_device_batch_size: 1
gradient_accumulation: 8
gpus: 2x T4
```

Continue to the complete 9,200/800 experiment only if:

- peak allocated memory remains below approximately 14 GiB per GPU;
- image–image examples do not cause OOM failures;
- the projected run fits within the Kaggle session limit;
- early validation is meaningfully better than the 4B result.

If memory is insufficient, reduce image area before reducing LoRA rank.

## Experiment R5: citation-focused auxiliary QLoRA

Status: planned if `related_papers` remains the weakest class.

Metadata retrieval cannot resolve the current Kaggle test objects, so DOI metadata must not be used as a test-time override. Use the public Hugging Face graph only to generate supervised training examples.

Initial caption-focused auxiliary set:

| Pair type | Rows |
|---|---:|
| Same paper | 20,000 |
| Related papers | 30,000 |
| Random unrelated papers | 15,000 |
| Hard unrelated papers | 15,000 |
| **Total** | **80,000** |

Hard unrelated examples should contain topically similar captions from papers with different DOIs and no citation edge.

Two-stage training:

```yaml
stage_1:
  data: 80000 generated caption pairs
  epochs: 1
  learning_rate: 5e-5
stage_2:
  data: Kaggle multimodal training split
  epochs: 1
  learning_rate: 2e-5
loss: restricted_four_class_cross_entropy
```

Decision rule: keep auxiliary training only if it improves both overall macro-F1 and `related_papers` F1 on the permanent validation set.

## Experiment R6: probability ensemble

Status: final-stage experiment.

Candidate members:

- existing full-10K 4B QLoRA (`0.70751`);
- best restricted-loss 4B adapter;
- restricted-loss 8B adapter or auxiliary-trained adapter.

Test simple validation-selected weights:

```text
0.25 / 0.75
0.50 / 0.50
0.75 / 0.25
```

Use only models with complementary validation errors. Apply the selected modality mask and calibration after probability averaging.

## Experiment R7: symmetry test-time augmentation

Status: optional final inference enhancement.

```python
p_forward = predict(obj_1, obj_2)
p_reverse = predict(obj_2, obj_1)
p_final = 0.5 * (p_forward + p_reverse)
```

Decision rule: use symmetry TTA on the complete test set only if validation macro-F1 improves by at least `0.005`, because it approximately doubles inference time.

## Lower-priority experiments

Run these only after the preceding experiments:

- Higher figure resolution
- Multi-seed ensembles
- Longer caption context

These are lower priority because they are more likely to provide incremental gains, while the current error profile is dominated by paper-level semantic relationships rather than `same_figure` recognition.

## Execution order

```text
R1: restricted loss on 9,200/800
  → R1A: compact LoRA rank and target-module ablation
  → R2: modality mask and validation calibration
  → R3: selected configuration refit on all 10K
  → submit and compare with 0.70751
  → R4: 8B feasibility pilot and full run if viable
  → R5: auxiliary citation training if related_papers remains weak
  → R6: probability ensemble
  → R7: symmetry TTA if its validation gain justifies the cost
```

## Experiment log

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

Before every Kaggle upload, verify:

- exactly 10,000 prediction rows;
- IDs are unique and match the test set;
- columns are exactly `id`, `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`;
- every class value is `0` or `1`;
- every row sums to exactly one;
- there are no missing values or saved DataFrame index columns.

## Stop conditions

- Stop any run that collapses to one or two predicted classes.
- Do not promote a costly change for a validation improvement below `0.005`.
- Do not run full-test inference for an unselected checkpoint.
- Stop optional work early enough to preserve time for the final 10K refit and submission.
- Do not resume direct metadata lookup as a test-time classifier unless an official dataset release materially changes test-object coverage.
