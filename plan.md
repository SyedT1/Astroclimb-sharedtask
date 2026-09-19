# AstroCLIMB Score-Improvement Plan

## Objective

Improve the current best user-reported Kaggle macro-F1 of **0.73725** under the Kaggle **T4×2** constraint. Future model and inference choices must be selected on a fixed validation set, independently of hidden test labels and leaderboard feedback.

The main error-reduction target remains discrimination among:

- `same_paper`;
- `related_papers`;
- `unrelated_papers`.

The evidence now favors Qwen3-VL-8B with language-attention QLoRA and the original full-vocabulary objective over broader visual adaptation or auxiliary citation pretraining.

## Completed work

### Scoreboard

| Run | Model | Rows | Epochs | Loss | Adapter targets | Kaggle macro-F1 |
|---|---|---:|---:|---|---|---:|
| Balanced 5K | Qwen3-VL-4B | 5,000 | 1 | Full-vocabulary CE | Language attention | 0.67856 |
| Full 10K baseline | Qwen3-VL-4B | 10,000 | 1 | Full-vocabulary CE | Language attention | 0.70751 |
| Language + projector | Qwen3-VL-4B | 10,000 | 1 | Full-vocabulary CE | Language attention + visual merger | 0.71016 |
| All-linear | Qwen3-VL-4B | 10,000 | 1 | Full-vocabulary CE | Vision and language linear layers | 0.71202 |
| Restricted four-class | Qwen3-VL-4B | 9,200 + 800 validation | 2 | Restricted four-class CE | Language attention | 0.71453 |
| 8B language attention, seed 42 | Qwen3-VL-8B | 10,000 | 1 | Full-vocabulary CE | Language attention | 0.73230 |
| 8B language attention, seed 123 | Qwen3-VL-8B | 10,000 | 1 | Full-vocabulary CE | Language attention | 0.73250 |
| 8B language attention, seed 17 | Qwen3-VL-8B | 10,000 | 1 | Full-vocabulary CE | Language attention | **0.73307** |
| 8B restricted four-class | Qwen3-VL-8B | 10,000 | 1.5 | Restricted four-class CE | Language attention | 0.73032 |
| Three-seed probability ensemble | Qwen3-VL-8B | 3 × 10,000 | 1 each | Full-vocabulary CE | Language attention | 0.73604 |
| 75% full ensemble + 25% restricted blend | Qwen3-VL-8B | Derived | — | Probability blend | — | 0.73590 |
| Three-seed ensemble + modality mask | Qwen3-VL-8B | Derived | — | Deterministic postprocessing | — | **0.73725** |
| Retired vision-only ablation | Qwen3-VL-4B | Image-containing rows | 1 | Full-vocabulary CE | Visual merger only | 0.48182 |

### Implemented notebooks

- [`astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb)
- [`astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb)
- [`astroclimb-restricted4class-qwen3vl-qlora.ipynb`](astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-qlora.ipynb`](astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-seed17-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-seed123-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-seed42-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed42_qlora/astroclimb-qwen3vl8b-language-seed42-qlora.ipynb)
- [`astroclimb-qwen3vl8b-full-restricted-blend.ipynb`](astroclimb-qwen3vl8b-full-restricted-blend.ipynb)
- [`astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb`](astroclimb-qwen3vl8b-three-seed-modality-mask/astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb)
- [`astroclimb-alllinear-vision-language-qlora.ipynb`](astroclimb_alllinear_vision_language_qlora/astroclimb-alllinear-vision-language-qlora.ipynb)
- [`astroclimb-language-projector-qlora.ipynb`](astroclimb_language_projector_qlora/astroclimb-language-projector-qlora.ipynb)

The vision-projector-only notebook was removed after its `0.48182` result showed that visual merger adaptation alone cannot learn the caption-heavy paper relationships.

## Evidence and decisions

### 1. Model scale is the strongest demonstrated lever

Moving from the 4B full-10K language-attention model to the best analogous 8B seed improved macro-F1 by `0.02556`. This is larger than any observed adapter-coverage change.

Decision: use Qwen3-VL-8B as the primary backbone for the next controlled experiment.

### 2. Full-vocabulary loss remains the primary 8B objective

The 4B restricted-loss run scored `0.71453`, an improvement of `0.00702` over the 4B full-10K baseline despite withholding 800 rows from training. However, the full-10K 8B restricted-loss refit reached `0.73032`, which is `0.00275` below the best full-vocabulary 8B result of `0.73307`.

Decision: retain full-vocabulary training for the seed replications and primary ensemble. Keep the restricted-loss model only as a possible complementary ensemble member.

### 3. Vision expansion has low priority

The language-plus-projector and all-linear variants improved on the 4B baseline by only `0.00265` and `0.00451`, respectively. The projector-only experiment collapsed to `0.48182`.

Decision: freeze vision modules for the primary 8B runs. Do not repeat projector-only adaptation. Test input resolution separately without expanding LoRA into visual modules.

### 4. Controlled validation is mandatory

Several completed leaderboard runs used all 10,000 labeled rows and therefore provide no local checkpoint-selection signal. Leaderboard scores must be treated as final measurements, not tuning data.

Decision: all new comparisons begin on the same 9,200/800 split and save raw validation probabilities.

## Fixed validation and reporting policy

```yaml
seed: 42
training_rows: 9200
validation_rows: 800
validation_examples_per_class: 200
validation_oversampling: false
```

For every new run, record:

- overall and per-class macro-F1;
- confusion matrix;
- macro-F1 for caption–caption, caption–image, and image–image subsets;
- class prediction counts;
- validation probabilities and row IDs;
- training and inference wall time;
- seconds per optimizer step and inference row;
- peak allocated memory per GPU;
- exact resolved LoRA module names.

Use the same validation IDs for every comparison. Treat improvements smaller than `0.005` as inconclusive unless repeated across seeds or bootstrap samples.

## Priority 1: 8B restricted four-class loss

Status: **completed**. The best checkpoint was 863 at 1.5 epochs with validation macro-F1 `0.729164`.

Notebook: [`astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_validation_qlora/astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb)

```yaml
model: Qwen/Qwen3-VL-8B-Instruct
training_rows: 9200
validation_rows: 800
loss: restricted_four_class_cross_entropy
epochs: 2
learning_rate: 5e-5
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]
image_area_budget: 448x448
max_text_chars: 3000
global_batch_size: 16
precision: FP16
quantization: 4-bit NF4 with double quantization
random_object_swap: true
execution: two-process DDP on two T4 GPUs
```

The training loss must normalize over only the digit-token logits:

```python
outputs = model(**model_inputs)
next_token_logits = outputs.logits[batch_indices, answer_positions - 1]
class_logits = next_token_logits.index_select(-1, label_token_ids)
loss = torch.nn.functional.cross_entropy(class_logits.float(), target_class)
```

Evaluate and save checkpoints at approximately 0.5, 1.0, 1.5, and 2.0 epochs. Restore the checkpoint with the highest validation macro-F1.

### Acceptance criteria

- At least `0.005` validation macro-F1 above the comparable 4B restricted model.
- No collapse in `related_papers` or `unrelated_papers` recall.
- Peak memory remains safe on both T4 GPUs.
- Projected training and inference fit inside Kaggle session limits.

If the 8B run cannot fit, reduce image area before reducing LoRA rank.

## Priority 2: inference constraints and calibration

Use saved probabilities from models evaluated on the same fixed split. No retraining is required for mask, calibration, or TTA comparisons, but postprocessing choices must not be selected from full-10K Kaggle scores.

### Modality mask

The `same_figure` class is impossible for caption–caption and image–image pairs:

```python
if modality in {"CC", "II"}:
    class_logits[0] = float("-inf")
```

Compare raw argmax against mask-only predictions.

### Class calibration

Tune additive biases only on validation:

```python
prediction = np.argmax(np.log(probabilities + 1e-12) + class_bias, axis=1)
```

Compare:

1. no bias;
2. global class biases;
3. modality-specific class biases;
4. modality mask plus modality-specific biases.

Retain calibration only if its gain is at least `0.005` and stable under validation bootstrapping.

### Symmetry TTA

```python
p_final = 0.5 * (
    predict(obj_1, obj_2) +
    predict(obj_2, obj_1)
)
```

Measure prediction disagreement between object orders. Use TTA for the complete test set only if it adds at least `0.005`, because it approximately doubles inference time.

## Priority 3: full-10K restricted-loss refit

Status: **completed**. The full-data restricted-loss refit scored `0.73032`, below the `0.73307` full-vocabulary system, so it is not the primary standalone model.

Notebook: [`astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_full10k_qlora/astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb)

After choosing the best checkpoint duration and inference rules, retrain from the original 8B base model on all 10,000 labeled rows.

```yaml
model: Qwen/Qwen3-VL-8B-Instruct
training_rows: 10000
epochs: validation_selected
loss: restricted_four_class_cross_entropy
learning_rate: 5e-5
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]
```

The completed result remains `0.00693` below the current `0.73725` modality-mask champion.

## Priority 4: language MLP target expansion

Status: **completed and not selected**. Its best macro-F1 was `0.729019` at 2.0 epochs versus `0.729164` for attention-only at 1.5 epochs. The `0.000145` difference is far below the `0.005` threshold, while MLP adaptation used more memory and training time.

Notebook: [`astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_language_mlp_qlora/astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb)

If restricted 8B training succeeds, test whether language MLP adaptation improves paper-level semantics.

| Variant | Target modules |
|---|---|
| Attention baseline | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Attention + MLP | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Use the same two-epoch half-epoch checkpoint schedule on the fixed split. Keep rank, alpha, dropout, learning rate, and loss unchanged. Promote the expanded target set only for a validation gain of at least `0.005`.

Do not combine this ablation with a rank or dropout change; otherwise the source of any gain will be ambiguous.

## Priority 5: probability ensemble

Status: **three-seed ensemble completed and promoted; restricted-loss blend tested and rejected**. Equal averaging of seeds 17, 42, and 123 scored `0.73604`. The `0.75/0.25` full/restricted blend scored `0.73590`, a decrease of `0.00014`, so the remaining blend weights should not be submitted merely to tune against the leaderboard. Applying the deterministic modality mask to the raw ensemble increased the score to the current best of `0.73725`.

Use only models with complementary validation errors. Primary candidates are:

- three full-vocabulary 8B language-attention seeds (`0.73230`, `0.73250`, and `0.73307`);
- new 8B restricted-loss model;
- 4B restricted-loss model (`0.71453`) if it contributes complementary errors.

Test weights:

```text
0.25 / 0.75
0.50 / 0.50
0.75 / 0.25
```

Average probabilities first, then apply the selected modality mask and biases. Save component probabilities and ensemble weights. Do not ensemble hard one-hot CSVs.

## Retired direction: citation-focused auxiliary QLoRA

The sequential citation-focused auxiliary run caused negative transfer: validation macro-F1 fell from `0.72916` to `0.71405`. The proposed follow-up auxiliary and hierarchical experiments were removed. Do not spend any of the remaining submission budget on this direction.

## Deferred ablations

Run these only after the priorities above:

### Rank and alpha

| Rank | Alpha |
|---:|---:|
| 8 | 16 |
| 16 | 32 |
| 32 | 64 |

### Dropout

Test `0.00`, `0.05`, and `0.10` only if checkpoint trajectories indicate underfitting or overfitting.

### Image and text budgets

- Benchmark higher image resolution on 200–400 validation examples before training.
- Increase caption context only after measuring truncation frequency.
- Prefer reducing two-image resolution first if 8B memory is tight.

## Updated execution order

```text
S1: seed 17 completed at 0.73307
  → S2: seed 123 completed at 0.73250
  → seed 42 probabilities recovered
  → S3: equal-probability three-seed ensemble completed at 0.73604
  → restricted-loss 0.75/0.25 blend rejected at 0.73590
  → modality mask promoted at 0.73725
  → evaluate swap TTA, first alone and then with the modality mask
  → test longer context and higher resolution one factor at a time
  → build the strongest validated heterogeneous ensemble
  → reserve the final upload for the completely frozen champion
```

## Remaining submission queue

Five of the twelve planned submissions have now been used. The modality-mask system is the current best at `0.73725`. Seven candidate slots remain; a slot is not an obligation, so skip candidates that fail their fixed-validation gate.

### Completed submission slots

| Slot | Submission | Kaggle macro-F1 | Status |
|---:|---|---:|---|
| 1 | Full-10K 8B full-vocabulary, seed 17 | **0.73307** | Completed |
| 2 | Full-10K 8B full-vocabulary, seed 123 | **0.73250** | Completed |
| 3 | Equal three-seed full-vocabulary ensemble | **0.73604** | Completed |
| 5 | Three-seed ensemble with modality mask | **0.73725** | Completed; current best |
| 7 | 75% full-vocabulary ensemble + 25% restricted-loss model | **0.73590** | Completed; rejected |

### Seven remaining candidate slots

| Slot | Candidate submission | Required gate |
|---:|---|---|
| 4 | Three-seed ensemble with swap TTA | Forward/reverse averaging improves the fixed validation set enough to justify doubled inference |
| 6 | Three-seed ensemble with swap TTA and modality mask | Combined rules beat slots 3--5 on validation |
| 8 | Modality-gated ensemble | Modality-specific weights for CC, CI/IC, and II are stable under bootstrap resampling |
| 9 | Longer-caption 8B full-vocabulary system | A 6,000-character limit improves validation, especially CC and paper-relation examples |
| 10 | Higher-resolution 8B full-vocabulary system | The selected image budget improves CI/IC and II validation subsets and fits T4×2 |
| 11 | Strongest heterogeneous ensemble | Every included system adds validation value; exclude weak models added only for diversity |
| 12 | Final locked champion | Reserve until models, weights, TTA, mask, ID order, and output checks are frozen |

### Completed prerequisite: seed-42 probability recovery

Raw probability files are available for seeds 17, 42, and 123. The isolated seed-42 rerun regenerated both probabilities and the one-hot output:

- [`astroclimb-qwen3vl8b-language-seed42-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed42_qlora/astroclimb-qwen3vl8b-language-seed42-qlora.ipynb)

This rerun was an artifact-recovery prerequisite, not a new Kaggle experiment. Its required artifact is:

```text
astroclimb_qwen3vl8b_language_seed42_qlora/submission_probabilities.csv
```

### Seed-replication configuration and results

The three systems differ only in random seed:

```yaml
model: Qwen/Qwen3-VL-8B-Instruct
training_rows: 10000
epochs: 1
learning_rate: 1e-4
loss: full_vocabulary_label_token_cross_entropy
lora_targets: [q_proj, k_proj, v_proj, o_proj]
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
global_batch_size: 16
image_area_budget: 448x448
max_text_chars: 3000
seeds: [17, 42, 123]
```

Notebooks:

- [`astroclimb-qwen3vl8b-language-seed17-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-seed123-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb)
- [`astroclimb-qwen3vl8b-language-seed42-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed42_qlora/astroclimb-qwen3vl8b-language-seed42-qlora.ipynb)

| Seed | Kaggle macro-F1 | Difference from seed 42 |
|---:|---:|---:|
| 17 | **0.73307** | +0.00077 |
| 123 | **0.73250** | +0.00020 |
| 42 | **0.73230** | — |

The mean across the three scored seeds is `0.73262`, with sample standard deviation `0.00040` and range `0.00077`.

### Three-seed ensemble

Slot 3 averaged the raw probability files:

$$
p_{\mathrm{seed}}=\frac{p_{42}+p_{17}+p_{123}}{3}.
$$

All three probability files must have identical IDs and ordering. Renormalize only if necessary for numerical precision, then take the four-class argmax.

The resulting raw ensemble scored `0.73604`. The modality-mask derivative scored `0.73725` by setting `same_figure` probability to zero for the 6,000 caption--caption and image--image test pairs and changing seven hard predictions.

### Context and resolution candidates

Slot 9 changes only `MAX_TEXT_CHARS` from `3000` to `6000`. Slot 10 changes only the maximum image-area budget, beginning with a small memory/timing pilot before full training. Do not change the seed, loss, LoRA targets, rank, alpha, dropout, or epoch count in either experiment.

### Final submission validation

Before every upload—and especially slot 12—verify:

- exactly 10,000 rows and 10,000 unique test IDs;
- IDs match the test manifest and appear in the required order;
- columns are exactly `id`, `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`;
- target cells are binary and every row sums to one;
- raw component probabilities, ensemble weights, and postprocessing switches are archived;
- no choice was made solely from Kaggle feedback.

## Stop conditions

- Stop any run that collapses to one or two predicted classes.
- Do not select checkpoints or biases from Kaggle feedback.
- Do not run full test inference for an unselected exploratory checkpoint.
- Do not accept substantial runtime cost for a validation gain below `0.005`.
- Prefer the smaller or simpler configuration when scores differ by less than `0.005`.
- Preserve enough compute time for the final full-10K refit and inference.

## Required final artifacts

- selected adapter and processor;
- exact configuration and resolved target-module list;
- training and validation metrics;
- validation and test probability files;
- sharded inference outputs;
- merged, validated submission;
- experiment-log entry with score, timing, memory, and notes.
