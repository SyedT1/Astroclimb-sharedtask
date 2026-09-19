# AstroCLIMB Shared Task

AstroCLIMB—**Astronomy Citation Linking from Illustrations, a Multimodal Benchmark**—asks whether a multimodal system can partially reconstruct the citation graph of astronomy papers using only scientific figures and their captions.

It is a shared task of the **4th Workshop on Artificial Intelligence for Scientific Publications (WASP 2026)**, co-located with **IJCNLP–AACL 2026**. The online workshop is scheduled for **November 9–10, 2026**. The competition is used to automate participant scoring and does not offer monetary rewards.

- [Official task page](https://ui.adsabs.harvard.edu/WIESP/2026/shared_task)
- [Kaggle competition](https://www.kaggle.com/competitions/astroclimb/overview)
- [Full dataset on Hugging Face](https://huggingface.co/datasets/adsabs/AstroCLIMB)
- [WASP 2026 OpenReview group](https://openreview.net/group?id=aclweb.org/AACL-IJCNLP/2026/Workshop/WASP)
- [ACL paper templates](https://github.com/acl-org/acl-style-files)

## Motivation

Figures are central to scientific communication, but the information they contain is difficult to parse, archive, and search automatically. AstroCLIMB, created in partnership with [astroexplorer.org](https://astroexplorer.org/), tests multimodal models on real astronomy figures rather than generic visual-language data.

The broader source collection contains more than 100,000 figure–caption pairs from recent open-access astronomy papers. The shared task turns part of that collection into a relationship-classification problem.

## Competition task

This is a **four-class, single-label classification** task. Each example contains two objects, and each object can be either:

- a scientific figure;
- an English-language figure caption.

Consequently, an input can be figure–figure, figure–caption, or caption–caption. A model must assign exactly one of these labels:

| Target column | Meaning |
|---|---|
| `same_figure` | The objects are the figure and caption belonging to the same scientific figure. This class is valid only for figure–caption pairs. |
| `same_paper` | The objects originate from different figures in the same paper and therefore share a DOI. |
| `related_papers` | The objects originate from different papers and one paper cites the other. |
| `unrelated_papers` | None of the preceding relationships applies. |

Every pair has only one label. The relationships are **symmetric**, so exchanging `obj_1` and `obj_2` must not change the correct class.

The intended class priority is implicit in the mutually exclusive labels: a matching figure–caption pair is `same_figure`; otherwise objects with the same DOI are `same_paper`; otherwise a citation connection is `related_papers`; all remaining pairs are `unrelated_papers`.

## Available datasets

### Full Hugging Face dataset

The source dataset exposes the citation graph as paper metadata and adjacency lists instead of enumerating every possible object pair. The dataset description in `details.txt` reports 94,233 currently listed rows and describes the overall collection as 100K+ figure–caption pairs.

Its fields are:

```text
image
UUID
Image ID
Paper DOI
Paper Title
Image Caption
Image Authors
References DOIs
Citing DOIs
```

- Images are PIL PNG objects.
- Captions are English-language text.
- DOI, title, author, UUID, and image ID fields provide source metadata.
- `References DOIs` and `Citing DOIs` encode citation-graph adjacency.
- Pair examples are not precomputed because the possible number of pairs is large.
- Additional training pairs can be generated from these metadata.
- The source description states that the full-dataset test set is planned for November 2026.

### Kaggle evaluation dataset

The Kaggle package is approximately **20.22 GB** and contains four files:

| File | Purpose |
|---|---|
| `train.csv` | 10,000 labeled object pairs used for model development and training. |
| `test.csv` | 10,000 object pairs on which predictions are required. The dataset explorer describes it as roughly 10.13 GB. |
| `sample_submission.csv` | Required submission columns and example formatting. |
| `solution.csv` | Competition scoring artifact; labels are not intended as test-time model inputs. |

The pair inputs can be long text captions or base64-encoded image strings. PNG images commonly begin with `iVBORw0KGgo`; other image encodings may also occur. They must be decoded and converted to RGB/PIL images before being supplied to a vision-language model.

## CSV schemas

### Training data

```text
id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2
```

The four target fields form a one-hot label: exactly one target should be `1` and the others `0`.

### Test data

The Kaggle data explorer describes the model-input test schema as:

```text
id,obj_1,obj_2
```

Only `id`, `obj_1`, and `obj_2` should be used for test inference. If a downloaded or displayed table includes extra target-shaped placeholder columns, predictions must still be produced independently and written in the submission format below.

### Object modalities

For either object column:

- ordinary English text represents a caption;
- an image-like base64 string represents an encoded figure.

A minimal PNG decoder is:

```python
import base64
import io
from PIL import Image


def decode_png(encoded: str) -> Image.Image:
    if encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
```

Real pipelines should also handle truncated images, validate modality detection, preserve aspect ratio while resizing, and cache decoded files so the large CSV strings are not decoded on every epoch.

## Evaluation

The leaderboard metric is **macro-averaged F1 across all four classes**:

```text
Macro-F1 = (F1_same_figure + F1_same_paper
            + F1_related_papers + F1_unrelated_papers) / 4
```

Each class contributes equally regardless of its frequency. A competitive workflow should therefore inspect per-class F1 and the confusion matrix rather than accuracy alone.

The `same_figure` validity rule can be applied deterministically at inference: its score should be masked for figure–figure and caption–caption examples. Object-order swapping is also a useful augmentation or test-time consistency check because the target relation is symmetric.

## Submission format

Submit a CSV containing the ID and four one-hot prediction columns:

```csv
id,same_figure,same_paper,related_papers,unrelated_papers
0,0,1,0,0
1,1,0,0,0
2,0,1,0,0
```

Before uploading, verify that:

- the file contains exactly one row per test ID;
- test IDs occur once and are not reordered incorrectly during distributed inference;
- the columns exactly match `sample_submission.csv`;
- every row has exactly one predicted class;
- there is no saved DataFrame index column.

## Recommended modeling workflow

AstroCLIMB is intended as a multimodal benchmark; participants are not expected to train a large foundation model from scratch. Suitable approaches include statistical baselines, embedding models, zero-shot VLM evaluation, and parameter-efficient fine-tuning.

This repository currently explores **Qwen3-VL-4B and Qwen3-VL-8B QLoRA** on Kaggle T4×2:

1. Decode and resize image strings once, then cache them as files.
2. Represent each pair consistently as two numbered objects.
3. Train a four-way classifier through constrained answer tokens `0`–`3`.
4. Load the base model in 4-bit NF4 and train LoRA adapters.
5. Use a fixed validation split and report macro-F1 plus per-class F1.
6. Run two-process distributed training, one process per T4.
7. Shard test inference across both GPUs and merge predictions by `id`.
8. Validate the final CSV against `sample_submission.csv`.

See [experiments.md](experiments.md) for the experiment sequence and ablations.

## Experiment results

The following scores were obtained by the completed QLoRA runs and their derived ensemble and postprocessing systems. They are user-reported Kaggle macro-F1 results.

### Frozen-feature non-VLM baseline

[`metadata-first-hybrid-training.ipynb`](notebooks/metadata-first-hybrid-training.ipynb) is a non-generative, frozen-feature classifier pipeline rather than a SPECTER-only model. It combines frozen [SPECTER2](https://aclanthology.org/2023.emnlp-main.338/) caption embeddings, [SigLIP2](https://arxiv.org/abs/2502.14786) image--text embeddings, and [DINO](https://arxiv.org/abs/2508.10104) visual embeddings with word/character TF-IDF, OCR overlap, and perceptual-hash features. Three modality-specific CatBoost models classify caption--caption, mixed, and image--image pairs. The resulting submission scored **0.48279 macro-F1** on the final Kaggle leaderboard.

| Run / artifact | Kaggle score | Model | Seed | Epochs | Training dataset | Validation dataset | LoRA rank (`r`) | LoRA alpha | LoRA dropout | LoRA target modules |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| [Three-seed ensemble + modality mask](astroclimb-qwen3vl8b-three-seed-modality-mask/astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb) | **0.73725** | `Qwen/Qwen3-VL-8B-Instruct` | 17, 42, 123 | 1 each | All 10,000 labeled pairs per seed | None (final fits) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`; inference-only modality mask |
| [Three-seed probability ensemble](astroclimb_qwen3vl8b_three_seed_ensemble/submission.csv) | **0.73604** | `Qwen/Qwen3-VL-8B-Instruct` | 17, 42, 123 | 1 each | All 10,000 labeled pairs per seed | None (final fits) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb) | **0.67856** | `Qwen/Qwen3-VL-4B-Instruct` | 42 | 1 | 5,000 balanced pairs (1,250 per class) | 400 pairs | 16 | 32 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb) | **0.70751** | `Qwen/Qwen3-VL-4B-Instruct` | 42 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-restricted4class-qwen3vl-qlora.ipynb`](astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb) | **0.71453** | `Qwen/Qwen3-VL-4B-Instruct` | 42 | 2 | 9,200 pairs | 800 balanced pairs (200 per class) | 16 | 32 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-qwen3vl8b-language-seed17-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb) | **0.73307** | `Qwen/Qwen3-VL-8B-Instruct` | 17 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-qwen3vl8b-language-seed123-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb) | **0.73250** | `Qwen/Qwen3-VL-8B-Instruct` | 123 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-qwen3vl8b-language-qlora.ipynb`](astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb) | **0.73230** | `Qwen/Qwen3-VL-8B-Instruct` | 42 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_full10k_qlora/astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb) | **0.73032** | `Qwen/Qwen3-VL-8B-Instruct` | 42 | 1.5 | All 10,000 labeled pairs | None (validation-selected final fit) | 16 | 32 | 0.05 | Language-attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-alllinear-vision-language-qlora.ipynb`](astroclimb_alllinear_vision_language_qlora/astroclimb-alllinear-vision-language-qlora.ipynb) | **0.71202** | `Qwen/Qwen3-VL-4B-Instruct` | 42 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | All eligible linear layers in the vision and language towers |
| [`astroclimb-language-projector-qlora.ipynb`](astroclimb_language_projector_qlora/astroclimb-language-projector-qlora.ipynb) | **0.71016** | `Qwen/Qwen3-VL-4B-Instruct` | 42 | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | Language-attention projections plus visual merger projectors |

### Three-seed Qwen3-VL-8B ensemble with modality mask

The current best user-reported Kaggle result is **0.73725 macro-F1**, produced by applying a deterministic modality constraint to the `0.73604` three-seed probability ensemble. This is an absolute improvement of `0.00121` over the raw ensemble and `0.00418` over the best individual seed. No model was retrained, and no additional model inference was required.

The `same_figure` relationship is possible only when one object is a figure and the other is a caption. For caption--caption (`CC`) and image--image (`II`) pairs, the system therefore sets the ensemble's `same_figure` probability to zero and renormalizes the other three probabilities. Let class $0$ denote `same_figure`. If $p(c\mid x)$ is the raw ensemble distribution and $m(x)$ is the pair modality, the postprocessed distribution is

$$
\widetilde p(c\mid x)=
\begin{cases}
0, & m(x)\in\{\mathrm{CC},\mathrm{II}\},\ c=0,\\
\dfrac{p(c\mid x)}{1-p(0\mid x)},
& m(x)\in\{\mathrm{CC},\mathrm{II}\},\ c\in\{1,2,3\},\\
p(c\mid x), & m(x)\in\{\mathrm{CI},\mathrm{IC}\}.
\end{cases}
$$

The test CSV is streamed rather than loaded into memory. An object is classified as an image from its encoded-image signature (`iVBORw0KGgo`, `/9j/`, `UklGR`, `R0lGOD`, or `data:image`); all other objects are treated as captions. The established test-set modality distribution is 3,000 `CC`, 4,000 cross-modal, and 3,000 `II` pairs, so the constraint applies to 6,000 of 10,000 rows. Cross-modal probability vectors are left unchanged.

| Setting | Value |
|---|---|
| Input probabilities | Equal arithmetic mean of the seed-17, seed-42, and seed-123 four-class probabilities |
| Kaggle input path used | `/kaggle/input/datasets/syedmohaiminulhoque/submission-probabilities/submission_probabilities.csv` |
| Postprocessing | Set `p_same_figure` to zero for `CC` and `II`, then renormalize |
| Prediction | Argmax of the postprocessed four-class probability vector |
| Swap TTA | Disabled |
| Class-bias calibration | Disabled |
| Restricted-loss blending | Disabled |
| Additional training or inference | None |
| Hardware | CPU only; Kaggle accelerator set to `None` |
| Rows masked / unchanged | 6,000 `CC` or `II` rows / 4,000 cross-modal rows |
| Predictions changed | 7, all removed from the impossible `same_figure` class |
| Raw prediction counts | `(998, 2480, 3495, 3027)` in class order |
| Masked prediction counts | `(991, 2487, 3495, 3027)` in class order |
| Notebook | [`astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb`](astroclimb-qwen3vl8b-three-seed-modality-mask/astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb) |
| Generated artifacts | [`submission.csv`](astroclimb-qwen3vl8b-three-seed-modality-mask/submission.csv) and [`submission_probabilities.csv`](astroclimb-qwen3vl8b-three-seed-modality-mask/submission_probabilities.csv) |
| Kaggle macro-F1 | **0.73725** |

The immediately preceding postprocessing results are:

| System | Kaggle macro-F1 | Change from raw ensemble |
|---|---:|---:|
| Raw three-seed ensemble | 0.73604 | — |
| 75% three-seed ensemble + 25% restricted-loss model | 0.73590 | -0.00014 |
| Three-seed ensemble + modality mask | **0.73725** | **+0.00121** |

The restricted-loss blend was not promoted because it slightly reduced the score. The modality mask becomes the current champion, although its `0.00121` gain remains below the project's usual `0.005` threshold for a practically decisive validation improvement.

### Three-seed Qwen3-VL-8B probability ensemble

The unmasked equal-weight probability ensemble obtains **0.73604 macro-F1** from full-vocabulary Qwen3-VL-8B language-attention models trained with seeds 17, 42, and 123. This improves on the best individual seed, seed 17 at `0.73307`, by `0.00297` and supplies the probabilities used by the modality-mask champion.

For each test pair, the ensemble aligns the three raw four-class probability vectors by `id`, averages them, and then takes the class argmax:

$$
p_{\mathrm{ensemble}}(c\mid x)
=\frac{p_{17}(c\mid x)+p_{42}(c\mid x)+p_{123}(c\mid x)}{3},
\qquad
\hat y=\arg\max_c p_{\mathrm{ensemble}}(c\mid x).
$$

This is probability averaging rather than majority voting over one-hot predictions. The scored ensemble applies neither swap test-time augmentation nor a modality mask.

| Setting | Value |
|---|---|
| Component seeds | 17, 42, and 123 |
| Component Kaggle macro-F1 | `0.73307`, `0.73230`, and `0.73250`, respectively |
| Backbone | `Qwen/Qwen3-VL-8B-Instruct` |
| Training data | All 10,000 labeled pairs independently for each seed; no oversampling |
| Objective | Full-vocabulary cross-entropy at the single answer-token position |
| Training duration | 1 epoch per seed |
| Learning rate and schedule | $10^{-4}$ with cosine decay and 0.05 warmup ratio |
| Quantization | 4-bit NF4 with double quantization and FP16 computation |
| LoRA | Rank 16, alpha 32, dropout 0.05, no bias |
| LoRA targets | Language-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` |
| Batch configuration | Per-device batch 1, two GPUs, 8 accumulation steps; global batch 16 |
| Training augmentation | Random object-order swap with probability 0.5 |
| Input limits | Image area from $256^2$ to $448^2$ pixels; 3,000 characters per caption object |
| Ensemble rule | Equal arithmetic mean of the three raw four-class probability vectors |
| Postprocessing | No swap TTA, modality mask, or class-bias calibration |
| Output artifacts | [`submission_probabilities.csv`](astroclimb_qwen3vl8b_three_seed_ensemble/submission_probabilities.csv) and [`submission.csv`](astroclimb_qwen3vl8b_three_seed_ensemble/submission.csv) |

### Full-10K Qwen3-VL-8B seed replications

The three full-vocabulary 8B runs differ only in random seed. Seed 17 is the best individual model at **0.73307 macro-F1**. Seed 123 obtains **0.73250**, which is `0.00020` above seed 42 and `0.00057` below seed 17. These small differences should be treated as seed variation rather than evidence that one seed is intrinsically superior.

| Seed | Kaggle macro-F1 | Difference from seed 42 |
|---:|---:|---:|
| 17 | **0.73307** | +0.00077 |
| 123 | **0.73250** | +0.00020 |
| 42 | **0.73230** | — |

Across the three seeds, the mean Kaggle macro-F1 is **0.73262** with sample standard deviation **0.00040** and range **0.00077**.

| Setting | Value |
|---|---|
| Scored replication notebooks | [Seed 17](astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb); [seed 123](astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb); [seed 42](astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb) |
| Kaggle macro-F1 | Seed 17: **0.73307**; seed 123: **0.73250**; seed 42: **0.73230** |
| Random seeds | Base seeds **17**, **123**, and **42**; each DDP worker seeds Python, NumPy, and PyTorch with `base_seed + local_rank`, while the Transformers trainer uses the base seed |
| Backbone | `Qwen/Qwen3-VL-8B-Instruct` |
| Training data | All 10,000 labeled pairs; no oversampling |
| Validation data | None; final fit |
| Objective | Full-vocabulary cross-entropy at the single answer-token position |
| Epochs | 1 |
| Learning rate | $10^{-4}$ |
| Schedule | Cosine decay with 0.05 warmup ratio |
| Optimizer | Paged AdamW 8-bit; weight decay 0.01; maximum gradient norm 1.0 |
| Quantization | 4-bit NF4 with double quantization and FP16 computation |
| LoRA | Rank 16, alpha 32, dropout 0.05, no bias |
| LoRA targets | Language-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` |
| Batch configuration | Per-device batch 1, two GPUs, 8 accumulation steps; global batch 16 |
| Training augmentation | Random object-order swap with probability 0.5 |
| Image area range | $256^2$ to $448^2$ pixels |
| Maximum caption length | 3,000 characters per object, retaining the beginning and end |
| Test inference | Four digit-token probabilities over classes 0--3; 10,000 rows |
| Swap TTA / modality mask | Disabled / disabled |
| Hardware | Two NVIDIA T4 GPUs with two-process DDP |
| Saved outputs | Adapter, training metrics, probability shards, `submission_probabilities.csv`, and one-hot `submission.csv` |

### Zero-shot baselines

These inference-only notebooks perform no training or parameter updates. Their user-reported Kaggle macro-F1 scores are:

| Notebook | Kaggle score | Model |
|---|---:|---|
| [`astroclimb-qwen3vl4b-zero-shot.ipynb`](baseline/astroclimb_qwen3vl4b_zero_shot/astroclimb-qwen3vl4b-zero-shot.ipynb) | **0.47956** | `Qwen/Qwen3-VL-4B-Instruct` |
| [`astroclimb-qwen3vl8b-zero-shot.ipynb`](baseline/astroclimb_qwen3vl8b_zero_shot/astroclimb-qwen3vl8b-zero-shot.ipynb) | **0.40112** | `Qwen/Qwen3-VL-8B-Instruct` |
| [`astroclimb-qwen25vl7b-zero-shot.ipynb`](baseline/astroclimb_qwen25vl7b_zero_shot/astroclimb-qwen25vl7b-zero-shot.ipynb) | **0.46745** | `Qwen/Qwen2.5-VL-7B-Instruct` |

All three notebooks use the same inference configuration apart from the model checkpoint:

| Setting | Value |
|---|---|
| Models | `Qwen/Qwen3-VL-4B-Instruct`; `Qwen/Qwen3-VL-8B-Instruct`; `Qwen/Qwen2.5-VL-7B-Instruct` |
| Loading | 4-bit NF4 with double quantization and FP16 computation |
| Attention implementation | SDPA |
| Hardware layout | One visible Kaggle T4 (`cuda:0`) |
| Image area range | $256^2$ to $448^2$ pixels |
| Maximum caption length | 3,000 characters per object |
| Validation sample | 256 balanced pairs (64 per class) |
| Prompt | Direct classification with the four relation definitions and a one-digit response |
| Prediction | Constrained next-token probabilities over tokens `0`, `1`, `2`, and `3` |
| Swap test-time augmentation | Disabled |
| Test inference | Streamed over all 10,000 test pairs with one-hot submission output |

The 8B restricted full-10K refit uses the attention-only configuration selected by the fixed 9,200/800 validation experiment: checkpoint 863 at 1.5 epochs with validation macro-F1 `0.729164`. It trains all 10,000 labeled rows with learning rate $5\times10^{-5}$, global batch size 16, 4-bit NF4 quantization, and restricted four-token cross-entropy. The run used 15,335,424 trainable parameters, completed 938 optimizer steps in 320.45 minutes, peaked at 11.53 GiB on rank 0, and completed two-GPU test inference in 78.15 minutes. Its test prediction counts were `(1035, 2668, 3267, 3030)` in class order. Its Kaggle score is `0.00275` below the best individual seed-17 full-vocabulary model at `0.73307`, `0.00572` below the raw three-seed ensemble at `0.73604`, and `0.00693` below the modality-mask champion at `0.73725`; therefore the full-vocabulary objective remains the primary configuration.

### Relation-focused auxiliary-training ablation

The 20K relation-focused auxiliary experiment is a validation-only negative ablation rather than a leaderboard submission. It first adapts `Qwen/Qwen3-VL-8B-Instruct` on 20,000 metadata-derived pairs with class allocation `(2000, 6000, 8000, 4000)`, then fine-tunes on the fixed 9,200-row competition split and evaluates on the same balanced 800-row validation set. The competition stage uses restricted four-token cross-entropy, language-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` LoRA targets, learning rate $10^{-5}$, and 1.5 epochs. Its selected final checkpoint is step 863.

| Validation metric | Restricted-loss baseline | Auxiliary experiment | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.729164 | 0.714051 | -0.015113 |
| `same_figure` F1 | 0.929134 | 0.894309 | -0.034825 |
| `same_paper` F1 | 0.669880 | 0.628297 | -0.041583 |
| `related_papers` F1 | 0.537897 | 0.540670 | +0.002773 |
| `unrelated_papers` F1 | 0.779747 | 0.792929 | +0.013182 |
| Mean F1 of classes 1 and 2 | 0.603889 | 0.584484 | -0.019405 |

The auxiliary model's validation confusion matrix, with true classes as rows and predicted classes as columns, is

$$
C=
\begin{bmatrix}
165 & 30 & 5 & 0 \\
4 & 131 & 62 & 3 \\
0 & 51 & 113 & 36 \\
0 & 5 & 38 & 157
\end{bmatrix}.
$$

The dominant error remains the `same_paper`/`related_papers` boundary: 62 `same_paper` examples are predicted as `related_papers`, while 51 errors occur in the reverse direction. A further 74 errors cross the `related_papers`/`unrelated_papers` boundary. Modality macro-F1 is `0.728454` for caption-caption (181 rows), `0.671083` for caption-image (402 rows), and `0.651020` for image-image (217 rows).

This experiment therefore does not satisfy its improvement criterion. It gains only `0.002773` F1 on `related_papers` and `0.013182` on `unrelated_papers`, while losing substantially more on `same_paper` and `same_figure`. The likely cause is an evidence mismatch: DOI and citation metadata can define a correct paper-level relationship even when the sampled figures or captions do not visibly express it. The 8,000 auxiliary `related_papers` pairs consequently shift predictions toward class 2 without establishing a cleaner class-1/class-2 boundary. The exact configuration should not be refitted on the full 10K training set. It is retained as a paper ablation showing that unfiltered metadata-derived sequential auxiliary supervision can cause negative transfer. A low-cost follow-up is to sweep probability blends between this model and the baseline; any new auxiliary run should instead use confidence-filtered, evidence-visible pairs and interleave a small auxiliary fraction with competition data.

### Mathematical formulation of the scored fine-tuning notebooks

For every object pair $x_i=(o_{i1},o_{i2})$, the one-hot target is converted to a class index

$$
y_i=\arg\max_{c\in\{0,1,2,3\}}Y_{ic},
$$

where classes $0,1,2,3$ denote `same_figure`, `same_paper`, `related_papers`, and `unrelated_papers`. Let $t_c$ be the tokenizer ID of the single digit token representing class $c$, and let $z_{i,v}$ be the model logit for vocabulary token $v$ at the position immediately before the supervised answer token. All other prompt positions are assigned the ignore index $-100$, so only this answer position contributes to the loss.

#### 5k balanced notebook

The training sample has equal class quotas,

$$
(N_0,N_1,N_2,N_3)=(1250,1250,1250,1250),\qquad N=5000.
$$

Because only 960 unique class-0 training rows remain after reserving validation data, 290 are sampled with replacement. The validation distribution is $(40,120,120,120)$. Training uses the standard causal-language-model cross-entropy over the complete vocabulary $\mathcal V$:

$$
\mathcal L_{\text{5k}}
=-\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp(z_{i,t_{y_i}})}{\sum_{v\in\mathcal V}\exp(z_{i,v})}.
$$

#### Full-10k notebook

This notebook uses every labeled row once, without oversampling, with class counts

$$
(N_0,N_1,N_2,N_3)=(1000,3000,3000,3000),\qquad N=10000.
$$

It uses the same full-vocabulary answer-token objective for one epoch:

$$
\mathcal L_{\text{10k}}
=-\frac{1}{10000}\sum_{i=1}^{10000}
\log\frac{\exp(z_{i,t_{y_i}})}{\sum_{v\in\mathcal V}\exp(z_{i,v})}.
$$

There is no held-out validation set in this final-fit notebook.

#### Qwen3-VL-8B full-10k language-attention notebook

The 8B notebook and its seed-17 and seed-123 replications use all 10,000 rows once, without oversampling or a validation holdout. Let $z^{(8\mathrm B)}_{i,v}$ denote the next-token logits from `Qwen3-VL-8B-Instruct`. Their label-only loss retains the original full-vocabulary objective:

$$
\mathcal L_{\text{8B}}
=-\frac{1}{10000}\sum_{i=1}^{10000}
\log\frac{\exp\left(z^{(8\mathrm B)}_{i,t_{y_i}}\right)}
{\sum_{v\in\mathcal V}\exp\left(z^{(8\mathrm B)}_{i,v}\right)}.
$$

Thus, the mathematical training objective and class distribution match the 4B full-10k notebook; the principal change is the larger 8B backbone. LoRA remains restricted to the language-attention projections. The scored 8B replications differ only in random seed: 42 produces `0.73230`, 123 produces `0.73250`, and 17 produces `0.73307`.

#### Qwen3-VL-8B restricted-loss full-10k notebook

The restricted 8B refit uses the complete class distribution

$$
(N_0,N_1,N_2,N_3)=(1000,3000,3000,3000),\qquad N=10000,
$$

without a validation holdout. Its 1.5-epoch duration was selected by the preceding fixed-split validation run rather than by the Kaggle leaderboard. Writing its next-token logits as $z^{(8\mathrm{B-R})}_{i,v}$, it normalizes only over the four label tokens:

$$
\mathcal L_{\text{8B-restricted}}
=-\frac{1}{10000}\sum_{i=1}^{10000}
\log\frac{\exp\left(z^{(8\mathrm{B-R})}_{i,t_{y_i}}\right)}
{\sum_{c=0}^{3}\exp\left(z^{(8\mathrm{B-R})}_{i,t_c}\right)}.
$$

The backbone remains frozen in 4-bit NF4 form, while rank-16 LoRA updates are learned only for the language-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj` matrices. Inference uses the same four label-token probabilities without swap TTA or a modality mask.

#### All-linear vision-and-language notebook

This notebook uses the same 10,000-row class distribution and full-vocabulary label-token objective as the 4B full-10k notebook, without oversampling or a validation holdout. It trains for one epoch at a learning rate of $10^{-4}$ with global batch size 16. Writing its logits as $z^{(\mathrm{AL})}_{i,v}$ gives

$$
\mathcal L_{\text{all-linear}}
=-\frac{1}{10000}\sum_{i=1}^{10000}
\log\frac{\exp\left(z^{(\mathrm{AL})}_{i,t_{y_i}}\right)}
{\sum_{v\in\mathcal V}\exp\left(z^{(\mathrm{AL})}_{i,v}\right)}.
$$

Its distinguishing feature is the adapter coverage. If $\mathcal M_{\mathrm{AL}}$ is the set of eligible linear weight matrices across both towers, then

$$
W_m^{\mathrm{eff}}
=Q_{\mathrm{NF4}}(W_{m,0})+\frac{\alpha}{r}B_mA_m
=Q_{\mathrm{NF4}}(W_{m,0})+2B_mA_m,
\qquad m\in\mathcal M_{\mathrm{AL}}.
$$

This expands LoRA beyond the four language-attention projections while keeping the quantized base weights frozen.

#### Language-attention plus vision-projector notebook

This configuration also trains on all 10,000 rows for one epoch without oversampling or a validation holdout. It uses learning rate $10^{-4}$, global batch size 16, and the same full-vocabulary label-token loss. If $z^{(\mathrm{LP})}_{i,v}$ denotes its logits, then

$$
\mathcal L_{\text{language+projector}}
=-\frac{1}{10000}\sum_{i=1}^{10000}
\log\frac{\exp\left(z^{(\mathrm{LP})}_{i,t_{y_i}}\right)}
{\sum_{v\in\mathcal V}\exp\left(z^{(\mathrm{LP})}_{i,v}\right)}.
$$

Its adapter target set is the union of the language-attention projections and the visual merger projectors:

$$
\mathcal M_{\mathrm{LP}}
=\mathcal M_{\mathrm{lang}}
\cup\mathcal M_{\mathrm{proj}},
$$

$$
\mathcal M_{\mathrm{lang}}=\{q_{\mathrm{proj}},k_{\mathrm{proj}},v_{\mathrm{proj}},o_{\mathrm{proj}}\},
\qquad
\mathcal M_{\mathrm{proj}}=\{\mathrm{linear\_fc1},\mathrm{linear\_fc2}\}_{\mathrm{merger,deepstack}}.
$$

For every $m\in\mathcal M_{\mathrm{LP}}$, the learned update is

$$
W_m^{\mathrm{eff}}
=Q_{\mathrm{NF4}}(W_{m,0})+\frac{\alpha}{r}B_mA_m
=Q_{\mathrm{NF4}}(W_{m,0})+2B_mA_m.
$$

#### Restricted four-class notebook

Holding out 200 examples from each class gives

$$
(N_0,N_1,N_2,N_3)=(800,2800,2800,2800),\qquad N=9200,
$$

and a balanced validation set of $4\times200=800$ examples. Like the 8B restricted full-10K refit, its training denominator contains only the four valid label-token logits:

$$
\mathcal L_{\text{restricted}}
=-\frac{1}{N}\sum_{i=1}^{N}
\log\frac{\exp(z_{i,t_{y_i}})}{\sum_{c=0}^{3}\exp(z_{i,t_c})}.
$$

This directly matches the four-way decision made at inference and avoids spending probability mass on irrelevant vocabulary tokens.

#### Shared QLoRA, augmentation, inference, and evaluation

Each notebook keeps the 4-bit NF4 base weights frozen and learns rank-16 LoRA updates. For each adapted matrix,

$$
W_{\text{eff}}=Q_{\text{NF4}}(W_0)+\frac{\alpha}{r}BA
=Q_{\text{NF4}}(W_0)+2BA,
$$

where $r=16$, $\alpha=32$, $A\in\mathbb R^{r\times d_{\text{in}}}$, and $B\in\mathbb R^{d_{\text{out}}\times r}$. Only $A$ and $B$ are optimized; LoRA dropout is $0.05$. Seven scored notebooks adapt only `q_proj`, `k_proj`, `v_proj`, and `o_proj`; the language-projector notebook adds the visual merger projectors, while the all-linear notebook adapts every eligible linear layer in both model towers.

Because the relation is symmetric, let $S$ denote the swap operation. Training applies it with probability $1/2$ while preserving the label:

$$
S(o_{i1},o_{i2})=(o_{i2},o_{i1}),
\qquad b_i\sim\mathrm{Bernoulli}\left(\frac{1}{2}\right),
\qquad \tilde{x}_i=S^{b_i}(x_i),
\qquad \tilde{y}_i=y_i.
$$

All nine scored fine-tuning notebooks restrict inference to the four digit tokens, even when training used full-vocabulary cross-entropy:

$$
p_i(c)=\frac{\exp(z_{i,t_c})}{\sum_{k=0}^{3}\exp(z_{i,t_k})},
\qquad
\hat y_i=\arg\max_{c\in\{0,1,2,3\}}p_i(c).
$$

The reported competition metric is macro-F1, which weights every class equally:

$$
F_{1,c}=\frac{2P_cR_c}{P_c+R_c},
\qquad
F_{1,\mathrm{macro}}=\frac{1}{4}\sum_{c=0}^{3}F_{1,c}.
$$

The base model was loaded using 4-bit NF4 quantization with double quantization and FP16 computation. Only the LoRA adapters were trained; the underlying model weights remained frozen. Training used random object-order swapping and two-process DDP on two NVIDIA T4 GPUs.

## Repository notebooks

| Notebook | Purpose |
|---|---|
| [`astroclimb_run0_qwen3vl_qlora.ipynb`](astroclimb_run0_qwen3vl_qlora.ipynb) | Small end-to-end QLoRA pipeline check. |
| [`astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb) | Balanced 5,000-example QLoRA experiment with validation. |
| [`astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb) | Final training on all 10,000 labeled pairs, full test inference, and submission generation. |
| [`astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb`](astroclimb_restricted4class_qwen3vl_qlora/astroclimb-restricted4class-qwen3vl-qlora.ipynb) | Restricted four-class-loss QLoRA experiment with a balanced 800-example validation split. |
| [`astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb`](astroclimb_qwen3vl8b_language_qlora/astroclimb-qwen3vl8b-language-qlora.ipynb) | Qwen3-VL-8B language-attention QLoRA trained on all 10,000 labeled pairs. |
| [`astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed17_qlora/astroclimb-qwen3vl8b-language-seed17-qlora.ipynb) | Seed-17 replication of the full-10K Qwen3-VL-8B language-attention system; best individual-model Kaggle macro-F1 `0.73307`. |
| [`astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed123_qlora/astroclimb-qwen3vl8b-language-seed123-qlora.ipynb) | Seed-123 replication of the full-10K Qwen3-VL-8B language-attention system; Kaggle macro-F1 `0.73250`. |
| [`astroclimb_qwen3vl8b_language_seed42_qlora/astroclimb-qwen3vl8b-language-seed42-qlora.ipynb`](astroclimb_qwen3vl8b_language_seed42_qlora/astroclimb-qwen3vl8b-language-seed42-qlora.ipynb) | Seed-42 artifact-recovery rerun used to preserve raw probabilities for the three-seed ensemble. |
| [`astroclimb_qwen3vl8b_three_seed_ensemble/submission.csv`](astroclimb_qwen3vl8b_three_seed_ensemble/submission.csv) | Equal-probability ensemble of seeds 17, 42, and 123; unmasked Kaggle macro-F1 `0.73604`. |
| [`astroclimb-qwen3vl8b-full-restricted-blend.ipynb`](astroclimb-qwen3vl8b-full-restricted-blend.ipynb) | CPU-only probability blending of the three-seed full-vocabulary ensemble and restricted-loss model; the tested 75%/25% blend scored `0.73590` and was rejected. |
| [`astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb`](astroclimb-qwen3vl8b-three-seed-modality-mask/astroclimb-qwen3vl8b-three-seed-modality-mask.ipynb) | CPU-only modality-mask postprocessing for the three-seed ensemble; current-best Kaggle macro-F1 `0.73725`. |
| [`astroclimb_alllinear_vision_language_qlora/astroclimb-alllinear-vision-language-qlora.ipynb`](astroclimb_alllinear_vision_language_qlora/astroclimb-alllinear-vision-language-qlora.ipynb) | Qwen3-VL-4B all-linear QLoRA across the vision and language towers, trained on all 10,000 labeled pairs. |
| [`astroclimb_language_projector_qlora/astroclimb-language-projector-qlora.ipynb`](astroclimb_language_projector_qlora/astroclimb-language-projector-qlora.ipynb) | Qwen3-VL-4B language-attention plus visual-projector QLoRA trained on all 10,000 labeled pairs. |
| [`astroclimb_qwen3vl8b_restricted_validation_qlora/astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_validation_qlora/astroclimb-qwen3vl8b-restricted-validation-qlora.ipynb) | Qwen3-VL-8B restricted-loss language-attention experiment on the fixed 9,200/800 split. |
| [`astroclimb_qwen3vl8b_restricted_full10k_qlora/astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_full10k_qlora/astroclimb-qwen3vl8b-restricted-full10k-qlora.ipynb) | Full-10K Qwen3-VL-8B restricted-loss refit for 1.5 validation-selected epochs; Kaggle macro-F1 `0.73032`. |
| [`astroclimb_qwen3vl8b_restricted_language_mlp_qlora/astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb`](astroclimb_qwen3vl8b_restricted_language_mlp_qlora/astroclimb-qwen3vl8b-restricted-language-mlp-qlora.ipynb) | Qwen3-VL-8B restricted-loss ablation with language attention and MLP LoRA targets. |
| [`astroclimb_qwen3vl8b_relation_auxiliary_qlora/astroclimb-qwen3vl8b-relation-auxiliary-qlora.ipynb`](astroclimb_qwen3vl8b_relation_auxiliary_qlora/astroclimb-qwen3vl8b-relation-auxiliary-qlora.ipynb) | Validation-only 20K relation-focused auxiliary-training pilot targeting `same_paper` and `related_papers`. |

For the full notebook, these settings request predictions for the entire test set:

```python
RUN_TEST_INFERENCE = True
TEST_LIMIT = None
```

## Participation and system papers

1. Join the competition on Kaggle to access the evaluation data and submit predictions.
2. Develop and score the system using macro-F1.
3. Participants may submit a system-description paper through the WASP 2026 OpenReview group using the ACL LaTeX template.
4. System papers receive light peer review; accepted papers are intended for the WASP 2026 proceedings in the ACL Anthology.

Participation and paper submission are separate: a strong leaderboard position is not required to submit a system description. Consult the official task and Kaggle pages for current registration requirements and exact competition deadlines.

## Organization and license

- Organizer: **NASA Science Explorer**.
- Dataset partner: **astroexplorer.org**.
- Dataset license reported by Kaggle: **MIT**.

## Source note

This README consolidates the competition description, Kaggle dataset explorer excerpt, and WASP call for participation stored in [`details.txt`](details.txt). Where relative countdown text such as “four days ago” or “three months to go” appeared, it was omitted because it becomes stale; use the official pages for live dates.
