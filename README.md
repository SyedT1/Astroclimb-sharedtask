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

This repository currently explores **Qwen3-VL-4B QLoRA** on Kaggle T4×2:

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

The following scores were obtained by the completed QLoRA notebooks. They are user-reported Kaggle macro-F1 results.

| Notebook | Kaggle score | Model | Epochs | Training dataset | Validation dataset | LoRA rank (`r`) | LoRA alpha | LoRA dropout | LoRA target modules |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| [`astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb) | **0.67856** | `Qwen/Qwen3-VL-4B-Instruct` | 1 | 5,000 balanced pairs (1,250 per class) | 400 pairs | 16 | 32 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| [`astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb) | **0.70751** | `Qwen/Qwen3-VL-4B-Instruct` | 1 | All 10,000 labeled pairs | None (final fit) | 16 | 32 | 0.05 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

The base model was loaded using 4-bit NF4 quantization with double quantization and FP16 computation. Only the LoRA adapters were trained; the underlying model weights remained frozen. Training used random object-order swapping and two-process DDP on two NVIDIA T4 GPUs.

## Repository notebooks

| Notebook | Purpose |
|---|---|
| [`astroclimb_run0_qwen3vl_qlora.ipynb`](astroclimb_run0_qwen3vl_qlora.ipynb) | Small end-to-end QLoRA pipeline check. |
| [`astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb`](astroclimb_5k_qwen3vl_qlora/astroclimb-5k-qwen3vl-qlora.ipynb) | Balanced 5,000-example QLoRA experiment with validation. |
| [`astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb`](astroclimb_full10k_qwen3vl_qlora/astroclimb-full10k-qwen3vl-qlora.ipynb) | Final training on all 10,000 labeled pairs, full test inference, and submission generation. |

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
