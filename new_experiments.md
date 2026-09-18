# AstroCLIMB: Next 12 Kaggle Submissions

This is the ordered queue for the remaining **12 Kaggle submissions**. Every candidate must first pass evaluation on the fixed validation split. Skip any candidate that fails its gate rather than spending a submission solely to inspect the leaderboard.

| # | Submission | Required condition |
|---:|---|---|
| 1 | Current 8B best + modality mask | Improves validation macro-F1 |
| 2 | Current 8B best + swap TTA | Beats the raw model locally |
| 3 | Current 8B best + modality mask + swap TTA | Beats Submissions 1--2 locally |
| 4 | Best postprocessed 8B + cross-validated calibration | Stable gain of at least 0.005 |
| 5 | Three-seed full-vocabulary 8B probability ensemble | Beats the best individual seed |
| 6 | Three-seed restricted-loss 8B probability ensemble | Competitive with Submission 5 |
| 7 | Full-vocabulary + restricted-loss probability ensemble | Weight selected only on validation |
| 8 | Evidence-aware auxiliary model | Improves overall and `related_papers` F1 |
| 9 | Best base model + auxiliary-model ensemble | Auxiliary errors must be complementary |
| 10 | OCR-enhanced or caption-specialist model | Improves image-containing subsets |
| 11 | Heterogeneous ensemble of the strongest systems | Beats Submission 7 locally |
| 12 | Final locked champion system | Reserved until every choice is frozen |

## Submission 1: Modality mask

Use the current `0.73230` Qwen3-VL-8B system. For caption--caption and image--image pairs, set

$$
z_{\texttt{same\_figure}}=-\infty.
$$

Everything else remains unchanged. This enforces the task constraint that `same_figure` is possible only for a figure--caption pair.

## Submission 2: Swap test-time augmentation

Average predictions from both object orders:

$$
p=\frac{1}{2}\left[p(o_1,o_2)+p(o_2,o_1)\right].
$$

Do not apply the modality mask in this submission so the effect of order averaging remains identifiable.

## Submission 3: Modality mask plus swap TTA

Combine both deterministic changes:

1. Average the forward and reversed probabilities.
2. Mask `same_figure` where it is impossible.
3. Renormalize the probabilities and take the argmax.

## Submission 4: Calibrated best system

Starting from the best of Submissions 1--3, apply validation-selected logit biases:

$$
\hat y=\arg\max_c\left[\log p_c+b_{m,c}\right],
$$

where $b_{m,c}$ may depend on modality $m$. Fit biases using cross-validation, not the complete validation set followed by evaluation on that same set. Submit only if the improvement is at least `0.005` and remains stable under bootstrap resampling.

## Submission 5: Full-vocabulary multi-seed ensemble

Train the full-vocabulary Qwen3-VL-8B system with seeds 42, 17, and 123. Average their probabilities:

$$
p_{\mathrm{full}}=\frac{p_{42}+p_{17}+p_{123}}{3}.
$$

Apply only postprocessing already selected on validation. Do not submit the individual seeds.

## Submission 6: Restricted-loss multi-seed ensemble

Train the restricted four-token system with the same three seeds:

$$
p_{\mathrm{restricted}}
=\frac{p_{42}+p_{17}+p_{123}}{3}.
$$

Submit this ensemble only if it is competitive with the full-vocabulary ensemble on validation.

## Submission 7: Objective ensemble

Blend the multi-seed full-vocabulary and restricted-loss systems:

$$
p=\lambda p_{\mathrm{full}}
+(1-\lambda)p_{\mathrm{restricted}}.
$$

Select $\lambda$ locally. Search from 0 to 1 in increments of 0.1 and choose a stable region rather than an isolated maximum.

## Submission 8: Evidence-aware auxiliary system

Train a revised auxiliary system using:

- validation DOIs excluded before pair construction;
- citation pairs containing observable evidence;
- topically similar hard negatives;
- approximately 10--20% interleaved auxiliary data;
- restricted four-class loss;
- no separate sequential auxiliary-pretraining stage.

Submit only if it improves overall macro-F1 and `related_papers` F1 without materially damaging the other classes.

## Submission 9: Base plus auxiliary ensemble

Blend the best base ensemble with Submission 8:

$$
p=\lambda p_{\mathrm{base}}
+(1-\lambda)p_{\mathrm{aux}}.
$$

This can be useful even when the auxiliary model has lower standalone macro-F1, provided that its errors are complementary. Select the weight only on validation.

## Submission 10: OCR-enhanced or caption-specialist model

The preferred candidate is an OCR-enhanced model:

- extract OCR from every figure;
- append cleaned OCR tokens to the object representation;
- retain the original image;
- train or adapt using the selected objective.

If OCR performs poorly, replace this candidate with a caption-specialist model whose probabilities are used only for caption--caption pairs. Submit only if the candidate improves the relevant modality subsets locally.

## Submission 11: Heterogeneous ensemble

Combine the strongest complementary systems, potentially including:

- the full-vocabulary multi-seed ensemble;
- the restricted-loss multi-seed ensemble;
- the evidence-aware auxiliary model;
- the OCR or caption specialist.

Choose members and weights exclusively on validation. Do not add a weak model merely for diversity.

## Submission 12: Final locked system

Reserve this slot until all choices are frozen. It should contain:

- the best validated model or ensemble;
- the selected modality-mask and swap-TTA behavior;
- the selected calibration;
- frozen ensemble weights;
- verified test ID order;
- exactly one one-hot prediction per row.

Do not use Submission 12 for another exploratory variant.

## Validation and submission policy

Before every upload:

- use the same fixed validation IDs;
- save raw four-class probabilities;
- report macro-F1, per-class F1, modality-level F1, and a confusion matrix;
- use paired bootstrap confidence intervals where possible;
- require a practically meaningful local gain, normally at least `0.005`;
- verify exactly 10,000 unique test IDs;
- verify the required column order;
- verify binary targets and exactly one predicted class per row;
- do not select class biases, ensemble weights, or epochs from Kaggle scores.

If Submissions 1--4 can all be evaluated reliably using saved validation probabilities, upload only the local winner. Unused submission slots should remain reserves rather than being spent on leaderboard-driven tuning.

## Candidates that should remain local experiments

Do not spend Kaggle submissions on:

- majority or random baselines;
- individual random seeds;
- text-removed or image-removed ablations;
- DOI-overlap and leakage analysis;
- isolated prompt variants;
- minor learning-rate changes;
- every individual ensemble weight;
- systems gaining less than `0.005` locally;
- any model selected primarily because an earlier Kaggle submission scored well.
