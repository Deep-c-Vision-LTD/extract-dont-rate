# Complement, Don't Replace

Code and data release for:

> Hamit Soyel. **Complement, Don't Replace: Vision-Language Model Attributes Add to a Frozen Encoder
> Where the Model Can Perceive the Target.** Under review.

## The result in one paragraph

We ask when a vision-language model's (VLM's) output improves a predictor of a subjective image property
trained from few labels. On **five datasets** (EmoSet, ArtPhoto and Abstract for evoked emotion; LaMem and
MemCat for memorability), with **three VLMs** (Claude Sonnet 4.6, and the open-weights Qwen2.5-VL-7B and
Gemma-3-4B) and up to 2,000 labels, we compare direct VLM rating, a head on VLM-extracted perceptual
attributes, a head on a frozen CLIP embedding, a label-free compression of that embedding, and the
compression with the attributes appended. The main comparisons use **30 training subsets per budget**, a
two-level bootstrap, and **Holm correction over 336 contrasts**. The attributes are **not a substitute** for
the encoder: from 160 labels no attribute vector beats the equally compact compression after correction. They
are a **complement where the VLM perceives the target**: for emotion, appending them to the compression
improves accuracy at every label budget with every VLM on EmoSet (**+5.9 to +14.0 points at 20 labels**); for
memorability, which no VLM rates well zero-shot (ρ ≤ 0.243), they add little or nothing with few labels (with the
open models on LaMem they lower accuracy at 20 labels).
Across the 12 dataset and VLM pairs, the gain at 20 labels tracks the VLM's zero-shot agreement with human
labels (**Spearman 0.84**, a relation observed after the predictions below were fixed). For the three datasets added last, four predictions were fixed before any label,
feature or result existed (`PREREGISTRATION_new_datasets.md`); three were confirmed and one (the small-budget gain against the
equal-size code, which failed on Abstract and in one ArtPhoto cell) was not.

![Learning curves](figures/fig_curves.png)

![When the attributes help](figures/fig_rule.png)

## Quick reproduction (CPU, no API key)

Every VLM output and image list is included, and the CLIP embeddings of LaMem, ArtPhoto, Abstract and MemCat.
Two pieces must come from the original datasets:

- **EmoSet** does not allow redistributing derived data, so its CLIP embeddings and the label file of its 20,000-image
  pool are not included. Download [EmoSet](https://vcc.tech/EmoSet), set `EMOSET_IMAGE_DIR` in `extract_features.py`
  and run `python extract_features.py emoset` (about 30 minutes on a laptop CPU); the image lists in `samples/` fix
  which images are used and in which order. Regenerated embeddings match ours to within 5 × 10⁻⁶ per coordinate.
  With 20 to 40 labels the inner cross-validation can then choose a different regularization strength on some training
  subsets, so EmoSet numbers at the smallest budgets can move by up to about 0.6 points (for example +14.5 instead of
  +14.0 points for Claude's added-information gain at 20 labels); no conclusion changes. Our exact EmoSet embedding files
  are available on request to researchers who have accepted EmoSet's terms. Every other number reproduces exactly.
- **MemCat**'s scores are not copied here: download `memcat_image_data.csv` from the
  [MemCat OSF project](https://osf.io/kvghu/) (CC BY 4.0) and point `MEMCAT_CSV` at it.

```bash
pip install -r requirements.txt
# main results, EmoSet and LaMem (Tables 2-4, A.8; Sections 5.1-5.4)
for v in claude qwen gemma; do REPS=30 PROBE=linear python analysis.py $v; done
# non-linear probe (Section 5.8)
for v in claude qwen; do REPS=10 PROBE=mlp python analysis.py $v; done
# new datasets: ArtPhoto, Abstract, MemCat (Sections 5.2-5.5)
for v in qwen gemma; do MEMCAT_CSV=/path/to/memcat_image_data.csv VLMS=qwen,gemma REPS=30 python analysis_new.py $v; done
python check_predictions.py       # verdict on the pre-registered predictions (Table 5)
python make_assets.py             # tables/tab_{direct,added,vsclip,subst,equal,prereg}.tex, their JSON sources, figures/
```

Each `analysis.py` run takes about an hour on a laptop CPU (the MLP runs longer). Outputs are identical to
the files in `results/`. `multitest.py` applies Holm and Benjamini-Hochberg corrections to any set of result
files; `make_assets.py` applies them over the whole study (`tables/family_holm.json`).

Analyses carried over from the first version of the study use its protocol (8 training subsets,
`analysis_v1.py`), so that their published numbers reproduce exactly:

```bash
python analysis_v1.py claude; python analysis_v1.py qwen    # results/final_results_<vlm>.json
python robustness.py claude; python robustness.py qwen      # fixed-regularizer comparison (Section 5.8)
python schema_distill.py                                     # random half-schema test, attribute recoverability
python extract_alt_encoders.py                               # DINOv2-L and SigLIP-L embeddings (needs raw images)
python encoder_robustness.py dinov2l claude                  # and dinov2l qwen, siglipl claude, siglipl qwen
python encoder_distill.py; python make_encoder_table.py      # tables/tab_encoders.tex (Table 7)
```

## Full reproduction (from raw images)

1. Download the images yourself (not redistributed here): [EmoSet](https://vcc.tech/EmoSet),
   [LaMem](http://memorability.csail.mit.edu/), [ArtPhoto and Abstract](https://www.imageemotion.org/),
   and [MemCat](https://osf.io/kvghu/).
2. Set the image folders: edit the `EDIT ME` constants in `make_splits.py`, `extract_features.py`,
   `extract_alt_encoders.py` and `label_with_vlm.py` (`label_with_qwen.py` uses the folders of
   `label_with_vlm.py`), and set the environment variables `EMOSET_DIR`, `LAMEM_DIR` and `NEW_DATA` for
   `label_with_open_vlm.py` (`prep_new_datasets.py` reads `NEW_DATA`).
3. Image lists and CLIP embeddings: `python make_splits.py` (rebuilds the EmoSet and LaMem lists and checks them
   against `samples/`), `python extract_features.py` (EmoSet and LaMem), `NEW_DATA=... python prep_new_datasets.py clip`.
4. Labels:
   - Claude (Anthropic Batch API, `ANTHROPIC_API_KEY`, roughly $40; EmoSet and LaMem only):
     `python label_with_vlm.py submit all` then `python label_with_vlm.py fetch all`.
   - Qwen2.5-VL-7B, EmoSet and LaMem: `python label_with_qwen.py all`.
   - Gemma-3-4B, EmoSet and LaMem: `python label_with_open_vlm.py gemma all` (the model is gated on
     Hugging Face: accept its licence and run `hf auth login` first).
   - Qwen and Gemma on the new datasets: `python label_with_open_vlm.py qwen new` and `... gemma new`.

   The open models are free but slow: about 11 s per attribute record and 6 s per direct rating on an
   Apple-silicon Mac.
5. Run the Quick reproduction steps above.

## Routes and tests

| Route | Inputs | Uses labels | Uses the VLM |
|---|---|---|---|
| Direct rating | none (zero-shot answer) | no | yes |
| VLM attributes | k attributes (k = 10 emotion, 8 memorability) | yes | yes |
| CLIP | 768-d frozen CLIP ViT-L/14 embedding | yes | no |
| CLIP-PCA*k* (dimension-matched) | first k principal components | yes | no |
| CLIP-PCA*2k* (equal-size) | first 2k principal components | yes | no |
| Hybrid | CLIP-PCA*k* + the k VLM attributes | yes | yes |

- *Substitute test*: attributes vs CLIP-PCA*k*.
- *Added-information test*: hybrid vs CLIP-PCA*k* (does appending the attributes help the same encoder code?).
- *Equal-size test*: hybrid vs CLIP-PCA*2k*. With 20 labels CLIP-PCA*2k* can fit worse than CLIP-PCA*k*,
  so this test alone can credit the attributes with a gain that belongs to the control.

## Data and protocol

| | Training pool | Test set | Unlabelled PCA pool |
|---|---|---|---|
| EmoSet | 1,917 (training split) | 1,522 (validation subset) | 20,000 training-split images |
| ArtPhoto | 548 (stratified 70%) | 241 | EmoSet pool |
| Abstract | 160 (stratified 70%) | 69 | EmoSet pool |
| LaMem | 1,997 (`train_1`) | 2,000 (`test_1`) | 6,000 `train_1` images |
| MemCat | 1,975 | 2,000 | the other 6,000 MemCat images |

- On each dataset every VLM is evaluated on the same images: those for which every VLM run on it returned a
  complete attribute record. Near-duplicates (CLIP cosine > 0.95) are removed: training images that match a
  test image, and PCA-pool images that match a training or test image.
- The MLP probe (10 subsets) and the carried-over encoder, schema and fixed-regularizer analyses (8 subsets,
  1,000 resamples, uncorrected intervals; `analysis_v1.py`) cover EmoSet and LaMem with Claude and Qwen and are
  not in the Holm family. The pre-registered verdicts use their own Holm family (`check_predictions.py`).
- Budgets n ∈ {20, 40, 80, 160, 320, 640, 1280} below the pool size, plus the full pool; 30 training
  subsets per budget (class-stratified for emotion). Logistic regression (emotion) or ridge regression
  (memorability) on standardized inputs, regularizer re-selected by inner cross-validation on every subset.
- Two-level paired bootstrap (2,000 resamples of training subsets and test images) with two-sided p-values;
  Holm and Benjamini-Hochberg corrections over the 336 contrasts of the study.
- All VLMs receive the same instructions and schema (`prompts.py`). Claude answers by forced tool call. For Qwen
  and Gemma, "then call <tool>" becomes "then answer" and the schema is appended as a JSON specification; the
  answer is validated against it, with greedy decoding and one sampled retry (temperature 0.7) if it is invalid.

## Repository layout

```
├── analysis.py                main results, EmoSet and LaMem (any VLM; linear or MLP probe)
├── analysis_new.py            ArtPhoto, Abstract and MemCat
├── multitest.py               Holm and Benjamini-Hochberg over result files
├── prompt_check.py            agreement and results under the reworded emotion prompts
├── check_predictions.py       verdict on the pre-registered predictions
├── make_assets.py             tables 1-5 and A.8 (tables/) and both figures (figures/)
├── PREREGISTRATION_new_datasets.md   predictions fixed before the new datasets were labelled
├── prep_new_datasets.py       splits and CLIP embeddings for the new datasets
├── label_with_open_vlm.py     Gemma (all jobs) and Qwen (new datasets), local
├── label_with_vlm.py          Claude (Anthropic Batch API)
├── label_with_qwen.py         Qwen, EmoSet and LaMem
├── prompts.py                 every prompt and schema
├── analysis_v1.py             protocol of the first version (8 subsets), used by:
├── robustness.py  schema_distill.py  encoder_robustness.py  encoder_distill.py  make_encoder_table.py
├── extract_features.py  extract_alt_encoders.py  make_splits.py
├── samples/   image lists (no MemCat scores)        ├── labels/   every VLM output
├── features/  CLIP ViT-L/14 embeddings (not EmoSet)  ├── results/  every result file
└── tables/  figures/
```

## Prompt-robustness check

`prompts.py` also holds a reworded version of the emotion prompts and schema descriptions (`*_P2`): the same
instructions, scale anchors, options and field names, reordered and reworded, chosen by a panel of independent
paraphrasers and judges. Gemma-3-4B labelled EmoSet with it (`python label_with_open_vlm.py gemma p2`,
`labels/gemma_*_p2_labels.csv`). To compare the two prompts on the same images:

```bash
IDS_VARIANT=_p2 REPS=30 python analysis.py gemma emoset                 # original prompts
VARIANT=_p2 IDS_VARIANT=_p2 REPS=30 python analysis.py gemma emoset     # reworded prompts
python prompt_check.py                                                   # results/prompt_check_gemma.json
```

Result: the attributes agree across wordings (Spearman 0.86 to 0.93 per attribute), the direct labels agree with
Cohen's κ = 0.83, and the gain from appending the attributes to CLIP-PCA*k* is unchanged (+8.2 vs +8.3 points at
20 labels; the 95% interval excludes zero at every budget under both wordings).

`PREREGISTRATION_new_datasets.md` is reproduced unchanged (SHA-256 `872f3467...`). The script it names,
`analysis_rev.py`, is released here as `analysis.py`; the 96-test basis it cites is
`results/multitest_claude_linear_r30_qwen_linear_r30.json` (written by `multitest.py` before Gemma's EmoSet and LaMem
labels existed, so its rows differ slightly from the final three-VLM files).

## License and data

Code: MIT (see `LICENSE`). The VLM outputs in `labels/` are released under the same licence; Claude outputs were
produced under Anthropic's terms and Gemma outputs under the Gemma Terms of Use. Files derived from the
benchmarks (the image lists in `samples/`, LaMem scores in `samples/` and `features/lamem_train6000_scores.json`,
ArtPhoto and Abstract labels, and the CLIP embeddings in `features/`) are **not** covered by the MIT licence: they are
provided for non-commercial research only, under the terms of the original datasets. No images are redistributed, and
neither are EmoSet-derived embeddings or labels or MemCat's scores.
