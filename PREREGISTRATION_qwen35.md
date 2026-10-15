# Pre-specified predictions for a fourth VLM, Qwen3.5-4B

Written 2026-10-15, before Qwen3.5-4B was downloaded or produced any output on any image of this study.
Basis: the relation reported in the manuscript between a VLM's zero-shot agreement with human labels and the gain from
appending its attributes to the compressed encoder (12 dataset and VLM pairs: Claude Sonnet 4.6, Qwen2.5-VL-7B and
Gemma-3-4B; tables/rule.json, Spearman 0.84).

## Fixed protocol
- Model: Qwen/Qwen3.5-4B (Hugging Face, Apache-2.0), run locally in bfloat16, thinking mode disabled
  (`enable_thinking=False` in the chat template), greedy decoding, at most 220 new tokens, one sampled retry
  (temperature 0.7) if an answer fails schema validation: the same policy as for Qwen2.5-VL-7B and Gemma-3-4B.
- Prompts and schemas: unchanged (`prompts.py`, original wording), on all five datasets: EmoSet and LaMem
  (`label_with_open_vlm.py qwen35 all`) and ArtPhoto, Abstract and MemCat (`label_with_open_vlm.py qwen35 new`).
- Analysis: unchanged (`analysis.py`, `analysis_new.py`): 30 training subsets per budget, two-level bootstrap, same
  splits, near-duplicate removal and compression pools. On each dataset every VLM is evaluated on the images for which
  every VLM run on it returned a complete attribute record, so earlier numbers may shift slightly if Qwen3.5-4B leaves
  records incomplete.
- Zero-shot agreement: Cohen's kappa of the direct label (emotion) or Spearman rho of the direct rating (memorability)
  on the test set, as in `make_assets.py`. Gain = hybrid minus CLIP-PCAk accuracy (points) or rho at 20 labels.
- Significance for Q2, Q3 and Q5: Holm correction over all of Qwen3.5-4B's hybrid minus CLIP-PCAk and attributes minus
  CLIP-PCAk contrasts (five datasets, every budget).

## Predictions
- **Q1 (ordering within each dataset):** on each of the five datasets, Qwen3.5-4B's gain at 20 labels has the same rank
  among the VLMs run on that dataset as its zero-shot agreement has among them.
- **Q2 (emotion):** on EmoSet and on ArtPhoto, if its zero-shot kappa is at least 0.30, its gain at 20 and at 40 labels
  is positive and Holm-significant.
- **Q3 (memorability):** on LaMem and on MemCat, if its zero-shot rho is below 0.30, its gain at 20 and at 40 labels is
  not significantly positive under Holm.
- **Q4 (across datasets):** with its five dataset cells added to the twelve, the Spearman correlation between zero-shot
  agreement and the gain at 20 labels over all seventeen cells is at least 0.70.
- **Q5 (no substitution):** on no dataset do its attributes alone beat CLIP-PCAk under Holm at 160 labels or more.

Each prediction will be reported as confirmed, not confirmed, or not applicable (if its condition is not met),
whatever the outcome.
