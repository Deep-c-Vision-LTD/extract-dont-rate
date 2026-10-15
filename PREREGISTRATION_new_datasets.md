# Pre-specified predictions for the new datasets (CVIU resubmission of CVIU-26-3197)

Written 2026-10-11, before any VLM label, CLIP feature or result exists for the datasets below.
Basis: the results on EmoSet and LaMem with Claude and Qwen2.5-VL-7B (30 training subsets per budget,
Holm correction over 96 tests; results/rev_*_linear_r30.json, results/multitest_*.json).

## Datasets (none used in the original submission)
- **ArtPhoto** (Machajdik & Hanbury, 2010): 806 artistic photographs, 8 emotion classes identical to EmoSet's.
- **Abstract** (Machajdik & Hanbury, 2010): abstract paintings with 8-class vote counts; label = unique
  majority class, images with tied maxima excluded.
- **MemCat** (Goetschalckx & Wagemans, 2019): 10,000 images with memorability scores (false-alarm corrected).

## Fixed protocol
- VLMs: Qwen2.5-VL-7B-Instruct and Gemma-3-4B-it, run locally, greedy decoding, the unchanged EmoSet
  emotion-attribute schema (ArtPhoto, Abstract) and LaMem memorability-attribute schema (MemCat).
  Claude is not run on these datasets.
- Encoder: CLIP ViT-L/14 image embeddings; PCA fitted only on unlabelled, disjoint images
  (EmoSet 20,000-image pool for ArtPhoto and Abstract; the MemCat images outside pool and test for MemCat).
- Routes, probes and statistics as in analysis_rev.py: attrs, pcak, pca2k, hybrid (pcak + attrs), full CLIP;
  linear probe with inner-CV regularisation; 30 training subsets per budget; two-level bootstrap; Holm
  correction over all primary contrasts of the new datasets.
- Splits: MemCat pool 2,000 and test 2,000 images, random, seed 2026. ArtPhoto and Abstract: stratified
  70/30 pool/test split, seed 2026. Budgets: 20, 40, 80, 160, 320, 640, 1280 and the full pool, each
  capped at the pool size.

## Predictions
- **P1 (complement):** hybrid > pca2k at 20 and 40 labels, for every dataset and VLM.
- **P2 (crossover):** the advantage of hybrid over full CLIP shrinks as labels grow; at the full pool
  hybrid is not significantly better than full CLIP on MemCat.
- **P3 (VLM quality):** on each dataset, the VLM with the higher zero-shot (direct) score shows the larger
  complement gain (hybrid - pca2k) at 20 labels.
- **P4 (no substitution):** attrs alone do not beat pcak at 160 labels or more.

## Practical recommendation under test
Add VLM-extracted attributes to a compressed frozen-encoder embedding, rather than replacing the encoder
with them, when only tens of labels are available; expect the benefit to fall as labels grow and to scale
with the VLM's zero-shot competence on the target. Each prediction will be reported as confirmed or not,
whatever the outcome.
