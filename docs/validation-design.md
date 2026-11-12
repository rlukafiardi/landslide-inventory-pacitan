# Validation design (ds-workflow phase 4), fixed before the first real model

Written 2026-11-11 after the EDA and the feature checks, before any model beyond the smoke tests listed under Evidence.
A change after a result must be stated as such in `docs/decisions-log.md`.

## Use and good-enough threshold
Screening aid: the model flags candidate scars, an expert reviews them. Good enough on the held-out blocks:
**object recall >= 80% for polygons >= 5 px (area >= 500 m2) at object precision >= 30%.**
Precision is a floor: unmapped real slides count as false alarms (the inventory started from citizen reports).

## Split
- Test set: 34 of 170 3 km blocks, carved once (seed 42), IDs in `data/interim/test_blocks.json`, loader refuses the test side
  unless `LANDSLIDE_FINAL_EVAL=1`; used only by the final evaluation script (phase 6).
- Selection and tuning: **repeated block K-fold on the training blocks**: 5 folds of whole 3 km blocks, balanced by positive pixels
  (`src/modeling/folds.py`: 3,220 to 3,221 positives per fold), 5 repeats (seeds 0 to 4). Sensitivity: 6 km blocks, seed 0.
- Why blocks: pixels of one scar sit in the same block. On the same logistic regression, random pixel folds flatter the weighted
  average precision by +0.059 (all 95 numeric features) and +0.018 (dNDVI + terrain) against 3 km blocks; AUC moves only +0.006
  to +0.009, so AUC alone would hide it. 6 km blocks give the same picture as 3 km (`reports/tables/leak_check.csv`).
- No time dimension (one event, one year), so no forward-in-time split; the EDA Moran's I (0.07 / 0.07 / 0.23 at 1.5 / 3 / 6 km) is why 6 km is the sensitivity check.

## Objects and metrics (`src/modeling/metrics.py`, tested on synthetic scenes in `src/modeling/check_metrics.py`)
- Predicted objects: pixels with probability >= threshold inside the scored blocks, 8-connected, objects smaller than `a_min` pixels dropped.
- Reference: polygon id raster (`all_touched`, so polygons without a pixel centre count), classes by area: <5 px (<500 m2), 5-20 px, >20 px.
  A polygon belongs to the fold or to the test side of its representative point's 3 km block.
- **Primary**: object recall for polygons >= 5 px (a polygon is detected when >= 1 predicted pixel touches it) and object precision
  (a predicted object is a true object when it overlaps a polygon or its 20 m exclusion ring).
- Also reported: recall by size class, strict recall (IoU >= 0.3 between a polygon and the union of the objects overlapping it),
  flagged share of the area (reviewer workload), weighted pixel average precision (negatives carry their sampling weight).
- Operating point per model: sweep thresholds (quantiles of the pooled out-of-fold probabilities) x `a_min` in {1, 2, 3, 5};
  take the highest object precision among the rows with recall (polygons >= 5 px) >= 0.80, ties by smaller flagged area
  (`choose_operating_point`). Models are compared at this matched recall, never at a fixed probability. The threshold is chosen on the pooled
  out-of-fold probabilities of the training blocks, then applied once to the test blocks with the model refit on all training blocks.

## Baselines
1. Naive: threshold on -dB (single pixel), and on -dB_min5 (strongest single feature), with the same object rules.
2. Floor: site factors only (logistic regression).
3. Best simple model: regularised logistic regression on dNDVI (pixel + neighbourhood) + terrain + land use / geology (one-hot), no embedding.

## Candidates (feature groups from `src/features/registry.py`)
- (a) ndvi_pixel + ndvi_nbr + terrain + site_cat + flags   (dNDVI + factors)
- (b) emb + terrain + site_cat + flags                     (2018 embedding + factors)
- (c) all groups
Models: regularised logistic regression first, then gradient boosting with monotone constraints; hyperparameters by a small search on
the block folds, never on the test blocks. Groups are added or dropped as groups in an ablation at the end.
With and without `distroad` and `landuse` (they carry reporting signal). Results also reported with the 5 likely-unmapped blocks put back.

## Comparing models
Identical folds for every model in a repeat. For each pair: difference in precision at 80% recall and in weighted average precision,
repeats won (of 5), and a 95% interval from resampling whole blocks. A difference counts only when the interval excludes zero and the
model wins at least 4 of 5 repeats. Phase 5 gate: the chosen model beats the best simple baseline by more than that noise on precision at 80% recall.
**Geometry guard (added 2026-11-12, after a result; approved by the user):** object precision with the "touches a polygon or ring" rule rewards
fewer, bigger merged objects. A candidate that changes object geometry (post-processing, a second stage, a new object rule) counts only
if, at its operating point, strict recall (IoU >= 0.3, polygons >= 5 px) and the share of flagged pixels on a polygon or its ring are both
at least those of the stage-1 reference. Trigger: option B's double-threshold objects raised object precision from 7.8% to 14.1% while
strict recall fell from 64% to 31% and the on-slide pixel share from 23.8% to 8.7%. The 30% target is unchanged.

## Evidence behind the design (smoke tests, not results)
- Random vs block folds: `reports/tables/leak_check.csv`.
- Label-permutation AUC 0.498 under block CV (`src/features/check.py`).
- Weighted average precision of a logistic regression with the embedding (0.52) vs without (0.14), 3 km blocks: AUC differs by only
  about 0.02, so models are judged on precision at fixed recall, not AUC. To be tested properly in phase 5.

## Known limits
Labels carry about one pixel of error (shift estimated from the data). The inventory is not wall-to-wall, so precision is understated.
Land-use and lithology class names are unknown. The threshold is chosen on out-of-fold data and applied to a refit model, which can shift calibration slightly.
