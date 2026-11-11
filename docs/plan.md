# Landslide detection, Pacitan (Cempaka 2017): plan after the POC

## Context
The POC (point level, 742 landslide vs 743 matched background points, 3 km spatial block CV) answered its question.
With the same 11 site factors in every set, Satellite Embedding change ties dNDVI (D1 fail) but adds to it (D2 pass, +0.02 AUC).
The best set was the 2018 embedding plus factors (0.94 AUC), but it wasn't one of the pre-set candidates.
Those are 1:1 balanced points, not a map. The goal is to replace manual digitizing with a **screening aid**: the model flags
candidate scars and an expert reviews them. The plan builds a full-area 10 m detector and scores it the way it will be used:
polygons against the Samodra inventory at true prevalence (about 0.13%). It follows the `ds-workflow` phases with their gates.

Decided with the user:
- Use case: screening aid.
- Compute: download the feature stack locally.
- Model: Opus for this design, Sonnet for building it.

## Fixed design (written into the notes before any real model, ds-workflow phase 4)
- **Grid**: the Mendeley factor-raster grid (EPSG:32749, 10 m, 5815 x 4051, extent of `map_landuse.tif`). Every layer is aligned to it.
- **Domain**: the area the inventory actually mapped. The default is the valid (non-nodata) land-use mask, to be confirmed against the
  Data in Brief paper (step 1a). Pixels outside the mapped area must not count as negatives.
- **Labels**: pixels whose centre falls inside a polygon = 1. A 20 m ring around each polygon is excluded from training and
  pixel metrics, to allow for edge and misregistration uncertainty. Everything else in the domain = 0.
- **Test set**: 3 km blocks, about 20% held out, stratified by landslide-pixel tercile. Carved once with a fixed seed.
  The IDs file is gitignored, the seed goes in the notes, and the loader returns train unless asked for `test`.
  Limit to note: the POC looked at all areas, so feature-set choice is slightly informed by test areas.
- **CV**: 5-fold GroupKFold over 3 km blocks on the training part (5 km reported as a sensitivity). Identical folds for every
  candidate. Report repeats won and intervals.
- **Metrics**, from the use:
  - Primary: object recall by size class (<5 px, 5-20 px, >20 px) and object precision, at the operating threshold.
  - A reference polygon counts as detected if at least 1 predicted pixel falls inside it (lenient), with IoU ≥ 0.3 also reported (strict).
  - Predicted objects are 8-connected components after the threshold and a minimum area (tuned in CV).
  - Secondary: pixel average precision at true prevalence, and total landslide area vs the inventory's 2.16 km².
- **Good-enough threshold (screening)**: object recall ≥ 80% for slides ≥ 5 px at object precision ≥ 30%, measured on the
  test blocks. The operating threshold is chosen in CV as the lowest threshold that reaches that recall.
- **Baselines**:
  - Naive: a dNDVI threshold with no ML.
  - Floor: site factors only.
  - Best simple model: dNDVI set + factors (logistic regression).
- **Candidates**: (a) 2018 embedding + factors, (b) 2018 embedding + embedding change + dNDVI + factors.
  Pick the simplest one that isn't worse within the noise. If (a) is chosen, the 2017 layer and the v1.0/v1.1 version issue drop out.
- **Models**: logistic regression and random forest without tuning first, then XGBoost with a small grouped-CV search.
  Class imbalance is handled with weights and negative sampling during training (sampling fraction recorded). Prediction runs on all pixels.

## Steps by phase

### Phase 1: data validation
a. **Mapping extent and land-use legend.** Read the Data in Brief article (DOI 10.1016/j.dib.2024.111155) for the mapped area,
   the imagery dates, and the land-use source and year. Check whether the land-use map is post-event or has a landslide or bare class,
   because then it would leak the label. If the paper is silent, the user asks the inventory author. Record it under Data in the notes.
b. **Grid check** (`src/data/check_grid.py`): every factor raster has the same CRS, transform, shape and nodata. Count invalid
   pixels per layer.
c. **Feature download** (`src/ingestion/download_stack.py`): tiled `ee.data.computePixels` into the factor grid (crsTransform
   taken from `map_landuse.tif`).
   - Layers: 2018 and 2017 embeddings as int8 (×127), plus NDVI pre, postA and postB as float16.
   - Output: one GeoTIFF per layer in `data/interim/`.
   - The loop is resumable, polite, and stops on a quota error (`crawling` skill pattern).
   - First, check the Community-tier limits and the int8 precision loss: on a sample, float vs dequantized must not change the AUC.
   - Reuse `ndvi_composite`, `S2_WINDOWS`, `COLLECTION` and `BANDS` from `src/embedding_utils.py`.
d. **Range checks**:
   - Embedding norm ≈ 1, NDVI in [-1, 1].
   - Per-layer coverage inside the domain; NDVI cloud gaps counted, not dropped silently.
e. **Labels** (`src/data/labels.py`): rasterize the polygons (centre rule) plus the 20 m exclusion ring. Count positive pixels
   per size class.
f. **Test carve** (`src/data/split.py`): 3 km blocks, stratified, seed fixed. Store the IDs and refuse to overwrite them.
   `src/data/load.py` returns train by default.
- **Gate:** one `make data` rebuild from raw, bad pixels counted, test IDs stored.

### Phase 2: EDA (training blocks only)
- Write `reports/eda.html` using `.claude/skills/ds-workflow/templates/html_report.py`, with charts per `chart-design` and a map per `gis-workflow`.
- Cover:
  - class balance per block
  - in-scar vs background feature distributions
  - NDVI gap pattern
  - polygon size vs pixel-grid fit
  - land cover of positives
  - likely false-positive classes: Cempaka floods (river-bank erosion and flood deposits), harvested fields, roads, quarries.
- **Gate:** cleaning rules with counts, modeling constraints, open questions in the notes.

### Phase 3: features
- A feature table builder working on the stacked rasters, with negative sampling for training.
- A catalogue of every feature: source, date window, resolution.
- A leakage review covering the land-use year, the 2017 embedding containing December 2017, and post-event factor layers if any.
- Turn the POC scripts into modules where they are reused; keep the shared helpers in `src/embedding_utils.py` and `src/cv_utils.py`.

### Phase 4: validation design
- Write the "Fixed design" above into the notes. Extend `src/cv_utils.py` with object-level metrics: connected components
  (`scipy.ndimage.label`), matching to inventory polygons, and recall/precision by size class.

### Phase 5: modeling
- Run the baselines and the candidates on identical folds, with paired comparisons.
- Choose the threshold and the minimum object area inside CV.
- Final model module in `src/modeling/`.
- **Gate:** beats the best simple baseline by more than the noise on the primary metric, or else we report that dNDVI + factors is enough.

### Phase 6: evaluation and reporting
- Score the test blocks once.
- Error-analysis table of misses and false alarms by size, slope, land use and distance to rivers.
- Write `reports/modeling.html`.
- Record the decision in the notes.

### Phase 7: delivery
- Predicted candidate polygons as GeoPackage, plus a map with attribution (Samodra et al. CC BY 4.0; Satellite Embedding licence to be verified in the catalogue).
- First architecture diagram (`architecture-diagram` skill) once the pipeline works.

## Folders and tools (confirm on start)
- New folders: `data/interim/` and `src/{data,ingestion,modeling}/`, created only when they hold code.
- `models/` (gitignored) appears at phase 5.
- New packages: `scipy` is already in the project via scikit-learn. `xgboost` gets added at phase 5. A Makefile provides `make data`.

## Out of scope
Other events, Sentinel-1 features (listed as an idea), deep-learning segmentation, susceptibility, deployment, dashboards.

## Risks, riskiest first
1. The inventory doesn't cover the whole domain, so false negatives are fabricated. Step 1a resolves it.
2. The land-use layer is post-event and leaks the label. Step 1a and the leakage review.
3. EE quota or download limits on about 3-4 GB. Check the limits first, and the job is resumable.
4. Flood and river-erosion scars appear as false positives the inventory never mapped. Error analysis; the user judges a sample.

## Verification
- `make data` rebuilds the stack, labels and split from raw, prints the per-layer coverage and bad-pixel counts, and refuses to overwrite the test IDs.
- Regression: point AUCs re-computed from the downloaded stack at the POC points match the POC (`features.csv`) within 0.01.
  This checks the alignment and the int8 quantization.
- Unit checks on the object metrics against known cases: a synthetic perfect prediction scores recall = precision = 1, an empty prediction scores recall 0,
  and a one-pixel hit counts as detected under the lenient rule.
- The permutation-label sanity check stays in every CV run (AUC ≈ 0.5).
- Each phase gate gets one line in `projects/landslide-inventory-pacitan/CLAUDE.md` (Decisions), with dated detail in `docs/decisions-log.md` once the notes near 15 KB.
