# Landslide detection, Pacitan (Cyclone Cempaka, Nov 2017)

## Goal
A screening aid that flags candidate landslide scars from satellite data, replacing most of the manual digitizing; an expert reviews the flags.
Success: on held-out 3 km blocks, object recall >= 80% for polygons >= 5 px (>= 500 m2) at object precision >= 30% (precision is a floor, see Data).
POC (passed in modest form): Satellite Embedding is complementary to Sentinel-2 dNDVI, not better alone. Full plan: docs/plan.md; metric and split definitions: docs/validation-design.md.

## Data
- Labels: Samodra et al. 2024 inventory, Mendeley DOI 10.17632/vrtx3w6mjd.1 (CC BY 4.0, credit the authors; article CC BY-NC), 743 polygons, EPSG:32749. Built from 328 BPBD citizen reports extended with 50 cm imagery and field checks; no stated footprint, so a missing polygon is not proof of no landslide.
- Site factors from the same release (10 m grid, 5815 x 4051): slope, elevation, curvatures, twi, spi (already log-scaled), distance to river and road, aspect, lithology (1:100,000 map, 1992) and land use (1:25,000 map); class names and map year unknown.
- Imagery on the same grid via Earth Engine (`.env` holds the project id and SSL_CERT_FILE; Community tier 150 EECU-hours per month): Sentinel-2 NDVI medians pre (Aug-Oct 2017), postA (Dec-Mar), postB (Apr-May) with Cloud Score+; Satellite Embedding 2018 (uint8, value = (v - 128) / 127).
- Derived in data/interim/ (gitignored): label.tif, block.tif, poly_id.tif, polygons.parquet, features_train.parquet, fill_values.json, test_blocks.json. `make data` rebuilds from raw, `make features` builds the table and runs the checks.
- Quirks: polygons carry a registration offset (see decisions); polygons are small (median 1,096 m2, 26% under 500 m2); prevalence 0.15%.

## Key decisions
- Objective: detection of Cempaka scars (not susceptibility), as a screening aid.
- Labels: polygons shifted +8.4 m E, +9.4 m N onto the Sentinel-2 grid (found from the data, cause unverified; shift predictions back to compare with the manual inventory); positives = pixel centre in a polygon, 20 m exclusion ring, polygons rasterized all_touched for detection.
- Domain 13.70 M px (1,370 km2): valid land use with geology, minus 5 likely-unmapped 3 km blocks (empty although lithology-class density predicts >= 4.6 slides); report with and without them.
- Test set: 34 of 170 3 km blocks (seed 42), IDs file gitignored, loader needs LANDSLIDE_FINAL_EVAL=1; unseen until phase 6.
- Features: 97 in 6 groups (ndvi_pixel, ndvi_nbr, terrain, site_cat, flags, emb), registry with hypotheses in src/features/registry.py; one tile-wise code path for training and full-area prediction. The 2017 embedding and the embedding change are dropped (version mismatch, dNDVI beats the change).
- EDA: vegetation loss is the strongest signal (-dB_min5 pixel AUC 0.91); the naive -dNDVI detector reaches only 7% pixel precision at 50% recall; its false alarms sit in land-use classes 10, 11 and 2, so land use and the embedding act as false-alarm filters; distroad and land use carry reporting signal (run with and without).
- Validation: repeated 5-fold block CV (3 km, 5 seeds; 6 km sensitivity), identical folds per repeat; compare models at 80% object recall on polygons >= 5 px (not AUC: random folds flatter AP by up to +0.06, AUC hides it).
- Baselines: naive -dB / -dB_min5 threshold; site factors only; regularised logistic on dNDVI + terrain + land use / geology. Candidates: (a) dNDVI + factors, (b) 2018 embedding + factors, (c) all.
- Terrain at a glance: distroad opposite to the prior, remote ground looks under-mapped; 24 neighbourhood features are collinear, so linear models need regularisation or a reduced set.

## Status
- Done: POC; phases 1 to 4 (data, EDA, features, validation design). All dated detail: docs/decisions-log.md (grep it, never load it whole).
- Phase 5 in progress. Gate PASSED over 5 repeats: c_hgb (all features, boosting) object precision 7.6% (CI 3.9-13.7) at 80% recall on polygons >= 5 px, flagging 1% of the area; +4.8 points over the best simple baseline (2.8%). The 30% target is NOT met, criterion unchanged. Ablation (repeat 0): no single group is separable from noise; dropping distroad from the final model is proposed, not decided. Error analysis: 77% of false alarms are 3-19 px; 24% are scar-sized (>= 20 px) and cluster near mapped slides, so possibly unmapped scars.
  Review done (40 blinded chips): 17 of 30 large false alarms are clearly not scars, 12 unsure (terrace farming, river bank, coast, road), 1 scar, so low precision is mainly a model/feature problem. Option B (object-level second stage, src/modeling/stage2.py) gave NO real gain: its higher object precision came from merged blobs (strict recall 64% to 31%), which led to the geometry guard in docs/validation-design.md (rule change after a result, approved). 30% target unchanged.
  Options B, A, A2 tested NULL (causes in decisions-log; code removed, features 97). Model c_hgb, protocol frozen before scoring. PHASE 6 DONE: test blocks scored once (decisions-log 2026-11-12): object recall 86.8% (79-93) MET, object precision 9.0% (6.7-11.5) vs 30% NOT MET, criterion unchanged; 3x the precision of the baselines (3%), 10 false objects per true object (baselines 31), strict recall 70% (34-51%). Next: reports/modeling.html (src/analysis/modeling_report.py, charts per chart-design, open every PNG), phase 7 delivery (candidate polygons GeoPackage shifted back by (-8.4, -9.4) m, attribution Samodra et al. CC BY 4.0, README, architecture diagram via the skill), model card limits (7.6-9% precision, scar-sized false alarms are mostly terraces, river banks, roads; test n = 106 polygons). The reviewer-workload framing is a labelled secondary view, not a criterion change. Commands: `set -a && . ./.env && set +a` first.
- Open: inventory footprint and completeness (paper silent, the inventory author unreachable); land-use and lithology class names and map year.
