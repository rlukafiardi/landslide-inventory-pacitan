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
- Features: 97 in 6 groups (ndvi_pixel, ndvi_nbr, terrain, site_cat, flags, emb), registry with hypotheses in src/features/registry.py; one tile-wise code path for training and full-area prediction. The 2017 embedding and the embedding change are dropped (version mismatch, dNDVI beats the change). pre, postA, postB, dA, dB are exactly dependent (dA = postA - pre, dB = postB - pre): VIF is infinite for them; dropping dA, dB leaves the logistic baselines unchanged (docs/decisions-log.md, 2026-11-14).
- EDA: vegetation loss is the strongest signal (-dB_min5 pixel AUC 0.91); the naive -dNDVI detector reaches only 7% pixel precision at 50% recall; its false alarms sit in land-use classes 10, 11 and 2, so land use and the embedding act as false-alarm filters; distroad and land use carry reporting signal (run with and without).
- Validation: repeated 5-fold block CV (3 km, 5 seeds; 6 km sensitivity), identical folds per repeat; compare models at 80% object recall on polygons >= 5 px (not AUC: random folds flatter AP by up to +0.06, AUC hides it).
- Baselines: naive -dB / -dB_min5 threshold; site factors only; regularised logistic on dNDVI + terrain + land use / geology. Candidates: (a) dNDVI + factors, (b) 2018 embedding + factors, (c) all.
- Terrain at a glance: distroad opposite to the prior, remote ground looks under-mapped; 24 neighbourhood features are collinear, so linear models need regularisation or a reduced set.

## Status
- Done: POC; phases 1 to 7 (data, EDA, features, validation design, modeling, test score, delivery). Dated detail: docs/decisions-log.md (grep it, never load it whole).
- Result: c_hgb (boosting, 97 features) on 34 held-out blocks, scored once: object recall 86.8% (79-93) MET, object precision 9.0% (6.7-11.5) vs 30% NOT MET, criterion unchanged; 10 false objects per true scar (baselines 31). Options B (object stage), A (prior-year NDVI), A2 (post-event persistence) were null and removed; an object-size prior (size_prior.py) gave +0.8 points, interval includes 0, not adopted. A blinded review showed large false alarms are mostly terraces, river banks, roads, not unmapped scars. Report: reports/modeling.html. Deliverable: data/processed/candidates_cempaka_pacitan.gpkg (5,865 candidates, review the 1,678 large ones first), README, docs/architecture.svg. Make targets: data, features, validation, model, rehearsal, deliver; the test score is run once by hand (README).
- Public repo (MIT code; data and reports keep their sources' terms, README). Pending: none. Block (8,10) review done (3 of 33 candidates judged scars vs 0 of 10 reference false alarms; reviewer recognises ~60% of real scars; not under-mapped beyond a handful, exclusion stands).
- Next, if the work continues: expert verdicts on the candidates (verdict column) as new labels; river-bank mask dropped (scars sit near rivers, precision is highest there), land-use class names; push to GitHub when the user says.
- Open: inventory footprint and completeness (paper silent, the inventory author unreachable; of the 5 likely-unmapped blocks only (8,10) shows excess candidates, 33 large; visual review found 3 scars, so no sign of under-mapping); land-use class names inferred from the BIG RBI25K 2022 layers (reports/tables/landuse_legend.csv: 3/4 shrub, 5 high forest, 6 plantation, 8 settlement, 10/11 paddy, 12 mixed crops, 13 dry field; unconfirmed by the authors); lithology: 1992 sheet legend read (13 units), codes named only at group level (reports/tables/lithology_legend.csv: 30 intrusive, 2 alluvium, 8 limestone, others low confidence; codes 12/13/16 lie off the sheet) and survey year (raster built Aug 2021, so no later than that).
