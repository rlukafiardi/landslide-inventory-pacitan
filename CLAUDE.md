# Landslide inventory mapping, Pacitan (Cempaka 2017)

## Goal
Detect landslides triggered by Cyclone Cempaka (28 Nov to 1 Dec 2017) in Pacitan with a data science approach instead of manual digitizing. Success measured by: TBD (polygon-level precision, recall, IoU vs a reference inventory).
POC: do Satellite Embedding change features (2018 vs 2017) detect Cempaka landslides at least as well as a Sentinel-2 dNDVI baseline? Slice: 1 to 2 sub-districts, a few hundred labeled polygons, spatial train/test split. Pass / stop: embeddings beat dNDVI recall at similar precision, or catch slides it misses; drop if no better or only works above a size cutoff that loses most of the inventory (to be written as a script check). Out of scope until it passes: all of Pacitan, susceptibility mapping, pipeline, dashboard, other events.

## Data
- Sources: reference inventory = Samodra et al. (UGM), Mendeley Data DOI 10.17632/vrtx3w6mjd.1, CC BY 4.0 (credit authors). Mendeley page footer reserves text/data-mining and AI-training rights; read as site content, not checked against the dataset licence. Embeddings: Earth Engine GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (2017 layer regenerated in v1.1; pin version, check live catalog for the upper year).
- Grain / key tables: `data/spatial/mendeley_vrtx3w6mjd/landslide_pacitan.gpkg`: 743 MultiPolygons, EPSG:32749, columns Id, kode, Area, X/Y_centroid. Also available, not downloaded: ls_center.gpkg, non_ls.gpkg, 11 factor rasters (~96 MB each).
- Known quirks: embeddings are annual (2017 is almost all pre-event; 2018 may be diluted by revegetation); 10 m pixels miss small slides: median polygon 1,068 m2, 15.9% under 3 px, 27.5% under 5 px. Extent is all of Pacitan (~43 x 39 km); landslides are ~0.13% of the bbox, so negatives must be sampled and pixel accuracy is meaningless.

## Decisions
- 2026-11-11: Project created. Type: modelling, spatial. Tools: geopandas, shapely, rasterio, pandas, matplotlib, earthengine-api (uv). scikit-learn added at the modelling step.
- 2026-11-11: Used the public Samodra inventory as labels (no need for the friend's data). Size check (`src/inventory_check.py`) done; pass/stop criteria unchanged. Report metrics at polygon and pixel level, split by size class (<5 px, 5 to 20 px, >20 px). Slice must shrink from all of Pacitan to one or two sub-districts.
- 2026-11-11: Embedding change check (cosine distance 2017->2018, point level, 743 landslide vs 743 background, AUC): random bg 0.725; slope+elevation matched 0.720; + land-use matched 0.705 (size classes 0.52 / 0.73 / 0.83 for <5 px / 5-20 px / >20 px). 2018->2019 control ~0.50. Risk still open: 2017 layer is v1.1, 2018 is v1.0, and background 17->18 change is ~2.1x a normal year (M3 fails every run).
- 2026-11-11: dNDVI baseline (S2 L1C + Cloud Score+, pre Aug-Oct 2017) on the same points: post Dec17-Mar18 AUC 0.698 (cloud-limited, tie with embeddings 0.704, diff +0.007, CI [-0.021, +0.037]); post Apr-May18 AUC 0.790, beats embeddings by 0.085 (CI [-0.116, -0.057]). Which post window counts as primary was not fixed beforehand; B2 should be read as requiring the CI to exclude zero. Unsupervised embedding distance is no better than dNDVI. Supervised use of embeddings untested.

## Status
- Works: scaffold, private repo (rlukafiardi/landslide-inventory-pacitan), inventory downloaded and size-checked, Earth Engine access (project <earthengine-project>; `.env` holds it and SSL_CERT_FILE), checks in src/ (embedding_change_check, matched_background_check, s2_dndvi_check).
- Next, riskiest first: (1) done: inventory found; (2) done: size check; (3) done, see Decisions: weak signal, no better than dNDVI; (4) usable Sentinel-2 scenes Dec 2017 to early 2018.
