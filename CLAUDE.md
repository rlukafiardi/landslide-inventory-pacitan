# Landslide inventory mapping, Pacitan (Cempaka 2017)

## Goal
Detect landslides triggered by Cyclone Cempaka (28 Nov to 1 Dec 2017) in Pacitan with a data science approach instead of manual digitizing. Success measured by: TBD (polygon-level precision, recall, IoU vs a reference inventory).
POC: do Satellite Embedding change features (2018 vs 2017) detect Cempaka landslides at least as well as a Sentinel-2 dNDVI baseline? Slice: 1 to 2 sub-districts, a few hundred labeled polygons, spatial train/test split. Pass / stop: embeddings beat dNDVI recall at similar precision, or catch slides it misses; drop if no better or only works above a size cutoff that loses most of the inventory (to be written as a script check). Out of scope until it passes: all of Pacitan, susceptibility mapping, pipeline, dashboard, other events.

## Data
- Sources: TBD
- Grain / key tables: TBD
- Known quirks: embeddings are annual (2017 is almost all pre-event; 2018 may be diluted by revegetation); 10 m pixels miss small slides.

## Decisions
- 2026-11-11: Project created. Type: modelling, spatial. Tools: geopandas, shapely, rasterio, pandas, matplotlib, earthengine-api (uv). scikit-learn added at the modelling step.

## Status
- Works: scaffold only.
- Next, riskiest first: (1) find a usable reference inventory and check terms; (2) share of polygons under 2 to 3 pixels; (3) do 2017 vs 2018 embeddings change at known scars; (4) usable Sentinel-2 scenes Dec 2017 to early 2018.
