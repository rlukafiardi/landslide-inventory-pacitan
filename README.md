# Landslide inventory mapping, Pacitan (Cyclone Cempaka, Nov 2017)

Goal: test whether Google Satellite Embedding (annual, 10 m) change features can detect Cempaka landslides
as well as or better than a Sentinel-2 dNDVI baseline, replacing manual digitizing.

## Data sources
- Google Satellite Embedding V1 annual (Earth Engine), Sentinel-2, DEM: to be confirmed.
- Reference landslide inventory: TBD (availability and terms unverified).

## How to run
```
uv sync
cp .env.example .env   # set EARTHENGINE_PROJECT
uv run earthengine authenticate
```
