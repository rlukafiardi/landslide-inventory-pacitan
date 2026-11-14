"""POC features re-sampled from the downloaded stack at the POC points, on the aligned grid.
Landslide points are moved onto the Sentinel-2 grid (same shift as src/data/labels.py); lon/lat stay in the inventory
frame, so the site factors (sampled in site_controlled_check.py) keep their original coordinates.
Usage: python -m src.data.features_from_stack <poc_features_csv> <out_csv>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.data.constants import SHIFT_E_M, SHIFT_N_M

poc = pd.read_csv(sys.argv[1], index_col="pid")
xy = gpd.GeoSeries(gpd.points_from_xy(poc.lon, poc.lat), index=poc.index, crs=4326).to_crs(32749)
land = (poc.grp == "landslide").to_numpy()
pts = list(zip(xy.x + np.where(land, SHIFT_E_M, 0), xy.y + np.where(land, SHIFT_N_M, 0), strict=True))
out = poc[["grp", "area", "lon", "lat"]].copy()
for y in (2017, 2018):
    with rasterio.open(f"data/interim/emb{y}.tif") as r:
        v = (np.array(list(r.sample(pts)), dtype=float) - 128) / 127
    out = out.join(pd.DataFrame(v, index=poc.index, columns=[f"e{y % 100}_A{i:02d}" for i in range(64)]))
with rasterio.open("data/interim/ndvi.tif") as r:
    nd = np.array(list(r.sample(pts)), dtype=float)
nd[nd == -9999] = np.nan
out["pre"], out["postA"], out["postB"] = nd[:, 0], nd[:, 1], nd[:, 2]
out = out.dropna(subset=[c for c in out.columns if c != "area"])
out.to_csv(sys.argv[2])
print("rows", len(out), "| landslide", int((out.grp == "landslide").sum()))
