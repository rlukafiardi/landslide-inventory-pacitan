"""Sample 2017 and 2018 embeddings and Sentinel-2 NDVI composites at the matched points -> features.csv.
Usage: python -m src.build_features <matched_points_csv> <out_csv>
"""
import sys

import ee
import geopandas as gpd
import pandas as pd

from src.embedding_utils import BANDS, COLLECTION, S2_WINDOWS, init_ee, ndvi_composite, sample_image

meta = pd.read_csv(sys.argv[1], index_col="pid")
pts = gpd.GeoDataFrame(meta.reset_index()[["pid"]], geometry=gpd.points_from_xy(meta.lon, meta.lat), crs=4326)
init_ee()
coll = ee.ImageCollection(COLLECTION)
parts = [meta[["grp", "area", "lon", "lat"]]]
for y in (2017, 2018):
    e = sample_image(coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic(), pts, 10, BANDS)
    parts.append(e.rename(columns=lambda b: f"e{y % 100}_{b}"))
for name, (s, e) in S2_WINDOWS.items():
    parts.append(sample_image(ndvi_composite(s, e), pts, 10, ["ndvi"]).ndvi.rename(name))
out = pd.concat(parts, axis=1, join="inner")
out = out.dropna(subset=[c for c in out.columns if c != "area"])
print("points with all features:", len(out), "of", len(meta), "| landslide", int((out.grp == "landslide").sum()), "| background", int((out.grp == "background").sum()))
out.to_csv(sys.argv[2])
