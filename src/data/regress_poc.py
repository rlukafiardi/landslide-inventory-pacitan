"""Phase 1 verification: AUCs from the downloaded stack at the POC points must match the POC within 0.01.
POC reference (features.csv, same 3 km block CV): A 0.945 logreg / 0.940 rf; B 0.890 / 0.894.
Usage: python -m src.data.regress_poc
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.cv_utils import blocks, oof, pooled_auc
from src.data.constants import SHIFT_E_M, SHIFT_N_M

poc = pd.read_csv("data/processed/features.csv", index_col="pid")
xy = gpd.GeoSeries(gpd.points_from_xy(poc.lon, poc.lat), index=poc.index, crs=4326).to_crs(32749)
# landslide points sit in the inventory frame: move them onto the Sentinel-2 grid like the polygons in labels.py
land = (poc.grp == "landslide").to_numpy()
pts = list(zip(xy.x + np.where(land, SHIFT_E_M, 0), xy.y + np.where(land, SHIFT_N_M, 0), strict=True))
with rasterio.open("data/interim/emb2018.tif") as r:
    e18 = (np.array(list(r.sample(pts)), dtype=float) - 128) / 127
with rasterio.open("data/interim/ndvi.tif") as r:
    nd = np.array(list(r.sample(pts)), dtype=float)
d = pd.DataFrame(e18, index=poc.index, columns=[f"e{i}" for i in range(64)])
d["pre"], d["postA"], d["postB"] = nd[:, 0], nd[:, 1], nd[:, 2]
d["dA"], d["dB"] = d.postA - d.pre, d.postB - d.pre
d = d.replace(-9999.0, np.nan)
print("points with a missing value:", int(d.isna().any(axis=1).sum()), "of", len(d))
ok = ~d.isna().any(axis=1)
d, y, g = d[ok], (poc.grp == "landslide").to_numpy()[ok.to_numpy()], blocks(xy.x, xy.y, 3000)[ok.to_numpy()]
sets = {"A": [f"e{i}" for i in range(64)], "B": ["pre", "postA", "postB", "dA", "dB"]}
ref = {("A", "logreg"): 0.945, ("A", "rf"): 0.940, ("B", "logreg"): 0.890, ("B", "rf"): 0.894}
allok = True
for (k, m), want in ref.items():
    got = pooled_auc(oof(d, sets[k], m, g, y), y)
    good = abs(got - want) <= 0.01
    allok &= good
    print(f"set {k} {m:<6}: stack {got:.3f} vs POC {want:.3f} (diff {got - want:+.3f}) -> {'PASS' if good else 'FAIL'}")
print("REGRESSION PASS" if allok else "REGRESSION FAIL")
