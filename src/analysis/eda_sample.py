"""Phase 2 sample from the TRAINING blocks only: every positive pixel and N_NEG uniformly drawn negative pixels.

Writes data/interim/eda_sample.csv.gz (labels, weights, NDVI and the 10 site factors; no embeddings) and
data/interim/eda_emb.npy (float16, 64 dequantized 2018 embedding bands, same row order).
Negative weight w = (training negatives) / N_NEG, so weighted sums recover population totals. Usage: python -m src.analysis.eda_sample
"""
import numpy as np
import pandas as pd
import rasterio

from src.data.load import block_mask

FDIR = "data/spatial/mendeley_vrtx3w6mjd"
N_NEG, SEED = 300_000, 42
FACTORS = ["aspect", "distriv", "distroad", "elev", "geologi", "landuse", "planc", "profc", "slope", "spi", "twi"]

train = block_mask("train")
with rasterio.open("data/interim/label.tif") as r:
    label = r.read(1)
with rasterio.open("data/interim/block.tif") as r:
    block = r.read(1)
pi, pj = np.nonzero(train & (label == 1))
ni, nj = np.nonzero(train & (label == 0))
rng = np.random.default_rng(SEED)
k = np.sort(rng.choice(len(ni), N_NEG, replace=False))
ri, cj = np.r_[pi, ni[k]], np.r_[pj, nj[k]]
y = np.r_[np.ones(len(pi), int), np.zeros(N_NEG, int)]
w = np.r_[np.ones(len(pi)), np.full(N_NEG, len(ni) / N_NEG)]
print(f"train positives {len(pi):,}; train negatives {len(ni):,}; sample negatives {N_NEG:,}; negative weight {len(ni) / N_NEG:.1f}")

d = pd.DataFrame({"row": ri, "col": cj, "y": y, "w": w, "block": block[ri, cj]})
with rasterio.open("data/interim/ndvi.tif") as r:
    nd = r.read().astype("float32")[:, ri, cj]
nd[nd == -9999] = np.nan
d["pre"], d["postA"], d["postB"] = nd
d["dA"], d["dB"] = d.postA - d.pre, d.postB - d.pre
for f in FACTORS:
    with rasterio.open(f"{FDIR}/map_{f}.tif") as r:
        v = r.read(1)[ri, cj].astype("float32")
        v[v == r.nodata] = np.nan
        d[f] = v
d.to_csv("data/interim/eda_sample.csv.gz", index=False, float_format="%.5g")
with rasterio.open("data/interim/emb2018.tif") as r:
    e = r.read()[:, ri, cj]
np.save("data/interim/eda_emb.npy", ((e.astype("float32") - 128) / 127).T.astype("float16"))
print("rows", len(d), "| missing per column:", {c: int(v) for c, v in d.isna().sum().items() if v})
