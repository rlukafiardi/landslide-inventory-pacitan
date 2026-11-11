"""Phase 5: out-of-fold probability rasters for one repeat (seed) of the block folds.

For every config in src.modeling.prep.CONFIGS and every fold k, fit on the training-table rows outside fold k (tuned parameters from
tune.py), then stream the training area tile by tile with the one feature code path (src.features.build.feature_window) and predict
every training-domain pixel with the model whose fold held its 3 km block out. Writes data/interim/oof/<config>_r<seed>.npy (float32,
NaN outside the training domain). Repeat 0 also writes the naive score rasters (-dB, -dB_min5), which need no model.
Usage: python -m src.modeling.oof <seed>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.data.constants import B
from src.data.load import block_mask
from src.features.build import INTERIM, TILE, feature_window
from src.features.registry import NAMES
from src.modeling.folds import block_folds, fold_of
from src.modeling.prep import CONFIGS, SETS, design, load_params, make_model

seed = int(sys.argv[1])
OUT = Path(INTERIM) / "oof"
OUT.mkdir(exist_ok=True)
t0 = time.time()
table = pd.read_parquet(f"{INTERIM}/features_train.parquet")
y = table.y.to_numpy() == 1
folds = block_folds(seed)
tfold = fold_of(table.row.to_numpy(), table.col.to_numpy(), folds)
assert (tfold >= 0).all()
params = load_params()
models = {}
for name, (kind, fset) in CONFIGS.items():
    cols = SETS[fset]
    X = design(table, cols, kind)
    models[name] = [make_model(kind, params[name], cols).fit(X[tfold != k], y[tfold != k]) for k in range(5)]
    print(f"fit {name} x5 ({time.time() - t0:.0f}s)", flush=True)

train = block_mask("train")
H, W = train.shape
lut = np.full(14000, -1)
lut[folds.group.to_numpy()] = folds.fold.to_numpy()
rasters = {n: np.lib.format.open_memmap(OUT / f"{n}_r{seed}.npy", mode="w+", dtype="float32", shape=(H, W)) for n in CONFIGS}
for r in rasters.values():
    r[:] = np.nan
if seed == 0:
    naive = {n: np.lib.format.open_memmap(OUT / f"{n}.npy", mode="w+", dtype="float32", shape=(H, W)) for n in ("naive_dB", "naive_dBmin5")}
    for r in naive.values():
        r[:] = np.nan
for r0 in range(0, H, TILE):
    for c0 in range(0, W, TILE):
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        m = train[r0:r0 + h, c0:c0 + w]
        if not m.any():
            continue
        feats = feature_window(r0, c0, h, w)
        ii, jj = np.nonzero(m)
        df = pd.DataFrame({n: feats[n][ii, jj] for n in NAMES})
        pf = lut[((ii + r0) // B) * 1000 + (jj + c0) // B]
        for name, (kind, fset) in CONFIGS.items():
            X = design(df, SETS[fset], kind)
            p = np.empty(len(df), dtype="float32")
            for k in range(5):
                s = pf == k
                if s.any():
                    p[s] = models[name][k].predict_proba(X[s])[:, 1]
            rasters[name][r0 + ii, c0 + jj] = p
        if seed == 0:
            naive["naive_dB"][r0 + ii, c0 + jj] = np.nan_to_num(-df["dB"].to_numpy(), nan=-9.0)
            naive["naive_dBmin5"][r0 + ii, c0 + jj] = np.nan_to_num(-df["dB_min5"].to_numpy(), nan=-9.0)
    print(f"  tile row {r0} done ({time.time() - t0:.0f}s)", flush=True)
for r in rasters.values():
    r.flush()
print(f"repeat {seed} written to {OUT} in {time.time() - t0:.0f}s")
