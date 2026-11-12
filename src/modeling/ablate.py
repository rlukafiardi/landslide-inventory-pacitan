"""Phase 5: variants of the chosen model (c_hgb: all features, boosting): one group or feature removed, or a new group added.

Each variant refits boosting with the tuned parameters of its base config (c_hgb, or a_hgb for the a-based variant) on the same block
folds of one repeat, and writes data/interim/oof/abl_<variant>_r<seed>.npy (same format as oof.py; the `abl_` prefix keeps them out of
the default evaluation). Hyperparameters are not retuned per variant, so a difference is the features, not the tuning.
Evaluate with: python -m src.modeling.evaluate_oof --out reports/tables/ablation_raw.csv c_hgb_r<seed> abl_<variant>_r<seed> ...
then python -m src.modeling.ablation_report <seed>.
Usage: python -m src.modeling.ablate <seed> [variant ...]   (default: the six "no_*" variants)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import B
from src.data.load import block_mask
from src.features.build import INTERIM, TILE, feature_window
from src.features.registry import GROUPS, NAMES
from src.modeling.folds import block_folds, fold_of
from src.modeling.prep import SETS, design, load_params, make_model


def minus(*drop):
    return [c for c in SETS["c"] if c not in drop]


# name -> (columns, base config whose tuned parameters are used)
VARIANTS = {
    "no_ndvi_pixel": (minus(*GROUPS["ndvi_pixel"]), "c_hgb"),
    "no_ndvi_nbr": (minus(*GROUPS["ndvi_nbr"]), "c_hgb"),
    "no_terrain": (minus(*GROUPS["terrain"]), "c_hgb"),
    "no_site_cat": (minus(*GROUPS["site_cat"]), "c_hgb"),
    "no_distroad": (minus("distroad"), "c_hgb"),
    "no_landuse": (minus("landuse"), "c_hgb"),
    "plus_ndvi_prev": (SETS["c"] + GROUPS["ndvi_prev"], "c_hgb"),  # option A: the year-before NDVI (normal seasonal swing)
    "a_plus_ndvi_prev": (SETS["a"] + GROUPS["ndvi_prev"], "a_hgb"),
    "plus_ndvi_post": (SETS["c"] + GROUPS["ndvi_post"], "c_hgb"),  # option A2: persistence one and two years after the event (retrospective)
    "a_plus_ndvi_post": (SETS["a"] + GROUPS["ndvi_post"], "a_hgb"),
}
DEFAULT = [v for v in VARIANTS if v.startswith("no_")]

if __name__ == "__main__":
    seed = int(sys.argv[1])
    chosen = sys.argv[2:] or DEFAULT
    OUT = Path(INTERIM) / "oof"
    t0 = time.time()
    table = pd.read_parquet(f"{INTERIM}/features_train.parquet")
    y = table.y.to_numpy() == 1
    folds = block_folds(seed)
    tfold = fold_of(table.row.to_numpy(), table.col.to_numpy(), folds)
    params = load_params()
    models = {}
    for v in chosen:
        cols, base = VARIANTS[v]
        X = design(table, cols, "hgb")
        models[v] = [make_model("hgb", params[base], cols).fit(X[tfold != k], y[tfold != k]) for k in range(5)]
        print(f"fit {v} ({len(cols)} features, parameters of {base}) x5 ({time.time() - t0:.0f}s)", flush=True)

    train = block_mask("train")
    H, W = train.shape
    lut = np.full(14000, -1)
    lut[folds.group.to_numpy()] = folds.fold.to_numpy()
    rasters = {v: np.lib.format.open_memmap(OUT / f"abl_{v}_r{seed}.npy", mode="w+", dtype="float32", shape=(H, W)) for v in chosen}
    for r in rasters.values():
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
            for v in chosen:
                X = design(df, VARIANTS[v][0], "hgb")
                p = np.empty(len(df), dtype="float32")
                for k in range(5):
                    s = pf == k
                    if s.any():
                        p[s] = models[v][k].predict_proba(X[s])[:, 1]
                rasters[v][r0 + ii, c0 + jj] = p
        print(f"  tile row {r0} done ({time.time() - t0:.0f}s)", flush=True)
    for r in rasters.values():
        r.flush()
    print(f"ablation repeat {seed} written in {time.time() - t0:.0f}s")
