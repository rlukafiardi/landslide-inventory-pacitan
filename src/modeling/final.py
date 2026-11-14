"""Phase 5 final models: refit on ALL training-table rows (unweighted, as the out-of-fold models were) and predict the whole domain.

Models: c_hgb (the chosen model), simple_logreg (best simple baseline), and the naive -dB_min5 score (no fit).
Writes
  models/<config>.joblib                  fitted model (gitignored)
  models/operating_points.json            frozen operating point per model: probability threshold (median over the out-of-fold repeats of the
                                          threshold that reaches 80% object recall) and minimum object size (most frequent over the repeats)
  data/interim/final_<config>.npy         float32 score raster over the whole domain (NaN outside). On the training blocks these
                                          scores are in-sample (the model saw their labels); only the test blocks are an honest score.
Predicting the test blocks here is not scoring them: no label or metric is touched. Scoring is evaluate_test.py (LANDSLIDE_FINAL_EVAL=1).
Usage: python -m src.modeling.final
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio

from src.features.build import INTERIM, TILE, feature_window
from src.features.registry import NAMES
from src.modeling.prep import CONFIGS, SETS, design, load_params, make_model

FINAL = ["c_hgb", "simple_logreg"]
NAIVE = "naive_dBmin5"


def operating_points():
    raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
    raw = raw[raw.recall_target == 0.8]
    out = {}
    for cfg in FINAL + [NAIVE]:
        d = raw[raw.config == cfg]
        out[cfg] = {
            "threshold": float(d.thr.median()),
            "a_min": int(d.a_min.mode().iloc[0]),
            "repeats": int(len(d)),
            "threshold_range": [float(d.thr.min()), float(d.thr.max())],
            "oof_precision_mean": float(d.precision_at_target.mean()),
        }
    return out


if __name__ == "__main__":
    t0 = time.time()
    Path("models").mkdir(exist_ok=True)
    table = pd.read_parquet(f"{INTERIM}/features_train.parquet")
    y = table.y.to_numpy() == 1
    params = load_params()
    models = {}
    for cfg in FINAL:
        kind, fset = CONFIGS[cfg]
        cols = SETS[fset]
        models[cfg] = make_model(kind, params[cfg], cols).fit(design(table, cols, kind), y)
        joblib.dump(models[cfg], f"models/{cfg}.joblib")
        print(f"fit {cfg} on {len(table):,} rows ({int(y.sum()):,} positives) ({time.time() - t0:.0f}s)", flush=True)
    ops = operating_points()
    json.dump(ops, open("models/operating_points.json", "w"), indent=1)
    print(json.dumps(ops, indent=1))

    with rasterio.open(f"{INTERIM}/block.tif") as r:
        dom = r.read(1) >= 0
    H, W = dom.shape
    out = {
        c: np.lib.format.open_memmap(f"{INTERIM}/final_{c}.npy", mode="w+", dtype="float32", shape=(H, W))
        for c in FINAL + [NAIVE]
    }
    for r in out.values():
        r[:] = np.nan
    for r0 in range(0, H, TILE):
        for c0 in range(0, W, TILE):
            h, w = min(TILE, H - r0), min(TILE, W - c0)
            m = dom[r0 : r0 + h, c0 : c0 + w]
            if not m.any():
                continue
            feats = feature_window(r0, c0, h, w)
            ii, jj = np.nonzero(m)
            df = pd.DataFrame({n: feats[n][ii, jj] for n in NAMES})
            for cfg in FINAL:
                kind, fset = CONFIGS[cfg]
                out[cfg][r0 + ii, c0 + jj] = models[cfg].predict_proba(design(df, SETS[fset], kind))[:, 1]
            out[NAIVE][r0 + ii, c0 + jj] = np.nan_to_num(-df["dB_min5"].to_numpy(), nan=-9.0)
        print(f"  tile row {r0} done ({time.time() - t0:.0f}s)", flush=True)
    for r in out.values():
        r.flush()
    print(f"final rasters written in {time.time() - t0:.0f}s")
