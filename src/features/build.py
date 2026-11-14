"""Phase 3: feature builder. One code path for the training table and (phase 5) full-area prediction.

feature_window(row0, col0, h, w) reads a window of the interim rasters with a halo, derives every feature in registry.NAMES
and returns {name: (h, w) float32 array, NaN where missing}. A pixel's features never depend on the tile it is computed in
(halo >= half the largest window; NaN neighbours are replaced by fixed constants, not tile medians).

CLI: python -m src.features.build   -> data/interim/features_train.parquet (training blocks only: every positive pixel and
N_NEG uniformly drawn negative pixels), data/interim/fill_values.json (medians from that table), and
reports/tables/feature_catalogue.csv. Missing values stay NaN in the table; apply_fills() fills them for linear models.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi

from src.data.load import block_mask
from src.features.registry import ANOM, BANDS, NAMES, WIN, catalogue

FDIR = "data/spatial/mendeley_vrtx3w6mjd"
INTERIM = "data/interim"
HALO = ANOM // 2
TILE = 512
N_NEG, SEED = 400_000, 42
# constants that stand in for NaN neighbours inside the neighbourhood filters (rounded EDA medians of background pixels)
NBR_FILL = {"dB": 0.09, "postB": 0.68}
LOG1P = ("distriv", "distroad")  # spi is already on a log scale in the source (range -8 to 7)
FACTORS = ["slope", "elev", "distriv", "distroad", "twi", "spi", "planc", "profc", "geologi", "landuse"]


def _read(path, win, bands=None, boundless=False, fill=None):
    with rasterio.open(path) as r:
        a = r.read(bands, window=win, boundless=boundless, fill_value=fill).astype("float32")
        nd = r.nodata
    return a, nd


def nbr_stats(x, w):
    """mean, min, std of a (H, W) array over a w x w window; edges use the nearest value."""
    mean = ndi.uniform_filter(x, w, mode="nearest")
    sq = ndi.uniform_filter(x * x, w, mode="nearest")
    return mean, ndi.minimum_filter(x, w, mode="nearest"), np.sqrt(np.maximum(sq - mean * mean, 0))


def feature_window(row0, col0, h, w):
    halo_win = Window(col0 - HALO, row0 - HALO, w + 2 * HALO, h + 2 * HALO)
    ndvi, _ = _read(f"{INTERIM}/ndvi.tif", halo_win, boundless=True, fill=-9999.0)
    ndvi[ndvi == -9999.0] = np.nan
    pre, post_a, post_b = ndvi
    d_a, d_b = post_a - pre, post_b - pre
    c = (slice(HALO, HALO + h), slice(HALO, HALO + w))
    out = {"pre": pre[c], "postA": post_a[c], "postB": post_b[c], "dA": d_a[c], "dB": d_b[c]}
    for base, arr in (("dB", d_b), ("postB", post_b)):
        filled = np.where(np.isnan(arr), NBR_FILL[base], arr)
        for win in WIN:
            mean, mn, std = nbr_stats(filled, win)
            out[f"{base}_mean{win}"], out[f"{base}_min{win}"], out[f"{base}_std{win}"] = mean[c], mn[c], std[c]
        out[f"{base}_anom{ANOM}"] = (filled - ndi.uniform_filter(filled, ANOM, mode="nearest"))[c]
    centre = Window(col0, row0, w, h)
    for name in FACTORS:
        a, nd = _read(f"{FDIR}/map_{name}.tif", centre, bands=1)
        a[a == nd] = np.nan
        if name in LOG1P:
            a = np.where(a > -1, np.log1p(np.where(a > -1, a, 0)), np.nan)
        out[name] = a
    asp, nd = _read(f"{FDIR}/map_aspect.tif", centre, bands=1)
    flat = (asp == nd) | (asp < 0)
    out["asp_sin"], out["asp_cos"] = (
        np.where(flat, 0, np.sin(np.deg2rad(asp))),
        np.where(flat, 0, np.cos(np.deg2rad(asp))),
    )
    out["postA_missing"] = np.isnan(out["postA"]).astype("float32")
    out["twi_missing"] = np.isnan(out["twi"]).astype("float32")
    emb, _ = _read(f"{INTERIM}/emb2018.tif", centre)
    emb_nodata = (emb == 0).all(axis=0)
    emb = (emb - 128) / 127
    emb[:, emb_nodata] = np.nan
    for i, b in enumerate(BANDS):
        out[f"e18_{b}"] = emb[i]
    return {k: np.asarray(v, dtype="float32") for k, v in out.items()}


def apply_fills(df, fills):
    """Fill NaN with the stored training medians (the missing flags keep the information)."""
    out = df.copy()
    for col, v in fills.items():
        out[col] = out[col].fillna(v)
    return out


def sample_pixels():
    train = block_mask("train")
    with rasterio.open(f"{INTERIM}/label.tif") as r:
        label = r.read(1)
    with rasterio.open(f"{INTERIM}/block.tif") as r:
        block = r.read(1)
    pi, pj = np.nonzero(train & (label == 1))
    ni, nj = np.nonzero(train & (label == 0))
    k = np.sort(np.random.default_rng(SEED).choice(len(ni), N_NEG, replace=False))
    rows, cols = np.r_[pi, ni[k]], np.r_[pj, nj[k]]
    y = np.r_[np.ones(len(pi), int), np.zeros(N_NEG, int)]
    w = np.r_[np.ones(len(pi)), np.full(N_NEG, len(ni) / N_NEG)]
    return pd.DataFrame({"row": rows, "col": cols, "y": y, "w": w, "block": block[rows, cols]})


def main():
    t0 = time.time()
    px = sample_pixels()
    with rasterio.open(f"{INTERIM}/label.tif") as r:
        H, W = r.height, r.width
    X = np.full((len(px), len(NAMES)), np.nan, dtype="float32")
    tile_id = (px.row // TILE) * 1000 + px.col // TILE
    for n, (tid, idx) in enumerate(px.groupby(tile_id).indices.items(), 1):
        r0, c0 = (tid // 1000) * TILE, (tid % 1000) * TILE
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        feats = feature_window(r0, c0, h, w)
        rr, cc = px.row.to_numpy()[idx] - r0, px.col.to_numpy()[idx] - c0
        for j, name in enumerate(NAMES):
            X[idx, j] = feats[name][rr, cc]
        if n % 20 == 0:
            print(f"  {n} tiles, {time.time() - t0:.0f}s", flush=True)
    table = pd.concat([px, pd.DataFrame(X, columns=NAMES)], axis=1)
    table.to_parquet(f"{INTERIM}/features_train.parquet", index=False)
    miss = table[NAMES].isna().sum()
    fills = {c: float(table.loc[table.y == 0, c].median()) for c in miss[miss > 0].index}
    json.dump(fills, open(f"{INTERIM}/fill_values.json", "w"), indent=1)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    catalogue().to_csv("reports/tables/feature_catalogue.csv", index=False)
    print(
        f"table {len(table):,} rows x {len(NAMES)} features (positives {int(table.y.sum()):,}, negatives {int((table.y == 0).sum()):,}, "
        f"negative weight {table.w.iloc[-1]:.1f}); built in {time.time() - t0:.0f}s"
    )
    print(
        "columns with missing values (rows), filled from training medians when a model needs it:",
        {c: int(v) for c, v in miss[miss > 0].items()},
    )


if __name__ == "__main__":
    main()
