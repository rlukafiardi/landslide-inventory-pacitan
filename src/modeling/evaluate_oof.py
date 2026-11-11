"""Phase 5: object-level evaluation of the out-of-fold probability rasters, paired across models (docs/validation-design.md).

Per raster: thresholds = quantiles giving a flagged-area share in SHARES, a_min in A_MINS; per (threshold, a_min) the per-block counts
(polygons >= 5 px detected, objects, true objects) are cached in data/interim/oof/counts_<name>.npz. Operating point = highest object
precision with recall (polygons >= 5 px) >= target. Intervals: the same block resamples for every model and repeat, so differences are
paired. Writes reports/tables/modeling_results.csv and modeling_pairs.csv.
Usage: python -m src.modeling.evaluate_oof [name ...]   (default: every raster found in data/interim/oof)
"""
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.metrics import EIGHT, object_metrics

OOF = Path(INTERIM) / "oof"
SHARES = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
A_MINS = (1, 2, 3, 5)
TARGETS = (0.7, 0.8, 0.9)
N_BOOT, SEED = 1000, 7
CLS = ["<5 px", "5-20 px", ">20 px"]

with rasterio.open(f"{INTERIM}/poly_id.tif") as r:
    poly = r.read(1)
with rasterio.open(f"{INTERIM}/block.tif") as r:
    block = r.read(1)
with rasterio.open(f"{INTERIM}/label.tif") as r:
    label = r.read(1)
train = block_mask("train")
ring = (label == 255) & (block >= 0)
pt = pd.read_parquet(f"{INTERIM}/polygons.parquet")
pt = pt[pt.in_train].reset_index(drop=True)
block_ids = np.unique(block[train])
bidx = {int(b): i for i, b in enumerate(block_ids)}
NB = len(block_ids)
pt["bi"] = pt.block.map(bidx)
big = pt.size_class != "<5 px"
n_poly_big = np.bincount(pt.bi[big], minlength=NB)
cls_id = pt.size_class.map({c: i for i, c in enumerate(CLS)}).to_numpy()
poly_flat, block_flat, ring_flat, train_flat = poly.ravel(), block.ravel(), ring.ravel(), train.ravel()
bi_of_flat = np.full(block.size, -1)
valid = block_flat >= 0
bi_of_flat[valid] = np.searchsorted(block_ids, block_flat[valid])
in_train_poly = np.zeros(int(poly.max()) + 1, bool)
in_train_poly[pt.id.to_numpy()] = True
poly_to_row = np.full(int(poly.max()) + 1, -1)
poly_to_row[pt.id.to_numpy()] = np.arange(len(pt))


def block_counts(prob):
    """counts[t, a, block, k]: k = 0 big polygons detected, 1 objects, 2 true objects, 3..5 detections per size class."""
    flat = prob.ravel()
    vals = flat[train_flat]
    thr = np.quantile(vals[~np.isnan(vals)], 1 - np.array(SHARES))
    out = np.zeros((len(thr), len(A_MINS), NB, 6), dtype=np.int32)
    for ti, th in enumerate(thr):
        pm = (prob >= th) & train
        lab, n = ndi.label(pm, structure=EIGHT)
        if n == 0:
            continue
        idx = np.flatnonzero(pm.ravel())
        labs = lab.ravel()[idx]
        sizes = np.bincount(labs, minlength=n + 1)
        uniq, first = np.unique(labs, return_index=True)
        obj_block = bi_of_flat[idx[first]]
        tp_lab = np.zeros(n + 1, bool)
        tp_lab[np.unique(labs[(poly_flat[idx] > 0) | ring_flat[idx]])] = True
        pid = poly_flat[idx]
        for ai, a in enumerate(A_MINS):
            keep_lab = sizes >= a
            keep_lab[0] = False
            ko = keep_lab[uniq]
            out[ti, ai, :, 1] = np.bincount(obj_block[ko], minlength=NB)
            out[ti, ai, :, 2] = np.bincount(obj_block[ko & tp_lab[uniq]], minlength=NB)
            hit_pid = np.unique(pid[keep_lab[labs] & (pid > 0)])
            rows = poly_to_row[hit_pid]
            rows = rows[rows >= 0]
            det = np.zeros(len(pt), bool)
            det[rows] = True
            out[ti, ai, :, 0] = np.bincount(pt.bi[det & big], minlength=NB)
            for ci in range(3):
                out[ti, ai, :, 3 + ci] = np.bincount(pt.bi[det & (cls_id == ci)], minlength=NB)
    return thr, out


def op_point(counts, target):
    """counts (..., T, A, K) already summed over blocks -> (precision, recall, t index, a index); NaN precision where recall never reaches target."""
    det, tot, obj, tp = counts[..., 0], counts[..., 6], counts[..., 1], counts[..., 2]
    recall = det / np.maximum(tot, 1)
    prec = np.where(obj > 0, tp / np.maximum(obj, 1), np.nan)
    ok = recall >= target
    p = np.where(ok, prec, -1.0)
    flat = p.reshape(*p.shape[:-2], -1)
    best = flat.argmax(-1)
    bp = np.take_along_axis(flat, best[..., None], -1)[..., 0]
    reached = bp >= 0
    ti, ai = np.divmod(best, p.shape[-1])
    return np.where(reached, bp, np.nan), reached, ti, ai


def with_totals(c):
    """Append the per-block polygon totals as an extra column (index 6) so the op_point sums stay in one array."""
    tot = np.broadcast_to(n_poly_big[None, None, :, None], c.shape[:3] + (1,))
    return np.concatenate([c, tot], axis=-1)


def main(names):
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(NB, np.full(NB, 1 / NB), size=N_BOOT).astype("float64")
    rows, store = [], {}
    y_pix = label[train]
    keep = y_pix != 255
    for name in names:
        f = OOF / f"counts_{name}.npz"
        if f.exists():
            z = np.load(f)
            thr, cnt = z["thr"], z["cnt"]
            ap = float(z["ap"])
        else:
            prob = np.load(OOF / f"{name}.npy", mmap_mode="r")
            prob = np.asarray(prob)
            thr, cnt = block_counts(prob)
            pv = prob[train][keep]
            ap = float(average_precision_score(y_pix[keep] == 1, np.nan_to_num(pv, nan=-1e9)))
            np.savez_compressed(f, thr=thr, cnt=cnt, ap=ap)
        store[name] = (thr, with_totals(cnt), ap)
    for name, (thr, cnt, ap) in store.items():
        model, rep = re.match(r"(.+?)(?:_r(\d+))?$", name).groups()
        full = cnt.sum(2)
        for tg in TARGETS:
            p, reached, ti, ai = op_point(full, tg)
            row = {"config": model, "repeat": -1 if rep is None else int(rep), "recall_target": tg, "precision_at_target": float(p) if reached else np.nan,
                   "reached": bool(reached), "weighted_pixel_AP": ap, "date": date.today().isoformat()}
            if reached:
                row.update(flagged_share=SHARES[int(ti)], a_min=A_MINS[int(ai)], thr=float(thr[int(ti)]),
                           recall_ge5=float(full[ti, ai, 0] / full[ti, ai, 6]), objects=int(full[ti, ai, 1]))
                for ci, c in enumerate(CLS):
                    row[f"recall_{c}"] = float(full[ti, ai, 3 + ci] / max((cls_id == ci).sum(), 1))
            rows.append(row)
        # paired bootstrap of the operating precision, same block resamples for every raster
        boot = {tg: op_point(np.einsum("bn,tank->btak", W, cnt), tg)[0] for tg in TARGETS}
        for tg in TARGETS:
            np.save(OOF / f"boot_{name}_{tg}.npy", boot[tg])
    res = pd.DataFrame(rows)
    res.to_csv("reports/tables/modeling_results_raw.csv", index=False)
    print(res[res.recall_target == 0.8][["config", "repeat", "precision_at_target", "reached", "flagged_share", "recall_ge5", "weighted_pixel_AP"]].round(3).to_string(index=False))


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in OOF.glob("*.npy") if not p.stem.startswith(("boot_", "counts_")))
    main(names)
