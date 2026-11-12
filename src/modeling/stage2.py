"""Phase 5, option B: object-level second stage on the c_hgb out-of-fold candidates (decision of 2026-11-12, after the blinded review).

Candidates = 8-connected objects of c_hgb_r<seed> >= the threshold of flagged share SHARE1 (the 90% recall region), >= AMIN1 pixels.
A boosted model re-scores each object from object features (shape, stage-1 probability statistics, NDVI change inside and around the
object, terrain and land-use context, density of other candidates within 500 m, optional mean embedding). Label = true object (overlaps a
polygon or its 20 m ring), the same definition as the metric. The objects are scored with the same block folds as stage 1 (fold k
objects by a model fitted on the other folds); their stage-1 probabilities are already out-of-fold. Limit: the stage-1 models that
produced the training objects' probabilities saw fold k's labels (the usual stacking leak, second order).
Evaluation reuses evaluate_oof's metric code and bootstrap resamples, so differences against stage 1 are paired. Before any number is
used, keeping every candidate must reproduce the stage-1 counts of the same threshold exactly (asserted).
Usage: python -m src.modeling.stage2 <seed>   -> reports/tables/stage2_results.csv (rows for this seed are replaced)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from src.cv_utils import SEED
from src.data.constants import B
from src.features.build import FDIR, INTERIM, NBR_FILL
from src.modeling import evaluate_oof as ev
from src.modeling.folds import block_folds
from src.modeling.metrics import EIGHT

OOF = ev.OOF
SHARE1, AMIN1, GATES = 0.08, 2, (2, 3, 5)
FACTORS = ("slope", "elev", "distriv", "distroad", "twi", "spi", "landuse", "geologi")
LOG1P = ("distriv", "distroad")
CAT = ["landuse", "geologi"]
PROBSTATS = ["pmax", "pmean", "size"]
BASE = PROBSTATS + ["elongation", "extent", "bbox_h", "bbox_w", "dB_mean", "dB_min", "dB_lt20", "postB_mean", "pre_mean", "dBenv", "dB_contrast",
                    "dBmin5", "dens", "penv", "slope_mean", "slope_std", "elev_mean", "distriv_min", "distroad_min", "twi_mean", "spi_mean"] + CAT
EMB = [f"e{i:02d}" for i in range(64)]
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=SEED)


def read1(path):
    with rasterio.open(path) as r:
        a = r.read(1).astype("float32")
        nd = r.nodata
    if nd is not None:
        a[a == nd] = np.nan
    return a


def build_objects(seed):
    t0 = time.time()
    prob = np.asarray(np.load(OOF / f"c_hgb_r{seed}.npy", mmap_mode="r"))
    # kept as np.float64: a Python float is cast to float32 against the float32 raster (NumPy 2 promotion) and shifts boundary pixels
    thr1 = np.load(OOF / f"counts_c_hgb_r{seed}.npz")["thr"][ev.SHARES.index(SHARE1)]
    H, W = prob.shape
    lab, n = ndi.label((prob >= thr1) & ev.train, structure=EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= AMIN1
    keep[0] = False
    flat = lab.ravel()
    idx = np.flatnonzero(keep[flat])
    labs = flat[idx]
    order = np.argsort(labs, kind="stable")  # stable: the first pixel of an object stays its first row-major pixel
    idx, labs = idx[order], labs[order]
    uniq, start = np.unique(labs, return_index=True)
    oi = np.searchsorted(uniq, labs)
    rows, cols = idx // W, idx % W
    print(f"stage 1 threshold {thr1:.4f}: {len(uniq):,} candidate objects, {len(idx):,} pixels ({time.time() - t0:.0f}s)", flush=True)

    tp_lab = np.zeros(n + 1, bool)
    tp_lab[np.unique(labs[(ev.poly_flat[idx] > 0) | ev.ring_flat[idx]])] = True
    pid = ev.poly_flat[idx]
    m = pid > 0
    pairs = np.unique(np.stack([oi[m], ev.poly_to_row[pid[m]]], axis=1), axis=0)
    pairs = pairs[pairs[:, 1] >= 0]
    first = idx[start]
    obj = pd.DataFrame({"size": np.diff(np.r_[start, len(idx)]), "tp": tp_lab[uniq], "bi": ev.bi_of_flat[first], "block": (first // W // B) * 1000 + (first % W) // B})

    with rasterio.open(f"{INTERIM}/ndvi.tif") as r:
        nd = r.read().astype("float32")
    nd[nd == -9999.0] = np.nan
    pre, post_b = nd[0], nd[2]
    dB = post_b - pre
    dBf = np.where(np.isnan(dB), NBR_FILL["dB"], dB)
    cm = np.zeros((H, W), "float32")
    cm.ravel()[idx] = 1
    px = {"p": prob.ravel()[idx], "dB": dB.ravel()[idx], "postB": post_b.ravel()[idx], "pre": pre.ravel()[idx],
          "dBenv": ndi.uniform_filter(dBf, 21, mode="nearest").ravel()[idx], "dBmin5": ndi.minimum_filter(dBf, 5, mode="nearest").ravel()[idx],
          "dens": ndi.uniform_filter(cm, 51, mode="constant").ravel()[idx], "penv": ndi.uniform_filter(np.nan_to_num(prob, nan=0.0), 51, mode="constant").ravel()[idx]}
    del nd, dB, dBf, cm
    for name in FACTORS:
        a = read1(f"{FDIR}/map_{name}.tif")
        if name in LOG1P:
            a = np.where(a > -1, np.log1p(np.where(a > -1, a, 0)), np.nan)
        px[name] = a.ravel()[idx]
    del a
    df = pd.DataFrame({k: v for k, v in px.items() if k not in CAT})
    df["o"] = oi
    df["dB_lt20"] = (df.dB < -0.2).astype("float32")
    df["r"], df["c"] = rows.astype("float64"), cols.astype("float64")
    df["rr"], df["cc"], df["rc"] = df.r ** 2, df.c ** 2, df.r * df.c
    g = df.groupby("o")
    a = g.agg(pmax=("p", "max"), pmean=("p", "mean"), dB_mean=("dB", "mean"), dB_min=("dB", "min"), dB_lt20=("dB_lt20", "mean"), postB_mean=("postB", "mean"),
              pre_mean=("pre", "mean"), dBenv=("dBenv", "mean"), dBmin5=("dBmin5", "min"), dens=("dens", "mean"), penv=("penv", "mean"),
              slope_mean=("slope", "mean"), slope_std=("slope", "std"), elev_mean=("elev", "mean"), distriv_min=("distriv", "min"),
              distroad_min=("distroad", "min"), twi_mean=("twi", "mean"), spi_mean=("spi", "mean"), r=("r", "mean"), c=("c", "mean"), rr=("rr", "mean"),
              cc=("cc", "mean"), rc=("rc", "mean"), rmin=("r", "min"), rmax=("r", "max"), cmin=("c", "min"), cmax=("c", "max"))
    vr, vc, cv = a.rr - a.r ** 2, a.cc - a.c ** 2, a.rc - a.r * a.c
    half, rad = (vr + vc) / 2, np.sqrt(((vr - vc) / 2) ** 2 + cv ** 2)
    a["elongation"] = np.sqrt((half + rad + 1 / 12) / np.maximum(half - rad + 1 / 12, 1e-6))  # 1/12: variance of one pixel
    a["bbox_h"], a["bbox_w"] = a.rmax - a.rmin + 1, a.cmax - a.cmin + 1
    a["dB_contrast"] = a.dB_mean - a.dBenv
    obj = pd.concat([obj.reset_index(drop=True), a.reset_index(drop=True)], axis=1)
    obj["extent"] = obj["size"] / (obj.bbox_h * obj.bbox_w)
    for name in CAT:
        s = pd.Series(px[name]).groupby([oi, px[name]]).size()
        obj[name] = s.groupby(level=0).idxmax().map(lambda t: t[1]).reindex(range(len(uniq))).to_numpy()
    E = np.zeros((len(idx), 64), "float32")
    valid = np.zeros(len(idx), "float32")
    with rasterio.open(f"{INTERIM}/emb2018.tif") as r:
        for r0 in range(0, H, 256):
            sel = np.flatnonzero((rows >= r0) & (rows < r0 + 256))
            if len(sel) == 0:
                continue
            v = r.read(window=Window(0, r0, W, min(256, H - r0)))[:, rows[sel] - r0, cols[sel]].astype("float32")
            ok = ~(v == 0).all(axis=0)  # all-zero = no embedding
            E[sel[ok]] = ((v - 128) / 127)[:, ok].T
            valid[sel[ok]] = 1
    n_ok = np.add.reduceat(valid, start)
    with np.errstate(invalid="ignore", divide="ignore"):
        emb = np.add.reduceat(E, start, axis=0) / n_ok[:, None]  # NaN where an object has no embedded pixel
    obj = pd.concat([obj, pd.DataFrame(emb, columns=EMB)], axis=1)
    print(f"object features built ({time.time() - t0:.0f}s)", flush=True)
    return obj, pairs


def counts(score, obj, pairs, thr, gates=GATES):
    """Same layout as evaluate_oof.block_counts: [threshold, gate, block, k]; k = big polygons detected, objects, true objects, 3 size classes."""
    pt, big, cls_id, NB = ev.pt, ev.big, ev.cls_id, ev.NB
    out = np.zeros((len(thr), len(gates), NB, 6), dtype=np.int32)
    size, tp, bi = obj["size"].to_numpy(), obj.tp.to_numpy(), obj.bi.to_numpy()
    for ti, t in enumerate(thr):
        for ai, gate in enumerate(gates):
            k = (score >= t) & (size >= gate)
            out[ti, ai, :, 1] = np.bincount(bi[k], minlength=NB)
            out[ti, ai, :, 2] = np.bincount(bi[k & tp], minlength=NB)
            det = np.zeros(len(pt), bool)
            det[np.unique(pairs[k[pairs[:, 0]], 1])] = True
            out[ti, ai, :, 0] = np.bincount(pt.bi[det & big], minlength=NB)
            for ci in range(3):
                out[ti, ai, :, 3 + ci] = np.bincount(pt.bi[det & (cls_id == ci)], minlength=NB)
    return out


def cv_scores(obj, cols, folds_of):
    X = obj[cols].to_numpy("float32")
    y = obj.tp.to_numpy()
    cat = [c in CAT for c in cols]
    score = np.full(len(obj), np.nan)
    for k in range(5):
        tr = folds_of != k
        m = HistGradientBoostingClassifier(categorical_features=cat if any(cat) else None, **PARAMS).fit(X[tr], y[tr])
        score[~tr] = m.predict_proba(X[~tr])[:, 1]
    return score


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    obj, pairs = build_objects(seed)
    obj.to_parquet(f"{INTERIM}/stage2_objects_r{seed}.parquet", index=False)
    folds = block_folds(seed)
    lut = np.full(14000, -1)
    lut[folds.group.to_numpy()] = folds.fold.to_numpy()
    fold = lut[obj.block.to_numpy()]
    assert (fold >= 0).all()

    # check: keeping every candidate reproduces the stage-1 counts at this threshold (objects, true objects, detections)
    ref = np.load(OOF / f"counts_c_hgb_r{seed}.npz")["cnt"][ev.SHARES.index(SHARE1)]
    mine = counts(np.zeros(len(obj)), obj, pairs, [0.0])[0]
    for ai, gate in enumerate(GATES):
        assert np.array_equal(mine[ai], ref[ev.A_MINS.index(gate)]), f"count mismatch at gate {gate}"
    print(f"check passed: all candidates reproduce the stage-1 counts at share {SHARE1}, gates {GATES}", flush=True)

    rng = np.random.default_rng(ev.SEED)
    Wb = rng.multinomial(ev.NB, np.full(ev.NB, 1 / ev.NB), size=ev.N_BOOT).astype("float64")  # the resamples evaluate_oof used
    p_s1 = float(ev.op_point(ev.with_totals(np.load(OOF / f"counts_c_hgb_r{seed}.npz")["cnt"]).sum(2), 0.8)[0])
    boots = {"stage1": np.load(OOF / f"boot_c_hgb_r{seed}_0.8.npy")}
    p_ref = {"stage1": p_s1}
    ncls = [max(int((ev.cls_id == i).sum()), 1) for i in range(3)]
    # obj_pmax: the same candidates ranked by stage-1 max probability on the same fine threshold grid, the like-for-like reference
    cols = {"obj_pmax": None, "s2_probstats": PROBSTATS, "s2_base": BASE, "s2_emb": BASE + EMB}
    size = obj["size"].to_numpy()
    rows = []
    for name, c in cols.items():
        score = obj.pmax.to_numpy() if c is None else cv_scores(obj, c, fold)
        thr = np.unique(np.quantile(score, np.linspace(0, 0.999, 150)))
        cnt = ev.with_totals(counts(score, obj, pairs, thr))
        full = cnt.sum(2)
        for tg in ev.TARGETS:
            p, reached, ti, ai = ev.op_point(full, tg)
            row = {"variant": name, "repeat": seed, "recall_target": tg, "precision": float(p) if reached else np.nan, "reached": bool(reached)}
            if reached:
                ti, ai = int(ti), int(ai)
                kept = (score >= thr[ti]) & (size >= GATES[ai])
                row.update(gate=GATES[ai], objects=int(full[ti, ai, 1]), true_objects=int(full[ti, ai, 2]),
                           flagged_share=float(size[kept].sum() / ev.train.sum()), recall_ge5=float(full[ti, ai, 0] / full[ti, ai, 6]),
                           **{f"recall_{c}": float(full[ti, ai, 3 + i] / ncls[i]) for i, c in enumerate(ev.CLS)})
            if tg == 0.8:
                boots[name] = ev.op_point(np.einsum("bn,tank->btak", Wb, cnt), tg)[0]
                np.save(OOF / f"boot_s2_{name}_r{seed}_0.8.npy", boots[name])
                if name == "obj_pmax":
                    p_ref["obj_pmax"] = row["precision"]
                row["stage1_precision"] = p_s1
                for ref in ("stage1", "obj_pmax"):
                    d = boots[name] - boots[ref]
                    d = d[~np.isnan(d)]
                    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
                    row.update({f"diff_vs_{ref}": row["precision"] - p_ref[ref], f"ci_low_vs_{ref}": lo, f"ci_high_vs_{ref}": hi})
            rows.append(row)
        r8 = rows[-2]
        print(f"{name:13s} precision at 80% recall {r8['precision']:.3f}  vs stage 1 {r8['diff_vs_stage1']:+.3f} ({r8['ci_low_vs_stage1']:+.3f} to {r8['ci_high_vs_stage1']:+.3f})"
              f"  vs obj_pmax {r8['diff_vs_obj_pmax']:+.3f} ({r8['ci_low_vs_obj_pmax']:+.3f} to {r8['ci_high_vs_obj_pmax']:+.3f})"
              f"  flagged {r8.get('flagged_share', np.nan):.4f}, objects {r8.get('objects', 0):,}", flush=True)
    out = pd.DataFrame(rows)
    path = Path("reports/tables/stage2_results.csv")
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old[old.repeat != seed], out], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"written {path}")
