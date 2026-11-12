"""Phase 5: error analysis of the chosen model's out-of-fold predictions (training blocks) at its 80%-recall operating point.

Takes c_hgb repeat 0 and the threshold / minimum object size that evaluate_oof chose (reports/tables/modeling_results_raw.csv).
Predicted objects are split into true objects (overlap a polygon or its 20 m ring) and false alarms, polygons >= 5 px into detected and
missed. Both are broken down by size, slope, land use and distance to river; false alarms are also placed by distance to the nearest
mapped polygon (a cluster of false alarms around mapped slides hints at unmapped neighbours: a hint, not proof).
Writes reports/tables/error_analysis_*.csv. Usage: python -m src.modeling.error_analysis [config_r<seed>]
"""
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

from src.data.load import block_mask
from src.features.build import FDIR, INTERIM
from src.modeling.metrics import EIGHT

NAME = sys.argv[1] if len(sys.argv) > 1 else "c_hgb_r0"
model, rep = NAME.rsplit("_r", 1)
raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
op = raw[(raw.config == model) & (raw.repeat == int(rep)) & (raw.recall_target == 0.8)].iloc[0]
THR, AMIN = np.float64(op.thr), int(op.a_min)  # np.float64, not float: see the NumPy 2 note in stage2.py
print(f"{NAME}: threshold {THR:.3f}, a_min {AMIN}, precision {op.precision_at_target:.3f}, recall >=5 px {op.recall_ge5:.3f}")


def read(path, mask_nodata=True):
    with rasterio.open(path) as r:
        a = r.read(1).astype("float32")
        nd = r.nodata
    if mask_nodata and nd is not None:
        a[a == nd] = np.nan
    return a


prob = np.load(f"{INTERIM}/oof/{NAME}.npy", mmap_mode="r")
prob = np.asarray(prob)
# label.tif uses 255 both as nodata and as the exclusion-ring code, so the label, polygon and block rasters are read as stored
poly, block, label = (read(f"{INTERIM}/{n}.tif", mask_nodata=False) for n in ("poly_id", "block", "label"))
poly = poly.astype(int)
ring = (label == 255) & (block >= 0)
train = block_mask("train")
slope, landuse, distriv = (read(f"{FDIR}/map_{n}.tif") for n in ("slope", "landuse", "distriv"))
pt = pd.read_parquet(f"{INTERIM}/polygons.parquet")
pt = pt[pt.in_train].set_index("id")

lab, n = ndi.label((prob >= THR) & train, structure=EIGHT)
sizes = np.bincount(lab.ravel(), minlength=n + 1)
keep = sizes >= AMIN
keep[0] = False
objs = np.flatnonzero(keep)
is_true = np.zeros(n + 1, bool)
is_true[np.unique(lab[((poly > 0) | ring) & (lab > 0)])] = True
dist_poly = ndi.distance_transform_edt(poly == 0) * 10  # metres to the nearest mapped polygon pixel
idx = ndi.find_objects(lab)


def stat(a, o, fn):
    sl = idx[o - 1]
    v = a[sl][lab[sl] == o]
    v = v[~np.isnan(v)]
    return fn(v) if len(v) else np.nan


rows = []
for o in objs:
    sl = idx[o - 1]
    m = lab[sl] == o
    lu = landuse[sl][m]
    lu = lu[~np.isnan(lu)]
    rows.append({"obj": o, "size_px": int(sizes[o]), "true": bool(is_true[o]), "slope": np.nanmean(slope[sl][m]),
                 "landuse": int(np.bincount(lu.astype(int)).argmax()) if len(lu) else -1, "distriv_m": np.nanmin(distriv[sl][m]),
                 "dist_poly_m": float(dist_poly[sl][m].min()), "pmax": float(prob[sl][m].max())})
ob = pd.DataFrame(rows)
fa = ob[~ob.true]
print(f"objects {len(ob):,}: true {int(ob.true.sum()):,}, false alarms {len(fa):,} (precision {ob.true.mean():.3f})")

SL = [0, 8, 15, 25, 40, 90]
SZ = [0, 2, 3, 5, 10, 20, 10_000]


def by(df, col, bins=None, labels=None):
    k = pd.cut(df[col], bins, right=False) if bins else df[col]
    g = df.groupby(k, observed=True).agg(objects=("true", "size"), true_objects=("true", "sum"))
    g["false_alarms"] = g.objects - g.true_objects
    g["precision"] = g.true_objects / g.objects
    g["share_of_false_alarms"] = g.false_alarms / g.false_alarms.sum()
    return g.round(3)


out = {"size_px": by(ob, "size_px", SZ), "slope_deg": by(ob, "slope", SL), "landuse": by(ob, "landuse")}
dom = pd.Series(landuse[train & ~np.isnan(landuse)].astype(int)).value_counts(normalize=True).sort_index().round(4)
out["landuse"]["share_of_domain_area"] = dom.reindex(out["landuse"].index).to_numpy()
out["distriv_m"] = by(ob, "distriv_m", [0, 25, 50, 100, 250, 500, 100_000])
near = pd.cut(fa.dist_poly_m, [0, 1, 50, 100, 300, 1000, 100_000], right=False)
dpn = pd.Series(dist_poly[train].ravel())
base = pd.cut(dpn.sample(2_000_000, random_state=0), [0, 1, 50, 100, 300, 1000, 100_000], right=False).value_counts(normalize=True).sort_index()
out["false_alarm_distance_to_polygon"] = pd.DataFrame({"false_alarm_share": near.value_counts(normalize=True).sort_index().round(3),
                                                       "domain_area_share": base.round(3)})

# misses: polygons >= 5 px of the training blocks, detected when >= 1 predicted pixel (kept object) touches them
pred = lab > 0
pred &= keep[lab]
hit = np.bincount(poly[pred], minlength=poly.max() + 1)
big = pt[pt.size_class != "<5 px"].copy()
big["detected"] = hit[big.index.to_numpy()] > 0
boxes = ndi.find_objects(poly, max_label=int(poly.max()))
feat = []
for pid in big.index:
    sl = boxes[pid - 1]
    m = poly[sl] == pid
    lu = landuse[sl][m]
    lu = lu[~np.isnan(lu)]
    feat.append((np.nanmean(slope[sl][m]), int(np.bincount(lu.astype(int)).argmax()) if len(lu) else -1, np.nanmin(distriv[sl][m]), float(np.nanmax(prob[sl][m]))))
big[["slope", "landuse", "distriv_m", "pmax"]] = feat


def det_by(col, bins=None):
    k = pd.cut(big[col], bins, right=False) if bins else big[col]
    g = big.groupby(k, observed=True).agg(polygons=("detected", "size"), detected=("detected", "sum"))
    g["recall"] = (g.detected / g.polygons).round(3)
    g["missed"] = g.polygons - g.detected
    return g


out["miss_by_area_m2"] = det_by("area_m2", [0, 500, 1000, 2000, 5000, 1e9])
out["miss_by_slope"] = det_by("slope", SL)
out["miss_by_landuse"] = det_by("landuse")
out["miss_by_distriv_m"] = det_by("distriv_m", [0, 25, 50, 100, 250, 500, 100_000])
print(f"polygons >= 5 px: {len(big)}, detected {int(big.detected.sum())} (recall {big.detected.mean():.3f})")
for k, df in out.items():
    df.to_csv(f"reports/tables/error_analysis_{k}.csv")
    print(f"\n== {k}\n{df.to_string()}")
