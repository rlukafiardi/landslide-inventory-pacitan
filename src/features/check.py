"""Phase 3 leakage and correctness review of the feature table. Usage: python -m src.features.check

  C1 column audit    every feature: non-missing share, zero share, distinct values, AUC; FAIL if constant, all-zero or AUC outside 0.05-0.95
  C2 no identifiers  no coordinate, row/col or block column among the features
  C3 tile invariance the same pixel gets the same features from two different windows (halo and NaN-fill logic)
  C4 brute force     neighbourhood features recomputed with plain loops on 300 sampled pixels
  C5 sign check      observed AUC direction vs the expected sign in the registry (reported, not a failure: distroad is a known surprise)
  C6 permutation     logistic regression on permuted labels under block CV stays near 0.5 (pipeline smoke test, not a result)
"""
import json

import numpy as np
import pandas as pd
import rasterio

from src.cv_utils import SEED, oof, pooled_auc
from src.embedding_utils import auc
from src.features.build import INTERIM, NBR_FILL, apply_fills, feature_window
from src.features.registry import ANOM, FEATURES, NAMES, WIN

t = pd.read_parquet(f"{INTERIM}/features_train.parquet")
y = t.y.to_numpy() == 1
ok = True


def report(tag, passed, msg):
    global ok
    ok &= passed
    print(f"{tag} {'PASS' if passed else 'FAIL'}: {msg}")


# C1
rows = []
for f in FEATURES:
    v = t[f.name]
    a = auc(t.loc[y, f.name].dropna(), t.loc[~y, f.name].dropna())
    rows.append((f.name, f.group, f.sign, 1 - v.isna().mean(), (v == 0).mean(), v.nunique(), a))
audit = pd.DataFrame(rows, columns=["feature", "group", "expected_sign", "non_missing", "zero_share", "distinct", "auc"])
bad = audit[(audit.distinct <= 1) | (audit.zero_share == 1) | (audit.auc > 0.95) | (audit.auc < 0.05)]
audit["observed_sign"] = np.where(audit.auc > 0.5, "+", "-")
audit["sign_ok"] = (audit.expected_sign == "?") | (audit.expected_sign == audit.observed_sign)
audit.to_csv("reports/tables/feature_checks.csv", index=False)
report("C1", bad.empty, f"{len(audit)} features audited; flagged: {bad.feature.tolist() or 'none'}; strongest |AUC-0.5|: "
       + ", ".join(f"{r.feature} {r.auc:.2f}" for r in audit.assign(d=(audit.auc - 0.5).abs()).nlargest(4, "d").itertuples()))
# C2
forbidden = [n for n in NAMES if n in {"row", "col", "block", "lon", "lat", "x", "y", "w"} or n.startswith("coord")]
report("C2", not forbidden and list(t[NAMES].columns) == NAMES, f"features are exactly the registry ({len(NAMES)}), no identifier columns")
# C3
a = feature_window(1200, 2400, 300, 300)
b = feature_window(1290, 2500, 200, 200)
worst = 0.0
for n in NAMES:
    x, z = a[n][90:290, 100:300], b[n]
    same_nan = np.array_equal(np.isnan(x), np.isnan(z))
    worst = max(worst, float(np.nanmax(np.abs(x - z))) if same_nan else np.inf)
report("C3", worst < 1e-5, f"max abs difference between two overlapping windows: {worst:.2g}")
# C4
with rasterio.open(f"{INTERIM}/ndvi.tif") as r:
    nd = r.read().astype("float32")
nd[nd == -9999.0] = np.nan
dB, postB = nd[2] - nd[0], nd[2]
with rasterio.open(f"{INTERIM}/ndvi_prev.tif") as r:
    pv = r.read().astype("float32")
pv[pv == -9999.0] = np.nan
dprev = pv[1] - pv[0]
dd = dB - dprev
rng = np.random.default_rng(SEED)
sel = t.sample(300, random_state=SEED)
H, W = dB.shape
worst = 0.0
for _, p in sel.iterrows():
    r0, c0 = int(p.row), int(p.col)
    for base, arr in (("dB", dB), ("postB", postB)):
        pad = np.pad(arr, ANOM, mode="edge")
        for win in WIN + (ANOM,):
            h = win // 2
            patch = pad[r0 + ANOM - h: r0 + ANOM + h + 1, c0 + ANOM - h: c0 + ANOM + h + 1]
            patch = np.where(np.isnan(patch), NBR_FILL[base], patch)
            if win == ANOM:
                got, want = p[f"{base}_anom{ANOM}"], patch[h, h] - patch.mean()
            else:
                for stat, fn in (("mean", np.mean), ("min", np.min), ("std", np.std)):
                    worst = max(worst, abs(p[f"{base}_{stat}{win}"] - fn(patch)))
                continue
            worst = max(worst, abs(got - want))
    for base, arr, wins in (("dd", dd, WIN), ("dprev", dprev, (5,))):
        pad = np.pad(arr, ANOM, mode="edge")
        for win in wins:
            h = win // 2
            patch = pad[r0 + ANOM - h: r0 + ANOM + h + 1, c0 + ANOM - h: c0 + ANOM + h + 1]
            patch = np.where(np.isnan(patch), NBR_FILL[base], patch)
            worst = max(worst, abs(p[f"{base}_mean{win}"] - patch.mean()))
            if base == "dd":
                worst = max(worst, abs(p[f"dd_min{win}"] - patch.min()))
report("C4", worst < 1e-4, f"max abs difference to a plain-loop recomputation on 300 pixels: {worst:.2g}")
# C5
off = audit[~audit.sign_ok]
print(f"C5 INFO: {len(off)} features with an observed sign opposite to the registry: "
      + (", ".join(f"{r.feature} (expected {r.expected_sign}, AUC {r.auc:.2f})" for r in off.itertuples()) or "none"))
# C6
fills = json.load(open(f"{INTERIM}/fill_values.json"))
num = [n for n in NAMES if n not in ("geologi", "landuse")]
sub = pd.concat([t[y], t[~y].sample(100_000, random_state=SEED)])
Xs = apply_fills(sub, fills)
ys = (sub.y == 1).to_numpy()
g = sub.block.to_numpy()
perm = np.random.default_rng(SEED).permutation(ys)
a_perm = pooled_auc(oof(Xs, num, "logreg", g, perm), perm)
a_real = pooled_auc(oof(Xs, num, "logreg", g, ys), ys)
report("C6", abs(a_perm - 0.5) < 0.03, f"permuted-label block-CV AUC {a_perm:.3f} (smoke test: real labels {a_real:.3f}, not a result)")
print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
