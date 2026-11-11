"""POC, supervised step: do Satellite Embedding features classify landslide vs matched background points as
well as Sentinel-2 dNDVI features, under spatial block cross-validation?

Feature sets
  A  2018 embedding (64)             A2  2017 + 2018 embeddings (128, extra, not in the criteria)
  B  NDVI pre, postA, postB, dA, dB (5)
  C  A + B
Models (no tuning): standardized logistic regression (C=1), random forest (300 trees, min_samples_leaf=3).
CV: 5-fold GroupKFold over 3 km spatial blocks (UTM 49S); 1 km and 5 km reported as sensitivity.
Metric: pooled out-of-fold AUC; 95% CI of AUC differences from a bootstrap that resamples blocks.

Criteria (fixed before the first run), each required for BOTH models:
  S0  sanity: label-permutation AUC of A (logistic) within 0.5 +/- 0.05
  S1  AUC(A) - AUC(B): CI lower bound > 0           (embeddings alone beat dNDVI)
  S2  AUC(C) - max(AUC(A), AUC(B)): CI lower bound > 0   (embeddings add to dNDVI)
  Continue with embeddings: S0 and (S1 or S2). Otherwise drop them for this task.
Usage: python -m src.supervised_check <features_csv>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.embedding_utils import BANDS, auc

import geopandas as gpd

SEED = 42
d = pd.read_csv(sys.argv[1], index_col="pid")
d = d.assign(dA=d.postA - d.pre, dB=d.postB - d.pre,
             **{f"x_{b}": d[f"e18_{b}"] - d[f"e17_{b}"] for b in BANDS})
E17 = [c for c in d.columns if c.startswith("e17_")]
E18 = [c for c in d.columns if c.startswith("e18_")]
NDVI = ["pre", "postA", "postB", "dA", "dB"]
DIFF = [f"x_{b}" for b in BANDS]
SETS = {"A": E18, "B": NDVI, "C": E18 + NDVI, "A2": E17 + E18}
# post-hoc controls (added after the first run, not part of the criteria): is it the scar or the site?
CONTROLS = {"2017 embedding only": E17, "pre-event NDVI only": ["pre"], "embedding change 2018-2017": DIFF}
y = (d.grp == "landslide").to_numpy()
area = d.area.to_numpy()

xy = gpd.GeoSeries(gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(32749)
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
}


def blocks(size_m):
    return (np.floor(xy.x / size_m).astype(int).astype(str) + "_" + np.floor(xy.y / size_m).astype(int).astype(str)).to_numpy()


def oof(cols, model, groups, labels):
    pred = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(d, labels, groups):
        m = MODELS[model]().fit(d.iloc[tr][cols], labels[tr])
        pred[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    return pred


def pooled_auc(p, labels=y):
    return auc(p[labels], p[~labels])


def block_boot(preds, groups, fn, n=1000):
    """95% CI of fn(preds) over bootstrap resamples of whole blocks."""
    rng = np.random.default_rng(SEED)
    ub = pd.unique(groups)
    idx_by = {b: np.flatnonzero(groups == b) for b in ub}
    out = []
    for _ in range(n):
        idx = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub))])
        if y[idx].all() or not y[idx].any():
            continue
        out.append(fn({k: v[idx] for k, v in preds.items()}, y[idx]))
    return np.percentile(out, [2.5, 97.5])


g3 = blocks(3000)
print(f"points {len(d)} (landslide {y.sum()}), 3 km blocks: {len(set(g3))}")
res, verdict = {}, {}
for model in MODELS:
    P = {k: oof(c, model, g3, y) for k, c in SETS.items()}
    res[model] = P
    print(f"\n=== {model}: out-of-fold AUC, 3 km blocks ===")
    print("  " + " | ".join(f"{k} {pooled_auc(v):.3f}" for k, v in P.items()))
    print("  by polygon size class        " + "   ".join(f"{k:>5}" for k in ("A", "B", "C")))
    for lab, lo, hi in [("<5 px (<500 m2)", 0, 500), ("5-20 px", 500, 2000), (">20 px (>2000 m2)", 2000, 1e12)]:
        m = ~y | ((area >= lo) & (area < hi))
        print(f"    {lab:<22} n={int((m & y).sum()):<4} " + "   ".join(f"{pooled_auc(P[k][m], y[m]):.3f}" for k in ("A", "B", "C")))
    ci1 = block_boot(P, g3, lambda p, yy: auc(p["A"][yy], p["A"][~yy]) - auc(p["B"][yy], p["B"][~yy]))
    ci2 = block_boot(P, g3, lambda p, yy: auc(p["C"][yy], p["C"][~yy]) - max(auc(p["A"][yy], p["A"][~yy]), auc(p["B"][yy], p["B"][~yy])))
    d1 = pooled_auc(P["A"]) - pooled_auc(P["B"])
    d2 = pooled_auc(P["C"]) - max(pooled_auc(P["A"]), pooled_auc(P["B"]))
    print(f"  S1 A - B: {d1:+.3f}, CI [{ci1[0]:+.3f}, {ci1[1]:+.3f}]")
    print(f"  S2 C - max(A, B): {d2:+.3f}, CI [{ci2[0]:+.3f}, {ci2[1]:+.3f}]")
    perm = oof(SETS["A"], model, g3, np.random.default_rng(SEED).permutation(y))
    perm_auc = pooled_auc(perm, np.random.default_rng(SEED).permutation(y))
    print(f"  S0 label-permutation AUC of A: {perm_auc:.3f}")
    verdict[model] = (abs(perm_auc - 0.5) <= 0.05, ci1[0] > 0, ci2[0] > 0)

print("\nsensitivity to block size, logreg / rf pooled AUC (A | B | C)")
for km in (1, 5):
    g = blocks(km * 1000)
    for model in MODELS:
        P = {k: oof(SETS[k], model, g, y) for k in ("A", "B", "C")}
        print(f"  {km} km {model:<6}: " + " | ".join(f"{k} {pooled_auc(v):.3f}" for k, v in P.items()))

print("\npost-hoc controls, 3 km blocks, pooled AUC (logreg | rf)")
for name, cols in CONTROLS.items():
    r = [pooled_auc(oof(cols, m, g3, y)) for m in MODELS]
    print(f"  {name:<28} {r[0]:.3f} | {r[1]:.3f}")

s0 = all(v[0] for v in verdict.values())
s1 = all(v[1] for v in verdict.values())
s2 = all(v[2] for v in verdict.values())
print(f"\nS0 {'PASS' if s0 else 'FAIL'} | S1 {'PASS' if s1 else 'FAIL'} | S2 {'PASS' if s2 else 'FAIL'} | "
      f"continue with embeddings: {'YES' if s0 and (s1 or s2) else 'NO'}")
