"""POC question: do Satellite Embedding changes (2017->2018) separate Cempaka landslide points from matched
background at least as well as a Sentinel-2 dNDVI baseline, on the same points?

Pass criteria (fixed before the first run):
  B1  coverage: >= 80% of landslide points have a valid pre and post NDVI in the post window (else the baseline
      is cloud-limited and the comparison is reported as not like-for-like)
  B2  AUC(embedding d1718) > AUC(dNDVI) on the points valid for both, paired bootstrap 95% CI of the difference reported
  B3  complementarity: rank-average of the two scores has AUC >= max(single AUC) + 0.02
  Overall "continue": B1 and (B2 or B3).
Score for dNDVI is -(post - pre): larger = more vegetation loss. Cloud mask: Cloud Score+ cs_cdf >= 0.6.
Usage: python -m src.s2_dndvi_check <matched_points_csv>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from src.embedding_utils import S2_WINDOWS, auc, init_ee, ndvi_composite, sample_image

POST = {"A Dec17-Mar18": S2_WINDOWS["postA"], "B Apr-May18": S2_WINDOWS["postB"]}

meta = pd.read_csv(sys.argv[1], index_col="pid")
pts = gpd.GeoDataFrame(meta.reset_index()[["pid"]], geometry=gpd.points_from_xy(meta.lon, meta.lat), crs=4326)

init_ee()
pre = sample_image(ndvi_composite(*S2_WINDOWS["pre"]), pts, 10, ["ndvi"]).ndvi.rename("pre")
rng = np.random.default_rng(42)


def boot_diff(y, a, b, n=1000):
    """Paired bootstrap of AUC(a) - AUC(b); y is a boolean array of landslide labels."""
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        out.append(auc(a[idx][yy], a[idx][~yy]) - auc(b[idx][yy], b[idx][~yy]))
    return np.percentile(out, [2.5, 97.5])


L = meta.grp == "landslide"
for name, (s, e) in POST.items():
    post = sample_image(ndvi_composite(s, e), pts, 10, ["ndvi"]).ndvi.rename("post")
    d = meta[["grp", "area", "d1718"]].join(pre).join(post)
    d["score_ndvi"] = -(d.post - d.pre)
    cov = d[L.reindex(d.index)].dropna(subset=["score_ndvi"]).shape[0] / L.sum()
    v = d.dropna(subset=["score_ndvi", "d1718"])
    y = (v.grp == "landslide").to_numpy()
    emb, nd = v.d1718.to_numpy(), v.score_ndvi.to_numpy()
    comb = (v.d1718.rank() + v.score_ndvi.rank()).to_numpy()
    a_emb, a_nd, a_comb = auc(emb[y], emb[~y]), auc(nd[y], nd[~y]), auc(comb[y], comb[~y])
    lo, hi = boot_diff(y, emb, nd)
    print(f"\n=== post window {name} ===")
    print(f"landslide points with valid pre+post NDVI: {cov:.1%}; points used (both scores): {len(v)} (landslide {y.sum()})")
    print(f"AUC embedding d1718 {a_emb:.3f} | dNDVI {a_nd:.3f} | rank-average {a_comb:.3f}")
    print(f"AUC difference (emb - dNDVI) {a_emb - a_nd:+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("AUC by polygon size class      embedding   dNDVI")
    for lab, lo_a, hi_a in [("<5 px (<500 m2)", 0, 500), ("5-20 px", 500, 2000), (">20 px (>2000 m2)", 2000, 1e12)]:
        m = ~y | ((v.area >= lo_a) & (v.area < hi_a)).to_numpy()
        yy = y[m]
        print(f"  {lab:<22} n={int((m & y).sum()):<4} {auc(emb[m][yy], emb[m][~yy]):.3f}       {auc(nd[m][yy], nd[m][~yy]):.3f}")
    b1, b2, b3 = cov >= 0.8, a_emb > a_nd, a_comb >= max(a_emb, a_nd) + 0.02
    print(f"B1 coverage >= 80%: {'PASS' if b1 else 'FAIL'} | B2 embedding > dNDVI: {'PASS' if b2 else 'FAIL'} | "
          f"B3 combined >= max + 0.02: {'PASS' if b3 else 'FAIL'} | continue: {'YES' if b1 and (b2 or b3) else 'NO'}")
