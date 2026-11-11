"""Phase 1: estimate the registration offset between the inventory polygons and the Sentinel-2 / embedding grid.

For every integer pixel offset (di, dj) in [-4, 4], the positive label pixels are moved by (di, dj) and the AUC of the
Sentinel-2 vegetation loss (-dNDVI, Apr-May 2018 minus pre-event) is computed against a fixed random sample of background
pixels. The sub-pixel peak comes from a 2D quadratic fit around the maximum. Reported for all polygons and for four
halves of the regency (N, S, E, W) to check that a single rigid shift is justified. Usage: python -m src.data.estimate_shift
"""
import numpy as np
import rasterio

from src.embedding_utils import auc

R = 4
with rasterio.open("data/interim/label.tif") as r:
    label = r.read(1)
with rasterio.open("data/interim/ndvi.tif") as r:
    nd = r.read().astype("float32")
nd[nd == -9999] = np.nan
loss = -(nd[2] - nd[0])  # larger = more vegetation loss
H, W = label.shape
pi, pj = np.nonzero(label == 1)
rng = np.random.default_rng(42)
ni, nj = np.nonzero(label == 0)
k = rng.choice(len(ni), 400_000, replace=False)
neg = loss[ni[k], nj[k]]
neg = neg[~np.isnan(neg)]


def surface(sel):
    out = np.full((2 * R + 1, 2 * R + 1), np.nan)
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            v = loss[np.clip(pi[sel] + di, 0, H - 1), np.clip(pj[sel] + dj, 0, W - 1)]
            v = v[~np.isnan(v)]
            out[di + R, dj + R] = auc(v, neg)
    return out


def peak(s):
    """Sub-pixel (row, col) shift in pixels of the best sampling offset; negative row = north."""
    i, j = np.unravel_index(np.nanargmax(s), s.shape)
    i, j = min(max(i, 1), 2 * R - 1), min(max(j, 1), 2 * R - 1)
    sub = s[i - 1:i + 2, j - 1:j + 2]
    y, x = np.mgrid[-1:2, -1:2]
    A = np.c_[np.ones(9), x.ravel(), y.ravel(), x.ravel() ** 2, y.ravel() ** 2, (x * y).ravel()]
    c = np.linalg.lstsq(A, sub.ravel(), rcond=None)[0]
    M = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    dx, dy = np.linalg.solve(M, -np.array([c[1], c[2]]))
    return (i - R + dy), (j - R + dx)


s_all = surface(np.ones(len(pi), bool))
print("AUC at offset (0,0):", round(s_all[R, R], 3), "| best integer offset:",
      tuple(int(v) - R for v in np.unravel_index(np.nanargmax(s_all), s_all.shape)), "AUC", round(np.nanmax(s_all), 3))
dr, dc = peak(s_all)
print(f"sub-pixel peak: {dr:+.2f} rows, {dc:+.2f} cols  ->  true scar is {-dr * 10:+.1f} m north and {dc * 10:+.1f} m east of the polygon")
print("\nstability across the regency (positive pixels split at the median row / col):")
for name, sel in {"north half": pi < np.median(pi), "south half": pi >= np.median(pi),
                  "west half": pj < np.median(pj), "east half": pj >= np.median(pj)}.items():
    s = surface(sel)
    r_, c_ = peak(s)
    print(f"  {name:<11} AUC(0,0) {s[R, R]:.3f} -> peak {np.nanmax(s):.3f}; shift {-r_ * 10:+.1f} m N, {c_ * 10:+.1f} m E")
