"""Phase 2 EDA report on the TRAINING blocks only -> reports/eda.html, reports/figures/eda_*.png, reports/tables/eda_*.csv.

Needs data/interim/eda_sample.csv.gz and eda_emb.npy (python -m src.analysis.eda_sample). Usage: python -m src.analysis.eda
The test blocks are only drawn as a grey "held out" area on the map; no statistic uses them.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
from plotly.subplots import make_subplots

from src.analysis.glossary import DATA, EDA, METHOD
from src.analysis.html_report import (
    BLUE,
    BLUES,
    GREY,
    INK,
    RED,
    page,
    plots_html,
    save_pngs,
    section,
    styled,
    table_html,
    tile,
)
from src.data.constants import SHIFT_E_M, SHIFT_N_M
from src.data.load import block_mask
from src.embedding_utils import auc

FDIR = "data/spatial/mendeley_vrtx3w6mjd"
FIG, TAB = Path("reports/figures"), Path("reports/tables")
TAB.mkdir(parents=True, exist_ok=True)
SEED = 42

# ---------------------------------------------------------------------------------------------------------------------
# HYPOTHESES, written before the first plot of this report. Expected direction: "+" = higher in landslide pixels.
# Size classes by |AUC - 0.5| of landslide vs background pixels: strong >= 0.20, moderate 0.10-0.20, weak 0.03-0.10, none < 0.03.
# Retrofit: slope, pre-event NDVI and the geology densities were already seen in the POC and the coverage check, so those
# rows test the domain story, not a blind prediction.
# ---------------------------------------------------------------------------------------------------------------------
HYPOTHESES = [
    (
        "slope",
        "+",
        "strong",
        "steeper ground fails more (retrofit: POC slope-only AUC about 0.65 against a random pool)",
    ),
    ("dB", "-", "strong", "NDVI Apr-May 2018 minus pre-event drops where a scar removed vegetation"),
    ("dA", "-", "moderate", "same signal in Dec-Mar, but cloud gaps and noise should make it weaker than dB"),
    ("postB", "-", "moderate", "scars are bare in Apr-May, so post-event NDVI is lower"),
    ("pre", "-", "weak", "pre-event vegetation barely separates sites (retrofit: POC pre-NDVI-only AUC about 0.6)"),
    ("distroad", "-", "weak", "slides nearer roads: cut slopes, and reporting bias towards people"),
    ("distriv", "-", "weak", "river banks undercut and flood: nearer rivers means more"),
    ("elev", "-", "weak", "no clear direction; mid-slope belts may dominate"),
    ("twi", "+", "weak", "wetter convergent sites fail more, but at 10 m the index is noisy"),
    ("spi", "+", "weak", "stream power marks channels; noisy at 10 m"),
    ("planc", "-", "none", "plan curvature at 10 m is noise-dominated"),
    ("profc", "-", "none", "profile curvature at 10 m is noise-dominated"),
    ("asp_sin", "+", "none", "no aspect preference expected (a monsoon-facing effect would show here or in asp_cos)"),
    ("asp_cos", "+", "none", "see asp_sin"),
]
OTHER_HYPOTHESES = [
    "Geology: classes 7 and 32 carry the most landslides per area and class 8 the fewest (retrofit: seen in the coverage check).",
    "Land use: tree-crop and forest classes above settlement and paddy; cannot be checked because the legend is unknown.",
    "Space: landslide share is strongly clustered; Moran's I of block shares positive and falling as blocks grow.",
    "Naive detector (-dNDVI threshold): pixel precision below 10% at 50% recall, because prevalence is 0.16% and the labels are a few pixels wide.",
    "False alarms of that detector pile up near rivers (flood and bank erosion) and on bare or cultivated land.",
]
SIZE_CUTS = [(0.20, "strong"), (0.10, "moderate"), (0.03, "weak"), (0.0, "none")]
UNITS = {
    "slope": "degrees",
    "elev": "m a.s.l.",
    "distriv": "m",
    "distroad": "m",
    "twi": "index",
    "spi": "index",
    "planc": "1/m",
    "profc": "1/m",
    "pre": "NDVI",
    "postA": "NDVI",
    "postB": "NDVI",
    "dA": "NDVI diff",
    "dB": "NDVI diff",
    "aspect": "degrees",
    "asp_sin": "sin(aspect)",
    "asp_cos": "cos(aspect)",
}
NUM = [
    "pre",
    "postA",
    "postB",
    "dA",
    "dB",
    "slope",
    "elev",
    "distriv",
    "distroad",
    "twi",
    "spi",
    "planc",
    "profc",
    "asp_sin",
    "asp_cos",
]
LOGGED = {"distriv", "distroad"}  # spi is already on a log scale in the source, so it is plotted as stored


def size_class(a):
    return next(n for c, n in SIZE_CUTS if abs(a - 0.5) >= c)


def wquantile(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return np.interp(qs, cw, x[o])


# ------------------------------------------------------------------ data (training blocks only)
s = pd.read_csv("data/interim/eda_sample.csv.gz")
emb = np.load("data/interim/eda_emb.npy").astype("float32")
asp = s.aspect.where(s.aspect >= 0)
s["asp_sin"], s["asp_cos"] = np.sin(np.deg2rad(asp)), np.cos(np.deg2rad(asp))
y = s.y.to_numpy() == 1
pos, neg = s[y], s[~y]
W_NEG = float(neg.w.iloc[0])
N_POS, N_NEG_POP = int(y.sum()), len(neg) * W_NEG
train = block_mask("train")
with rasterio.open("data/interim/label.tif") as r:
    label = r.read(1)
with rasterio.open("data/interim/block.tif") as r:
    block = r.read(1)
H, Wd = label.shape
n_blocks = len(np.unique(block[train]))

ls = gpd.read_file(f"{FDIR}/landslide_pacitan.gpkg").to_crs(32749)
ls["geometry"] = ls.geometry.translate(SHIFT_E_M, SHIFT_N_M)
with rasterio.open("data/interim/label.tif") as r:
    tr = r.transform
rp = ls.geometry.representative_point()
pr = np.clip(((tr.f - rp.y) / 10).astype(int), 0, H - 1)
pc = np.clip(((rp.x - tr.c) / 10).astype(int), 0, Wd - 1)
ls["in_train"] = train[pr, pc]
ls_tr = ls[ls.in_train].copy()
ls_tr["area_m2"] = ls_tr.area

figs: dict[str, go.Figure] = {}
tables: dict[str, pd.DataFrame] = {}
sections: list[str] = []

# ------------------------------------------------------------------ 1. dictionary
dictionary = pd.DataFrame(
    [
        (
            "label",
            "1 = pixel centre inside an inventory polygon, 0 = background",
            "0/1",
            "Samodra et al. 2024 (CC BY 4.0), shifted +8.4 m E, +9.4 m N onto the Sentinel-2 grid",
        ),
        (
            "pre / postA / postB",
            "median NDVI, cloud-masked (Cloud Score+ >= 0.6): Aug-Oct 2017 / Dec 2017-Mar 2018 / Apr-May 2018",
            "NDVI",
            "Sentinel-2 L1C via Earth Engine, 10 m",
        ),
        ("dA / dB", "postA - pre / postB - pre (negative = vegetation loss)", "NDVI diff", "derived"),
        (
            "emb 2018 (64 bands)",
            "Satellite Embedding annual vector, year 2018, unit length",
            "unit",
            "Google Satellite Embedding V1, 10 m",
        ),
        (
            "slope, elev, planc, profc, twi, spi, aspect",
            "terrain from DEMNAS",
            "see profile",
            "Mendeley release (SAGA GIS), 10 m grid",
        ),
        ("distriv, distroad", "distance to river / road line in the 1:25,000 topographic map", "m", "Mendeley release"),
        (
            "geologi",
            "lithology class from the 1:100,000 Pacitan geological map (codes only)",
            "class code",
            "Mendeley release",
        ),
        (
            "landuse",
            "land-use class from the 1:25,000 topographic map (codes 1-13, names unknown)",
            "class code",
            "Mendeley release",
        ),
    ],
    columns=["column", "meaning", "unit", "source"],
)
tables["dictionary"] = dictionary
sections.append(
    section(
        "Terms used in this report",
        "Plain-language definitions; the recall and precision used for the model are defined in the modeling report.",
        table_html(pd.DataFrame([t for t in METHOD if t[0] == "AUC"] + DATA + EDA, columns=["term", "what it means"])),
    )
)
sections.append(
    section(
        "Data dictionary",
        "Every column used; the label is the only thing derived from the inventory.",
        table_html(dictionary),
    )
)


# ------------------------------------------------------------------ 2. profile (population = weighted negatives)
def profile(df, cols):
    rows = []
    for c in cols:
        v = df[c]
        q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            (
                c,
                UNITS.get(c, ""),
                int(v.notna().sum()),
                int(v.isna().sum()),
                f"{v.isna().mean():.3%}",
                int(v.nunique()),
                v.mean(),
                v.std(),
                v.min(),
                q[0.05],
                q[0.25],
                q[0.5],
                q[0.75],
                q[0.95],
                v.max(),
                v.skew(),
                f"{(v == 0).mean():.1%}",
                pos[c].median() if c in pos else np.nan,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "unit",
            "non_null",
            "missing",
            "missing_%",
            "distinct",
            "mean",
            "std",
            "min",
            "p5",
            "p25",
            "median",
            "p75",
            "p95",
            "max",
            "skew",
            "zeros",
            "median_in_landslides",
        ],
    )


prof = profile(neg, NUM)
cat_rows = []
for c in ("geologi", "landuse"):
    vc = neg[c].value_counts(normalize=True)
    cat_rows.append(
        (
            c,
            int(neg[c].nunique()),
            int(vc.index[0]),
            f"{vc.iloc[0]:.1%}",
            ", ".join(f"{int(k)} ({v:.0%})" for k, v in vc.head(3).items()),
            int((vc < 0.01).sum()),
            int(neg[c].isna().sum()),
        )
    )
cat_prof = pd.DataFrame(
    cat_rows, columns=["feature", "distinct", "top_value", "top_share", "top3", "rare_<1%_values", "missing"]
)
tables["profile_numeric"], tables["profile_categorical"] = prof, cat_prof
hist = make_subplots(
    rows=4,
    cols=4,
    vertical_spacing=0.09,
    horizontal_spacing=0.05,
    subplot_titles=[f"log1p({c})" if c in LOGGED else c for c in NUM],
)
for i, c in enumerate(NUM):
    f = (lambda v: np.log1p(v)) if c in LOGGED else (lambda v: v)
    a, b = f(neg[c].dropna()), f(pos[c].dropna())
    lo, hi = np.nanpercentile(a, [0.5, 99.5])
    edges = np.linspace(lo, hi, 31)
    mid = (edges[:-1] + edges[1:]) / 2
    r_, c_ = i // 4 + 1, i % 4 + 1
    hist.add_trace(
        go.Bar(
            x=mid, y=np.histogram(a, edges, density=True)[0], marker_color=GREY, name="all pixels", showlegend=i == 0
        ),
        r_,
        c_,
    )
    hist.add_trace(
        go.Scatter(
            x=mid,
            y=np.histogram(b, edges, density=True)[0],
            mode="lines",
            line_color=BLUE,
            name="landslide pixels",
            showlegend=i == 0,
        ),
        r_,
        c_,
    )
hist.update_annotations(font_size=11)
figs["hist_grid"] = styled(
    hist,
    "Distribution of every numeric feature: all pixels (grey) and landslide pixels (blue)",
    height=760,
    legend=True,
)
hist.update_layout(legend=dict(orientation="v", x=0.8, y=0.1, xanchor="left", yanchor="bottom"))
sections.append(
    section(
        "Data profile",
        "Numeric profile on the negative sample (population proxy; weights are uniform within it). Heavy tails in the two distances, so they are shown on a log1p axis (spi is already log-scaled in the source).",
        table_html(prof.round(3)) + table_html(cat_prof) + "<div>@@hist_grid@@</div>",
    )
)

# ------------------------------------------------------------------ 3. origin, grain, target
grain_overlap = int(
    sum(1 for i, g in enumerate(ls_tr.geometry) for j in ls_tr.sindex.query(g, predicate="intersects") if j > i)
)
origin = (
    "<p>The 743 polygons come from 328 BPBD citizen reports re-delineated on 50 cm imagery, extended with Google Earth, Pleiades and drone "
    "images, and field-checked (Samodra et al. 2024). The paper gives no mapping footprint, so a missing polygon is not proof of no landslide. "
    "We removed five 3 km blocks that look unmapped. The label is <b>a mapped scar polygon in the inventory frame, shifted onto the Sentinel-2 grid</b>, "
    "not a ground-truth pixel state.</p><p>The model can claim to find <b>scars that resemble the ones the inventory mapped</b>, near people and roads as the "
    "reporting allowed. It cannot claim to measure true landslide density or to find slides nobody reported. Precision is a floor: real unmapped slides count as false alarms.</p>"
)
sections.append(
    section(
        "Origin and label meaning",
        "Inventory built from reports and imagery, not wall-to-wall; the label is a polygon, shifted 12.6 m.",
        origin,
    )
)

hyp_df = pd.DataFrame(HYPOTHESES, columns=["feature", "expected", "expected_size", "why"])
other = "<ul>" + "".join(f"<li>{h}</li>" for h in OTHER_HYPOTHESES) + "</ul>"
sections.append(
    section(
        "Hypotheses written before the plots",
        "Expected direction (+ = higher in landslides) and size by |AUC - 0.5|; compared with the results in the feature section.",
        table_html(hyp_df) + other,
    )
)

# block share of landslide area (training blocks, full raster, not the sample)
bi = block[train]
tot = np.bincount(bi, minlength=block.max() + 1)
npos = np.bincount(block[train & (label == 1)], minlength=block.max() + 1)
ok = tot > 0
share = pd.DataFrame({"block": np.nonzero(ok)[0], "px": tot[ok], "pos_px": npos[ok]})
share = share[share.px >= 0.2 * 300 * 300]
share["pct"] = share.pos_px / share.px * 100
edges = [-1, 0, 0.05, 0.2, 0.5, 1, 100]
labels_ = ["none", "<0.05%", "0.05-0.2%", "0.2-0.5%", "0.5-1%", ">1%"]
share["cls"] = pd.cut(share.pct.where(share.pct > 0, 0), edges, labels=labels_)
bc = share.cls.value_counts().reindex(labels_)
fig = go.Figure(
    go.Bar(x=labels_, y=bc.values, marker_color=[GREY] + [BLUE] * 5, text=bc.values, textposition="outside")
)
fig.update_yaxes(title="training blocks")
figs["block_share"] = styled(
    fig,
    f"Landslide share of block area: {bc.iloc[0] / bc.sum():.0%} of training blocks have none, {int(bc.iloc[-1])} have over 1%",
    height=340,
)
tables["block_share"] = share
fig = go.Figure(go.Histogram(x=np.log10(ls_tr.area_m2), xbins=dict(size=0.15), marker_color=BLUE))
for a, t in ((500, "5 px"), (2000, "20 px")):
    fig.add_vline(x=np.log10(a), line_color=INK, line_dash="dot", annotation_text=t, annotation_position="top")
fig.update_xaxes(title="polygon area, log10 m2")
fig.update_yaxes(title="polygons")
figs["poly_size"] = styled(
    fig,
    f"Polygon size: median {ls_tr.area_m2.median():,.0f} m2, {(ls_tr.area_m2 < 500).mean():.0%} under 500 m2 (5 px)",
    height=340,
)
sz = (
    pd.cut(ls_tr.area_m2, [0, 500, 2000, 1e9], labels=["<5 px", "5-20 px", ">20 px"])
    .value_counts()
    .reindex(["<5 px", "5-20 px", ">20 px"])
)
tables["polygon_size_classes"] = sz.rename("polygons").reset_index()
sections.append(
    section(
        "Grain, duplicates and target",
        f"One row per pixel (10 m). {len(ls_tr)} polygons sit in training blocks; {grain_overlap} overlapping polygon pair(s). Prevalence {N_POS / (N_POS + N_NEG_POP):.3%}.",
        "<div>@@block_share@@</div><div>@@poly_size@@</div>" + table_html(tables["polygon_size_classes"]),
    )
)

# ------------------------------------------------------------------ 4. missingness
miss = []
for c in ["pre", "postA", "postB", "dA", "dB", "twi"]:
    m = s[c].isna()
    miss.append(
        (
            c,
            int(m.sum()),
            f"{(s.loc[m & ~y, 'w'].sum()):,.0f}",
            f"{s.loc[m & ~y, 'w'].sum() / N_NEG_POP:.3%}",
            int((m & y).sum()),
            f"{(m & y).sum() / N_POS:.3%}",
        )
    )
miss = pd.DataFrame(
    miss,
    columns=[
        "column",
        "missing_in_sample",
        "population_px (weighted)",
        "population_%",
        "missing_landslide_px",
        "landslide_%",
    ],
)
tables["missingness"] = miss
sections.append(
    section(
        "Missingness",
        f"Few gaps: postA has no cloud-free observation for {miss.loc[miss.column == 'postA', 'population_%'].iloc[0]} of training pixels ({miss.loc[miss.column == 'postA', 'landslide_%'].iloc[0]} of landslide pixels); twi {miss.loc[miss.column == 'twi', 'population_%'].iloc[0]}; pre and dB about 36 pixels. Landslide and background gaps are similar, so the missingness does not carry the label.",
        table_html(miss)
        + "<p>Domain exclusions earlier (counted in the Phase 1 gate): 50,669 px without geology, 431,721 px in five likely-unmapped blocks.</p>",
    )
)

# ------------------------------------------------------------------ 5. features vs target
res = []
for c in NUM:
    a, b = pos[c].dropna(), neg[c].dropna()
    v = auc(a, b)
    res.append((c, v, "+" if v > 0.5 else "-", size_class(v)))
resdf = pd.DataFrame(res, columns=["feature", "auc", "direction", "size"])
hyp = hyp_df.merge(resdf, on="feature")


def verdict(r):
    if r.expected_size == "none":
        return "match" if r["size"] == "none" else "surprise"
    if r["size"] == "none":
        return "surprise (no effect)"
    if r.direction != r.expected:
        return "surprise (opposite direction)"
    return "match" if r["size"] == r.expected_size else "direction ok, size differs"


hyp["verdict"] = hyp.apply(verdict, axis=1)
tables["hypotheses_vs_results"] = hyp
order = resdf.assign(d=resdf.auc - 0.5).sort_values("d", key=abs)
fig = go.Figure(
    go.Bar(
        y=order.feature,
        x=order.d,
        base=0.5,
        orientation="h",
        marker_color=[BLUE if v > 0.5 else RED for v in order.auc],
        text=[f"{v:.2f}" for v in order.auc],
        textposition="outside",
    )
)
fig.add_vline(x=0.5, line_color=INK, line_width=1)
fig.update_xaxes(title="AUC, landslide vs background (right of 0.5: higher in landslides, left: lower)", range=[0, 1])
top3 = order.iloc[::-1].head(3)
figs["auc_bars"] = styled(
    fig, "Strongest single features: " + ", ".join(f"{r.feature} ({r.auc:.2f})" for r in top3.itertuples()), height=480
)

# embedding: PCA and per-band AUC (balanced subsample)
rng = np.random.default_rng(SEED)
sel = np.r_[np.nonzero(y)[0], rng.choice(np.nonzero(~y)[0], 30_000, replace=False)]
X = emb[sel] - emb[sel].mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
ev = S**2 / (S**2).sum()
pcs = X @ Vt.T
ysel = y[sel]
pc_auc = [auc(pcs[ysel, k], pcs[~ysel, k]) for k in range(10)]
band_auc = np.array([auc(emb[y, k], emb[~y, k]) for k in range(64)])
top = np.argsort(-np.abs(band_auc - 0.5))[:10]
tables["embedding_pca"] = pd.DataFrame({"pc": np.arange(1, 11), "explained": ev[:10], "auc": pc_auc})
tables["embedding_top_bands"] = pd.DataFrame({"band": [f"A{k:02d}" for k in top], "auc": band_auc[top]})
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "variance explained by the first 15 components",
        "AUC of components 1-10 (landslide vs background)",
    ),
)
fig.add_trace(go.Bar(x=np.arange(1, 16), y=ev[:15], marker_color=GREY), 1, 1)
fig.add_trace(
    go.Bar(
        x=np.arange(1, 11),
        y=np.array(pc_auc) - 0.5,
        base=0.5,
        marker_color=[BLUE if abs(v - 0.5) >= 0.1 else GREY for v in pc_auc],
    ),
    1,
    2,
)
fig.add_hline(y=0.5, line_color=INK, line_width=1, row=1, col=2)
fig.update_yaxes(range=[0, None], row=1, col=1)
fig.update_yaxes(range=[0.1, 0.9], row=1, col=2)
strong_pcs = [k + 1 for k, v in enumerate(pc_auc) if abs(v - 0.5) >= 0.1]
figs["emb_pca"] = styled(
    fig,
    f"2018 embedding: {int(np.searchsorted(np.cumsum(ev), 0.9)) + 1} components hold 90% of the variance; components {strong_pcs} separate landslides (|AUC - 0.5| >= 0.1)",
    height=340,
)
n_match = int((hyp.verdict == "match").sum())
with_dir = hyp[hyp.expected_size != "none"]
dir_ok = int((with_dir.expected == with_dir.direction).sum())
size_ok = int((with_dir["size"] == with_dir.expected_size).sum())
surp = "; ".join(
    f"{r.feature} (expected {r.expected} {r.expected_size}, got AUC {r.auc:.2f})"
    for r in hyp[hyp.verdict != "match"].itertuples()
)
sections.append(
    section(
        "Features vs target",
        f"AUC of each feature alone (landslide vs background pixels, training blocks). Of the {len(with_dir)} features with an expected effect, {dir_ok} had the expected direction and {size_ok} the expected size (most size misses are stronger than expected); of the {len(hyp) - len(with_dir)} expected to do nothing, {int((hyp[hyp.expected_size == 'none']['size'] != 'none').sum())} showed a weak effect. Differences: {surp}.",
        "<div>@@auc_bars@@</div>"
        + table_html(hyp[["feature", "expected", "expected_size", "auc", "direction", "size", "verdict"]].round(3))
        + "<div>@@emb_pca@@</div>"
        + table_html(tables["embedding_top_bands"].round(3)),
    )
)

# binned views with a block-bootstrap interval
KEY = ["slope", "dB", "postB", "distroad", "distriv", "elev"]
bfig = make_subplots(rows=2, cols=3, subplot_titles=[f"{c} ({UNITS[c]})" for c in KEY], vertical_spacing=0.18)
bin_rows = []
blocks_u = np.unique(s.block)
bidx = {b: np.nonzero(s.block.to_numpy() == b)[0] for b in blocks_u}
for i, c in enumerate(KEY):
    v = s[c].to_numpy()
    okk = ~np.isnan(v)
    q = wquantile(v[okk], s.w.to_numpy()[okk], np.linspace(0, 1, 11))
    q[0], q[-1] = q[0] - 1e-9, q[-1] + 1e-9
    k = np.digitize(v, q[1:-1])
    k[~okk] = -1
    num = np.zeros((len(blocks_u), 10))
    den = np.zeros((len(blocks_u), 10))
    for bi_, b in enumerate(blocks_u):
        ix = bidx[b]
        for kk in range(10):
            m = ix[k[ix] == kk]
            num[bi_, kk] = (s.w.to_numpy()[m] * y[m]).sum()
            den[bi_, kk] = s.w.to_numpy()[m].sum()
    rate = num.sum(0) / den.sum(0) * 100
    boots = []
    for _ in range(300):
        pick = rng.integers(0, len(blocks_u), len(blocks_u))
        boots.append(num[pick].sum(0) / np.maximum(den[pick].sum(0), 1e-9) * 100)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    mid = (q[:-1] + q[1:]) / 2
    r_, c_ = i // 3 + 1, i % 3 + 1
    bfig.add_trace(
        go.Scatter(
            x=np.arange(1, 11),
            y=rate,
            mode="lines+markers",
            line_color=BLUE,
            name=c,
            showlegend=False,
            error_y=dict(type="data", symmetric=False, array=hi - rate, arrayminus=rate - lo, color=GREY),
            hovertext=[
                f"bin {j + 1}: {q[j]:.2f} to {q[j + 1]:.2f}<br>{rate[j]:.2f}% [{lo[j]:.2f}, {hi[j]:.2f}]"
                for j in range(10)
            ],
            hoverinfo="text",
        ),
        r_,
        c_,
    )
    for j in range(10):
        bin_rows.append((c, j + 1, q[j], q[j + 1], rate[j], lo[j], hi[j]))
bfig.update_yaxes(title_text="landslide % of pixels", col=1)
bfig.update_xaxes(title_text="decile of the feature")
figs["binned"] = styled(bfig, "Landslide share by decile of each key feature (95% interval over blocks)", height=620)
tables["binned_rates"] = pd.DataFrame(
    bin_rows, columns=["feature", "decile", "from", "to", "rate_%", "ci_low", "ci_high"]
)
br = tables["binned_rates"]


def g(f, d):
    return float(br[(br.feature == f) & (br.decile == d)]["rate_%"].iloc[0])


binned_take = (
    f"Landslide share by decile; intervals resample whole 3 km blocks. dB: {g('dB', 1):.2f}% in the lowest decile against {br[(br.feature == 'dB') & (br.decile >= 3)]['rate_%'].mean():.2f}% "
    f"on average from decile 3 up, so the signal sits in the bottom 10 to 20%. slope: {g('slope', 1):.2f}% to {g('slope', 10):.2f}% from the flattest to the steepest decile. "
    f"distroad: share rises to {g('distroad', 9):.2f}% in decile 9 then drops to {g('distroad', 10):.2f}% in the most remote decile: remote ground may be under-mapped."
)
sections.append(section("Binned view of the key features", binned_take, "<div>@@binned@@</div>"))

# categorical: geology and land use, rate per class
cat_figs = make_subplots(
    rows=1, cols=2, subplot_titles=("geology class", "land-use class (names unknown)"), horizontal_spacing=0.12
)
cat_tab = []
for i, c in enumerate(("geologi", "landuse")):
    g = s.dropna(subset=[c])
    g = g.assign(code=g[c].astype(int))
    t = (
        g.groupby("code")
        .apply(
            lambda d: pd.Series(
                {
                    "landslide_px": int(d.y.sum()),
                    "area_km2": d.w.sum() * 100 / 1e6,
                    "rate_%": (d.w * d.y).sum() / d.w.sum() * 100,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    t.insert(0, "feature", c)
    cat_tab.append(t)
    t = t.sort_values("rate_%", ascending=False)
    cat_figs.add_trace(
        go.Bar(
            x=t.code.astype(str),
            y=t["rate_%"],
            marker_color=[BLUE if n >= 30 else GREY for n in t.landslide_px],
            hovertext=[
                f"class {k}: {n} landslide px, {a:.0f} km2"
                for k, n, a in zip(t.code, t.landslide_px, t.area_km2, strict=True)
            ],
            hoverinfo="text",
        ),
        1,
        i + 1,
    )
cat_figs.update_yaxes(title_text="landslide % of pixels", col=1)
cat_figs.update_xaxes(type="category")
figs["categorical"] = styled(
    cat_figs, "Rate by class: blue bars have 30 or more landslide pixels, grey bars are thin", height=360
)
tables["categorical_rates"] = pd.concat(cat_tab)
ct = tables["categorical_rates"]
geo_t = ct[(ct.feature == "geologi") & (ct.landslide_px >= 30)].sort_values("rate_%", ascending=False)
lu_t = ct[(ct.feature == "landuse") & (ct.landslide_px >= 30)].sort_values("rate_%", ascending=False)
cat_take = (
    f"Pixel rate by class (classes with 30 or more landslide pixels). Geology: highest {', '.join(f'{int(c)} ({r:.2f}%)' for c, r in zip(geo_t.code.head(3), geo_t['rate_%'].head(3), strict=True))}, "
    f"lowest {', '.join(f'{int(c)} ({r:.2f}%)' for c, r in zip(geo_t.code.tail(2), geo_t['rate_%'].tail(2), strict=True))}. "
    f"Land use: highest {', '.join(f'{int(c)} ({r:.2f}%)' for c, r in zip(lu_t.code.head(3), lu_t['rate_%'].head(3), strict=True))}, lowest {int(lu_t.code.iloc[-1])} ({lu_t['rate_%'].iloc[-1]:.2f}%)."
)
sections.append(section("Categorical features vs target", cat_take, "<div>@@categorical@@</div>"))

# ------------------------------------------------------------------ 6. correlations, VIF, outliers
ref_cols = [c for c in NUM]
# Both coefficients on the non-landslide sample pixels (99.85% of the domain): Spearman is robust to the heavy tails (planc, distriv, twi),
# Pearson is what a linear model's collinearity depends on, so VIF is computed from the Pearson matrix (diag of its inverse).
# dA and dB are exact differences (postA - pre, postB - pre; registry: "derived"), so their VIF is infinite by construction: VIF is computed on
# the other features, complete cases only (pairwise-complete correlations need not be a valid matrix), and the derived two are listed as exact.
DERIVED = {"dA": "postA - pre", "dB": "postB - pre"}
rho_s, rho_p = neg[ref_cols].rank().corr(), neg[ref_cols].corr()
pairs = [
    (a, b, rho_s.loc[a, b], rho_p.loc[a, b])
    for i, a in enumerate(ref_cols)
    for b in ref_cols[i + 1 :]
    if max(abs(rho_s.loc[a, b]), abs(rho_p.loc[a, b])) >= 0.7
]
vif_cols = [c for c in ref_cols if c not in DERIVED]
rc = neg[vif_cols].dropna().corr()
vif = (
    pd.Series(np.diag(np.linalg.inv(rc.to_numpy())), index=vif_cols, name="vif")
    .reset_index()
    .rename(columns={"index": "feature"})
)
vif["note"] = ""
vif = pd.concat(
    [vif, pd.DataFrame({"feature": list(DERIVED), "vif": np.nan, "note": [f"exact: {v}" for v in DERIVED.values()]})],
    ignore_index=True,
)
tables["vif"] = vif
tables["high_corr_pairs"] = pd.DataFrame(pairs, columns=["a", "b", "spearman", "pearson"])
fig = make_subplots(rows=1, cols=2, subplot_titles=("Spearman (rank)", "Pearson"), horizontal_spacing=0.08)
for k, m in enumerate((rho_s, rho_p), start=1):
    fig.add_trace(
        go.Heatmap(
            z=m.to_numpy(),
            x=ref_cols,
            y=ref_cols,
            zmin=-1,
            zmax=1,
            colorscale=[[0, RED], [0.5, "#f0efec"], [1, BLUE]],
            text=np.round(m.to_numpy(), 2),
            texttemplate="%{text}",
            textfont_size=8,
            showscale=False,
        ),
        1,
        k,
    )
fig.update_yaxes(autorange="reversed")
figs["corr"] = styled(
    fig,
    "Correlation between features on non-landslide pixels: the NDVI features move together under both coefficients",
    height=560,
)
hi_vif = ", ".join(f"{r.feature} ({r.vif:.1f})" for r in vif[vif.vif >= 10].itertuples()) or "none"
n_vif = len(neg[vif_cols].dropna())
pair_txt = ", ".join(f"{a}-{b} (Spearman {sp:.2f}, Pearson {pe:.2f})" for a, b, sp, pe in pairs) or "none"
sections.append(
    section(
        "Feature-feature correlations",
        f"Pairs with |r| >= 0.7 under either coefficient: {pair_txt}. dA and dB are exact differences (postA - pre, postB - pre), so a linear model cannot take them together with pre, postA and postB. VIF (Pearson matrix, {n_vif:,} complete rows, derived features left out) of 10 or more means a feature is redundant for a linear model: {hi_vif}.",
        "<div>@@corr@@</div>"
        + table_html(tables["high_corr_pairs"].round(2))
        + table_html(vif.round(1).astype({"vif": object}).fillna({"vif": "infinite"})),
    )
)

out_rows = []
for c in NUM:
    v = neg[c].dropna()
    t = np.log1p(v) if c in LOGGED else v
    med = t.median()
    mad = (t - med).abs().median() * 1.4826
    flag = ((t - med).abs() / mad > 3.5).mean() if mad > 0 else np.nan
    tp = np.log1p(pos[c].dropna()) if c in LOGGED else pos[c].dropna()
    flag_pos = ((tp - med).abs() / mad > 3.5).mean() if mad > 0 else np.nan
    out_rows.append((c, f"{flag:.2%}", f"{flag_pos:.2%}"))
outliers = pd.DataFrame(out_rows, columns=["feature", "flagged_all_pixels", "flagged_landslide_pixels"])
tables["outliers"] = outliers
o = outliers.set_index("feature")
sections.append(
    section(
        "Outliers",
        f"Rule: robust z (median and MAD, log1p for the two distances) above 3.5. Nothing is removed. For the NDVI features the flagged pixels are the signal: {o.loc['dB', 'flagged_landslide_pixels']} of landslide pixels are flagged on dB and {o.loc['postB', 'flagged_landslide_pixels']} on postB, against {o.loc['dB', 'flagged_all_pixels']} and {o.loc['postB', 'flagged_all_pixels']} of all pixels. Terrain features flag few landslide pixels.",
        table_html(outliers),
    )
)

# ------------------------------------------------------------------ 7. space: map and Moran's I
test_ids = set(json.load(open("data/interim/test_blocks.json"))["test_blocks"])
bmax_r, bmax_c = H // 300 + 1, Wd // 300 + 1
z = np.full((bmax_r, bmax_c), np.nan)
held = np.full((bmax_r, bmax_c), np.nan)
for _, r_ in share.iterrows():
    z[int(r_.block) // 100, int(r_.block) % 100] = r_.pct
for t_ in test_ids:
    held[t_ // 100, t_ % 100] = 1
fig = go.Figure()
fig.add_trace(
    go.Heatmap(z=held, colorscale=[[0, "#d8d7d2"], [1, "#d8d7d2"]], showscale=False, hoverinfo="skip", name="held out")
)
fig.add_trace(
    go.Heatmap(
        z=z,
        colorscale=BLUES,
        zmin=0,
        zmax=float(np.nanpercentile(z, 95)),
        colorbar=dict(title="landslide % of block area", len=0.6),
        hovertemplate="block row %{y}, col %{x}: %{z:.2f}%<extra></extra>",
    )
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
fig.update_xaxes(visible=False)
figs["map"] = styled(
    fig,
    "Landslide share of area by 3 km block; grey = held-out test blocks, blank = outside the domain or dropped",
    height=560,
)


def morans(vals, valid, n_perm=499):
    """Moran's I with queen contiguity on a regular grid; permutation p-value."""
    idx = np.argwhere(valid)
    v = vals[valid]
    zv = v - v.mean()
    pos_ = {tuple(p): k for k, p in enumerate(idx)}
    ii, jj = [], []
    for k, (a, b) in enumerate(idx):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if (da or db) and (a + da, b + db) in pos_:
                    ii.append(k)
                    jj.append(pos_[(a + da, b + db)])
    ii, jj = np.array(ii), np.array(jj)

    def stat(zz):
        return len(zz) / len(ii) * (zz[ii] * zz[jj]).sum() / (zz**2).sum()

    obs = stat(zv)
    perm = [stat(rng.permutation(zv)) for _ in range(n_perm)]
    return obs, (1 + sum(p >= obs for p in perm)) / (n_perm + 1), len(zv)


mi_rows = []
for px in (150, 300, 600):
    nr, nc = -(-H // px), -(-Wd // px)
    pad = ((0, nr * px - H), (0, nc * px - Wd))
    tp_, pp_ = (np.pad(m.astype(float), pad) for m in (train & (label != 255), train & (label == 1)))
    tsum = tp_.reshape(nr, px, nc, px).sum((1, 3))
    psum = pp_.reshape(nr, px, nc, px).sum((1, 3))
    valid = tsum >= 0.5 * px * px
    obs, p, n = morans(np.where(valid, psum / np.maximum(tsum, 1), 0), valid)
    mi_rows.append((f"{px * 10 / 1000:g} km", n, obs, p))
mi = pd.DataFrame(mi_rows, columns=["block size", "blocks", "Moran_I", "p_value"])
tables["morans_i"] = mi
mi_text = ", ".join(f"{r['block size']}: {r.Moran_I:.2f} (p {r.p_value:.2f})" for _, r in mi.iterrows())
sections.append(
    section(
        "Space",
        f"Moran's I of block landslide share (queen contiguity, training pixels only): {mi_text}. Local clustering is weak and the regional gradient dominates, so a 3 km split removes neighbours but not the regional trend: report 6 km blocks as a sensitivity check.",
        "<div>@@map@@</div>" + table_html(mi.round(3)),
    )
)

# ------------------------------------------------------------------ 8. naive detector and false-alarm classes (all training pixels)
with rasterio.open("data/interim/ndvi.tif") as r:
    pre_a, postb_a = r.read(1).astype("float32"), r.read(3).astype("float32")
dB_full = np.where((pre_a == -9999) | (postb_a == -9999), np.nan, postb_a - pre_a)
del pre_a, postb_a
tm = train & (label != 255) & ~np.isnan(dB_full)
score = -dB_full[tm]
lab = label[tm]
edges_ = np.linspace(-1.5, 1.5, 3001)
hp, _ = np.histogram(score[lab == 1], edges_)
hn, _ = np.histogram(score[lab == 0], edges_)
tp = np.cumsum(hp[::-1])[::-1]
fp = np.cumsum(hn[::-1])[::-1]
recall = tp / hp.sum()
precision = tp / np.maximum(tp + fp, 1)
pr_tab = []
for r_t in (0.2, 0.5, 0.8):
    k = np.nonzero(recall >= r_t)[0].max()  # the highest threshold that still reaches this recall
    pr_tab.append(
        (f"{r_t:.0%}", edges_[k], precision[k], int(tp[k] + fp[k]), f"{(tp[k] + fp[k]) / (hp.sum() + hn.sum()):.2%}")
    )
pr_df = pd.DataFrame(
    pr_tab, columns=["pixel recall", "threshold on -dB", "pixel precision", "flagged px", "flagged share of area"]
)
tables["naive_detector"] = pr_df
fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line_color=BLUE))
fig.add_hline(y=(hp.sum()) / (hp.sum() + hn.sum()), line_dash="dot", line_color=GREY, annotation_text="prevalence")
fig.update_xaxes(title="pixel recall", range=[0, 1])
fig.update_yaxes(title="pixel precision", range=[0, None])
figs["naive_pr"] = styled(
    fig, "Naive detector (-dNDVI threshold): precision stays low at any useful recall", height=360
)
thr = float(pr_df.loc[pr_df["pixel recall"] == "50%", "threshold on -dB"].iloc[0])
flag = np.zeros_like(label, bool)
flag[tm] = score >= thr
fa = flag & (label == 0)
comp = []
with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    lu = r.read(1)
with rasterio.open(f"{FDIR}/map_distriv.tif") as r:
    dr = r.read(1)
dr_bins = np.digitize(np.where(dr < 0, np.nan, dr), [100, 300, 1000])
dr_lab = ["<100 m", "100-300 m", "300-1000 m", ">1000 m"]
pop_dr = np.bincount(dr_bins[tm], minlength=4) / tm.sum()
fa_dr = np.bincount(dr_bins[fa], minlength=4) / fa.sum()
pos_dr = np.bincount(dr_bins[tm & (label == 1)], minlength=4) / (tm & (label == 1)).sum()
fig = go.Figure()
for nme, v, col in (
    ("all pixels", pop_dr, GREY),
    ("landslide pixels", pos_dr, BLUE),
    ("false alarms at 50% recall", fa_dr, RED),
):
    fig.add_trace(go.Bar(x=dr_lab, y=v * 100, name=nme, marker_color=col))
fig.update_yaxes(title="share of pixels, %")
fig.update_layout(barmode="group")
figs["fa_river"] = styled(
    fig, "Where the naive detector's false alarms sit, by distance to the nearest river", height=360, legend=True
)
lu_codes = np.unique(lu[tm])
pop_lu = np.array([(lu[tm] == c).mean() for c in lu_codes])
fa_lu = np.array([(lu[fa] == c).mean() for c in lu_codes])
pos_lu = np.array([(lu[tm & (label == 1)] == c).mean() for c in lu_codes])
fig = go.Figure()
for nme, v, col in (
    ("all pixels", pop_lu, GREY),
    ("landslide pixels", pos_lu, BLUE),
    ("false alarms at 50% recall", fa_lu, RED),
):
    fig.add_trace(go.Bar(x=[str(c) for c in lu_codes], y=v * 100, name=nme, marker_color=col))
fig.update_xaxes(type="category", title="land-use class code")
fig.update_yaxes(title="share of pixels, %")
fig.update_layout(barmode="group")
figs["fa_landuse"] = styled(
    fig, "Where the naive detector's false alarms sit, by land-use class", height=360, legend=True
)
tables["false_alarm_river"] = pd.DataFrame(
    {"distance to river": dr_lab, "all_%": pop_dr * 100, "landslide_%": pos_dr * 100, "false_alarm_%": fa_dr * 100}
)
tables["false_alarm_landuse"] = pd.DataFrame(
    {"landuse": lu_codes, "all_%": pop_lu * 100, "landslide_%": pos_lu * 100, "false_alarm_%": fa_lu * 100}
)
ratio = (
    tables["false_alarm_landuse"]
    .assign(r=lambda d: d["false_alarm_%"] / d["all_%"].replace(0, np.nan))
    .query("`false_alarm_%` >= 5")
    .sort_values("r", ascending=False)
)
fa_cls = ", ".join(
    f"{int(r.landuse)} ({r['false_alarm_%']:.0f}% of false alarms, {r['all_%']:.0f}% of pixels)"
    for _, r in ratio.head(3).iterrows()
)
rv = tables["false_alarm_river"]
naive_take = (
    f"A plain -dNDVI threshold sets the bar: pixel precision {pr_df['pixel precision'].iloc[0]:.0%} at 20% recall, {pr_df['pixel precision'].iloc[1]:.0%} at 50%, "
    f"{pr_df['pixel precision'].iloc[2]:.1%} at 80% (flagging {pr_df['flagged share of area'].iloc[2]} of the area). Its false alarms concentrate in land-use classes {fa_cls}. "
    f"They are no closer to rivers than the landslides themselves ({rv['false_alarm_%'].iloc[0]:.0f}% against {rv['landslide_%'].iloc[0]:.0f}% within 100 m), so distance to a river does not tell false alarms from real scars."
)
sections.append(
    section(
        "Naive detector and where it fails",
        naive_take,
        table_html(pr_df.round(4)) + "<div>@@naive_pr@@</div><div>@@fa_river@@</div><div>@@fa_landuse@@</div>",
    )
)

# ------------------------------------------------------------------ 9. leakage review
leak = pd.DataFrame(
    [
        (
            "dNDVI, NDVI windows",
            "yes, after the event",
            "retrospective detection: post-event imagery is the input, no target derived",
        ),
        (
            "embedding 2018",
            "yes, after the event",
            "year-long summary; includes the scar period, fine for retrospective detection",
        ),
        ("embedding 2017", "partly", "contains Dec 2017, about a month after the event; dropped from the candidates"),
        ("slope, elev, curvature, twi, spi, aspect", "yes", "DEMNAS terrain, not event specific"),
        (
            "distriv, distroad",
            "yes",
            "lines from the 1:25,000 topographic map; roads may be post-event cuts: unknown date",
        ),
        ("geologi", "yes", "1:100,000 map from 1992"),
        ("landuse", "yes", "1:25,000 polygons cannot hold 100 to 1,000 m2 scars; year unknown"),
        ("block id, coordinates", "no", "never a feature: they would let a model memorise the inventory footprint"),
    ],
    columns=["feature", "exists at prediction time", "note"],
)
tables["leakage_review"] = leak
sections.append(
    section(
        "Leakage review",
        "No feature is derived from the label. Roads of unknown date and the 2017 embedding are the open points.",
        table_html(leak),
    )
)


# ------------------------------------------------------------------ 10. gate: cleaning rules, constraints, ideas, open questions
pop_missing = {c: (s.loc[s[c].isna() & ~y, "w"].sum()) for c in ("postA", "twi", "pre")}
small = int((ls_tr.area_m2 < 500).sum())
cleaning = pd.DataFrame(
    [
        ("Domain: valid land use with geology, minus 5 likely-unmapped 3 km blocks", "50,669 + 431,721 px (Phase 1)"),
        ("Exclusion ring of 20 m around polygons (not positives, not negatives)", "37,929 px in the domain (Phase 1)"),
        (
            "Pixels with no cloud-free postA NDVI: keep, median-fill dA/postA and add a missing flag",
            f"about {pop_missing['postA']:,.0f} training px",
        ),
        ("Pixels with no twi: keep, median-fill and flag", f"about {pop_missing['twi']:,.0f} training px"),
        ("Pixels with no pre / dB value: drop", f"about {pop_missing['pre']:,.0f} training px"),
        ("Outliers: none removed (rule: robust z above 3.5 is reported only)", "0 px"),
    ],
    columns=["rule", "pixels"],
)
constraints = [
    f"The effective sample is the polygons ({len(ls_tr)} in training), not the {N_POS:,} positive pixels, which sit in clumps: keep models regularised and use block-grouped CV.",
    f"Prevalence is {N_POS / (N_POS + N_NEG_POP):.2%}: choose thresholds in CV, judge by precision-recall and object metrics, not ROC or accuracy; negatives are sampled and weighted.",
    f"{small} of {len(ls_tr)} training polygons ({small / len(ls_tr):.0%}) are under 5 pixels and 14 polygons have no centre pixel: the labels carry about one pixel of error, so object-level scoring by overlap is needed.",
    "Moran's I grows with block size: use 3 km blocks and report 6 km as a sensitivity check.",
    "distroad and land use carry reporting signals (remote ground looks under-mapped): treat a model that leans on them with suspicion and run it with and without.",
]
ideas = pd.DataFrame(
    [
        (
            1,
            "Neighbourhood statistics of dB and postB (3x3 and 5x5 mean, min, std)",
            "labels are about one pixel off and scars are about 10 pixels wide, while a single-pixel dB already reaches AUC "
            + f"{1 - resdf.set_index('feature').auc['dB']:.2f}",
        ),
        (
            2,
            "Land-use class as a false-alarm filter, with dB interactions",
            "false alarms of the naive detector are concentrated in a few classes: " + fa_cls,
        ),
        (
            3,
            "Gradient boosting with monotone constraints (slope up, dB down) next to a regularised logistic baseline",
            "few polygons and clumped pixels favour constrained, simple models",
        ),
        (
            4,
            "2018 embedding through its principal components or a regularised logistic regression",
            f"components {strong_pcs} already separate landslides; raw 64 bands risk overfitting few polygons",
        ),
        (
            5,
            "Geology class as a target-encoded feature inside each CV training fold",
            "class rates differ by an order of magnitude (see categorical chart); encode inside folds to avoid leakage",
        ),
        (
            6,
            "Slope mask or flag for pixels below about 5 degrees",
            "the flattest decile of slope holds the lowest rate; check before adding",
        ),
    ],
    columns=["rank", "idea", "finding behind it"],
)
questions = [
    "What are the land-use and lithology class names and the year of the topographic map? (needs the author or the map sheets)",
    "Is the drop in landslide share in the most remote decile from roads real or under-mapping? The mapping footprint decides.",
    "Which land-use classes are 10, 11 and 2, and are the false alarms bare soil, agriculture or something the inventory ignored?",
    "Are roads in distroad pre-event or include post-cyclone cuts? Dates unknown.",
]
tables["cleaning_rules"], tables["ideas"] = cleaning, ideas
sections.append(
    section(
        "Gate: cleaning rules, modeling constraints, ideas and open questions",
        "The same list goes into the project notes before any modeling approach is chosen.",
        "<h3>Cleaning rules</h3>"
        + table_html(cleaning)
        + "<h3>Modeling constraints</h3><ul>"
        + "".join(f"<li>{c}</li>" for c in constraints)
        + "</ul>"
        + "<h3>Ranked feature and model ideas</h3>"
        + table_html(ideas)
        + "<h3>Open questions</h3><ul>"
        + "".join(f"<li>{q}</li>" for q in questions)
        + "</ul>",
    )
)

# ------------------------------------------------------------------ write
n_match = int((hyp.verdict == "match").sum())
tiles = [
    tile(f"{n_blocks}", "training blocks (3 km, incl. coast slivers)"),
    tile(f"{N_POS:,}", "landslide pixels (training)"),
    tile(f"{N_POS / (N_POS + N_NEG_POP):.2%}", "prevalence in training pixels"),
    tile(f"{len(ls_tr)}", "polygons in training blocks"),
    tile(f"{dir_ok} of {len(with_dir)}", "expected directions confirmed"),
    tile(f"{precision[np.nonzero(recall >= 0.5)[0].max()]:.1%}", "naive pixel precision at 50% recall"),
]
for name, df in tables.items():
    df.to_csv(TAB / f"eda_{name}.csv", index=False)
html_parts = plots_html(figs)
rendered = []
for sec in sections:
    for k, v in html_parts.items():
        sec = sec.replace(f"@@{k}@@", v)
    rendered.append(sec)
doc = page(
    "Landslide detection, Pacitan: EDA (training blocks only)",
    "Cyclone Cempaka, Nov 2017. Pixels at 10 m, EPSG:32749. The 34 held-out blocks are not used by any number here.",
    tiles,
    rendered,
)
Path("reports/eda.html").write_text(doc)
save_pngs(figs, FIG, "eda")
print("written reports/eda.html,", len(figs), "figures,", len(tables), "tables")
print(hyp[["feature", "expected", "expected_size", "auc", "size", "verdict"]].round(3).to_string(index=False))
print(pr_df.round(4).to_string(index=False))
print(mi.round(3).to_string(index=False))
