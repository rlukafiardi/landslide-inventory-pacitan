"""Phase 5: blinded visual review of large false alarms, as a self-contained HTML page (no GIS software needed).

Samples N_FA large false alarms (>= MIN_PX pixels, no overlap with a mapped polygon or its ring) and N_CTRL large true objects from the
c_hgb out-of-fold predictions at the 80%-recall operating point, shuffles them, and renders for each a 64 x 64 px chip (640 m) of
Sentinel-2 true colour before (Aug-Oct 2017) and after (Apr-May 2018), the same with the flagged object outlined, and the NDVI change.
The page asks for a verdict per chip (landslide scar / not a scar / unsure), keeps answers in the browser and downloads them as CSV.
The key (which chips are false alarms or controls) is written to data/interim/review_key.csv and not shown in the page.
Imagery is the same Sentinel-2 composite as the model input, not the 50 cm imagery behind the inventory, so only scar-sized objects
are judgeable; the page links each chip to a satellite view in Google Maps (grid coordinates are Sentinel-2-true, no shift needed).
Needs the Earth Engine login (.env). Usage: python -m src.modeling.review_sample [config_r<seed>]
"""
import base64
import io
import json
import sys

import ee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

from src.data.load import block_mask
from src.embedding_utils import AOI_BOUNDS, CS_MIN, S2_WINDOWS, init_ee
from src.features.build import FDIR, INTERIM
from src.modeling.metrics import EIGHT

NAME = sys.argv[1] if len(sys.argv) > 1 else "c_hgb_r0"
MIN_PX, N_FA, N_CTRL, CHIP, SEED = 20, 30, 10, 64, 11
model, rep = NAME.rsplit("_r", 1)
raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
op = raw[(raw.config == model) & (raw.repeat == int(rep)) & (raw.recall_target == 0.8)].iloc[0]
THR, AMIN = np.float64(op.thr), int(op.a_min)  # np.float64, not float: see the NumPy 2 note in stage2.py


def read(path):
    with rasterio.open(path) as r:
        return r.read(1)


prob = np.asarray(np.load(f"{INTERIM}/oof/{NAME}.npy", mmap_mode="r"))
poly, block, label = (read(f"{INTERIM}/{n}.tif") for n in ("poly_id", "block", "label"))  # label: 255 is the ring, read as stored
ring = (label == 255) & (block >= 0)
train = block_mask("train")
lab, n = ndi.label((prob >= THR) & train, structure=EIGHT)
sizes = np.bincount(lab.ravel(), minlength=n + 1)
keep = sizes >= AMIN
keep[0] = False
is_true = np.zeros(n + 1, bool)
is_true[np.unique(lab[((poly > 0) | ring) & (lab > 0)])] = True
big = np.flatnonzero(keep & (sizes >= MIN_PX))
rng = np.random.default_rng(SEED)
fa = rng.choice(big[~is_true[big]], N_FA, replace=False)
ctrl = rng.choice(big[is_true[big]], N_CTRL, replace=False)
sample = rng.permutation(np.r_[fa, ctrl])
print(f"{NAME}: threshold {THR:.3f}; large objects {len(big)} ({int((~is_true[big]).sum())} false alarms, {int(is_true[big].sum())} true); sampling {N_FA} + {N_CTRL}")

with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    tr, H, W = r.transform, r.height, r.width
with rasterio.open(f"{INTERIM}/ndvi.tif") as r:
    ndvi = r.read().astype("float32")
ndvi[ndvi == -9999.0] = np.nan
dB = ndvi[2] - ndvi[0]
slope = read(f"{FDIR}/map_slope.tif").astype("float32")
to_ll = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

init_ee()
cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")


def rgb(key):
    start, end = S2_WINDOWS[key]
    c = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(ee.Geometry.Rectangle(AOI_BOUNDS)).filterDate(start, end)
         .linkCollection(cs, ["cs_cdf"]).map(lambda i: i.updateMask(i.select("cs_cdf").gte(CS_MIN)).select(["B4", "B3", "B2"])))
    return c.median().rename([f"{key}_{b}" for b in ("R", "G", "B")])


image = ee.Image.cat([rgb("pre"), rgb("postB")]).toFloat()


def stretch(a):
    return np.clip(a / 3000.0, 0, 1) ** 0.8


def chip_png(o):
    ii, jj = np.nonzero(lab == o)
    ci, cj = int(ii.mean()), int(jj.mean())
    r0 = int(np.clip(ci - CHIP // 2, 0, H - CHIP))
    c0 = int(np.clip(cj - CHIP // 2, 0, W - CHIP))
    grid = {"dimensions": {"width": CHIP, "height": CHIP}, "crsCode": "EPSG:32749",
            "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": tr.c + c0 * 10, "shearY": 0, "scaleY": -10, "translateY": tr.f - r0 * 10}}
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    pre = np.dstack([arr[f"pre_{b}"] for b in "RGB"])
    post = np.dstack([arr[f"postB_{b}"] for b in "RGB"])
    mask = lab[r0:r0 + CHIP, c0:c0 + CHIP] == o
    fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.6), dpi=90)
    panels = [("Before (Aug-Oct 2017)", stretch(pre)), ("After (Apr-May 2018)", stretch(post)), ("After, flagged object", stretch(post)), ("NDVI change", None)]
    for a, (t, img) in zip(ax, panels):
        if img is None:
            a.imshow(dB[r0:r0 + CHIP, c0:c0 + CHIP], cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="nearest")
        else:
            a.imshow(np.nan_to_num(img, nan=0.8), interpolation="nearest")
        if t in ("After, flagged object", "NDVI change"):
            a.contour(mask, levels=[0.5], colors="#ffd400" if t.startswith("After") else "black", linewidths=1.1)
        a.set_title(t, fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].plot([4, 4 + 10], [CHIP - 4, CHIP - 4], color="white", lw=2)
    ax[0].text(4, CHIP - 7, "100 m", color="white", fontsize=7)
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    x, y = tr.c + (jj.mean() + 0.5) * 10, tr.f - (ii.mean() + 0.5) * 10
    lon, lat = to_ll.transform(x, y)
    return base64.b64encode(buf.getvalue()).decode(), lat, lon, float(np.nanmean(slope[lab == o])), float(prob[lab == o].max())


rows, cards = [], []
for k, o in enumerate(sample, 1):
    png, lat, lon, sl, pmax = chip_png(o)
    rows.append({"id": k, "object": int(o), "kind": "control_true" if is_true[o] else "false_alarm", "size_px": int(sizes[o]),
                 "lat": round(lat, 6), "lon": round(lon, 6), "slope_deg": round(sl, 1), "pmax": round(pmax, 3)})
    cards.append(f"""<section class="card" data-id="{k}"><h2>#{k}</h2>
<img src="data:image/png;base64,{png}" alt="chip {k}">
<p class="meta">{int(sizes[o])} px (~{int(sizes[o]) * 100:,} m2) · mean slope {sl:.0f}° · max probability {pmax:.2f} ·
<a href="https://www.google.com/maps?q={lat:.6f},{lon:.6f}&t=k&z=18" target="_blank" rel="noopener">open satellite view</a></p>
<div class="ans"><label><input type="radio" name="a{k}" value="scar"> landslide scar</label>
<label><input type="radio" name="a{k}" value="not_scar"> not a scar</label>
<label><input type="radio" name="a{k}" value="unsure"> unsure</label>
<input type="text" name="n{k}" placeholder="note (optional)"></div></section>""")
key = pd.DataFrame(rows)
key.to_csv(f"{INTERIM}/review_key.csv", index=False)
json.dump(key[["id", "lat", "lon", "size_px"]].to_dict("records"), open(f"{INTERIM}/review_points.json", "w"))

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Landslide review sample</title><style>
:root{--ink:#0b0b0b;--muted:#52514e;--bg:#fcfcfb;--line:#e8e7e3;--accent:#2a78d6}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}.lead{color:var(--muted);max-width:70ch}
.card{border-top:1px solid var(--line);padding:12px 0}.card h2{font-size:16px;margin:0 0 6px}
a{color:var(--accent)}img{max-width:100%;height:auto;display:block}.meta{color:var(--muted);margin:6px 0;font-size:13px}
.ans{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.ans input[type=text]{flex:1;min-width:180px;padding:5px;border:1px solid var(--line);border-radius:4px}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}
#bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;display:flex;gap:12px;align-items:center}
@media (prefers-color-scheme: dark){:root{--ink:#f2f2f0;--muted:#b9b8b2;--bg:#161615;--line:#34332f}}
</style></head><body><main>
<h1>Are these flagged objects landslide scars?</h1>
<p class="lead">__N__ objects flagged by the model, in random order. Each row: true colour before (Aug-Oct 2017) and after (Apr-May 2018) the storm,
the same view with the flagged object outlined in yellow, and the NDVI change (red = vegetation lost; black outline = the object). A scar looks like a
fresh bare, light patch after the event that was vegetated before. Sentinel-2 has 10 m pixels, so judge only what is clearly visible; use the satellite
link for a closer look. Answers save in this browser as you go; press the button to download them as CSV and send the file back.</p>
<div id="bar"><button id="dl">Download answers (CSV)</button><span id="count"></span></div>
__CARDS__
</main><script>
const KEY="landslide_review_v1";let S={};
try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}
 const done=document.querySelectorAll('input[type=radio]:checked').length;document.getElementById('count').textContent=done+" of __N__ answered"}
document.querySelectorAll('.card').forEach(c=>{const id=c.dataset.id;
 const r=c.querySelectorAll('input[type=radio]'),t=c.querySelector('input[type=text]');
 const s=S[id]||{};r.forEach(x=>{if(s.a===x.value)x.checked=true;x.addEventListener('change',()=>{S[id]=Object.assign(S[id]||{},{a:x.value});save()})});
 if(s.n)t.value=s.n;t.addEventListener('input',()=>{S[id]=Object.assign(S[id]||{},{n:t.value});save()})});
save();
document.getElementById('dl').onclick=()=>{let o="id,answer,note\\n";
 document.querySelectorAll('.card').forEach(c=>{const id=c.dataset.id,s=S[id]||{};o+=id+","+(s.a||"")+",\\""+(s.n||"").replace(/"/g,'""')+"\\"\\n"});
 const b=new Blob([o],{type:"text/csv"}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download="landslide_review_answers.csv";a.click()};
</script></body></html>"""
open("reports/review_sample.html", "w").write(PAGE.replace("__N__", str(len(rows))).replace("__CARDS__", "\n".join(cards)))
print(f"written reports/review_sample.html ({len(rows)} chips); key in {INTERIM}/review_key.csv")
