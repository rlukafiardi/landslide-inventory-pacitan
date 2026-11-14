"""Which lithology code is which unit? Overlay the 1992 Geological Map of the Pacitan Quadrangle (1:100,000, Samodra, Gafoer, Tjokrosapoetro; a scan the user
downloaded from Badan Geologi GeoMap, not in the repo) on the release's map_geologi.tif and cross-tabulate.
The scan is not georeferenced: its graticule (111 deg 00' to 111 deg 30' E, 8 deg 00' to 8 deg 25' S) gives a first affine, refined by a small search that maximises
agreement between the scan's sea colour and the raster's no-data area (coastline). Each raster pixel then takes the nearest legend colour of the scan pixel
under it (legend swatch colours sampled from the scan); per code, the share of each unit is written to reports/tables/lithology_legend_crosstab.csv.
The names are inferred from colour and position, with the placement error (reported) of the fit; the authors have not confirmed them.
The raster carries more codes than the sheet shows (15 against 13 units): codes with no pixel inside the sheet lie west of 111 deg E; pale and near-identical
fills (the two limestones, the yellow tuffs) cannot be told apart by colour, so only groups are named. Terrain (median elevation, slope) is added as a second signal.
Usage: python -m src.analysis.lithology_legend <path to the scan .jpg>
"""
import glob
import sys

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
SCAN = sys.argv[1]
img = np.asarray(Image.open(SCAN).convert("RGB")).astype("float32")
f = glob.glob("data/spatial/mendeley_vrtx3w6mjd/map_geologi.tif")[0]
with rasterio.open(f) as r:
    code, tr = r.read(1), r.transform
    nd = r.nodata
land = code != nd
H, W = code.shape
to_ll = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
rows_i, cols_i = np.mgrid[0:H:5, 0:W:5]  # 50 m subsample for the fit
x = tr.c + (cols_i + 0.5) * 10
y = tr.f - (rows_i + 0.5) * 10
lon, lat = to_ll.transform(x, y)
lon_f, lat_f = lon - 111.0, -8.0 - lat  # degrees east of the left edge, south of the top edge
sea = (img[..., 2] - img[..., 0] >= 3) & (img.mean(2) > 215)  # the sea is a pale blue; land units and paper are not bluer than red
print("share of the scan classed as sea:", round(float(sea.mean()), 3))


def sample(params, arr):
    x0, x1, y0, y1 = params
    px = np.round(x0 + (x1 - x0) * lon_f / 0.5).astype(int)
    py = np.round(y0 + (y1 - y0) * lat_f / (25 / 60)).astype(int)
    ok = (px >= 0) & (px < arr.shape[1]) & (py >= 0) & (py < arr.shape[0])
    out = np.zeros(px.shape + arr.shape[2:], arr.dtype)
    out[ok] = arr[py[ok], px[ok]]
    return out, ok


def score(params):
    s, ok = sample(params, sea)
    land_s = land[::5, ::5]
    inside = ok & (lon_f >= 0) & (lon_f <= 0.5) & (lat_f >= 0) & (lat_f <= 25 / 60)
    return float((s[inside] == ~land_s[inside]).mean())


p = np.array([364.0, 5638.0, 551.0, 4890.0])
best = score(p)
print("initial agreement of land/sea:", round(best, 4))
for step in (40, 20, 10, 5, 2):
    for _ in range(3):
        for i in range(4):
            for d in (-step, step):
                q = p.copy()
                q[i] += d
                sc = score(q)
                if sc > best:
                    best, p = sc, q
print("fitted corners (x0, x1, y0, y1):", p.round(1), "agreement", round(best, 4))
np.save("data/interim/lithology_scan_corners.npy", p)

# legend swatch centres in the scan (pixels), read off the legend panel; colours are medians over a small box without dark (text) pixels
SWATCH = {"Qa": (2377, 6066), "Qpk": (2377, 6151), "Tmwl": (2380, 6233), "Tmo": (2380, 6359), "Tmn": (2380, 6499), "Tmw": (2380, 6635), "Tmj": (2382, 6820),
          "Tmcl": (2386, 6982), "Tms": (2389, 7064), "Toma": (2389, 7145), "Tomw": (2393, 7333), "Tomm": (2393, 7437), "Tomi": (2396, 7614)}
sw = {}
for n, (cx, cy) in SWATCH.items():
    box = img[cy - 8:cy + 9, cx - 8:cx + 9].reshape(-1, 3)
    sw[n] = np.median(box[box.mean(1) > 110], axis=0)
names = list(sw)
sw_arr = np.array([sw[n] for n in names])

# every raster pixel (stride 2) -> scan pixel under it; a 5 x 5 median removes lines and text

x0, x1, y0, y1 = p
smooth = np.stack([ndi.median_filter(img[int(y0) - 20:int(y1) + 20, int(x0) - 20:int(x1) + 20, c], size=5) for c in range(3)], axis=2)
ri, ci = np.mgrid[0:H:2, 0:W:2]
X = tr.c + (ci + 0.5) * 10
Y = tr.f - (ri + 0.5) * 10
lo, la = to_ll.transform(X, Y)
px = np.round(x0 + (x1 - x0) * (lo - 111.0) / 0.5).astype(int) - (int(x0) - 20)
py = np.round(y0 + (y1 - y0) * (-8.0 - la) / (25 / 60)).astype(int) - (int(y0) - 20)
ok = land[::2, ::2] & (px >= 0) & (px < smooth.shape[1]) & (py >= 0) & (py < smooth.shape[0])
col = smooth[py[ok], px[ok]]
cd = code[::2, ::2][ok]
dist = np.linalg.norm(col[:, None, :] - sw_arr[None], axis=2)
cls = np.array(names)[dist.argmin(1)]
ct = pd.crosstab(cd, cls)
share = ct.div(ct.sum(axis=1), axis=0)
mean_rgb = pd.DataFrame(col, columns=list("RGB")).groupby(cd).mean().round(0)
out = pd.DataFrame({"pixels_x4": ct.sum(axis=1)})
out["pct_of_area"] = (100 * ct.sum(axis=1) / ct.sum().sum()).round(2)
for i, k_ in enumerate(("1st", "2nd", "3rd")):
    top = share.apply(lambda r, i=i: r.sort_values(ascending=False).index[i], axis=1)
    out[f"{k_}_unit"] = top
    out[f"{k_}_share"] = [round(float(share.loc[c, u]), 2) for c, u in top.items()]
out = out.join(mean_rgb)
out.index.name = "code"


def layer(name):
    with rasterio.open(f"data/spatial/mendeley_vrtx3w6mjd/{name}") as rr:
        a = rr.read(1).astype("float64")
        a[a == rr.nodata] = np.nan
    return a


terrain = pd.DataFrame({"code": code.ravel(), "elev": layer("map_elev.tif").ravel(), "slope": layer("map_slope.tif").ravel()})
terrain = terrain[terrain.code != nd].groupby("code").agg(raster_pct=("elev", "size"), elev_median_m=("elev", "median"), slope_median_deg=("slope", "median")).round(1)
terrain["raster_pct"] = (100 * terrain.raster_pct / terrain.raster_pct.sum()).round(2)
out = terrain.join(out, how="left")
out.to_csv("reports/tables/lithology_legend_crosstab.csv")
pd.set_option("display.width", 250)
print(out.to_string())
print("\nswatch colours:", {n: sw[n].astype(int).tolist() for n in names})
