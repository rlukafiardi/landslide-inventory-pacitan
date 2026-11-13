"""Which land-use code is which class? Cross-tabulate the release's map_landuse.tif against the BIG RBI 1:25,000 land-cover layers of Kab. Pacitan
(2022 edition, downloaded by the user from BIG; not in the repo). Each polygon layer is rasterized onto the release grid; per land-use code the
share of its pixels falling in each RBI layer is written to reports/tables/landuse_legend_crosstab.csv, with the dominant layer and its share.
The RBI edition (2022) is newer than the release raster (built Aug 2021), so a code-to-class match is an inference, graded by how dominant the layer is.
Usage: python -m src.analysis.landuse_legend <path to 2022_RBI25K_KABPACITAN.gdb>
"""
import sys
import warnings

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize

from src.features.build import FDIR

GDB = sys.argv[1]  # path to 2022_RBI25K_KABPACITAN.gdb
LAYERS = ["PERMUKIMAN_AR_25K", "PERUMAHAN_AR_25K", "INDUSTRIPARIWISATA_AR_25K", "SAWAH_AR_25K", "LADANG_AR_25K", "PERKEBUNAN_AR_25K", "TANAMANCAMPUR_AR_25K",
          "SEMAKBELUKAR_AR_25K", "HERBADANRUMPUT_AR_25K", "HUTANLAHANTINGGI_AR_25K", "HUTANLAHANRENDAH_AR_25K", "HUTANTANAMAN_AR_25K",
          "SUNGAI_AR_25K", "WADUK_AR_25K", "DANAU_AR_25K", "EMBUNG_AR_25K", "EMPANG_AR_25K", "TAMBAK_AR_25K", "PASIR_AR_25K", "PESISIR_AR_25K", "DERMAGA_AR_25K"]
warnings.filterwarnings("ignore")

with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    lu, tr, shape = r.read(1).astype("float64"), r.transform, r.shape
    nodata = r.nodata
lu[lu == nodata] = np.nan
rbi = np.zeros(shape, "int16")  # 0 = no RBI polygon
for i, name in enumerate(LAYERS, 1):
    g = pyogrio.read_dataframe(GDB, layer=name, columns=[])
    g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs("EPSG:32749")
    got = rasterize(((geom, i) for geom in g.geometry), out_shape=shape, transform=tr, dtype="int16", fill=0)
    new = (got > 0) & (rbi == 0)  # first layer wins where polygons overlap
    rbi[new] = i
    print(f"{name:28s} {len(g):6d} polygons, {int(new.sum()):>10,} px", flush=True)
names = np.array(["(none)"] + [n.replace("_AR_25K", "") for n in LAYERS])
ok = ~np.isnan(lu)
ct = pd.crosstab(lu[ok].astype(int), names[rbi[ok]])
share = ct.div(ct.sum(1), axis=0).round(3)
share.insert(0, "pixels", ct.sum(1))
share.insert(1, "dominant", ct.idxmax(1))
share.insert(2, "dominant_share", ct.max(1) / ct.sum(1))
share.round(3).to_csv("reports/tables/landuse_legend_crosstab.csv")
pd.set_option("display.width", 250)
print(share[["pixels", "dominant", "dominant_share"]].round(3).to_string())
print("\nshare of valid land-use pixels with no RBI polygon:", round(float((rbi[ok] == 0).mean()), 3))
