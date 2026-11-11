"""Sentinel-2 scene availability and cloudiness over Pacitan around Cyclone Cempaka (28 Nov 2017)."""
import os

import ee

from src.embedding_utils import init_ee

init_ee()
aoi = ee.Geometry.Rectangle([111.0, -8.28, 111.39, -7.93])
months = [f"2017-{m:02d}" for m in range(8, 13)] + [f"2018-{m:02d}" for m in range(1, 7)]
for cid in ("COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S2_HARMONIZED"):
    print(cid)
    for m in months:
        start = ee.Date(m + "-01")
        c = ee.ImageCollection(cid).filterBounds(aoi).filterDate(start, start.advance(1, "month"))
        n = c.size().getInfo()
        med = c.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").reduce(ee.Reducer.median()).getInfo() if n else None
        print(f"  {m}: scenes={n}, median cloudy % = {None if med is None else round(med)}")
