"""Shared helpers for the embedding change checks."""

import os

import ee
import numpy as np
import pandas as pd

BANDS = [f"A{i:02d}" for i in range(64)]
COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def auc(pos, neg):
    """Mann-Whitney AUC: P(score_pos > score_neg), ties count half."""
    s = pd.concat([pd.Series(pos), pd.Series(neg)], ignore_index=True)
    r = s.rank()
    n_pos, n_neg = len(pos), len(neg)
    return (r.iloc[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def init_ee():
    ee.Initialize(project=os.environ["EARTHENGINE_PROJECT"])


def sample_image(img, pts, scale, props, chunk=4000):
    """Sample an ee.Image at points (GeoDataFrame in EPSG:4326 with a 'pid' column); returns a DataFrame by pid."""
    out = []
    for i in range(0, len(pts), chunk):
        part = pts.iloc[i : i + chunk]
        fc = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([g.x, g.y]), {"pid": pid})
                for pid, g in zip(part.pid, part.geometry, strict=True)
            ]
        )
        feats = img.sampleRegions(collection=fc, scale=scale, geometries=False).getInfo()["features"]
        out += [{"pid": f["properties"]["pid"], **{p: f["properties"][p] for p in props}} for f in feats]
    return pd.DataFrame(out).set_index("pid")


def embedding_distances(pts, years=(2017, 2018, 2019)):
    """Cosine distance between consecutive annual embeddings at each point; only points valid in all years."""
    coll = ee.ImageCollection(COLLECTION)
    emb = {}
    for y in years:
        img = coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic()
        emb[y] = sample_image(img, pts, 10, BANDS)
        print(y, "sampled", len(emb[y]), "of", len(pts))
    common = emb[years[0]].index
    for y in years[1:]:
        common = common.intersection(emb[y].index)
    out = pd.DataFrame(index=common)
    for a, b in zip(years[:-1], years[1:], strict=True):
        va, vb = emb[a].loc[common, BANDS].to_numpy(), emb[b].loc[common, BANDS].to_numpy()
        va, vb = va / np.linalg.norm(va, axis=1, keepdims=True), vb / np.linalg.norm(vb, axis=1, keepdims=True)
        out[f"d{a % 100}{b % 100}"] = 1 - (va * vb).sum(axis=1)
    return out


S2_WINDOWS = {
    "pre": ("2017-08-01", "2017-11-01"),
    "postA": ("2017-12-01", "2018-04-01"),
    "postB": ("2018-04-01", "2018-06-01"),
}
CS_MIN = 0.6
AOI_BOUNDS = [111.0, -8.28, 111.39, -7.93]


def ndvi_composite(start, end):
    """Median NDVI of Sentinel-2 L1C scenes with Cloud Score+ cs_cdf >= CS_MIN."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    c = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(ee.Geometry.Rectangle(AOI_BOUNDS))
        .filterDate(start, end)
        .linkCollection(cs, ["cs_cdf"])
        .map(lambda i: i.updateMask(i.select("cs_cdf").gte(CS_MIN)).normalizedDifference(["B8", "B4"]).rename("ndvi"))
    )
    return c.median()
