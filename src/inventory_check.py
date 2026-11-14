"""Size and extent check of the Cempaka landslide inventory (Samodra et al., Mendeley vrtx3w6mjd)."""

import sys

import geopandas as gpd
import pyogrio

path = sys.argv[1]
print(pyogrio.list_layers(path))
gdf = gpd.read_file(path)
print("rows", len(gdf), "| crs", gdf.crs, "| geom types", gdf.geom_type.value_counts().to_dict())
print("columns", [c for c in gdf.columns if c != "geometry"])
print("invalid geoms", int((~gdf.is_valid).sum()), "| empty", int(gdf.is_empty.sum()))

m = gdf.to_crs(32749)
area = m.area
print("\nbounds (EPSG:4326)", [round(v, 4) for v in gdf.to_crs(4326).total_bounds])
print(
    "bbox size km",
    [round(v / 1000, 1) for v in (m.total_bounds[2] - m.total_bounds[0], m.total_bounds[3] - m.total_bounds[1])],
)
print("\narea m2 quantiles\n", area.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())
for px, label in [
    (1, "<1 px (100 m2)"),
    (2, "<2 px (200 m2)"),
    (3, "<3 px (300 m2)"),
    (5, "<5 px (500 m2)"),
    (10, "<10 px (1000 m2)"),
]:
    n = int((area < px * 100).sum())
    print(f"{label}: {n} polygons, {n / len(area):.1%}")
print("\nlargest 3 m2", area.nlargest(3).round(0).tolist())
print("total landslide area km2", round(area.sum() / 1e6, 2))
