"""Read-only Earth Engine connectivity check: embedding collection for 2017 and 2018 over Pacitan."""
import ee

from src.config import EE_PROJECT

ee.Initialize(project=EE_PROJECT)
coll = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
pacitan = ee.Geometry.Rectangle([111.0, -8.28, 111.39, -7.93])
for year in (2017, 2018):
    img = coll.filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(pacitan)
    print(year, "tiles:", img.size().getInfo(), "| bands:", len(img.first().bandNames().getInfo()),
          "| version:", img.first().get("DATASET_VERSION").getInfo())
print("latest year in collection:", ee.Date(coll.aggregate_max("system:time_start")).get("year").getInfo())
