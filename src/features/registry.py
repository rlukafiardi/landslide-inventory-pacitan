"""Feature registry: one row per feature (or family), written before the code that builds it (ds-workflow phase 3).

The sign is the expected direction in landslide pixels ("-" = lower, "+" = higher, "?" = no direction), taken from the EDA
(reports/eda.html). `build.py` produces exactly the columns listed in NAMES; `catalogue()` turns the registry into
reports/tables/feature_catalogue.csv. Groups are the unit of the phase 5 ablations.
"""

from dataclasses import dataclass

import pandas as pd

BANDS = [f"A{i:02d}" for i in range(64)]
WIN = (3, 5)  # neighbourhood windows in pixels (30 m and 50 m); the anomaly window is ANOM
ANOM = 11


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    source: str
    definition: str
    sign: str
    hypothesis: str
    missing: str = "none expected"


def _nbr():
    out = []
    for base, why in (("dB", "vegetation loss around the pixel"), ("postB", "bare ground around the pixel")):
        for w in WIN:
            r = f"{w * 10} m"
            out += [
                Feature(
                    f"{base}_mean{w}",
                    "ndvi_nbr",
                    "Sentinel-2",
                    f"mean of {base} in a {w}x{w} window ({r})",
                    "-",
                    f"{why}; labels are about one pixel off and scars are about 10 pixels wide, so the window mean absorbs the shift",
                ),
                Feature(
                    f"{base}_min{w}",
                    "ndvi_nbr",
                    "Sentinel-2",
                    f"minimum of {base} in a {w}x{w} window ({r})",
                    "-",
                    "strongest loss within reach of the pixel; robust to a one-pixel label shift",
                ),
                Feature(
                    f"{base}_std{w}",
                    "ndvi_nbr",
                    "Sentinel-2",
                    f"standard deviation of {base} in a {w}x{w} window ({r})",
                    "+",
                    "scars are patchy, so the spread inside a window is higher than on uniform cover",
                ),
            ]
    for base in ("dB", "postB"):
        out.append(
            Feature(
                f"{base}_anom{ANOM}",
                "ndvi_nbr",
                "Sentinel-2",
                f"{base} minus the mean of {base} in an {ANOM}x{ANOM} window ({ANOM * 10} m)",
                "-",
                "contrast with the surroundings removes regional phenology and cloud bias; a scar is a local anomaly",
            )
        )
    return out


FEATURES: list[Feature] = [
    Feature(
        "pre",
        "ndvi_pixel",
        "Sentinel-2 L1C + Cloud Score+",
        "median NDVI Aug-Oct 2017 (pre-event)",
        "-",
        "weak: pre-event cover barely separates sites (EDA AUC 0.48); kept as the baseline for dA and dB",
        "70 px: NaN, kept for trees, median-fill + flag for linear",
    ),
    Feature(
        "postA",
        "ndvi_pixel",
        "Sentinel-2 L1C + Cloud Score+",
        "median NDVI Dec 2017-Mar 2018",
        "-",
        "scars are bare soon after the event; cloud-limited, noisier than postB (EDA AUC 0.13)",
        "0.11% of pixels: median-fill + postA_missing flag",
    ),
    Feature(
        "postB",
        "ndvi_pixel",
        "Sentinel-2 L1C + Cloud Score+",
        "median NDVI Apr-May 2018",
        "-",
        "clearest window; scars still bare (EDA AUC 0.11)",
    ),
    Feature(
        "dA",
        "ndvi_pixel",
        "derived",
        "postA - pre",
        "-",
        "vegetation loss in the cloudy window (EDA AUC 0.19)",
        "as postA",
    ),
    Feature(
        "dB",
        "ndvi_pixel",
        "derived",
        "postB - pre",
        "-",
        "vegetation loss in the clear window; strongest single feature (EDA AUC 0.14)",
    ),
    *_nbr(),
    Feature(
        "slope",
        "terrain",
        "DEMNAS via Mendeley release",
        "slope in degrees",
        "+",
        "steeper ground fails more (EDA AUC 0.71)",
    ),
    Feature(
        "elev", "terrain", "DEMNAS via Mendeley release", "elevation in m", "?", "no clear direction (EDA AUC 0.53)"
    ),
    Feature(
        "distriv",
        "terrain",
        "1:25,000 topographic map via Mendeley release",
        "log1p of distance to river in m",
        "-",
        "banks undercut and flood (EDA AUC 0.39)",
    ),
    Feature(
        "distroad",
        "terrain",
        "1:25,000 topographic map via Mendeley release",
        "log1p of distance to road in m",
        "+",
        "opposite to the prior; the share drops in the most remote decile, a reporting signal (EDA AUC 0.56)",
    ),
    Feature(
        "twi",
        "terrain",
        "DEMNAS via Mendeley release",
        "topographic wetness index",
        "?",
        "weak (EDA AUC 0.52)",
        "1,746 px: median-fill + twi_missing flag",
    ),
    Feature(
        "spi",
        "terrain",
        "DEMNAS via Mendeley release",
        "stream power index as stored (already log-scaled, range -8 to 7)",
        "+",
        "collinear with slope (rho 0.58), AUC 0.71",
    ),
    Feature(
        "planc", "terrain", "DEMNAS via Mendeley release", "plan curvature", "-", "weak but not null (EDA AUC 0.41)"
    ),
    Feature("profc", "terrain", "DEMNAS via Mendeley release", "profile curvature", "-", "weak (EDA AUC 0.45)"),
    Feature("asp_sin", "terrain", "DEMNAS via Mendeley release", "sin of aspect (0 where flat)", "?", "weak"),
    Feature("asp_cos", "terrain", "DEMNAS via Mendeley release", "cos of aspect (0 where flat)", "?", "weak"),
    Feature(
        "geologi",
        "site_cat",
        "1:100,000 Pacitan geological map via Mendeley release",
        "lithology class code (categorical)",
        "?",
        "class rates differ by an order of magnitude (EDA); one-hot for few levels, target encoding only out-of-fold",
    ),
    Feature(
        "landuse",
        "site_cat",
        "1:25,000 topographic map via Mendeley release",
        "land-use class code (categorical)",
        "?",
        "false alarms of the naive detector pile up in classes 10, 11 and 2 (EDA); names unknown",
    ),
    Feature(
        "postA_missing",
        "flags",
        "derived",
        "1 where postA had no cloud-free observation",
        "?",
        "missingness is cloud-driven, not label-driven (EDA)",
    ),
    Feature("twi_missing", "flags", "derived", "1 where twi is missing", "?", "terrain edge cases"),
    *[
        Feature(
            f"e18_{b}",
            "emb",
            "Google Satellite Embedding V1 annual, 2018",
            f"embedding band {b} (dequantized, unit-length vector)",
            "?",
            "learned features; components 1 to 3 separate landslides (EDA AUC 0.31, 0.70, 0.17)",
        )
        for b in BANDS
    ],
]
NAMES = [f.name for f in FEATURES]
GROUPS = {g: [f.name for f in FEATURES if f.group == g] for g in dict.fromkeys(f.group for f in FEATURES)}
CATEGORICAL = ["geologi", "landuse"]


def catalogue() -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        if f.name.startswith("e18_") and f.name != "e18_A00":
            continue
        name = "e18_A00 .. e18_A63 (64 bands)" if f.name == "e18_A00" else f.name
        rows.append(
            (
                name,
                f.group,
                f.source,
                f.definition if f.name != "e18_A00" else "Satellite Embedding 2018, bands A00 to A63, dequantized",
                f.sign,
                f.hypothesis,
                f.missing,
                "yes: built from imagery and maps available after the event",
                "no label, coordinate or block id used",
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "group",
            "source",
            "definition",
            "expected_sign",
            "hypothesis",
            "missing_handling",
            "exists_at_prediction_time",
            "leakage_note",
        ],
    )
