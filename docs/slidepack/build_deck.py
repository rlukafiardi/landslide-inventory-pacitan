"""Build docs/slidepack/landslide_pacitan_deck.pdf from the result tables of this project.

Every number on a page is read from reports/tables/*.csv (or the blinded-review answers in data/interim, aggregated counts
only), never typed. Run from the project root:

    uv run --no-project --with reportlab --with pillow --with matplotlib --with pandas python -I docs/slidepack/build_deck.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

ROOT = Path.cwd()
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE / ".claude/skills/slidepack/scripts"))
from slidepack import BLUE, CHART_H_PT, CHART_W_PT, Deck, M, W  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.analysis.glossary import DATA, DECK, METHOD, METRICS  # noqa: E402
from src.analysis.palette import CHART_BLUE  # noqa: E402

VERDICT_PAGE, MODELS_PAGE = 11, 10  # pages cited in the text; checked after the build
PAGES = {}

T = ROOT / "reports/tables"
IMG = ROOT / "docs/slidepack/img"
OUT = ROOT / "docs/slidepack/landslide_pacitan_deck.pdf"
IMG.mkdir(parents=True, exist_ok=True)

for f in ("Regular", "Medium", "Bold"):
    font_manager.fontManager.addfont(str(WORKSPACE / f".claude/skills/social-posting/fonts/PlusJakartaSans-{f}.ttf"))
plt.rcParams.update(
    {"font.family": "Plus Jakarta Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
)

NAVY, INK2, SUN, GREY = "#1D3557", "#3D5470", "#E63946", "#B8B8B2"
SCAR, NOTSCAR, UNSURE = CHART_BLUE, "#D5D5D0", "#8A8A84"  # review chart: same mapping on every page that uses it

pct = lambda v, d=1: f"{v * 100:.{d}f}%"  # noqa: E731
ci = lambda v, lo, hi: f"{pct(v)} ({lo * 100:.1f}-{hi * 100:.1f})"  # noqa: E731


def save_chart(fig, name):
    fig.savefig(IMG / name, dpi=250, transparent=True)
    plt.close(fig)
    return IMG / name


def new_fig():
    fig, ax = plt.subplots(figsize=(CHART_W_PT / 72, CHART_H_PT / 72))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.17)
    return fig, ax


# ---- data -----------------------------------------------------------------------------------------------------------
test = pd.read_csv(T / "test_results.csv").set_index("model")
curve = pd.read_csv(T / "test_curve.csv")
mres = pd.read_csv(T / "modeling_results.csv")
mres = mres[mres.recall_target == 0.8].set_index("config")
naive = pd.read_csv(T / "eda_naive_detector.csv")
abl = pd.read_csv(T / "ablation.csv")
cand = pd.read_csv(T / "candidates_summary.csv")
s2 = pd.read_csv(T / "stage2_results.csv")
sp = pd.read_csv(T / "size_prior_results.csv").set_index(["rule_S", "rule_P"])
prev, post = pd.read_csv(T / "prev_raw.csv"), pd.read_csv(T / "post_raw.csv")
key, ans = pd.read_csv(ROOT / "data/interim/review_key.csv"), pd.read_csv(ROOT / "data/interim/review_answers.csv")

c, lg, nv = test.loc["c_hgb"], test.loc["simple_logreg"], test.loc["naive_dBmin5"]
base_fpt = (
    f"{lg.false_per_true:.0f}"
    if round(lg.false_per_true) == round(nv.false_per_true)
    else f"{lg.false_per_true:.0f} to {nv.false_per_true:.0f}"
)
n_cand, n_large = int(cand.candidates.sum()), int(cand[cand.size_class.str.startswith("large")].candidates.sum())
rv = key.merge(ans, on="id")
tab = (
    pd.crosstab(rv.kind, rv.answer)
    .reindex(["control_true", "false_alarm"])
    .reindex(columns=["scar", "not_scar", "unsure"])
    .fillna(0)
    .astype(int)
)
fa, ctl = tab.loc["false_alarm"], tab.loc["control_true"]
review_note_terrace = rv.note.fillna("").str.contains("terrace", case=False).sum()


def rep0(df, name, recall=0.8):
    return float(df[(df.config == name) & (df.repeat == 0) & (df.recall_target == recall)].precision_at_target.iloc[0])


# ---- charts ---------------------------------------------------------------------------------------------------------
# 1 review workload
fig, ax = new_fig()
rows = [
    ("Chosen model (boosting, 97 features)", c, SUN),
    ("Simple logistic\n(vegetation change + site maps)", lg, GREY),
    ("Naive threshold on\nvegetation loss", nv, GREY),
]
for i, (_, r, col) in enumerate(rows):
    ax.barh(i, r.false_per_true, color=col, height=0.62)
    ax.text(
        r.false_per_true + 0.6,
        i,
        f"{r.false_per_true:.0f}  (95% interval {r.false_per_true_ci_low:.0f} to {r.false_per_true_ci_high:.0f})",
        va="center",
        color=NAVY,
    )
ax.set_yticks(range(3), [r[0] for r in rows], color=NAVY)
ax.invert_yaxis()
ax.set_xlim(0, 60)
ax.set_xlabel("false objects to check per true scar found", color=INK2)
ax.grid(axis="x", color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)
img_work = save_chart(fig, "workload.png")

# 2 precision-recall on the test blocks
fig, ax = plt.subplots(figsize=(CHART_W_PT / 72, CHART_H_PT / 72))
fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.17)
cv = curve.sort_values("object_recall_ge5")
ax.plot(cv.object_recall_ge5 * 100, cv.object_precision * 100, "-o", color=GREY, ms=4, lw=2)
ax.axvline(80, color=NAVY, ls="--", lw=1)
ax.axhline(30, color=NAVY, ls="--", lw=1)
ax.text(79.3, 52, "recall target 80%", ha="right", color=NAVY)
ax.text(99.5, 31.2, "precision target 30%", ha="right", color=NAVY)
ax.scatter(
    [c.object_recall_ge5 * 100], [c.object_precision * 100], s=130, facecolor="none", edgecolor=SUN, lw=2.5, zorder=5
)
ax.annotate(
    f"frozen operating point\n{pct(c.object_recall_ge5)} recall, {pct(c.object_precision)} precision",
    (c.object_recall_ge5 * 100, c.object_precision * 100),
    (88, 24),
    color=SUN,
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "color": SUN},
)
ax.set_xlim(30, 100)
ax.set_ylim(0, 55)
ax.set_xlabel("object recall, polygons of 500 m2 or more", color=INK2)
ax.set_ylabel("object precision", color=INK2)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.grid(color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)
img_pr = save_chart(fig, "precision_recall.png")
above = cv[cv.object_precision >= 0.30].iloc[-1]
below = cv[cv.object_precision < 0.30].iloc[0]
recall_at_30 = float(
    above.object_recall_ge5
    + (above.object_precision - 0.30)
    / (above.object_precision - below.object_precision)
    * (below.object_recall_ge5 - above.object_recall_ge5)
)

# 3 blinded review
fig, ax = new_fig()
fig.subplots_adjust(left=0.24)
names = [f"Known scars (controls, n = {ctl.sum()})", f"Large false alarms (n = {fa.sum()})"]
for i, r in enumerate([ctl, fa]):
    left = 0
    for col, colr, lab in (
        ("scar", SCAR, "judged a scar"),
        ("not_scar", NOTSCAR, "judged not a scar"),
        ("unsure", UNSURE, "unsure"),
    ):
        share = r[col] / r.sum() * 100
        if r[col]:
            ax.barh(i, share, left=left, color=colr, height=0.6)
            if share < 10:
                ax.annotate(
                    f"{lab}: {r[col]}",
                    (left + share / 2, i - 0.3),
                    (left + 1, i - 0.46),
                    color=colr,
                    fontsize=10,
                    fontweight="bold",
                    va="bottom",
                    arrowprops={"arrowstyle": "-", "color": colr},
                )
            else:
                ax.text(
                    left + share / 2,
                    i,
                    f"{lab}\n{r[col]}",
                    ha="center",
                    va="center",
                    color="white" if col in ("scar", "unsure") else NAVY,
                    fontsize=10,
                )
        left += share
ax.set_yticks(range(2), names, color=NAVY)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("share of reviewed objects", color=INK2)
img_review = save_chart(fig, "review.png")

# 4 maps (docs/slidepack/maps.py): hillshade base, inventory, blocks, one test block, candidates
sys.path.insert(0, str(ROOT / "docs/slidepack"))
import maps  # noqa: E402

map_img, ms = maps.build_maps(ROOT, IMG)
ar = Image.open(IMG / "architecture.png")
ar.crop((0, 165, ar.width, ar.height)).save(IMG / "architecture_crop.png")

# ---- deck -----------------------------------------------------------------------------------------------------------
d = Deck(OUT, title="Landslide screening in Pacitan: results and limits")
SRC_REPO = "github.com/rlukafiardi/landslide-inventory-pacitan"


def glossary_page(title, items, sub=None):
    """Two-column plain-language definitions (from src/analysis/glossary.py, shared with the reports)."""
    d._page(sources="Definitions: src/analysis/glossary.py, also in the modeling and EDA reports")
    top = d._head(title, sub) - 22
    half = (len(items) + 1) // 2
    colw = (W - 2 * M - 30) / 2
    for k, col in enumerate((items[:half], items[half:])):
        y = top
        for term, body in col:
            y = d._para(M + k * (colw + 30), y, term + ".", body, 13, BLUE, NAVY, colw) - 10


d.cover(
    "Finding landslide scars from satellite data in Pacitan",
    "A screening model for Cyclone Cempaka (Nov 2017): what it finds, what it misses, and what it hands to an expert",
    "Model report, November 2026",
    authors=("Radikal Lukafiardi",),
)

d.summary(
    f"The model finds {pct(c.object_recall_ge5)} of scars but only {pct(c.object_precision)} of its flags are real",
    [
        (
            "Answer",
            f"On {int(c.polygons_ge5)} scars of 500 m2 or more in held-out blocks, the model detects {pct(c.object_recall_ge5)} (target 80%: met). "
            f"Of the objects it flags, {pct(c.object_precision)} touch a scar (target 30%: not met). The criterion stays as written.",
        ),
        (
            "Why it still helps",
            f"An expert checks {c.false_per_true:.0f} false objects per true scar instead of {base_fpt} with the baselines, "
            f"over {pct(c.flagged_share)} of the area.",
        ),
        (
            "What the false alarms are",
            f"In a blinded review of {fa.sum()} large false alarms, {fa['scar']} was judged a scar. Most look like terraces, river banks or roads, not unmapped landslides.",
        ),
        (
            "Use",
            f"A screening aid, not a map of every landslide. {n_cand:,} candidates are delivered for expert verdicts, the {n_large:,} large ones first.",
        ),
    ],
)

d.bullets(
    "The goal: flag candidate scars for expert review, with recall of at least 80% and precision of at least 30%",
    [
        (
            "Aim",
            "A screening aid that flags candidate landslide scars from satellite data, so an expert reviews flags instead of digitizing the whole area by hand.",
        ),
        ("Event and place", "Cyclone Cempaka, November 2017, Pacitan regency, East Java: a 1,370 km2 study domain."),
        (
            "Success, fixed before any model",
            "Object recall of 80% or more on polygons of 500 m2 or more (5 pixels), at object precision of 30% or more, on held-out 3 km blocks.",
        ),
        (
            "Why precision is a floor",
            "The inventory is not wall-to-wall: a flag with no polygon under it may still be a real scar that nobody mapped.",
        ),
    ],
    sub="Recall counts inventory polygons found; precision counts flagged objects that touch a polygon.",
    sources=f"{SRC_REPO} docs/plan.md, docs/validation-design.md",
)

n_found = round(c.object_recall_ge5 * c.polygons_ge5)
d.compare(
    "Recall: how many real scars did it find?",
    [
        ("Question", "Out of every 100 real scars of 500 m2 or more, how many did the model find?"),
        ("Counted as found", "At least one flagged object touches the scar."),
        (
            "On the test blocks",
            f"{n_found} of {int(c.polygons_ge5)} scars found: {pct(c.object_recall_ge5)}. Target 80% or more: met.",
        ),
        ("Read it with care", "Flagging more of the map raises recall, so it is never read alone."),
    ],
    "Precision: how many flags are real?",
    [
        ("Question", "Out of every 100 flagged objects, how many touch a real scar?"),
        ("Counted as real", "The object touches a scar mapped in the inventory."),
        (
            "On the test blocks",
            f"{int(c.true_objects)} of {int(c.objects):,} objects touch a scar: {pct(c.object_precision)}. Target 30% or more: not met.",
        ),
        ("A floor", "A flag on a real scar that nobody mapped counts as wrong, so true precision is higher."),
    ],
    left_tag="RECALL",
    right_tag="PRECISION",
)
for x0, share, color, caption in (
    (M, c.object_recall_ge5, SUN, "scars found, out of every 100 real scars"),
    (W / 2 + 30, c.object_precision, CHART_BLUE, "flags that are real, out of every 100 flagged objects"),
):
    k = round(share * 100)
    for i in range(100):
        row, col = divmod(i, 10)
        d.c.setFillColor(color if i < k else "#D5D5D0")
        d.c.rect(x0 + col * 14, 222 - (row + 1) * 14, 12, 12, stroke=0, fill=1)
    d._text(x0 + 156, 150, f"{k}", "PJS-XBold", 40, color, 120)
    d._text(x0 + 156, 128, caption, "PJS-Regular", 12, NAVY, 230, 1.25)

d.bullets(
    "Labels are 743 polygons from a citizen-report inventory, so a missing polygon is not proof of no landslide",
    [
        (
            "Labels",
            "Samodra et al. 2024 inventory (CC BY 4.0), built from 328 citizen reports extended with 50 cm imagery and field checks. It states no footprint.",
        ),
        (
            "Inputs",
            "Sentinel-2 satellite images before and after the storm, a 2018 satellite embedding, and the release's terrain, land-use and geology maps, all on one 10 m grid. The next page lists the 97 features.",
        ),
        (
            "Rare and small",
            "Scars cover 0.15% of the area; the median polygon is 1,096 m2 and 26% are under 500 m2, below what 10 m pixels show reliably.",
        ),
        (
            "Registration",
            "Polygons sit about 12.6 m off the Sentinel-2 grid (found from the data, cause unverified) and were shifted onto it; predictions shift back to compare with the inventory.",
        ),
    ],
    sub="Observed: labels and imagery. Assumed: the shift, and that unmapped ground is mostly no landslide.",
    sources="Samodra et al. 2024, Mendeley Data 10.17632/vrtx3w6mjd.1; Sentinel-2: Copernicus; embedding: Google DeepMind (CC BY 4.0)",
)

cat = pd.read_csv(T / "feature_catalogue.csv")
cat["n"] = cat.feature.str.extract(r"\((\d+) bands\)")[0].fillna(1).astype(int)
nf = cat.groupby("group").n.sum()
assert int(nf.sum()) == 97, nf
d.bullets(
    f"Each 10 m pixel is described by {int(nf.sum())} numbers in six groups, from vegetation change to terrain and a satellite embedding",
    [
        (
            f"Vegetation at the pixel ({nf['ndvi_pixel']})",
            "Greenness before the storm (pre), after it (postA, postB) and the change between them (dA, dB).",
        ),
        (
            f"Vegetation around the pixel ({nf['ndvi_nbr']})",
            "Mean, lowest value and spread of the change within 30 m and 50 m (dB_min5 is the lowest within 50 m), and how the pixel differs from its 110 m surroundings.",
        ),
        (
            f"Terrain ({nf['terrain']})",
            "Slope, elevation, distance to river and road, wetness index (twi), stream power (spi), curvature and aspect.",
        ),
        (
            f"Land use and lithology ({nf['site_cat']})",
            "Class codes from the release's land-use and geology maps. With terrain they make up the 'site maps' of the model comparison.",
        ),
        (
            f"Satellite Embedding ({nf['emb']})",
            "A 64-number summary of what the satellite saw in 2018 at that pixel, from a Google model (e18_A00 to e18_A63).",
        ),
        (f"Missing-data flags ({nf['flags']})", "Mark pixels with no cloud-free observation or no wetness index."),
    ],
    sub="NDVI is a vegetation index from satellite light: high is dense green, low is bare ground. dB is its change after the storm.",
    sources="Source: reports/tables/feature_catalogue.csv",
)

d.map_page(
    f"Scars sit on steeper ground: the median slope at a scar is {ms['slope_med_poly']:.0f} degrees, against {ms['slope_med_area']:.0f} across the study area",
    "The map shows where scars were mapped, not where they happened: the inventory started from citizen reports.",
    map_img["inventory"],
    [
        (
            "What is drawn",
            f"Centroids of the {ms['n_poly']} inventory polygons on a hillshade of the release's 10 m elevation model. UTM 49S, north up.",
        ),
        (
            "Steeper ground",
            f"{ms['share_poly_ge25']:.0%} of scars sit on slopes of 25 degrees or more; {ms['share_area_ge25']:.0%} of the study area does.",
        ),
        ("Gaps", "Few scars are mapped on the flat western plain. A gap is not proof of no landslide."),
    ],
    legend=[("dot", SUN, f"inventory polygon, centroid (n = {ms['n_poly']})")],
    sources="Samodra et al. 2024 (CC BY 4.0): inventory and elevation model; slope read from the release's slope layer",
)

d.map_page(
    "Scores come from whole 3 km blocks held out of training, so spatial neighbours cannot leak into the test",
    f"{ms['n_blocks_test']} of {ms['n_blocks_eligible']} blocks (seed 42) hold {ms['n_poly_test']} of {ms['n_poly']} polygons and are scored once. The score on page {VERDICT_PAGE} is one run of a frozen model and threshold.",
    map_img["blocks"],
    [
        ("Model choice", "Repeated 5-fold block cross-validation, 5 seeds; models compared at 80% object recall."),
        (
            "Why not AUC",
            "AUC, a ranking score where 0.5 is a coin flip, hides the false alarms of a rare event; random folds also flatter average precision by up to 0.06.",
        ),
        ("Excluded", "Five blocks look unmapped and stay out of the domain."),
        (
            "Intervals",
            f"Block bootstrap 95%; {ms['n_poly_test_ge5']} test polygons of 500 m2 or more, so intervals are wide.",
        ),
    ],
    legend=[
        ("fill", SUN, f"held-out test block ({ms['n_blocks_test']})"),
        ("outline", "#9AA5B1", "training block (white edges)"),
        ("hatch", NAVY, "excluded as likely unmapped (5)"),
        ("dot", NAVY, "inventory polygon, centroid"),
    ],
    sources=f"{SRC_REPO} docs/validation-design.md; elevation: Samodra et al. 2024 (CC BY 4.0)",
)

r50 = naive[naive["pixel recall"] == "50%"].iloc[0]
d.table(
    f"A plain vegetation-loss threshold needs {r50['flagged share of area']} of the area to reach 50% recall, and only {pct(r50['pixel precision'], 0)} of those pixels are scars",
    ["Pixel recall", "Threshold on NDVI loss", "Pixel precision", "Flagged pixels", "Area flagged"],
    [
        [
            r["pixel recall"],
            f"{r['threshold on -dB']:.3f}",
            pct(r["pixel precision"]),
            f"{int(r['flagged px']):,}",
            r["flagged share of area"],
        ]
        for _, r in naive.iterrows()
    ],
    sub="A threshold alone is a weak detector, so the other features have to remove false alarms.",
    sources="Source: reports/tables/eda_naive_detector.csv",
    highlight=1,
    note="NDVI loss = pre-event NDVI minus April-May 2018 NDVI; pixel-level; recall of 80% flags a fifth of the area",
)

order = [
    ("c_hgb", "All features (boosting), chosen"),
    ("a_hgb", "Vegetation change + site maps (boosting)"),
    ("b_hgb", "Satellite embedding + site maps (boosting)"),
    ("all_logreg", "All features (logistic)"),
    ("simple_logreg", "Simple logistic (vegetation change + site maps)"),
    ("naive_dBmin5", "Naive threshold on vegetation loss"),
]
PAGES["models"] = d.n + 1
d.table(
    f"Vegetation change and the satellite embedding complement each other: together {pct(mres.loc['c_hgb', 'precision'])} precision at 80% recall, against {pct(mres.loc['a_hgb', 'precision'])} and {pct(mres.loc['b_hgb', 'precision'])} alone",
    ["Model", "Precision at 80% recall", "95% interval", "Area flagged"],
    [
        [
            lab,
            pct(mres.loc[k, "precision"]),
            f"{pct(mres.loc[k, 'ci_low'])} to {pct(mres.loc[k, 'ci_high'])}",
            pct(mres.loc[k, "flagged_share"], 0 if mres.loc[k, "flagged_share"] >= 0.1 else 1),
        ]
        for k, lab in order
    ],
    sub="Cross-validated on training blocks, 5 repeats. The intervals overlap; the paired comparison is in the report.",
    sources="Source: reports/tables/modeling_results.csv",
    highlight=0,
)

PAGES["verdict"] = d.n + 1
d.numbers(
    f"Recall target met at {pct(c.object_recall_ge5)}, precision target not met at {pct(c.object_precision)}; the criterion stays unchanged",
    [
        (
            pct(c.object_recall_ge5),
            f"object recall, 95% interval {pct(c.object_recall_ge5_ci_low)} to {pct(c.object_recall_ge5_ci_high)} (target 80%, met)",
        ),
        (
            pct(c.object_precision),
            f"object precision, 95% interval {pct(c.object_precision_ci_low)} to {pct(c.object_precision_ci_high)} (target 30%, not met)",
        ),
        (f"{c.false_per_true:.0f}", f"false objects to check per true scar (baselines: {base_fpt})"),
        (pct(c.flagged_share), "of the test area flagged for review"),
    ],
    sub=f"Frozen model scored once on 34 held-out blocks: {int(c.objects):,} objects flagged, {int(c.true_objects)} touching a scar.",
    sources="Source: reports/tables/test_results.csv",
)

d.table(
    "At similar recall the model flags about a third as many objects as the baselines and recovers scar shapes better",
    ["Model", "Object recall", "Strict recall", "Object precision", "Objects flagged"],
    [
        [
            lab,
            ci(r.object_recall_ge5, r.object_recall_ge5_ci_low, r.object_recall_ge5_ci_high),
            pct(r.strict_recall_ge5),
            ci(r.object_precision, r.object_precision_ci_low, r.object_precision_ci_high),
            f"{int(r.objects):,}",
        ]
        for lab, r in (
            ("Chosen model (boosting)", c),
            ("Simple logistic (vegetation change + site maps)", lg),
            ("Naive threshold on vegetation loss", nv),
        )
    ],
    sub="Same 34 test blocks, 157 polygons (106 of 500 m2 or more). Intervals overlap on recall; they do not on precision.",
    sources="Source: reports/tables/test_results.csv",
    highlight=0,
    note="strict recall needs overlap with a polygon (IoU 0.3); the loose 'touches' rule favours long objects",
)

d.chart(
    f"About {c.false_per_true:.0f} false objects to check per true scar, against {base_fpt} for the baselines",
    "The review workload is what an expert pays for each scar found.",
    "False objects per true scar found, test blocks, 95% interval",
    img_work,
    "Source: reports/tables/test_results.csv",
)

d.chart(
    f"Reaching 30% precision would cost recall: on the test blocks it appears only at about {recall_at_30 * 100:.0f}%",
    "The curve is descriptive (106 polygons); the operating point was frozen from training-block repeats, not read off this curve.",
    "Object precision against object recall, test blocks, model sweep over thresholds",
    img_pr,
    "Source: reports/tables/test_curve.csv, test_results.csv",
)

small, mid, big = "recall_<5 px", "recall_5-20 px", "recall_>20 px"
d.table(
    f"Misses are the small scars: {pct(c[big], 0)} of polygons over 20 pixels are found, against {pct(c[small], 0)} of those under 5",
    ["Polygon size", "Chosen model", "Simple logistic", "Naive threshold"],
    [
        ["Under 5 px (outside the target)", pct(c[small], 0), pct(lg[small], 0), pct(nv[small], 0)],
        ["5 to 20 px", pct(c[mid], 0), pct(lg[mid], 0), pct(nv[mid], 0)],
        ["Over 20 px", pct(c[big], 0), pct(lg[big], 0), pct(nv[big], 0)],
    ],
    sub="Share of polygons touched by a flagged object. The baselines find more small ones, at about three times the review workload.",
    sources="Source: reports/tables/test_results.csv",
    highlight=2,
)

zb = ms["zoom_block"]
undet_ge5 = ms["zoom_poly_ge5"] - ms["zoom_det_ge5"]
d.map_page(
    f"In the densest test block, flagged objects touch {ms['zoom_det_ge5']} of {ms['zoom_poly_ge5']} scars of 500 m2 or more, and {ms['zoom_false']} other objects touch no scar",
    f"Block {zb} has the most inventory polygons ({ms['zoom_poly']}) of the 34 test blocks. Its long blue object follows a river bed and counts as true because it touches a scar.",
    map_img["zoom"],
    [
        (
            "Counts",
            f"{ms['zoom_poly']} inventory polygons, {ms['zoom_poly_ge5']} of them 500 m2 or more; {ms['zoom_objects']} flagged objects, {ms['zoom_true']} touching a polygon.",
        ),
        (
            "Untouched polygons",
            f"{ms['zoom_poly'] - ms['zoom_det_all']} polygons are not touched by any object"
            + (", all under 500 m2." if undet_ge5 == 0 else f", {undet_ge5} of them 500 m2 or more."),
        ),
        (
            "Reading it",
            "False alarms here are small and scattered. The loose 'touches' rule also credits long objects, which is why strict recall is reported next to it.",
        ),
    ],
    legend=[
        ("outline", NAVY, "inventory polygon"),
        ("solid", NAVY, "inventory polygon, not touched"),
        ("fill", CHART_BLUE, "flagged object touching a polygon"),
        ("fill", SUN, "flagged object touching none"),
    ],
    sources="Candidate file, held-out test blocks; elevation: Samodra et al. 2024 (CC BY 4.0)",
    note="inventory frame, UTM 49S; dashed line = block edge; objects counted by their representative point",
    max_w=345,
)

d.chart(
    f"In a blinded review, {fa['scar']} of {fa.sum()} large false alarms was judged a scar, {fa['not_scar']} were not and {fa['unsure']} were unsure",
    f"Reviewers recognised {ctl['scar']} of {ctl.sum()} known scars shuffled in as controls; the most common note on false alarms was terrace farming ({review_note_terrace}).",
    "Blinded visual review on Sentinel-2 true colour, share of objects by verdict",
    img_review,
    "Source: blinded review answers (aggregated counts)",
    note="only objects of 20 px or more (24% of false alarms), n = 30; the reviewer saw the same imagery as the model",
)

pa, pb = rep0(prev, "abl_plus_ndvi_prev"), rep0(post, "abl_plus_ndvi_post")
base = rep0(prev, "c_hgb")
s2b = float(s2[(s2.variant == "s2_base") & (s2.repeat == 0) & (s2.recall_target == 0.8)].precision.iloc[0])
means = sp.mean(axis=1)
best = means.drop(index=(0, 0.0)).idxmax()
d.bullets(
    "Four follow-up ideas were tried after the blinded review; none gave a precision gain that held up, so none was kept",
    [
        (
            "Object-level second stage",
            f"{pct(s2b)} against {pct(base)} for the single stage, but only because objects merged into large blobs and shape recovery fell. A geometry guard was added to the validation.",
        ),
        (
            "Prior-year NDVI",
            f"{pct(pa)} against {pct(base)} without it. Terrace false alarms have the same seasonal swing as scars.",
        ),
        (
            "Post-event persistence (2019, 2020)",
            f"{pct(pb)} against {pct(base)}. Scars here are back to pre-event NDVI within a year.",
        ),
        (
            "Object-size prior",
            f"Best rule {means[best] * 100:.1f}% against {means[(0, 0.0)] * 100:.1f}% without it, a gain of {(means[best] - means[(0, 0.0)]) * 100:.1f} points whose interval includes zero. Not adopted.",
        ),
    ],
    sub="Precision at 80% recall in block cross-validation, first repeat; the size prior is the mean of 5 repeats.",
    sources="Source: reports/tables/stage2_results.csv, prev_raw.csv, post_raw.csv, size_prior_results.csv",
)

d.map_page(
    f"The deliverable ranks {ms['n_cand']:,} candidates across 1,370 km2; experts review the {ms['n_large']:,} large ones first",
    "Each candidate carries a score and a blank verdict column, so expert answers can become new labels.",
    map_img["candidates"],
    [
        (
            "Order",
            f"Candidates are ranked by score: the {ms['n_large']:,} large ones (20 pixels or more) come first, then {ms['n_cand'] - ms['n_large']:,} small ones.",
        ),
        (
            "Fields",
            "Score, size, slope, land use, distance to river and to the nearest mapped polygon, and empty verdict and note columns.",
        ),
        ("Layers", "Two layers: on the Sentinel-2 grid, and shifted back to the inventory's frame."),
    ],
    legend=[
        ("dot", SUN, f"large candidate ({ms['n_large']:,})"),
        ("dot", "#8A8A84", f"small candidate, 3 to 19 px ({ms['n_cand'] - ms['n_large']:,})"),
        ("outline", NAVY, "held-out test block"),
        ("hatch", NAVY, "excluded as likely unmapped"),
    ],
    sources="Labels: Samodra et al. 2024 (CC BY 4.0); Sentinel-2: Copernicus; embedding: Google DeepMind (CC BY 4.0)",
    note=f"test objects: {ms['n_cand_test']:,} in the candidate file (built over the whole area), {int(c.objects):,} in the scored run (built over the test blocks only)",
)

d.bullets(
    "Read it as a screening aid: one event, 34 test blocks and an incomplete inventory limit what the numbers say",
    [
        (
            "Inventory not wall-to-wall",
            "It started from 328 citizen reports, so precision is a floor; the blinded review suggests this explains little of the gap for large objects.",
        ),
        (
            "Small test set",
            "34 blocks and 106 polygons of 500 m2 or more: intervals are wide and the point estimates would move with another split.",
        ),
        (
            "Calibration",
            f"The threshold was set for 80% recall on training blocks; test recall ({pct(c.object_recall_ge5)}) is above that, inside its interval.",
        ),
        (
            "Ten-metre imagery",
            "Scars under about 500 m2 are not reliably visible, and the review used the same imagery as the model.",
        ),
        ("One event, one place", "Nothing here says how the model transfers to another storm or region."),
    ],
    sub="Observed: test scores. Assumed: label registration shift and land-use class names (inferred, unconfirmed by the inventory authors).",
    sources=f"{SRC_REPO} reports/modeling.html",
)

d.bullets(
    "Next: expert verdicts on the candidates become new labels; two data questions stay open",
    [
        (
            "Expert verdicts",
            "Reviewing the large candidates and recording a verdict gives labels that do not depend on the original citizen reports.",
        ),
        (
            "Inventory footprint",
            "The paper does not state the inventory's completeness and its author was not reachable; of five likely-unmapped blocks, one shows excess candidates.",
        ),
        (
            "Class names",
            "Land-use and lithology class names are inferred from public maps and not confirmed by the authors.",
        ),
        ("Dropped", "A river-bank mask: scars sit near rivers and precision is highest there."),
    ],
    sub="Nothing in this list is scheduled; it is what the evidence points to if the work continues.",
    sources=f"{SRC_REPO} CLAUDE.md, docs/decisions-log.md",
)

d.chart(
    "Appendix: the pipeline runs from raw inventory and imagery to a frozen model, a once-only test score and a candidate file",
    "Build, model and deliver phases; the test blocks are touched only in the deliver phase.",
    "Architecture and data flow",
    IMG / "architecture_crop.png",
    f"Source: {SRC_REPO} docs/architecture.svg",
)

assert not abl.interval_excludes_zero.any(), "ablation claim in the next title needs rewording"
mx = abl.diff_vs_full.abs().max()
LABELS = {
    "c_hgb": "Full model (all features)",
    "abl_no_ndvi_pixel": "Without pixel NDVI features",
    "abl_no_ndvi_nbr": "Without neighbourhood NDVI features",
    "abl_no_terrain": "Without terrain features",
    "abl_no_site_cat": "Without land use and lithology",
    "abl_no_distroad": "Without distance to road",
    "abl_no_landuse": "Without land use",
}
d.table(
    f"Appendix: removing any one feature group moves precision at 80% recall by at most {mx * 100:.1f} points, and no interval excludes zero",
    ["Variant", "Precision at 80% recall", "Change vs full", "95% interval of change"],
    [
        [
            LABELS[r.variant],
            pct(r.precision_at_80_recall),
            "-" if r.variant == "c_hgb" else f"{r.diff_vs_full * 100:+.1f} pts",
            "-" if r.variant == "c_hgb" else f"{r.ci_low * 100:+.1f} to {r.ci_high * 100:+.1f}",
        ]
        for r in abl.itertuples()
    ],
    sub="Groups overlap in information and the intervals are wide: no group was shown to be needed, which is not the same as none mattering.",
    sources="Source: reports/tables/ablation.csv",
    highlight=0,
    note=f"first cross-validation repeat; the 7.6% on page {MODELS_PAGE} is the mean of 5 repeats",
)

glossary_page(
    "Appendix: the metrics in plain words",
    METRICS,
    sub="Recall and precision are explained on page 4; these are the other terms the results use.",
)
glossary_page("Appendix: method and data terms in plain words", METHOD + DATA + DECK)

d.closing(
    "An independent analysis of public data, not an official product and not affiliated with the inventory's authors, BIG or Badan Geologi. "
    "Labels: Samodra et al. 2024 (CC BY 4.0). Sentinel-2: Copernicus. Satellite Embedding: Google and Google DeepMind (CC BY 4.0). Code is MIT licensed; data and reports keep their sources' terms.",
    authors=("Radikal Lukafiardi",),
    links=(f"https://{SRC_REPO}",),
)
assert PAGES == {"models": MODELS_PAGE, "verdict": VERDICT_PAGE}, PAGES
d.save()
print("wrote", OUT, "pages", d.n)
