"""Phase 6 report: reports/modeling.html + reports/figures/modeling_*.png, built only from the result tables and the blinded-review answers.

Inputs: reports/tables/{test_results,test_curve,rehearsal_results,rehearsal_curve,test_error_analysis_*,modeling_results,modeling_pairs,
        stage2_results,prev_raw,post_raw,ablation}.csv, data/interim/{review_key,review_answers}.csv (aggregated counts only).
Usage: python -m src.analysis.modeling_report
"""
import base64
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.analysis.html_report import BLUE, GREY, GRID, INK, MUTED, RED, page, plots_html, save_pngs, section, styled, table_html, tile

T = "reports/tables"
AMBER = "#e6a23c"
LABEL = {"c_hgb": "Chosen model (boosting, 97 features)", "simple_logreg": "Simple logistic (dNDVI + factors)", "naive_dBmin5": "Naive threshold on -dB min5"}
COLOR = {"c_hgb": BLUE, "simple_logreg": GREY, "naive_dBmin5": GREY}

test = pd.read_csv(f"{T}/test_results.csv").set_index("model")
oof = pd.read_csv(f"{T}/rehearsal_results.csv").set_index("model")
tcurve, ocurve = pd.read_csv(f"{T}/test_curve.csv"), pd.read_csv(f"{T}/rehearsal_curve.csv")
c = test.loc["c_hgb"]
pct = lambda v: f"{100 * v:.1f}%"
ci = lambda row, k: f"{pct(row[k + '_ci_low'])} to {pct(row[k + '_ci_high'])}"


def recall_at_precision(curve, p=0.30):
    """Recall where the (recall-ordered) curve first falls to precision p, linear between the two nearest curve points."""
    d = curve.sort_values("object_recall_ge5")
    r, q = d.object_recall_ge5.to_numpy(), d.object_precision.to_numpy()
    for i in range(len(r) - 1):
        if q[i] >= p > q[i + 1]:
            return float(r[i] + (q[i] - p) / (q[i] - q[i + 1]) * (r[i + 1] - r[i]))
    return np.nan


r30_test, r30_oof = recall_at_precision(tcurve), recall_at_precision(ocurve)


def image(path, alt):
    """A PNG embedded in the page, so the report is one file."""
    return f'<img alt="{html.escape(alt)}" style="max-width:100%;height:auto" src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}">'

figs = {}
# 1 criterion
f = go.Figure()
rows = [("Object precision", c.object_precision, c.object_precision_ci_low, c.object_precision_ci_high, 0.30),
        ("Object recall (polygons >= 5 px)", c.object_recall_ge5, c.object_recall_ge5_ci_low, c.object_recall_ge5_ci_high, 0.80)]
for y, (name, v, lo, hi, tgt) in enumerate(rows):
    f.add_trace(go.Scatter(x=[100 * v], y=[y], mode="markers+text", text=[pct(v)], textposition="top center", showlegend=False,
                           marker=dict(color=BLUE, size=13), error_x=dict(type="data", symmetric=False, array=[100 * (hi - v)], arrayminus=[100 * (v - lo)], color=BLUE, thickness=2)))
    f.add_trace(go.Scatter(x=[100 * tgt], y=[y], mode="markers+text", text=[f"target {100 * tgt:.0f}%"], textposition="bottom center", showlegend=False,
                           marker=dict(color=INK, size=14, symbol="line-ns-open", line=dict(width=3))))
f.update_xaxes(range=[0, 100], ticksuffix="%", title="test blocks, 34 blocks, 95% interval from resampling whole blocks")
f.update_yaxes(tickvals=[0, 1], ticktext=[r[0] for r in rows], range=[1.7, -0.7])
figs["criterion"] = styled(f, "Recall clears the 80% target; precision stays well below the 30% target", 300)

# 2 workload
f = go.Figure()
ms = list(test.index)
fpt, lo, hi = test.false_per_true, test.false_per_true_ci_low, test.false_per_true_ci_high
f.add_trace(go.Bar(y=[LABEL[m] for m in ms], x=fpt, orientation="h", marker_color=[COLOR[m] for m in ms], text=[f"{v:.0f}" for v in fpt], textposition="inside", insidetextanchor="start", textfont=dict(size=14, color=["white", INK, INK]), showlegend=False,
                   error_x=dict(type="data", symmetric=False, array=(hi - fpt).to_numpy(), arrayminus=(fpt - lo).to_numpy(), color=INK, thickness=1.3)))
f.update_xaxes(range=[0, 55], title="false objects to check per true scar found (95% interval)")
f.update_yaxes(autorange="reversed")
figs["workload"] = styled(f, "About 10 false objects to check per true scar, against about 31 for the baselines", 300)

# 3 curve
f = go.Figure()
for name, d, col, dash in (("Out-of-fold (training blocks)", ocurve, GREY, "solid"), ("Test blocks", tcurve, BLUE, "solid")):
    d = d.sort_values("object_recall_ge5")
    f.add_trace(go.Scatter(x=100 * d.object_recall_ge5, y=100 * d.object_precision, mode="lines+markers", name=name, line=dict(color=col, width=2.5), marker=dict(size=6)))
for name, row, col in (("frozen point, test", test.loc["c_hgb"], BLUE), ("frozen point, out-of-fold", oof.loc["c_hgb"], GREY)):
    f.add_trace(go.Scatter(x=[100 * row.object_recall_ge5], y=[100 * row.object_precision], mode="markers", name=name, showlegend=False,
                           marker=dict(size=15, color="white", line=dict(color=col, width=3))))
f.add_annotation(x=100 * test.loc["c_hgb"].object_recall_ge5, y=100 * test.loc["c_hgb"].object_precision, text="frozen point, test", ax=45, ay=-75, arrowcolor=BLUE, font=dict(color=BLUE))
f.add_annotation(x=100 * oof.loc["c_hgb"].object_recall_ge5, y=100 * oof.loc["c_hgb"].object_precision, text="frozen point, out-of-fold", ax=-95, ay=26, arrowcolor=MUTED, font=dict(color=MUTED))
f.add_hline(y=30, line=dict(color=INK, dash="dash", width=1), annotation_text="precision target 30%", annotation_position="top right")
f.add_vline(x=80, line=dict(color=INK, dash="dash", width=1), annotation_text="recall target 80%", annotation_position="top left")
f.update_xaxes(range=[30, 100], ticksuffix="%", title="object recall, polygons >= 5 px")
f.update_yaxes(range=[0, 55], ticksuffix="%", title="object precision")
figs["curve"] = styled(f, "Both curves cross 30% precision only at about 60% to 70% recall", 420, legend=True)

# 4 review
key, ans = pd.read_csv("data/interim/review_key.csv"), pd.read_csv("data/interim/review_answers.csv")
rv = key.merge(ans, on="id")
tab = pd.crosstab(rv.kind, rv.answer).reindex(["control_true", "false_alarm"])[["scar", "not_scar", "unsure"]]
names = {"control_true": f"Known scars (controls, n = {int(tab.loc['control_true'].sum())})", "false_alarm": f"Large false alarms (n = {int(tab.loc['false_alarm'].sum())})"}
f = go.Figure()
for col, lab, colr in (("scar", "judged a scar", BLUE), ("not_scar", "judged not a scar", GREY), ("unsure", "unsure", AMBER)):
    share = 100 * tab[col] / tab.sum(axis=1)
    f.add_trace(go.Bar(y=[names[k] for k in tab.index], x=share, orientation="h", name=lab, marker_color=colr,
                       text=[f"{int(n)}" if n else "" for n in tab[col]], textposition="inside"))
f.update_layout(barmode="stack", legend=dict(traceorder="normal"))
f.update_xaxes(range=[0, 100], ticksuffix="%", title="share of reviewed objects (counts inside the bars)")
f.update_yaxes(autorange="reversed")
figs["review"] = styled(f, "Reviewers recognise known scars, and almost no large false alarm is a scar", 300, legend=True)

# 5 size of objects (test)
sz = pd.read_csv(f"{T}/test_error_analysis_size_px.csv")
labels = ["3-4 px", "5-9 px", "10-19 px", "20 px or more"]
f = go.Figure(go.Bar(x=labels, y=100 * sz.precision, marker_color=[GREY, GREY, GREY, BLUE], text=[f"{100 * p:.0f}%<br>{n:,} objects" for p, n in zip(sz.precision, sz.objects)],
                     textposition="outside", showlegend=False))
f.update_yaxes(range=[0, 30], ticksuffix="%", title="object precision")
f.update_xaxes(title="predicted object size (10 m pixels)")
figs["size"] = styled(f, "Small objects are almost always false alarms; objects of 20 pixels or more are right about one time in five", 400)

# 6 misses by size class
f = go.Figure()
classes = [("recall_<5 px", "under 500 m2"), ("recall_5-20 px", "500 to 2,000 m2"), ("recall_>20 px", "over 2,000 m2")]
f.add_trace(go.Bar(x=[n for _, n in classes], y=[100 * c[k] for k, _ in classes], marker_color=[GREY, BLUE, BLUE], showlegend=False,
                   text=[pct(c[k]) for k, _ in classes], textposition="inside", insidetextanchor="start", textfont=dict(size=14, color=[INK, "white", "white"]),
                   error_y=dict(type="data", symmetric=False, array=[100 * (c[k + "_ci_high"] - c[k]) for k, _ in classes], arrayminus=[100 * (c[k] - c[k + "_ci_low"]) for k, _ in classes], color=INK, thickness=1.3)))
f.add_hline(y=80, line=dict(color=INK, dash="dash", width=1), annotation_text="80% target (applies to polygons of 500 m2 and over)", annotation_position="top left")
f.update_yaxes(range=[0, 115], ticksuffix="%", title="share of polygons detected")
f.update_xaxes(title="inventory polygon area (95% interval)")
figs["misses"] = styled(f, "Large scars are found almost always; the misses are small polygons", 400)

# land use of the test-block objects, with class names inferred from the BIG RBI25K layers (src/analysis/landuse_legend.py)
leg = pd.read_csv(f"{T}/landuse_legend.csv").set_index("code").proposed_name.str.replace(r" \(.*", "", regex=True)
lu = pd.read_csv(f"{T}/test_error_analysis_landuse.csv").set_index("landuse")
lm = pd.read_csv(f"{T}/test_error_analysis_miss_by_landuse.csv").set_index("landuse")
lu_tab = pd.DataFrame({"land-use class (inferred name)": [f"{k}: {leg[k]}" for k in lu.index], "objects": lu.objects.astype(int).to_numpy(),
                       "true": lu.true_objects.astype(int).to_numpy(), "object precision": [pct(v) for v in lu.precision],
                       "share of false alarms": [pct(v) for v in lu.share_of_false_alarms], "share of test area": [pct(v) for v in lu.share_of_domain_area],
                       "polygons >= 5 px detected": [f"{int(lm.detected.get(k, 0))} of {int(lm.polygons.get(k, 0))}" for k in lu.index]})
paddy = lu.loc[[k for k in (10, 11) if k in lu.index]]
lu_text = (f"<h3>By land-use class (test blocks)</h3>{table_html(lu_tab)}"
           f"<p>The false alarms follow the share of the area rather than a class: paddy (codes 10 and 11) holds {pct(paddy.share_of_false_alarms.sum())} of the false alarms and "
           f"{pct(paddy.share_of_domain_area.sum())} of the area; dry field (13) {pct(lu.loc[13].share_of_false_alarms)} against {pct(lu.loc[13].share_of_domain_area)}. "
           f"Precision is lowest in class 10 ({int(lu.loc[10].true_objects)} true of {int(lu.loc[10].objects)}) and highest in plantation (6, {pct(lu.loc[6].precision)}). "
           "The class names are inferred from the BIG RBI 1:25,000 layers (2022 edition) by overlay and are not confirmed by the inventory authors (reports/tables/landuse_legend.csv); the classes have 7 to 318 objects, so read the shares, not single precision values.</p>")

# tables
res = pd.DataFrame({"model": [LABEL[m] for m in test.index], "object recall >= 5 px": [f"{pct(test.loc[m, 'object_recall_ge5'])} ({ci(test.loc[m], 'object_recall_ge5')})" for m in test.index],
                    "strict recall (IoU >= 0.3)": [pct(test.loc[m, "strict_recall_ge5"]) for m in test.index],
                    "object precision": [f"{pct(test.loc[m, 'object_precision'])} ({ci(test.loc[m], 'object_precision')})" for m in test.index],
                    "flagged share of area": [pct(test.loc[m, "flagged_share"]) for m in test.index],
                    "objects to review": [f"{int(test.loc[m, 'objects']):,}" for m in test.index]})
# Counts behind the two rates. Recall counts polygons, precision counts predicted objects: two units (one polygon can be touched by several
# objects), and background has no objects to count, so a confusion matrix does not apply; this table gives each unit its own counts.
detected = (test.object_recall_ge5 * test.polygons_ge5).round().astype(int)
counts = pd.DataFrame({"model": [LABEL[m] for m in test.index],
                       "polygons >= 500 m2 detected": [f"{detected[m]} of {int(test.loc[m, 'polygons_ge5'])}" for m in test.index],
                       "polygons missed": [int(test.loc[m, "polygons_ge5"]) - detected[m] for m in test.index],
                       "objects flagged": [f"{int(test.loc[m, 'objects']):,}" for m in test.index],
                       "objects touching a polygon": [int(test.loc[m, "true_objects"]) for m in test.index],
                       "objects touching none (false alarms)": [f"{int(test.loc[m, 'objects'] - test.loc[m, 'true_objects']):,}" for m in test.index]})
mr = pd.read_csv(f"{T}/modeling_results.csv")
mr = mr[mr.recall_target == 0.8].copy()
names_map = {"naive_dB": "Naive threshold on -dB", "naive_dBmin5": "Naive threshold on -dB min5", "floor_logreg": "Site factors only (logistic)", "simple_logreg": "Simple logistic (dNDVI + factors)",
             "emb_logreg": "Embedding + factors (logistic)", "all_logreg": "All features (logistic)", "a_hgb": "dNDVI + factors (boosting)", "b_hgb": "Embedding + factors (boosting)", "c_hgb": "All features (boosting), chosen"}
hist = pd.DataFrame({"model": [names_map[m] for m in mr.config], "reached 80% recall": [f"{int(r)} of {int(n)}" for r, n in zip(mr.reached, mr.repeats)],
                     "object precision (95% interval)": [f"{pct(p)} ({pct(a)} to {pct(b)})" if p == p else "not reached" for p, a, b in zip(mr.precision, mr.ci_low, mr.ci_high)],
                     "flagged share": [pct(v) if v == v else "-" for v in mr.flagged_share], "weighted pixel AP": [f"{v:.2f}" for v in mr.weighted_pixel_AP]}).sort_values("model")
pairs = pd.read_csv(f"{T}/modeling_pairs.csv")
gate = pairs[(pairs.config == "c_hgb") & (pairs.baseline == "simple_logreg")].iloc[0]


def rep0(path, a, b):
    d = pd.read_csv(path)
    d = d[(d.recall_target == 0.8) & (d.repeat == 0)].set_index("config").precision_at_target
    return d[a], d[b]


b_obj, b_s2 = pd.read_csv(f"{T}/stage2_results.csv").query("recall_target == 0.8 and repeat == 0").set_index("variant").precision[["obj_pmax", "s2_base"]]
a1, a2 = rep0(f"{T}/prev_raw.csv", "abl_plus_ndvi_prev", "c_hgb")
p1, p2 = rep0(f"{T}/post_raw.csv", "abl_plus_ndvi_post", "c_hgb")
tried = pd.DataFrame([
    ["Object-level second stage (re-score each candidate object)", f"{pct(b_s2)} (plain candidates {pct(b_obj)}, single stage {pct(oof.loc['c_hgb'].object_precision)})", "Higher than the single stage only because objects merged into big blobs: strict recall fell from 64% to 31%. A geometry guard was added to the validation design"],
    ["Prior-year NDVI (separate cyclic farmland)", f"{pct(a1)} vs {pct(a2)} without", "Terrace false alarms have the same normal seasonal swing as scars"],
    ["Post-event persistence, 2019 and 2020 (scars stay bare)", f"{pct(p1)} vs {pct(p2)} without", "Scars here are back to pre-event NDVI within a year"]],
    columns=["tried after the review", "object precision at 80% recall (repeat 0)", "why it did not help"])

n_test_poly = int(c.polygons_ge5)
sections = [
    section("Result on the held-out blocks", f"Recall target met ({pct(c.object_recall_ge5)}, interval {ci(c, 'object_recall_ge5')}); precision target not met ({pct(c.object_precision)}, interval {ci(c, 'object_precision')}). "
            "The criterion was fixed before any model and is unchanged.",
            "<div>@@criterion@@</div>" + table_html(res) +
            f"<p>Test blocks: {int(c.polygons_all)} inventory polygons, {n_test_poly} of 500 m2 or more. The model flagged {int(c.objects):,} objects, {int(c.true_objects)} of them true. "
            "One score on the frozen model and the operating point chosen from the out-of-fold repeats; nothing was tuned on these blocks.</p>"
            "<h3>The counts behind the two rates</h3>" + table_html(counts) +
            "<p>There is no confusion matrix because the two rates count different things: recall counts polygons, precision counts predicted objects, and one polygon "
            "can be touched by several objects (the chosen model has more true objects than detected polygons). Background has no objects to count as true negatives. "
            "The polygons detected are recall times the polygons of 500 m2 or more (rounded to a whole polygon).</p>"),
    section("Where the model flags", "The candidates are spread over the steep interior; on the test blocks the reference polygons are mostly covered by true objects, with scattered small false alarms around them.",
            image("reports/figures/candidates_map.png", "Overview map of the candidates") +
            "<p>Overview of all candidates (the GeoPackage deliverable). Test blocks are outlined.</p>" +
            image("reports/figures/test_blocks_map.png", "Six test blocks: reference polygons, true objects and false alarms") +
            "<p>The six test blocks with the most reference polygons (one rule, fixed before looking at outcomes; not a hand-picked set). "
            "In block 613 the long blue object along the river counts as true only because one end touches a polygon: the \"touches\" rule favours long objects, "
            "which is why strict recall is reported next to precision.</p>"),
    section("Against the baselines", f"At similar recall the chosen model needs about a third of the review effort and recovers the polygon shape far better (strict recall {pct(test.loc['c_hgb', 'strict_recall_ge5'])} against "
            f"{pct(test.loc['simple_logreg', 'strict_recall_ge5'])} and {pct(test.loc['naive_dBmin5', 'strict_recall_ge5'])}).",
            "<div>@@workload@@</div><p>This reviewer-workload view is a secondary framing added after the result; the criterion itself is unchanged.</p>"),
    section("Precision against recall", f"Recall can be traded for precision, but 30% precision comes only at about {100 * r30_oof:.0f}% recall out of fold and about {100 * r30_test:.0f}% on the test blocks. "
            "The test curve is descriptive (106 polygons); no operating point was chosen from it.", "<div>@@curve@@</div>"),
    section("What the false alarms are", "A blinded visual review of 40 objects (30 large false alarms, 10 known scars as controls, shuffled, judged on Sentinel-2 true colour) shows that most large false alarms are bare ground that is not a landslide.",
            "<div>@@review@@</div><p>Reviewer notes on the false alarms: terrace farming (7), river bank (2), coastal wave erosion (1), road (1). Two known scars were also called unsure (terrace farming, mining area), "
            "so a few inventory polygons are hard to tell from look-alikes at 10 m. The review covers only objects of 20 pixels or more (24% of the false alarms); n = 30.</p>"
            "<div>@@size@@</div>" + lu_text),
    section("What it misses", "Misses are the small polygons: nearly every polygon over 2,000 m2 is found; polygons under 500 m2 are mostly not found and are outside the target.", "<div>@@misses@@</div>"),
    section("How the model was chosen", f"Five repeats of block cross-validation on the training blocks, compared at 80% object recall. The chosen model beat the best simple baseline by "
            f"{100 * gate.mean_diff_precision:.1f} points (interval {100 * gate.ci_low:+.1f} to {100 * gate.ci_high:+.1f}, {gate.repeats_won} repeats won).",
            table_html(hist) + "<h3>Tried after the blinded review, none kept</h3>" + table_html(tried)),
    section("Limits", "Read the numbers as a screening aid, not a map of every landslide.",
            "<ul><li><b>Inventory not wall-to-wall.</b> It started from 328 citizen reports, so precision is a floor. The blinded review shows this explains little of the gap for large objects.</li>"
            "<li><b>Small test set.</b> 34 blocks and 106 polygons of 500 m2 or more; the intervals are wide and the point estimates will move with another split.</li>"
            "<li><b>Calibration.</b> The threshold comes from models fitted on 80% of the training blocks and is applied to a refit on all of them; test recall (86.8%) is above the 80% the threshold was chosen for, inside its interval.</li>"
            "<li><b>Label registration.</b> Polygons sit about 12.6 m from the Sentinel-2 grid (shift estimated from the data, cause unverified) and were shifted onto it; predictions shift back by (-8.4 m, -9.4 m) to compare with the manual inventory.</li>"
            "<li><b>Ten-metre imagery.</b> Scars under about 500 m2 are not reliably visible; the review used the same imagery as the model.</li>"
            "<li><b>One event, one place.</b> Nothing here says how it transfers to another storm or region.</li></ul>"),
    section("Reproduce", "Commands from the project root (the Earth Engine login is in .env).",
            "<pre>make data features validation\nset -a &amp;&amp; . ./.env &amp;&amp; set +a\nuv run python -m src.modeling.tune\nfor s in 0 1 2 3 4; do uv run python -m src.modeling.oof $s; done\n"
            "uv run python -m src.modeling.evaluate_oof &amp;&amp; uv run python -m src.modeling.summarize\nuv run python -m src.modeling.final\n"
            "uv run python -m src.modeling.evaluate_test --rehearsal\nLANDSLIDE_FINAL_EVAL=1 uv run python -m src.modeling.evaluate_test   # once\nuv run python -m src.analysis.modeling_report</pre>"),
]
tiles = [tile(pct(c.object_recall_ge5), "object recall on test blocks (target 80%: met)"), tile(pct(c.object_precision), "object precision (target 30%: not met)"),
         tile(f"{c.false_per_true:.0f}", "false objects to check per true scar (baselines: 31)"), tile(pct(c.flagged_share), "of the test area flagged for review")]
Path("reports/figures").mkdir(parents=True, exist_ok=True)
save_pngs(figs, Path("reports/figures"), "modeling")
body = page("Landslide detection, Cyclone Cempaka (Pacitan): evaluation on held-out blocks",
            "Phase 6 of the project: the frozen model scored once on 34 held-out 3 km blocks (<code>reports/tables/test_*.csv</code>). Labels: Samodra et al. 2024 (CC BY 4.0).", tiles, sections)
for k, v in plots_html(figs).items():
    body = body.replace(f"@@{k}@@", v)
Path("reports/modeling.html").write_text(body)
print(f"reports/modeling.html written ({len(body) / 1e6:.1f} MB), figures: {', '.join(figs)}")
