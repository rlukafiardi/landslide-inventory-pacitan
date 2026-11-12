# Phase 1 rebuild from raw (the 2017 embedding is no longer part of it; the POC scripts need `python -m src.ingestion.download_stack emb2017`): grid check, feature stack (resumable), range checks, labels, test carve (once).
FDIR = data/spatial/mendeley_vrtx3w6mjd
RUN = set -a && . ./.env && set +a && uv run python -W ignore -m

.PHONY: data
data:
	$(RUN) src.data.check_grid $(FDIR)
	for L in emb2018 ndvi; do $(RUN) src.ingestion.download_stack $$L; done
	$(RUN) src.data.check_stack
	$(RUN) src.data.labels $(FDIR)
	test -f data/interim/test_blocks.json || ( $(RUN) src.data.split )

# Phase 3: training feature table (97 features, one code path with full-area prediction), then the leakage and correctness checks.
.PHONY: features
features:
	$(RUN) src.features.build
	$(RUN) src.features.check

# Phase 4: object-metric and fold checks on synthetic scenes and the real split; the random-vs-block leak measurement.
.PHONY: validation
validation:
	$(RUN) src.data.polygon_ids $(FDIR)
	$(RUN) src.modeling.check_metrics
	$(RUN) src.modeling.leak_check

# Phase 5: tuning, five out-of-fold repeats, evaluation, summary, final fit (about 1 hour; needs `make data features validation` first).
.PHONY: model
model:
	$(RUN) src.modeling.tune
	for s in 0 1 2 3 4; do $(RUN) src.modeling.oof $$s; done
	$(RUN) src.modeling.evaluate_oof
	$(RUN) src.modeling.summarize
	$(RUN) src.modeling.final

# Phase 6 rehearsal on training blocks (no test data). The real test score is run once by hand, see README: LANDSLIDE_FINAL_EVAL=1 ... evaluate_test.
.PHONY: rehearsal
rehearsal:
	$(RUN) src.modeling.check_objects
	$(RUN) src.modeling.evaluate_test --rehearsal

# Phase 7: candidate polygons, overview map, report (after the test score).
.PHONY: deliver
deliver:
	$(RUN) src.delivery.candidates
	$(RUN) src.delivery.overview_map
	$(RUN) src.analysis.modeling_report
