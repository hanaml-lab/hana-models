# hana-models (pipeline)

The glue of the monorepo: `hana_models.pipeline` runs distill-kit's data steps, calibrates small-lm's router on the
student's results and drives the serve-bench binary. `hana_models.student` (a rule-based extractor) and
`hana_models.simserver` (a simulated OpenAI-compatible server) are offline stand-ins so the pipeline can be run and
tested on a laptop. See the [top-level README](../../README.md).
