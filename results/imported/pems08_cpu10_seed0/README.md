# PeMS08 imported experiment record

This folder preserves the comparison tables from an earlier project run:
`flow_10epoch_batch8_seed0` from `ISY5002-Traffic-Recognition`.

`comparison.csv` is the original result table. `reference_comparison.csv` places
those rows next to the current PEMSD8 flow/seed-0 results. The earlier run used
a different training protocol, so the two groups are reported together only for
reference and are not combined into a single ranking.

Raw weights, prediction arrays and source data are not included here. The
source data came from the same public PeMS08 benchmark, but the historical
training files were not retained and their hashes are unavailable.
`source_snapshot/` records the settings used by the original run; its paths
follow the earlier project layout and are not an entry point for this
repository.
