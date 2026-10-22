# Included result snapshot

These files are copied from the completed locked `core_v1` run so a GitHub
reader can inspect the actual benchmark outcome:

| File | Purpose |
| --- | --- |
| `SEALED_RUN_REPORT.md` | Human-readable final report with the primary table. |
| `primary_endpoint.json` | Machine-readable effect estimate and paired CI. |
| `test_lock.json` | Hash-based provenance snapshot of the sealed run. |
| `metrics.parquet` | Full final metrics table. |
| `figures/` | Selected final-run figures. |

The primary endpoint did not show an improvement for the learned VoR router
over cost-normalized entropy uncertainty. Read `../docs/results.md` before
quoting or comparing these numbers.

The result snapshot is for inspection and reproducibility context. It does not
contain raw MIIC or Carinthia-S data and it must not be interpreted as a
production SEM, yield, throughput, or commercial-tool result.
