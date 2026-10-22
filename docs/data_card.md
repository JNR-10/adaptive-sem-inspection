# Data card

## MIIC

MIIC is the primary benchmark. It is non-commercial research data under CC
BY-NC 4.0. The public `Anomaly_test` archive is a subset of `Anomaly_train`;
the pipeline removes all decoded-pixel matches from detector-reference
candidates before split construction. MIIC's repetitive SEM texture makes
pHash+SSIM form false "near-duplicate" components, so the locked audit uses
decoded-pixel identity only. Reused inpainting-source components that touch
final-test images are quarantined, and the remaining inpainting pairs are
partitioned component-wise. Final test data must not enter model selection.

The benchmark uses real SEM image regions but simulates lower-resolution
observations. Its normalized costs are information-cost assumptions, not SEM
tool measurements.

## Carinthia-S

Carinthia-S is a separate CC BY 4.0 secondary domain-shift analysis. Its class
6 is no visible defect/misalignment and receives image-level label 0 regardless
of nonempty mask pixels. RGB and RGBA masks are converted to luminance after
discarding alpha. One released class-3 JPEG has no final 21 encoded bytes; the
ingestion code permits it only after matching its pinned file SHA-256 and still
requires a 480×480 grayscale decode. It is never mixed into MIIC training.

## Excluded claims

No dataset provides wafer coordinates, fab throughput, scan-time, electron-dose
or yield labels. The project cannot support those claims.
