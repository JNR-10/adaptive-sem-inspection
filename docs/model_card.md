# Model and policy card

`core_v1` uses frozen `dinov2_vits14_reg` at source commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The final run records the exact
downloaded weight SHA-256. It builds independent Q0–Q3 normal patch and CLS
memory banks to avoid treating ordinary resolution loss as anomalous.

The separately named `core_v1_resnet` configuration is a feasibility fallback,
not a model-selection option. It is permitted only when the pre-test DINOv2
resolution-feasibility gate fails, and it rebuilds all detector, calibration,
router, threshold, and test-lock artifacts in an isolated runtime root.

The router is not a defect detector. It predicts the clipped next-resolution
risk reduction using only current and historical observations. Its training
labels come from paired synthetic MIIC images, so synthetic-to-real transfer is
the central limitation. Oracle gain is a privileged diagnostic only and must
never be shown as deployable.
