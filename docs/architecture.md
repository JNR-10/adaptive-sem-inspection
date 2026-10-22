# Architecture and design choices

## The decision being studied

The system processes a batch of SEM image regions under a fixed normalized
information budget. Every region begins at Q0, a low-information survey
observation. The policy can leave it there or advance it one step at a time to
Q1, Q2, or Q3. Higher quality gives the detector more information, but consumes
more of the batch budget.

This is intentionally a **sequential allocation** problem, not a claim that a
software policy can command an SEM. The resolution ladder is simulated from
public 512×512 SEM images: effective sides are 56, 112, 224, and 448 pixels,
then each observation is restored to the encoder input size. Its costs are
information-cost assumptions: Q0=0.015625, Q1=0.0625, Q2=0.25, Q3=1.0.

## Four separable parts

| Part | What it does | What it deliberately does not see |
| --- | --- | --- |
| Observation simulator | Creates deterministic Q0–Q3 views and optional stress views. | Labels, masks, policy outcome. |
| Detector | Uses a frozen DINOv2 patch/CLS representation and a quality-specific normal memory bank to form anomaly evidence. | Test labels during fitting. |
| Router | Estimates clipped next-step cost-sensitive risk reduction from current state features. | Labels, masks, future features, acquisition cost, remaining global budget. |
| Allocator | Advances the best available candidate under a fixed budget, then emits final quality/score. | Ground truth. |

Keeping those pieces separate is the key implementation choice. It prevents a
route-selection model from accidentally using information that would not be
available at inspection time, and it allows the learned policy to be compared
fairly with score and uncertainty baselines.

## Detector: normal memory rather than defect-class training

The primary detector is a frozen `dinov2_vits14_reg` backbone. For each quality
level, normal training images contribute patch and CLS embeddings to a separate
memory bank. A test patch is anomalous when it is dissimilar to its nearest
normal-memory representation; patch evidence is pooled to an image score and
then calibrated per quality level.

Why use separate banks? A low-resolution normal pattern should not be called a
defect just because it differs from a high-resolution normal pattern. This
choice is especially important in periodic SEM structures where legitimate
texture changes dramatically with observation quality.

## Router: Value of Resolution

The router does not replace the detector. It predicts the incremental value of
the next observation based on 53 current-state features:

- calibrated probability and raw detector score;
- image-quality statistics;
- patch anomaly-map statistics;
- DINO CLS embedding compressed with a development-fit PCA;
- one-hot current quality and history/change features.

Training targets are paired synthetic MIIC observations: for a known label,
how much does the cost-sensitive risk change from the current quality to the
next? Ridge and multi-seed MLP routers are trained strictly on development
partitions, then the selected MLP ranks candidate next steps. The allocator is
deterministic so any policy difference is traceable to its information, not
random scheduling.

## Calibration and evaluation

Policy-specific thresholds are selected only from a dedicated calibration lot
of normal images. The final decision is `score > threshold`; no test label is
used to tune that threshold. The final test is locked with hashes of the data
manifest, duplicate decisions, configuration, weights, memory banks,
calibrators, router, threshold set, source revision, and final-test IDs.

This is a useful pattern for inspection research: detector feasibility, routing
utility, and final evaluation are different questions and should be auditable
separately.
