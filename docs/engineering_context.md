# Engineering context

## The practical question underneath the benchmark

Inspection systems routinely face a capacity problem: a batch may contain many
regions that are inexpensive to survey, but only a fraction can receive a more
informative or costly review. A useful vision system must therefore be judged
on more than a ranking score. It needs to preserve sensitivity, control the
review burden created by false positives, and make its allocation decision
auditable.

This repository models one small part of that problem. Given a set of SEM image
regions and a fixed information budget, it asks which region should receive the
next simulated resolution level. It is deliberately relevant to semiconductor
inspection engineering without pretending to be a fab tool:

- **Sensitivity:** can additional information recover an ambiguous defect?
- **Review capacity:** which candidates are worth spending the next increment
  of attention on?
- **False-positive discipline:** are thresholds calibrated before the final
  evaluation rather than tuned on it?
- **Traceability:** can an engineer reconstruct the data split, feature
  extraction, policy threshold, and source revision that produced a result?

## What the project contributes

The main contribution is not a claim of a new best detector. It is an
evaluation-ready implementation of a **decision layer on top of a visual
anomaly detector**:

1. observe each region at low quality;
2. estimate anomaly evidence and uncertainty;
3. predict whether one more quality step is likely to reduce decision risk;
4. allocate a limited batch budget deterministically; and
5. evaluate the result against cost-matched baselines under a sealed test.

That decomposition is useful in real engineering discussions because it makes
failure modes visible. A strong detector may still be paired with a weak policy;
a high-recall change may be a false-positive trade-off; and an attractive
simulation result may not transfer to true tool operation.

## What it does not claim

The project has no wafer coordinates, process context, scan recipe, electron
dose, physical tool timing, die-level aggregation, yield labels, or production
feedback loop. Its normalized information cost must not be converted into
wafer-per-hour, yield, cost-of-ownership, defect excursion, or commercial
inspection claims.

The data are public SEM image regions, and Q0–Q3 are simulated observations.
The code should therefore be read as a rigorous research prototype for
multi-resolution visual decision-making, not as a deployed semiconductor
inspection system.

## Why the negative primary result is valuable

The sealed comparison found that the learned VoR router did not beat a
cost-normalized uncertainty baseline. In an engineering setting, that is not
an embarrassment to edit away; it is a decision boundary. It says the current
synthetic-pair training target is not yet a sufficient replacement for the
simpler baseline on real MIIC anomalies.

The result still demonstrates several useful capabilities: SEM-specific visual
features, anomaly maps, quality-aware calibration, leakage-safe data handling,
cost-matched policies, and an auditable final-test lock. The next research
iteration should improve the router target or add realistically paired
observations before making a stronger utility claim.
