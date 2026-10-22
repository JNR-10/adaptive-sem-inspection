"""Current calibrated-anomaly-score comparator."""

from seminspect.policies.allocator import Observation, Policy


def anomaly_score_policy() -> Policy:
    return Policy("anomaly_score", lambda current, previous: current.calibrated_probability)
