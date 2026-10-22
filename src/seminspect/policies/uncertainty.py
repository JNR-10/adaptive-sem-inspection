"""Cost-normalized entropy and margin uncertainty comparators."""

from seminspect.policies.allocator import Observation, Policy, entropy


def entropy_uncertainty_policy() -> Policy:
    return Policy("uncertainty_entropy", lambda current, previous: entropy(current.calibrated_probability))


def margin_uncertainty_policy() -> Policy:
    return Policy("uncertainty_margin", lambda current, previous: 1.0 - max(current.calibrated_probability, 1.0 - current.calibrated_probability))
