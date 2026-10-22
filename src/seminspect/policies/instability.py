"""Resolution-instability comparator."""

from seminspect.policies.allocator import Policy, entropy


def instability_policy() -> Policy:
    def signal(current, previous):
        return entropy(current.calibrated_probability) if previous is None else abs(current.calibrated_probability - previous.calibrated_probability)

    return Policy("resolution_instability", signal)
