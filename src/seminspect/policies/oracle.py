"""Privileged non-deployable myopic gain diagnostic."""

from seminspect.policies.allocator import Policy


def oracle_gain_policy() -> Policy:
    def signal(current, previous):
        if current.oracle_gain is None:
            raise ValueError("Oracle policy requires true next-step gain")
        return current.oracle_gain

    return Policy("oracle_gain", signal, stop_on_nonpositive=True, non_deployable=True)
