"""Learned local Value-of-Resolution policies."""

from seminspect.policies.allocator import Policy


def predicted_gain_policy(name: str = "vor_mlp") -> Policy:
    def signal(current, previous):
        if current.predicted_gain is None:
            raise ValueError(f"{name} observation is missing predicted_gain")
        return current.predicted_gain

    return Policy(name, signal, stop_on_nonpositive=True)
