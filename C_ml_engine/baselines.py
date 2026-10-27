"""
baselines.py - simple-rule dispatchers to compare against MaskablePPO.
Each policy takes the environment (to read action_masks and its state) and
returns an action index. They only use information the agent could know:
masks, believed travel times (obs) and the request queue.
"""
import numpy as np

from bloodchain_env import HOSPITALS, N_VANS


def _first_free_van(mask):
    for v in range(N_VANS):
        if mask[1 + v * len(HOSPITALS): 1 + (v + 1) * len(HOSPITALS)].any():
            return v
    return None


def random_policy(env, rng):
    """Random among VALID actions (masked)."""
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def nearest_first(env, rng=None):
    """Send the first free van to the hospital with pending requests and the
    shortest believed travel time."""
    mask = env.action_masks()
    v = _first_free_van(mask)
    if v is None:
        return 0
    best, best_t = 0, 1e9
    for k, h in enumerate(HOSPITALS):
        a = 1 + v * len(HOSPITALS) + k
        if mask[a]:
            t = env._path_steps(env._path(0, h))
            if t < best_t:
                best, best_t = a, t
    return best


def most_urgent_first(env, rng=None):
    """Send the first free van to the hospital whose most urgent request has the
    earliest deadline (urgent before routine). This is also Fallback level L1."""
    mask = env.action_masks()
    v = _first_free_van(mask)
    if v is None:
        return 0
    best, key = 0, None
    for k, h in enumerate(HOSPITALS):
        a = 1 + v * len(HOSPITALS) + k
        if mask[a]:
            pend = env._pending(h)
            kk = (not any(q.urgent for q in pend), min(q.deadline for q in pend))
            if key is None or kk < key:
                best, key = a, kk
    return best


BASELINES = {"random": random_policy, "nearest": nearest_first, "urgent": most_urgent_first}
