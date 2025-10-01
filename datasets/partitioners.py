"""Partitioning and drift utilities for federated simulations.

This keeps a minimal surface: deterministic per-client seeds and optional drift hooks.
"""

from __future__ import annotations
import random
import numpy as np


def seed_for_client(client_id: int, alpha: float = 0.5) -> None:
    """Derive and set deterministic RNG seeds per client.

    Parameters
    - client_id: stable identifier for the client
    - alpha: kept for API parity with non-IID Dirichlet; unused in this stub
    """
    base = 1337 + int(client_id) * 1009
    np.random.seed(base)
    random.seed(base)


def maybe_apply_drift(step: int, drift_steps: int) -> None:
    """Optionally perturb RNG to simulate temporal drift.

    If drift_steps > 0, reseed periodically using step to change sequences.
    """
    if drift_steps and drift_steps > 0 and step > 0 and (step % drift_steps == 0):
        seed = 4242 + step
        np.random.seed(seed)
        random.seed(seed)


