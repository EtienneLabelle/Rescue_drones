"""Observation shaping utilities shared across training and federation.

Source of truth: replicate the exact vector fed to agents today.
"""

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def build_obs(raw_obs: Any, cfg) -> np.ndarray:
    """Convert a raw per-agent observation into a 1-D vector of length cfg.STATE_DIM.

    The current code path already constructs a flat numpy array of the correct size
    in env.get_state(...) and pads/truncates there. We therefore enforce type/shape and
    return as-is, asserting the configured size.
    """
    vec = np.asarray(raw_obs, dtype=np.float32).ravel()
    assert vec.shape[0] == cfg.STATE_DIM, f"STATE_DIM mismatch: got {vec.shape[0]} vs {cfg.STATE_DIM}"
    return vec


def build_obs_dict(raw_states: Dict[str, Any], cfg) -> Dict[str, np.ndarray]:
    """Apply build_obs to each agent entry of the raw states dict."""
    return {aid: build_obs(obs, cfg) for aid, obs in raw_states.items()}


