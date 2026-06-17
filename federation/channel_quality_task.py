"""Channel quality prediction task for the FL gossip simulation.

Each UAV generates its own local dataset by sampling positions around itself
and computing SINR via the probabilistic LoS path-loss model in coms.py.

Features  (N_FEATURES = 4):
  dx / ENV_WIDTH   — normalised horizontal offset
  dy / ENV_HEIGHT  — normalised vertical offset
  d2d / radius     — normalised 2-D distance
  P_LoS            — line-of-sight probability

Label:
  SINR_dB / SINR_SCALE  — normalised to roughly [-1, 1]
"""

import numpy as np
import torch

from coms import air_to_ground_path_loss, prob_los

N_FEATURES = 4
SINR_SCALE = 50.0   # dB, normalises SINR to ≈[-1, 1]


def make_channel_dataset(uav_pos, n_samples: int, cfg, sample_radius: float = None):
    """Local channel-quality dataset for a UAV at uav_pos.

    Args:
        uav_pos:       [x, y] position of the UAV
        n_samples:     number of training samples
        cfg:           Config instance (provides FREQUENCY, TRANSMIT_POWER, etc.)
        sample_radius: radius (m) within which user positions are sampled;
                       defaults to min(ENV_WIDTH, ENV_HEIGHT) / 4

    Returns:
        list with one (X, y) tensor batch, compatible with Client.train()
    """
    if sample_radius is None:
        sample_radius = min(cfg.ENV_WIDTH, cfg.ENV_HEIGHT) / 4.0

    h_uav = getattr(cfg, 'UAV_HEIGHT', 100.0)
    freq  = cfg.FREQUENCY
    P_tx  = cfg.TRANSMIT_POWER
    N_0   = cfg.NOISE_POWER

    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii  = sample_radius * np.sqrt(np.random.uniform(0, 1, n_samples))
    dx     = radii * np.cos(angles)
    dy     = radii * np.sin(angles)

    X_rows, y_rows = [], []
    for k in range(n_samples):
        d2d  = max(float(np.hypot(dx[k], dy[k])), 1.0)
        pl   = air_to_ground_path_loss(d2d, h_uav, freq)
        sinr = (P_tx - pl - N_0) / SINR_SCALE
        p    = prob_los(d2d, h_uav)
        X_rows.append([dx[k] / cfg.ENV_WIDTH,
                        dy[k] / cfg.ENV_HEIGHT,
                        d2d   / sample_radius,
                        p])
        y_rows.append([sinr])

    X = torch.tensor(X_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.float32)
    return [(X, y)]


def make_global_test_set(n_samples: int, cfg, sample_radius: float = None):
    """Shared test set sampled uniformly across the whole environment.

    Used to evaluate global prediction quality independently of any UAV's
    local coverage area.
    """
    if sample_radius is None:
        sample_radius = min(cfg.ENV_WIDTH, cfg.ENV_HEIGHT) / 4.0

    # Use a dummy UAV at centre for dx/dy normalisation
    centre = [cfg.ENV_WIDTH / 2, cfg.ENV_HEIGHT / 2]
    return make_channel_dataset(centre, n_samples, cfg, sample_radius=sample_radius)
