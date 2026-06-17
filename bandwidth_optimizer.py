"""Bandwidth allocation optimizer for FL + DRL co-training.

Closed-form B_FL* optimizer derived from a Lagrangian that balances
sum-rate throughput (weighted by α) against FL convergence time
(weighted by 1-α):

    B_FL* = sqrt( (1-α) · D  /  (α · log2(1+SINR_users) · log2(1+SINR_uav)) )

clipped to [0, B_total], where
    D           — actor model size in bits
    SINR_users  — average linear SINR from UAVs to their assigned users
    SINR_uav    — average linear SINR on neighboring inter-UAV links
    α           — tradeoff weight ∈ (0, 1)
    B_total     — total bandwidth per UAV (Hz)

Public API
----------
compute_optimal_bfl(alpha, sinr_users_lin, sinr_uav_lin, model_size_bits, B_total)
compute_sinr_uav(uav_positions, config)
run_bcd(env, agents, alpha, n_iterations, episodes_per_block)
"""

from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from coms import air_to_ground_path_loss, fspl_db
from utils import calculate_distance
from config import Config

# ---------------------------------------------------------------------------
# Closed-form optimizer
# ---------------------------------------------------------------------------

def compute_optimal_bfl(
    alpha: float,
    sinr_users_lin: float,
    sinr_uav_lin: float,
    model_size_bits: float,
    B_total: float,
) -> float:
    """Return B_FL* in Hz, clipped to [0, B_total].

    Args:
        alpha:           tradeoff ∈ (0,1); α→0 allocates all BW to FL,
                         α→1 allocates all BW to users.
        sinr_users_lin:  average linear SINR of UAV→user links.
        sinr_uav_lin:    average linear SINR on inter-UAV links.
        model_size_bits: actor network size in bits (params × 32).
        B_total:         total bandwidth per UAV in Hz.
    """
    if alpha <= 0.0:
        return float(B_total)
    if alpha >= 1.0:
        return 0.0

    log_users = np.log2(1.0 + max(sinr_users_lin, 0.0))
    log_uav   = np.log2(1.0 + max(sinr_uav_lin,   0.0))

    denom = alpha * max(log_users, 1e-12) * max(log_uav, 1e-12)
    numer = (1.0 - alpha) * max(model_size_bits, 0.0)

    B_fl = float(np.sqrt(numer / denom))
    return float(np.clip(B_fl, 0.0, B_total))


# ---------------------------------------------------------------------------
# SINR helpers
# ---------------------------------------------------------------------------

def compute_sinr_uav(uav_positions: List[List[float]], config) -> float:
    """Average linear SINR between every neighboring UAV pair (free-space PL).

    All unique pairs (i,j) are included regardless of distance.
    Returns 0.0 if fewer than 2 UAVs are present.
    """
    n = len(uav_positions)
    if n < 2:
        return 0.0

    N_lin = 10 ** (config.NOISE_POWER / 10)   # mW (noise floor)
    freq  = config.FREQUENCY
    P_tx  = config.TRANSMIT_POWER              # dBm

    sinr_values: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = uav_positions[i][0] - uav_positions[j][0]
            dy = uav_positions[i][1] - uav_positions[j][1]
            d  = max(float(np.sqrt(dx ** 2 + dy ** 2)), 1.0)

            pl         = fspl_db(d, freq)
            signal_lin = 10 ** ((P_tx - pl) / 10)
            sinr_values.append(signal_lin / N_lin)

    return float(np.mean(sinr_values)) if sinr_values else 0.0


def _get_avg_sinr_users(env) -> float:
    """Average linear SINR from each UAV to its currently assigned users."""
    h_uav = getattr(env.config, 'UAV_HEIGHT', 100.0)
    N_lin = 10 ** (env.config.NOISE_POWER / 10)
    sinr_values: List[float] = []

    for uav in env.uavs:
        assigned = env.current_assignment.get(uav.id, [])
        for user in assigned:
            d  = calculate_distance(user.pos, uav.pos)
            pl = air_to_ground_path_loss(d, h_uav, env.config.FREQUENCY)
            signal_lin = 10 ** ((env.config.TRANSMIT_POWER - pl) / 10)
            sinr_values.append(signal_lin / N_lin)

    return float(np.mean(sinr_values)) if sinr_values else 1.0


def model_size_bits(agents: dict) -> float:
    """Count total actor parameters across all agents and convert to bits (×32)."""
    first_agent = next(iter(agents.values()))
    n_params = sum(p.numel() for p in first_agent.actor.parameters())
    return float(n_params * 32)


def _apply_bfl_to_config(config, B_fl: float) -> None:
    """Update config.B_USERS_FRACTION so that FL bandwidth = B_fl Hz."""
    B_total = config.BANDWIDTH_PER_UAV
    fraction_fl = float(np.clip(B_fl / B_total, 0.0, 1.0))
    config.B_USERS_FRACTION = float(np.clip(1.0 - fraction_fl, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Compact episode-level training loop (does not touch shared_ddpg_main)
# ---------------------------------------------------------------------------

def _train_block(env, agents: dict, n_episodes: int, seed: int | None = None) -> Tuple[List[float], List[float]]:
    """Run `n_episodes` episodes of MADDPG training.

    Returns:
        (reward_history, sum_rate_history) — one entry per episode.
    """
    from features.obs import build_obs_dict
    from RL import collect_step

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    reward_history:   List[float] = []
    sum_rate_history: List[float] = []

    for episode in range(n_episodes):
        states_vec      = build_obs_dict(env.reset(), Config)
        episode_rewards = []
        episode_sr      = []

        noise_scale = max(
            Config.FINAL_NOISE,
            Config.INITIAL_NOISE - episode * Config.NOISE_DECAY,
        )

        for step in range(Config.EPISODE_LENGTH):
            actions = {
                aid: agent.select_action(states_vec[aid], explore=True, noise_scale=noise_scale)
                for aid, agent in agents.items()
                if aid in states_vec
            }

            prev_states  = states_vec
            new_states, rewards, done = env.step(actions)
            states_vec   = build_obs_dict(new_states, Config)
            episode_sr.append(env.global_sum_rate_mbps)

            collect_step(agents, prev_states, actions, rewards, states_vec, done)

            if step % 5 == 0:
                for agent in agents.values():
                    agent.update(batch_size=Config.BATCH_SIZE)

            episode_rewards.append(float(np.mean(list(rewards.values()))))

        reward_history.append(float(np.mean(episode_rewards)))
        sum_rate_history.append(float(np.mean(episode_sr)))

    return reward_history, sum_rate_history


# ---------------------------------------------------------------------------
# BCD loop
# ---------------------------------------------------------------------------

def run_bcd(
    env,
    agents: dict,
    alpha: float,
    n_iterations: int = 3,
    episodes_per_block: int = 500,
) -> dict:
    """Block-Coordinate Descent: alternate between DRL training and B_FL* update.

    Each BCD iteration:
      1. Train MADDPG for `episodes_per_block` episodes with the current B_FL.
      2. Compute B_FL* analytically from the current environment state.
      3. Apply B_FL* to config for the next iteration.

    Args:
        env:               DisasterCoverageEnvironment instance.
        agents:            Dict {agent_id: IDDPGAgent}.
        alpha:             Tradeoff parameter ∈ (0,1).
        n_iterations:      Number of BCD outer iterations.
        episodes_per_block: MADDPG episodes trained per iteration.

    Returns:
        dict with keys:
            'bfl_history'      — B_FL* value chosen before each iteration (Hz)
            'reward_history'   — flat list of per-episode avg rewards
            'sum_rate_history' — flat list of per-episode avg sum rates (Mbps)
            'alpha'            — the alpha used
    """
    D          = model_size_bits(agents)
    B_total    = env.config.BANDWIDTH_PER_UAV
    bfl_history:      List[float] = []
    reward_history:   List[float] = []
    sum_rate_history: List[float] = []

    # Initialise B_FL: start with the config default
    current_bfl = (1.0 - getattr(env.config, 'B_USERS_FRACTION', 0.9)) * B_total

    print(f"\n{'='*60}")
    print(f"  BCD bandwidth optimization  |  α={alpha}  |  {n_iterations} iters × {episodes_per_block} eps")
    print(f"  Actor model size: {D/1e3:.1f} kbits  |  B_total: {B_total/1e3:.0f} kHz")
    print(f"{'='*60}")

    for iteration in range(n_iterations):
        # --- apply current B_FL* to config ---
        _apply_bfl_to_config(env.config, current_bfl)
        bfl_history.append(current_bfl)

        uav_positions = [list(uav.pos) for uav in env.uavs]
        sinr_uav      = compute_sinr_uav(uav_positions, env.config)

        print(f"\n[BCD iter {iteration+1}/{n_iterations}]  "
              f"B_FL={current_bfl:.2f} Hz  "
              f"B_users_frac={env.config.B_USERS_FRACTION:.3f}  "
              f"SINR_uav(lin)={sinr_uav:.2f}")

        # --- train for one block ---
        t0 = time.time()
        r_hist, sr_hist = _train_block(env, agents, episodes_per_block)
        reward_history.extend(r_hist)
        sum_rate_history.extend(sr_hist)
        elapsed = time.time() - t0

        avg_r  = float(np.mean(r_hist[-50:])) if len(r_hist) >= 50 else float(np.mean(r_hist))
        avg_sr = float(np.mean(sr_hist[-50:])) if len(sr_hist) >= 50 else float(np.mean(sr_hist))
        print(f"  Block done in {elapsed:.0f}s  |  "
              f"avg_reward(last50)={avg_r:.3f}  avg_SR={avg_sr:.2f}Mbps")

        # --- update B_FL* from current env state ---
        sinr_users = _get_avg_sinr_users(env)
        uav_positions = [list(uav.pos) for uav in env.uavs]
        sinr_uav      = compute_sinr_uav(uav_positions, env.config)
        current_bfl   = compute_optimal_bfl(alpha, sinr_users, sinr_uav, D, B_total)

        print(f"  B_FL* updated → {current_bfl:.2f} Hz  "
              f"(SINR_users={sinr_users:.2f}, SINR_uav={sinr_uav:.2f})")

    return {
        'bfl_history':      bfl_history,
        'reward_history':   reward_history,
        'sum_rate_history': sum_rate_history,
        'alpha':            alpha,
        'final_bfl':        current_bfl,
    }
