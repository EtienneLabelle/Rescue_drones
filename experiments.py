"""Bandwidth-allocation experiments.

Run all three:
    python experiments.py

Or run individual experiments:
    python experiments.py alpha
    python experiments.py distance
    python experiments.py baseline

Results are written to results/ as CSV files.

Experiment 1 — Alpha sweep
    Fix UAV positions after a short warm-up, vary α ∈ [0,1], compute B_FL*
    at each α, record B_FL*, sum_rate, estimated FL convergence time.

Experiment 2 — Distance sweep
    Fix α=0.5, vary inter-UAV separation by changing ENV_WIDTH/ENV_HEIGHT,
    compute B_FL* at each configuration.

Experiment 3 — Baseline comparison
    α=0.5, train 1 000 episodes under four bandwidth policies and record
    per-episode sum rate and reward:
        (a) Adaptive B_FL* updated every 20 episodes
        (b) Fixed B_FL = 0       (no FL bandwidth)
        (c) Fixed B_FL = 25% B_total
        (d) Fixed B_FL = 50% B_total
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch

from config import Config
from env import DisasterCoverageEnvironment
from RL import IDDPGAgent, collect_step
from features.obs import build_obs_dict
from bandwidth_optimizer import (
    compute_optimal_bfl,
    compute_sinr_uav,
    _get_avg_sinr_users,
    _apply_bfl_to_config,
    model_size_bits,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(config) -> Dict[str, IDDPGAgent]:
    agents = {}
    for i in range(config.NUM_UAVS):
        aid = f"uav_{i+1}"
        agents[aid] = IDDPGAgent(
            agent_id=aid,
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            actor_lr=config.ACTOR_LR,
            critic_lr=config.CRITIC_LR,
            buffer_size=config.BUFFER_SIZE,
        )
    return agents


def _train_episodes(
    env,
    agents: dict,
    n_episodes: int,
    bfl_update_fn=None,
    bfl_update_interval: int = 20,
) -> List[dict]:
    """Train for n_episodes, returning a list of per-episode metric dicts.

    Args:
        bfl_update_fn:       if not None, called every bfl_update_interval episodes
                             as bfl_update_fn(env, agents) → new B_FL (Hz), which is
                             then applied to env.config.
        bfl_update_interval: how often (in episodes) to call bfl_update_fn.
    """
    records: List[dict] = []

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
            prev_states = states_vec
            new_states, rewards, done = env.step(actions)
            states_vec = build_obs_dict(new_states, Config)
            episode_sr.append(env.global_sum_rate_mbps)

            collect_step(agents, prev_states, actions, rewards, states_vec, done)

            if step % 5 == 0:
                for agent in agents.values():
                    agent.update(batch_size=Config.BATCH_SIZE)

            episode_rewards.append(float(np.mean(list(rewards.values()))))

        avg_reward   = float(np.mean(episode_rewards))
        avg_sum_rate = float(np.mean(episode_sr))
        bfl          = (1.0 - getattr(env.config, 'B_USERS_FRACTION', 0.9)) * env.config.BANDWIDTH_PER_UAV

        # Optional adaptive B_FL* update
        if bfl_update_fn is not None and (episode + 1) % bfl_update_interval == 0:
            new_bfl = bfl_update_fn(env, agents)
            _apply_bfl_to_config(env.config, new_bfl)
            bfl = new_bfl

        records.append({
            'episode':      episode + 1,
            'avg_reward':   avg_reward,
            'avg_sum_rate': avg_sum_rate,
            'B_FL_hz':      bfl,
        })

        if (episode + 1) % 100 == 0:
            print(f"  ep {episode+1:4d}  R={avg_reward:.3f}  SR={avg_sum_rate:.2f}Mbps  B_FL={bfl:.2f}Hz")

    return records


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Experiment 1 — Alpha sweep
# ---------------------------------------------------------------------------

def experiment_alpha_sweep(seed: int = 0, warmup_episodes: int = 200) -> str:
    """Vary α, compute B_FL* from fixed post-warmup state, record metrics.

    Returns path to the saved CSV.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Alpha sweep")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env    = DisasterCoverageEnvironment(Config)
    agents = _make_agents(Config)

    # Short warm-up so UAVs have meaningful positions / assignments
    print(f"  Warm-up: {warmup_episodes} episodes …")
    _train_episodes(env, agents, warmup_episodes)

    D        = model_size_bits(agents)
    B_total  = Config.BANDWIDTH_PER_UAV

    # Run a single evaluation episode to get stable SINR estimates
    states_vec = build_obs_dict(env.reset(), Config)
    for step in range(Config.EPISODE_LENGTH):
        actions = {
            aid: agent.select_action(states_vec[aid], explore=False, noise_scale=0.0)
            for aid, agent in agents.items()
            if aid in states_vec
        }
        new_states, rewards, done = env.step(actions)
        states_vec = build_obs_dict(new_states, Config)

    sinr_users = _get_avg_sinr_users(env)
    uav_positions = [list(uav.pos) for uav in env.uavs]
    sinr_uav      = compute_sinr_uav(uav_positions, Config)
    sum_rate_eval = env.global_sum_rate_mbps

    alphas = [round(a, 1) for a in np.arange(0.0, 1.05, 0.1)]

    rows: List[dict] = []
    for alpha in alphas:
        B_fl = compute_optimal_bfl(alpha, sinr_users, sinr_uav, D, B_total)
        b_users = B_total - B_fl

        # Estimated convergence time (seconds): D / (B_FL * log2(1+SINR_uav))
        log_uav = np.log2(1.0 + max(sinr_uav, 0.0))
        conv_time = (D / (B_fl * max(log_uav, 1e-12))) if B_fl > 0 else float("inf")

        rows.append({
            "alpha":             alpha,
            "B_FL_hz":           round(B_fl, 4),
            "B_users_hz":        round(b_users, 4),
            "B_FL_fraction":     round(B_fl / B_total, 4),
            "sum_rate_mbps":     round(sum_rate_eval, 4),
            "sinr_users_lin":    round(sinr_users, 4),
            "sinr_uav_lin":      round(sinr_uav, 4),
            "conv_time_s":       round(conv_time, 6) if not np.isinf(conv_time) else "inf",
        })
        print(f"  α={alpha:.1f}  B_FL={B_fl:.2f}Hz  conv_time={conv_time:.4f}s")

    path = os.path.join(RESULTS_DIR, "exp1_alpha_sweep.csv")
    _write_csv(path, rows, list(rows[0].keys()))
    return path


# ---------------------------------------------------------------------------
# Experiment 2 — Distance sweep
# ---------------------------------------------------------------------------

def experiment_distance_sweep(seed: int = 0, warmup_episodes: int = 200) -> str:
    """Fix α=0.5, vary inter-UAV distance via ENV_WIDTH/ENV_HEIGHT scaling.

    Returns path to the saved CSV.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Distance sweep (α=0.5)")
    print("=" * 60)

    alpha   = 0.5
    scales  = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # multipliers on base ENV_WIDTH/HEIGHT
    base_w  = Config.ENV_WIDTH
    base_h  = Config.ENV_HEIGHT

    rows: List[dict] = []

    for scale in scales:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Temporarily override environment size via a shallow config copy
        cfg = deepcopy(Config)
        cfg.ENV_WIDTH  = int(base_w * scale)
        cfg.ENV_HEIGHT = int(base_h * scale)

        env    = DisasterCoverageEnvironment(cfg)
        agents = _make_agents(cfg)

        print(f"\n  Scale={scale:.2f}  ENV={cfg.ENV_WIDTH}×{cfg.ENV_HEIGHT}m  warm-up …")
        _train_episodes(env, agents, warmup_episodes)

        # Evaluation pass
        states_vec = build_obs_dict(env.reset(), cfg)
        for step in range(cfg.EPISODE_LENGTH):
            actions = {
                aid: agent.select_action(states_vec[aid], explore=False, noise_scale=0.0)
                for aid, agent in agents.items()
                if aid in states_vec
            }
            new_states, rewards, done = env.step(actions)
            states_vec = build_obs_dict(new_states, cfg)

        uav_positions  = [list(uav.pos) for uav in env.uavs]
        avg_inter_dist = float(np.mean([
            np.sqrt((uav_positions[i][0] - uav_positions[j][0]) ** 2 +
                    (uav_positions[i][1] - uav_positions[j][1]) ** 2)
            for i in range(len(uav_positions))
            for j in range(i + 1, len(uav_positions))
        ])) if len(uav_positions) >= 2 else 0.0

        sinr_users = _get_avg_sinr_users(env)
        sinr_uav   = compute_sinr_uav(uav_positions, cfg)
        D          = model_size_bits(agents)
        B_total    = cfg.BANDWIDTH_PER_UAV
        B_fl       = compute_optimal_bfl(alpha, sinr_users, sinr_uav, D, B_total)

        log_uav   = np.log2(1.0 + max(sinr_uav, 0.0))
        conv_time = (D / (B_fl * max(log_uav, 1e-12))) if B_fl > 0 else float("inf")

        rows.append({
            "env_scale":         scale,
            "env_width_m":       cfg.ENV_WIDTH,
            "env_height_m":      cfg.ENV_HEIGHT,
            "avg_inter_uav_dist_m": round(avg_inter_dist, 2),
            "sinr_users_lin":    round(sinr_users, 4),
            "sinr_uav_lin":      round(sinr_uav, 4),
            "B_FL_hz":           round(B_fl, 4),
            "B_FL_fraction":     round(B_fl / B_total, 4),
            "sum_rate_mbps":     round(env.global_sum_rate_mbps, 4),
            "conv_time_s":       round(conv_time, 6) if not np.isinf(conv_time) else "inf",
        })
        print(f"  → avg_dist={avg_inter_dist:.0f}m  B_FL={B_fl:.2f}Hz  SINR_uav={sinr_uav:.2f}")

    path = os.path.join(RESULTS_DIR, "exp2_distance_sweep.csv")
    _write_csv(path, rows, list(rows[0].keys()))
    return path


# ---------------------------------------------------------------------------
# Experiment 3 — Baseline comparison
# ---------------------------------------------------------------------------

def experiment_baseline_comparison(
    seed: int = 0,
    n_episodes: int = 1000,
    bfl_update_interval: int = 20,
) -> str:
    """Compare four bandwidth policies over 1 000 training episodes.

    Policies:
        adaptive   — B_FL* recomputed every bfl_update_interval episodes
        fixed_0    — B_FL = 0 (all bandwidth to users, no FL)
        fixed_25pct — B_FL = 25% B_total
        fixed_50pct — B_FL = 50% B_total

    Returns path to the saved CSV.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Baseline comparison (α=0.5)")
    print("=" * 60)

    alpha   = 0.5
    B_total = Config.BANDWIDTH_PER_UAV

    def _adaptive_update(env, agents):
        D          = model_size_bits(agents)
        sinr_users = _get_avg_sinr_users(env)
        sinr_uav   = compute_sinr_uav([list(u.pos) for u in env.uavs], env.config)
        return compute_optimal_bfl(alpha, sinr_users, sinr_uav, D, B_total)

    policies = [
        ("adaptive",    None,             bfl_update_interval),   # update_fn set below
        ("fixed_0",     0.0,              None),
        ("fixed_25pct", 0.25 * B_total,   None),
        ("fixed_50pct", 0.50 * B_total,   None),
    ]

    all_rows: List[dict] = []

    for policy_name, fixed_bfl, interval in policies:
        print(f"\n  Policy: {policy_name}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        cfg = deepcopy(Config)
        env    = DisasterCoverageEnvironment(cfg)
        agents = _make_agents(cfg)

        if policy_name == "adaptive":
            _apply_bfl_to_config(cfg, 0.0)   # start from B_FL=0
            update_fn = _adaptive_update
            update_iv = interval
        else:
            _apply_bfl_to_config(cfg, fixed_bfl)
            update_fn = None
            update_iv = None

        # Swap the env config reference so helpers read the right cfg
        env.config = cfg

        records = _train_episodes(
            env, agents, n_episodes,
            bfl_update_fn=update_fn,
            bfl_update_interval=update_iv if update_iv else 1,
        )

        for rec in records:
            all_rows.append({
                "policy":       policy_name,
                "episode":      rec["episode"],
                "avg_reward":   round(rec["avg_reward"],   4),
                "avg_sum_rate": round(rec["avg_sum_rate"], 4),
                "B_FL_hz":      round(rec["B_FL_hz"],      4),
            })

    path = os.path.join(RESULTS_DIR, "exp3_baseline_comparison.csv")
    _write_csv(path, all_rows, ["policy", "episode", "avg_reward", "avg_sum_rate", "B_FL_hz"])
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_VALID_EXPERIMENTS = {"alpha", "distance", "baseline"}


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    targets = set(argv) & _VALID_EXPERIMENTS
    if not targets:
        targets = _VALID_EXPERIMENTS   # run all

    t_start = time.time()

    if "alpha" in targets:
        experiment_alpha_sweep()

    if "distance" in targets:
        experiment_distance_sweep()

    if "baseline" in targets:
        experiment_baseline_comparison()

    elapsed = time.time() - t_start
    print(f"\nAll experiments done in {elapsed/60:.1f} min.  Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
