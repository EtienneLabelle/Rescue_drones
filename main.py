"""
main.py — RL-for-FL entry point.

A single PPO agent (HybridPPOAgent) flies a UAV relay to minimise the
energy and time-to-accuracy of a federated learning workload running on
N ground clients.

Episode  = one FL run from scratch until loss < FL_TARGET_EPS or round cap.
RL step  = one FL round (client train → uplink → relay → BS aggregate → downlink).

Outputs (all under results/ and models/):
  rl_for_fl_training_<ts>.png   — reward / energy / rounds per episode
  rl_for_fl_comparison_<ts>.png — learned vs static-baseline bar chart
  models/hybrid_ppo_best.pt     — best checkpoint (by smoothed reward)
  runs/rl_for_fl_<ts>/          — TensorBoard logs
"""
from __future__ import annotations

import os
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import Config
from FL import FLWorkload
from env import FLRelayEnvironment
from RL import HybridPPOAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def smooth(values: list, w: int = 20) -> np.ndarray:
    if len(values) < w:
        return np.array(values)
    return np.convolve(values, np.ones(w) / w, mode='valid')


def run_episode(agent: HybridPPOAgent, env: FLRelayEnvironment,
                training: bool = True) -> dict:
    """Run one full episode.

    If training=True the trajectory is stored and agent.update() is called
    at the end.  If False the episode is just evaluated (no gradient update).
    """
    state      = env.reset()
    ep_reward  = 0.0
    ep_energy  = 0.0
    ep_latency = 0.0
    rounds     = 0
    info       = {}

    while True:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        if training:
            agent.store(state, action, log_prob, reward, done, value)

        ep_reward  += reward
        ep_energy  += info['energy_round']
        ep_latency += info['latency_round']
        rounds     += 1
        state       = next_state

        if done:
            break

    metrics = dict(
        reward=ep_reward,
        energy=ep_energy,
        latency=ep_latency,
        rounds=rounds,
    )
    if training:
        update_metrics = agent.update()
        metrics.update(update_metrics)
    return metrics


# ---------------------------------------------------------------------------
# Static baseline
# ---------------------------------------------------------------------------

def run_static_baseline(config, n_episodes: int = 10, seed: int = 0) -> dict:
    """UAV fixed at centre, all clients selected, uniform bandwidth each round.

    This isolates the value of mobility and dynamic selection over a static
    allocation that only optimises through position.
    """
    results = []
    for ep in range(n_episodes):
        fl  = FLWorkload(config)
        env = FLRelayEnvironment(config, fl, fixed_uav=True)
        env.reset(seed=seed + ep)

        ep_energy, ep_latency, rounds = 0.0, 0.0, 0
        N = env.n_clients

        # Fixed action: no movement, select all clients, uniform bandwidth
        action = np.zeros(2 + 2 * N, dtype=np.float32)
        action[2:2 + N] = 1.0   # sel_i = 1 → all selected

        while True:
            _, _, done, info = env.step(action)
            ep_energy  += info['energy_round']
            ep_latency += info['latency_round']
            rounds     += 1
            if done:
                break

        results.append(dict(energy=ep_energy, latency=ep_latency, rounds=rounds))

    return dict(
        energy =float(np.mean([r['energy']  for r in results])),
        latency=float(np.mean([r['latency'] for r in results])),
        rounds =float(np.mean([r['rounds']  for r in results])),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(reward_h, energy_h, rounds_h, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, data, title, ylabel in [
        (axes[0], reward_h, 'Episode Reward',        'Reward'),
        (axes[1], energy_h, 'Energy per Episode',    'Energy (J)'),
        (axes[2], rounds_h, 'FL Rounds per Episode', 'Rounds'),
    ]:
        s = smooth(data)
        ax.plot(data, alpha=0.18, color='tab:blue', linewidth=0.7)
        ax.plot(np.arange(len(s)) + (len(data) - len(s)) // 2,
                s, color='tab:blue', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(baseline: dict, learned: dict, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    labels = ['Static Baseline', 'Learned (PPO)']
    pairs  = [
        (baseline['energy'],  learned['energy'],  'Total Energy (J)'),
        (baseline['latency'], learned['latency'], 'Total Latency (s)'),
        (baseline['rounds'],  learned['rounds'],  'FL Rounds to Eps'),
    ]
    colors = ['tab:orange', 'tab:blue']
    for ax, (b_val, l_val, title) in zip(axes, pairs):
        bars = ax.bar(labels, [b_val, l_val], color=colors)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, [b_val, l_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    TRAIN_EPISODES = getattr(Config, 'FL_RL_TRAIN_EPISODES', 500)
    EVAL_EVERY     = getattr(Config, 'FL_RL_EVAL_EVERY',     50)
    SEED           = Config.RANDOM_SEED if Config.RANDOM_SEED is not None else 42
    set_seeds(SEED)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/rl_for_fl_{ts}"
    writer  = SummaryWriter(log_dir)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ---- Build training environment + agent ----
    fl_workload = FLWorkload(Config)
    env         = FLRelayEnvironment(Config, fl_workload)

    agent = HybridPPOAgent(
        state_dim    = env.state_dim,
        n_clients    = env.n_clients,
        learning_rate= getattr(Config, 'PPO_LR',           3e-4),
        gamma        = getattr(Config, 'PPO_GAMMA',        0.99),
        gae_lambda   = getattr(Config, 'PPO_GAE_LAMBDA',   0.95),
        clip_ratio   = getattr(Config, 'PPO_CLIP',         0.2),
        entropy_coef = getattr(Config, 'PPO_ENTROPY_COEF', 0.01),
        ppo_epochs   = getattr(Config, 'PPO_EPOCHS',       4),
        batch_size   = getattr(Config, 'PPO_BATCH_SIZE',   64),
        hidden_dim   = getattr(Config, 'PPO_HIDDEN_DIM',   256),
    )

    print(f"\n{'='*64}")
    print(f"  RL-for-FL | PPO | N_clients={env.n_clients} | "
          f"state_dim={env.state_dim} | action_dim={env.action_dim}")
    print(f"  {TRAIN_EPISODES} episodes | seed={SEED}")
    print(f"{'='*64}")

    reward_history  = []
    energy_history  = []
    latency_history = []
    rounds_history  = []

    best_reward = -float('inf')
    best_path   = "models/hybrid_ppo_best.pt"
    final_path  = "models/hybrid_ppo_final.pt"

    # ---- Training loop ----
    for episode in range(TRAIN_EPISODES):
        m = run_episode(agent, env, training=True)

        reward_history.append(m['reward'])
        energy_history.append(m['energy'])
        latency_history.append(m['latency'])
        rounds_history.append(m['rounds'])

        writer.add_scalar('PPO/Reward',    m['reward'],  episode)
        writer.add_scalar('PPO/Energy_J',  m['energy'],  episode)
        writer.add_scalar('PPO/Latency_s', m['latency'], episode)
        writer.add_scalar('PPO/FL_Rounds', m['rounds'],  episode)
        if 'actor_loss' in m:
            writer.add_scalar('PPO/Actor_Loss',  m['actor_loss'],  episode)
            writer.add_scalar('PPO/Critic_Loss', m['critic_loss'], episode)
            writer.add_scalar('PPO/Entropy',     m['entropy'],     episode)

        smoothed = float(np.mean(reward_history[-20:]))
        if smoothed > best_reward:
            best_reward = smoothed
            agent.save_model(best_path)

        if episode % EVAL_EVERY == 0 or episode == TRAIN_EPISODES - 1:
            print(f"Ep {episode+1:5d}/{TRAIN_EPISODES} | "
                  f"R={m['reward']:8.3f}  E={m['energy']:.1f}J  "
                  f"T={m['latency']:.1f}s  rounds={m['rounds']}")

    agent.save_model(final_path)
    writer.close()

    # ---- Static baseline ----
    print("\nRunning static baseline (UAV fixed, all clients, uniform BW)...")
    baseline = run_static_baseline(Config, n_episodes=10, seed=SEED)
    print(f"Baseline → energy={baseline['energy']:.1f}J  "
          f"latency={baseline['latency']:.1f}s  rounds={baseline['rounds']:.1f}")

    # ---- Evaluate best learned policy ----
    print("Evaluating best learned policy...")
    agent.load_model(best_path)
    eval_results = []
    for ep in range(10):
        set_seeds(SEED + ep)
        eval_fl  = FLWorkload(Config)
        eval_env = FLRelayEnvironment(Config, eval_fl)
        m = run_episode(agent, eval_env, training=False)
        eval_results.append(m)

    learned = dict(
        energy =float(np.mean([r['energy']  for r in eval_results])),
        latency=float(np.mean([r['latency'] for r in eval_results])),
        rounds =float(np.mean([r['rounds']  for r in eval_results])),
    )
    print(f"Learned  → energy={learned['energy']:.1f}J  "
          f"latency={learned['latency']:.1f}s  rounds={learned['rounds']:.1f}")

    # ---- Summary ----
    print(f"\n{'='*52}")
    print(f"  {'':20s}  {'Baseline':>12}  {'Learned':>12}")
    print(f"  {'Energy (J)':20s}  {baseline['energy']:>12.1f}  {learned['energy']:>12.1f}")
    print(f"  {'Latency (s)':20s}  {baseline['latency']:>12.1f}  {learned['latency']:>12.1f}")
    print(f"  {'FL Rounds':20s}  {baseline['rounds']:>12.1f}  {learned['rounds']:>12.1f}")
    print(f"{'='*52}")

    # ---- Plots ----
    train_plot = f"results/rl_for_fl_training_{ts}.png"
    cmp_plot   = f"results/rl_for_fl_comparison_{ts}.png"
    plot_training(reward_history, energy_history, rounds_history, train_plot)
    plot_comparison(baseline, learned, cmp_plot)

    print(f"\nTraining plot    → {train_plot}")
    print(f"Comparison plot  → {cmp_plot}")
    print(f"Best model       → {best_path}")
    print(f"TensorBoard      → tensorboard --logdir={log_dir}")
