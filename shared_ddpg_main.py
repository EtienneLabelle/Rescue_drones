from env import DisasterCoverageEnvironment
from display import run_episode_viz
import numpy as np
from RL import IDDPGAgent, collect_step
from features.obs import build_obs_dict
from config import Config
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import random
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for comparison plot
import matplotlib.pyplot as plt

from federation.gossip import actor_gossip_step
from model_io import save_agents, load_agents, MODELS_DIR

EVAL_SEED = 64553


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(values, window=30):
    """Centered moving average for cleaner reward curves."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


# ---------------------------------------------------------------------------
# Training function — called twice (with FL, without FL)
# ---------------------------------------------------------------------------

def run_training(use_fl: bool, seed: int):
    """Run one full shared-DDPG session, optionally with gossip FL of actor weights.

    Every Config.FL_INTERVAL episodes, each UAV gossip-averages its actor
    network with neighbors within Config.R_COMM metres.

    Returns:
        (episode_rewards_history, agents, env)
    """
    label   = "FL" if use_fl else "NoFL"
    log_dir = f"runs/drone_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer  = SummaryWriter(log_dir)

    # Reproducible init for each run
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = DisasterCoverageEnvironment(Config)

    agents = {}
    for i in range(Config.NUM_UAVS):
        aid = f"uav_{i+1}"
        agents[aid] = IDDPGAgent(
            agent_id=aid,
            state_dim=Config.STATE_DIM,
            action_dim=Config.ACTION_DIM,
            actor_lr=Config.ACTOR_LR,
            critic_lr=Config.CRITIC_LR,
            buffer_size=Config.BUFFER_SIZE,
        )

    print(f"\n{'='*62}")
    print(f"  Shared DDPG {'WITH' if use_fl else 'WITHOUT':7s} gossip FL"
          f"  |  {Config.TRAINING_EPISODES} episodes"
          + (f"  |  gossip every {Config.FL_INTERVAL} eps  |  R={Config.R_COMM}m" if use_fl else ""))
    print(f"{'='*62}")

    episode_rewards_history   = []
    episode_losses_history    = []
    episode_sum_rate_history  = []

    best_smoothed = -float('inf')
    best_path     = os.path.join(MODELS_DIR, f"{label}_best.pt")
    final_path    = os.path.join(MODELS_DIR, f"{label}_final.pt")

    for episode in range(Config.TRAINING_EPISODES):
        states_vec = build_obs_dict(env.reset(), Config)
        episode_rewards       = []
        episode_losses        = []
        episode_sum_rate      = []
        episode_agent_rewards = {aid: [] for aid in agents}

        noise_scale = max(Config.FINAL_NOISE,
                          Config.INITIAL_NOISE - episode * Config.NOISE_DECAY)

        for step in range(Config.EPISODE_LENGTH):
            actions = {
                aid: agent.select_action(states_vec[aid], explore=True, noise_scale=noise_scale)
                for aid, agent in agents.items() if aid in states_vec
            }

            prev_states_vec     = states_vec
            new_states, rewards, done = env.step(actions)
            states_vec          = build_obs_dict(new_states, Config)
            episode_sum_rate.append(env.global_sum_rate_mbps)

            collect_step(agents, prev_states_vec, actions, rewards, states_vec, done)

            if step % 5 == 0:
                for agent in agents.values():
                    loss = agent.update(batch_size=Config.BATCH_SIZE,
                                        return_losses=True)
                    if loss is not None:
                        episode_losses.append(loss)

            episode_rewards.append(np.mean(list(rewards.values())))
            for aid in agents:
                if aid in rewards:
                    episode_agent_rewards[aid].append(rewards[aid])

        avg_reward   = float(np.mean(episode_rewards))
        avg_loss     = float(np.mean(episode_losses)) if episode_losses else 0.0
        avg_sum_rate = float(np.mean(episode_sum_rate))

        episode_rewards_history.append(avg_reward)
        episode_losses_history.append(avg_loss)
        episode_sum_rate_history.append(avg_sum_rate)

        smoothed = float(np.mean(episode_rewards_history[-30:]))
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            save_agents(agents, best_path)

        writer.add_scalar(f'{label}/Average_Reward',   avg_reward,   episode)
        writer.add_scalar(f'{label}/Average_Loss',     avg_loss,     episode)
        writer.add_scalar(f'{label}/Global_Sum_Rate_Mbps', avg_sum_rate, episode)
        for aid in agents:
            if episode_agent_rewards[aid]:
                writer.add_scalar(f'Agents_{label}/{aid}_Reward',
                                  np.mean(episode_agent_rewards[aid]), episode)

        # ---- gossip FL step ----------------------------------------
        gossip_fired = False
        if use_fl and episode > 0 and episode % Config.FL_INTERVAL == 0:
            uav_positions = [list(uav.pos) for uav in env.uavs]
            n_exchanged   = actor_gossip_step(agents, uav_positions, Config.R_COMM)
            writer.add_scalar(f'{label}/GossipExchanged', n_exchanged, episode)
            gossip_fired = True

        if episode % 50 == 0 or episode == Config.TRAINING_EPISODES - 1:
            print(f"Ep {episode+1:5d}/{Config.TRAINING_EPISODES} | "
                  f"R={avg_reward:6.3f}  L={avg_loss:.4f}  SR={avg_sum_rate:.2f}Mbps"
                  + ("  [gossip]" if gossip_fired else ""))

    writer.add_scalar(f'{label}/Final_Avg_Reward',
                      np.mean(episode_rewards_history[-100:]), 0)
    writer.add_scalar(f'{label}/Final_Avg_Sum_Rate_Mbps',
                      np.mean(episode_sum_rate_history[-100:]), 0)
    writer.close()
    print(f"TensorBoard logs → {log_dir}")

    save_agents(agents, final_path)
    print(f"Models saved → {final_path}  (best → {best_path})")

    return episode_rewards_history, agents, env


# ---------------------------------------------------------------------------
# Main — run both, plot comparison, then final visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    train_seed = Config.RANDOM_SEED if Config.RANDOM_SEED is not None else random.randint(0, 2**31 - 1)
    print(f"Training seed: {train_seed}  (set Config.RANDOM_SEED to reproduce)")

    rewards_fl,   agents_fl,   env_fl   = run_training(use_fl=True,  seed=train_seed)
    rewards_nofl, agents_nofl, env_nofl = run_training(use_fl=False, seed=train_seed)

    # ---- comparison training curve ----
    fig, ax = plt.subplots(figsize=(13, 5))

    for rewards, color, label in [
        (rewards_fl,   'tab:blue',   f'Gossip FL  (every {Config.FL_INTERVAL} eps, R={Config.R_COMM:.0f}m)'),
        (rewards_nofl, 'tab:orange', 'No Gossip FL'),
    ]:
        s = _smooth(rewards)
        x = np.arange(len(s)) + (len(rewards) - len(s)) // 2
        ax.plot(rewards, color=color, alpha=0.15, linewidth=0.7)
        ax.plot(x, s, color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average reward (smoothed)')
    ax.set_title('Shared DDPG: Gossip FL vs No FL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f"fl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining plot saved → {plot_path}")
    plt.close()

    # ---- validation — reload best checkpoints ----
    agents_fl_best   = load_agents(os.path.join(MODELS_DIR, "FL_best.pt"))
    agents_nofl_best = load_agents(os.path.join(MODELS_DIR, "NoFL_best.pt"))

    reward_fl,   sr_fl   = env_fl.run_validation_eval(
        agents_fl_best,   EVAL_SEED, "Final validation (FL best model)")
    reward_nofl, sr_nofl = env_nofl.run_validation_eval(
        agents_nofl_best, EVAL_SEED, "Final validation (NoFL best model)")

    # ---- comparison summary ----
    print(f"\n{'='*52}")
    print(f"  {'':20s}  {'Gossip FL':>12}  {'No FL':>12}")
    print(f"  {'Validation reward':20s}  {reward_fl:>12.3f}  {reward_nofl:>12.3f}")
    print(f"  {'Validation SR (Mbps)':20s}  {sr_fl:>12.2f}  {sr_nofl:>12.2f}")
    fl_last  = float(np.mean(rewards_fl[-100:]))
    nfl_last = float(np.mean(rewards_nofl[-100:]))
    print(f"  {'Last-100 ep reward':20s}  {fl_last:>12.3f}  {nfl_last:>12.3f}")
    print(f"{'='*52}")
    print(f"Training plot saved → {plot_path}")
    print("To view TensorBoard: tensorboard --logdir=runs")

    # ---- final simulation with the better model ----
    if reward_fl >= reward_nofl:
        best_agents, best_env, best_label = agents_fl_best, env_fl, "FL"
    else:
        best_agents, best_env, best_label = agents_nofl_best, env_nofl, "NoFL"

    print(f"\nRunning final simulation with {best_label}-trained agents (no noise)...")
    run_episode_viz(best_agents, best_env, EVAL_SEED, label=best_label)
