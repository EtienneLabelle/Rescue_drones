from env import DisasterCoverageEnvironment
from display import SimpleAnimator
import tkinter as tk
import numpy as np
from RL import MADDPGAgent, CentralizedReplayBuffer, collect_step
from features.obs import build_obs_dict
from config import Config
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from datetime import datetime
import random
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for comparison plot
import matplotlib.pyplot as plt

from federation.gossip import actor_gossip_step
from model_io import save_agents, load_agents, MODELS_DIR

EVAL_SEED = 4242


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
    label   = "maddpg_fl" if use_fl else "maddpg_nofl"
    log_dir = f"runs/drone_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer  = SummaryWriter(log_dir)

    # Reproducible init for each run
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = DisasterCoverageEnvironment(Config)

    shared_buffer = CentralizedReplayBuffer(max_size=Config.BUFFER_SIZE)
    agents = {}
    for i in range(Config.NUM_UAVS):
        aid = f"uav_{i+1}"
        agents[aid] = MADDPGAgent(
            agent_id=aid,
            state_dim=Config.STATE_DIM,
            action_dim=Config.ACTION_DIM,
            num_agents=Config.NUM_UAVS,
            actor_lr=Config.ACTOR_LR,
            critic_lr=Config.CRITIC_LR,
            shared_buffer=shared_buffer,
        )

    print(f"\n{'='*62}")
    print(f"  MADDPG {'WITH' if use_fl else 'WITHOUT':7s} gossip FL"
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
                target_actors = [agents[f"uav_{i+1}"].target_actor for i in range(Config.NUM_UAVS)]
                for agent in agents.values():
                    loss = agent.update(batch_size=Config.BATCH_SIZE,
                                        all_target_actors=target_actors,
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

    rewards_fl, agents_fl, env_fl = run_training(use_fl=False, seed=train_seed)

    # ---- training curve ----
    fig, ax = plt.subplots(figsize=(13, 5))
    s_fl = _smooth(rewards_fl)
    x_fl = np.arange(len(s_fl)) + (len(rewards_fl) - len(s_fl)) // 2
    ax.plot(rewards_fl, color='tab:blue', alpha=0.15, linewidth=0.7)
    ax.plot(x_fl, s_fl, color='tab:blue', linewidth=2,
            label=f'Gossip FL (every {Config.FL_INTERVAL} eps, R={Config.R_COMM:.0f}m)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average reward (smoothed)')
    ax.set_title('MADDPG with gossip FL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f"fl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining plot saved → {plot_path}")
    plt.close()

    # ---- final simulation ----
    print("\nRunning final simulation with FL-trained agents (no noise)...")

    np.random.seed(EVAL_SEED)
    random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    torch.cuda.manual_seed_all(EVAL_SEED)

    fixed_positions = []
    for i in range(Config.NUM_UAVS):
        np.random.seed(EVAL_SEED + i)
        margin = Config.SPAWN_MARGIN
        fixed_positions.append([np.random.uniform(margin, Config.ENV_WIDTH  - margin),
                                 np.random.uniform(margin, Config.ENV_HEIGHT - margin)])

    states = env_fl.reset(deterministic=True, fixed_positions=fixed_positions)

    root     = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), sys.exit(0)))
    animator = SimpleAnimator(root, env_fl.sim.obstacles,
                              disaster_zones=env_fl.disaster_zones)
    animator.env = env_fl

    for step in range(Config.EPISODE_LENGTH):
        actions = {
            aid: agent.select_action(states[aid], explore=False, noise_scale=0.0)
            for aid, agent in agents_fl.items() if aid in states
        }
        new_states, rewards, done = env_fl.step(actions)

        if Config.COMMS_ENABLED:
            env_fl.sim.update_links()

        animator.record_positions(
            env_fl.uavs, env_fl.sim.links,
            ground_users=env_fl.ground_users,
            uav_users=getattr(env_fl, 'current_assignment', {}),
        )
        states = new_states

    # ---- validation — reload best checkpoint ----
    agents_fl_best = load_agents(os.path.join(MODELS_DIR, "maddpg_nofl_best.pt"), distributed=False)
    reward_fl, cov_fl = env_fl.run_validation_eval(
        agents_fl_best, EVAL_SEED, "Final validation (FL best model)")

    print(f"\nValidation — FL: reward={reward_fl:.3f}  coverage={cov_fl:.3f}")
    print(f"Training plot saved → {plot_path}")
    print("To view TensorBoard: tensorboard --logdir=runs")

    root.mainloop()
