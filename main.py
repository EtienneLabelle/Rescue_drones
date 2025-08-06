from env import DisasterCoverageEnvironment
from display import SimpleAnimator
import tkinter as tk
import numpy as np
from experiment_framework import ExperimentConfig
from RL import MADDPGAgent, CentralizedReplayBuffer
from config import Config
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Create TensorBoard writer
log_dir = f"runs/drone_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)

# Configuration
config = ExperimentConfig(
    num_uavs=Config.NUM_UAVS, 
    num_episodes=10, 
    episode_length=Config.EPISODE_LENGTH
)
env = DisasterCoverageEnvironment(config)

# Create shared centralized replay buffer
shared_buffer = CentralizedReplayBuffer(max_size=Config.BUFFER_SIZE)

# Create RL agents
agents = {}
for i in range(Config.NUM_UAVS):
    agent_id = f"uav_{i+1}"
    agent = MADDPGAgent(
        agent_id=agent_id,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        num_agents=Config.NUM_UAVS,
        learning_rate=Config.LEARNING_RATE,
        shared_buffer=shared_buffer
    )
    agents[agent_id] = agent

# Setup visualization
root = tk.Tk()
animator = SimpleAnimator(root, env.sim.obstacles, disaster_zones=env.disaster_zones)
animator.env = env



# Training configuration
print(f"Training {len(agents)} agents for {Config.TRAINING_EPISODES} episodes...")
print(f"TensorBoard logs will be saved to: {log_dir}")

# Training metrics tracking
episode_rewards_history = []
episode_losses_history = []
episode_coverage_history = []


for episode in range(Config.TRAINING_EPISODES):
    # Reset environment
    states = env.reset()
    episode_rewards = []
    episode_losses = []
    episode_coverage = []
    episode_agent_rewards = {agent_id: [] for agent_id in agents.keys()}
    
    # Decay exploration noise
    noise_scale = max(Config.FINAL_NOISE, Config.INITIAL_NOISE - episode * Config.NOISE_DECAY)
    
    for step in range(config.episode_length):
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            if agent_id in states:
                action = agent.select_action(states[agent_id], noise_scale=noise_scale)
                actions[agent_id] = action
        
        # Execute actions
        new_states, rewards, done = env.step(actions)
        
        
        # Calculate coverage
        drone_positions = [uav.pos for uav in env.uavs]
        total_coverage = 0.0
        for zone in env.disaster_zones:
            coverage = zone.update_coverage(drone_positions)
            total_coverage += coverage * zone.severity
        avg_coverage = total_coverage / len(env.disaster_zones) if env.disaster_zones else 0.0
        episode_coverage.append(avg_coverage)
        
        # Create joint experience for centralized buffer
        joint_state = []
        joint_action = []
        joint_reward = []
        joint_next_state = []
        joint_done = []
        
        for i in range(config.num_uavs):
            agent_id = f"uav_{i+1}"
            if agent_id in states and agent_id in new_states:
                joint_state.extend(states[agent_id])
                joint_action.extend(actions.get(agent_id, [0, 0]))
                joint_reward.append(rewards.get(agent_id, 0))
                joint_next_state.extend(new_states[agent_id])
                joint_done.append(done)
            else:
                # Fill with zeros if agent not present
                joint_state.extend([0] * config.state_dim)
                joint_action.extend([0] * config.action_dim)
                joint_reward.append(0)
                joint_next_state.extend([0] * config.state_dim)
                joint_done.append(done)
        
        # Store joint experience
        shared_buffer.push(
            np.array(joint_state),
            np.array(joint_action),
            np.array(joint_reward),
            np.array(joint_next_state),
            np.array(joint_done)
        )
        
        # Update agents periodically and track losses
        if step % 5 == 0:
            for agent in agents.values():
                loss = agent.update(batch_size=Config.BATCH_SIZE, return_losses=True)
                if loss is not None:
                    episode_losses.append(loss)
        
        states = new_states
        episode_rewards.append(np.mean(list(rewards.values())))
        
        # Track individual agent rewards
        for agent_id in agents.keys():
            if agent_id in rewards:
                episode_agent_rewards[agent_id].append(rewards[agent_id])
    
    # Calculate episode metrics
    avg_reward = np.mean(episode_rewards)
    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    avg_coverage = np.mean(episode_coverage)

    
    # Store metrics for TensorBoard
    episode_rewards_history.append(avg_reward)
    episode_losses_history.append(avg_loss)
    episode_coverage_history.append(avg_coverage)

    
    # Log to TensorBoard
    writer.add_scalar('Training/Average_Reward', avg_reward, episode)
    writer.add_scalar('Training/Average_Loss', avg_loss, episode)
    writer.add_scalar('Training/Average_Coverage', avg_coverage, episode)

    
    # Log individual agent rewards
    for agent_id, agent in agents.items():
        if episode_agent_rewards[agent_id]:
            avg_agent_reward = np.mean(episode_agent_rewards[agent_id])
            writer.add_scalar(f'Agents/{agent_id}_Reward', avg_agent_reward, episode)
    
    # Progress reporting
    if episode % 50 == 0 or episode == Config.TRAINING_EPISODES - 1:
        print(f"Episode {episode + 1}/{Config.TRAINING_EPISODES}: "
              f"Avg Reward = {avg_reward:.2f}, "
              f"Avg Loss = {avg_loss:.4f}, "
              f"Coverage = {avg_coverage:.3f}, ")

# Log final training summary
writer.add_scalar('Training/Final_Average_Reward', np.mean(episode_rewards_history[-100:]), 0)
writer.add_scalar('Training/Final_Average_Coverage', np.mean(episode_coverage_history[-100:]), 0)


print("Training complete! Running final simulation...")

# Run final simulation with trained agents
print("Running final simulation...")
states = env.reset()

final_simulation_rewards = []
final_simulation_coverage = []

for step in range(config.episode_length):
    # Get actions from trained agents
    actions = {}
    for agent_id, agent in agents.items():
        if agent_id in states:
            action = agent.select_action(states[agent_id], noise_scale=0.01)
            actions[agent_id] = action
    
    # Execute actions
    new_states, rewards, done = env.step(actions)
    
    # Update environment
    env.sim.update_links()
    drone_positions = [uav.pos for uav in env.uavs]
    for zone in env.disaster_zones:
        zone.update_coverage(drone_positions)
    
    # Record for visualization
    animator.record_positions(env.uavs, env.sim.links)
    
    # Calculate coverage for final simulation
    total_coverage = 0.0
    for zone in env.disaster_zones:
        coverage = zone.update_coverage(drone_positions)
        total_coverage += coverage * zone.severity
    avg_coverage = total_coverage / len(env.disaster_zones) if env.disaster_zones else 0.0
    final_simulation_coverage.append(avg_coverage)
    
    # Progress reporting
    if step % 50 == 0:
        avg_reward = np.mean(list(rewards.values()))
        final_simulation_rewards.append(avg_reward)
        print(f"Step {step}: Avg Reward = {avg_reward:.2f}")
        
        # Log to TensorBoard
        writer.add_scalar('Final_Simulation/Average_Reward', avg_reward, step)
        writer.add_scalar('Final_Simulation/Average_Coverage', avg_coverage, step)
    
    states = new_states

# Log final simulation summary
if final_simulation_rewards:
    writer.add_scalar('Final_Simulation/Final_Average_Reward', np.mean(final_simulation_rewards), 0)
if final_simulation_coverage:
    writer.add_scalar('Final_Simulation/Final_Average_Coverage', np.mean(final_simulation_coverage), 0)

# Close TensorBoard writer
writer.close()

print("Simulation complete! Use buttons to control animation playback.")
print(f"TensorBoard logs saved to: {log_dir}")
print("To view logs, run: tensorboard --logdir=runs")
root.mainloop()





