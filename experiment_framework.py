#!/usr/bin/env python3
"""
Experiment Framework for Federated MADDPG Disaster Coverage

This contains the core experiment infrastructure:
- Configuration management
- Federated learning utilities  
- Episode testing framework
- Experiment orchestration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from env import DisasterCoverageEnvironment
from RL import MADDPGAgent
from display import SimpleAnimator

class ExperimentConfig:
    """Configuration for federated MADDPG experiments"""
    def __init__(self, num_uavs=5, num_episodes=1000, episode_length=200):
        self.num_uavs = num_uavs
        self.state_dim = 13  # Updated for relative positioning
        self.action_dim = 2
        self.total_state_dim = num_uavs * 13
        self.total_action_dim = num_uavs * 2
        self.simulation_steps = episode_length
        self.frequency = 2.4e9
        self.bandwidth = 20e6
        self.noise_power = -90
        
        # Experiment parameters
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.federated_exchange_interval = 10
        self.batch_size = 32
        self.learning_rate = 0.001

def create_agents(config):
    """Create MADDPG agents for all UAVs"""
    agents = {}
    for i in range(config.num_uavs):
        agent_id = f"uav_{i+1}"
        agent = MADDPGAgent(
            agent_id=agent_id,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            total_state_dim=config.total_state_dim,
            total_action_dim=config.total_action_dim,
            learning_rate=config.learning_rate
        )
        agents[agent_id] = agent
    return agents

def federated_policy_exchange(agents):
    """Exchange policies between agents for federated learning"""
    if len(agents) < 2:
        return
    
    # Get all agent actors
    agent_actors = [agent.actor for agent in agents.values()]
    
    # Simple averaging of policy parameters
    with torch.no_grad():
        # Get the first agent's state dict as template
        avg_state_dict = {}
        first_agent = list(agents.values())[0]
        
        for key in first_agent.actor.state_dict().keys():
            avg_state_dict[key] = torch.zeros_like(first_agent.actor.state_dict()[key])
        
        # Average all agent parameters
        for agent in agents.values():
            for key in avg_state_dict.keys():
                avg_state_dict[key] += agent.actor.state_dict()[key]
        
        # Divide by number of agents
        for key in avg_state_dict.keys():
            avg_state_dict[key] /= len(agents)
        
        # Update all agents with averaged parameters
        for agent in agents.values():
            agent.actor.load_state_dict(avg_state_dict)
            agent.target_actor.load_state_dict(avg_state_dict)

def run_experiment_with_display(config, federated=True):
    """Run experiment with real-time display like main.py"""
    print(f"Starting experiment with {config.num_uavs} UAVs")
    print(f"Episodes: {config.num_episodes}, Federated: {federated}")
    
    # Create environment and agents
    env = DisasterCoverageEnvironment(config)
    agents = create_agents(config)
    
    # Create GUI like main.py
    root = tk.Tk()
    root.title("Disaster Coverage Experiment")
    
    # Create display using SimpleAnimator pattern
    display = SimpleAnimator(root, env.obstacles, disaster_zones=env.disaster_zones)
    
    # Track metrics
    episode_rewards = []
    episode_coverage = []
    federated_exchanges = 0
    current_episode = 0
    
    # Simulation loop like main.py
    def simulation_step():
        nonlocal current_episode, federated_exchanges
        
        if current_episode < config.num_episodes:
            # Run one episode
            states = env.reset()
            episode_rewards_dict = {agent_id: 0.0 for agent_id in agents.keys()}
            episode_steps = 0
            
            while episode_steps < config.episode_length:
                # Select actions
                actions = {}
                for agent_id, state in states.items():
                    if agent_id in agents:
                        agent = agents[agent_id]
                        action = agent.select_action(state, noise_scale=0.1)
                        actions[agent_id] = action
                
                # Execute actions
                new_states, rewards, done = env.step(actions)
                
                # Record positions for display like main.py
                if episode_steps % 5 == 0:  # Record every 5 steps
                    display.record_positions(env.uavs, env.sim.links if hasattr(env, 'sim') else [])
                
                # Store experience for each agent
                for agent_id in agents.keys():
                    if agent_id in states and agent_id in new_states:
                        agent = agents[agent_id]
                        
                        # Get other agents' states and actions
                        other_states = []
                        other_actions = []
                        other_next_states = []
                        
                        for other_id in agents.keys():
                            if other_id != agent_id:
                                if other_id in states and other_id in new_states:
                                    other_states.append(states[other_id])
                                    other_actions.append(actions.get(other_id, [0, 0]))
                                    other_next_states.append(new_states[other_id])
                        
                        # Store experience
                        agent.remember(
                            states[agent_id], actions[agent_id], rewards[agent_id],
                            new_states[agent_id], done, other_states, other_actions, other_next_states
                        )
                        
                        episode_rewards_dict[agent_id] += rewards[agent_id]
                
                states = new_states
                episode_steps += 1
                
                if done:
                    break
            
            # Update agents
            for agent_id, agent in agents.items():
                other_agents = [a for a_id, a in agents.items() if a_id != agent_id]
                agent.update(batch_size=config.batch_size, other_agents=other_agents)
            
            # Calculate final coverage
            drone_positions = [uav.pos for uav in env.uavs]
            total_coverage = 0.0
            
            for zone in env.disaster_zones:
                coverage = zone.update_coverage(drone_positions)
                total_coverage += coverage * zone.severity
            
            final_coverage = total_coverage / len(env.disaster_zones)
            
            # Track metrics
            episode_rewards.append(episode_rewards_dict)
            episode_coverage.append(final_coverage)
            
            # Federated exchange
            if federated and (current_episode + 1) % config.federated_exchange_interval == 0:
                federated_policy_exchange(agents)
                federated_exchanges += 1
                print(f"Episode {current_episode + 1}: Federated exchange #{federated_exchanges}")
            
            # Progress update
            if (current_episode + 1) % 10 == 0:
                avg_reward = np.mean([sum(r.values()) for r in episode_rewards[-10:]])
                avg_coverage = np.mean(episode_coverage[-10:])
                print(f"Episode {current_episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Coverage = {avg_coverage:.3f}")
            
            current_episode += 1
            
            # Schedule next step
            root.after(100, simulation_step)  # 100ms delay between episodes
    
    # Start simulation
    simulation_step()
    
    # Start GUI
    root.mainloop()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage,
        'federated_exchanges': federated_exchanges,
        'final_coverage': episode_coverage[-1] if episode_coverage else 0.0,
        'final_rewards': episode_rewards[-1] if episode_rewards else {}
    }

def run_experiment(config, federated=True):
    """Run a complete experiment (original version without display)"""
    print(f"Starting experiment with {config.num_uavs} UAVs")
    print(f"Episodes: {config.num_episodes}, Federated: {federated}")
    
    # Create environment and agents
    env = DisasterCoverageEnvironment(config)
    agents = create_agents(config)
    
    # Track metrics
    episode_rewards = []
    episode_coverage = []
    federated_exchanges = 0
    
    for episode in range(config.num_episodes):
        # Run episode
        rewards, coverage, steps = run_single_episode(env, agents, config)
        
        # Track metrics
        episode_rewards.append(rewards)
        episode_coverage.append(coverage)
        
        # Federated exchange
        if federated and (episode + 1) % config.federated_exchange_interval == 0:
            federated_policy_exchange(agents)
            federated_exchanges += 1
            print(f"Episode {episode + 1}: Federated exchange #{federated_exchanges}")
        
        # Progress update
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean([sum(r.values()) for r in episode_rewards[-10:]])
            avg_coverage = np.mean(episode_coverage[-10:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Coverage = {avg_coverage:.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage,
        'federated_exchanges': federated_exchanges,
        'final_coverage': episode_coverage[-1],
        'final_rewards': episode_rewards[-1]
    }

def run_single_episode(env, agents, config):
    """Run a single episode with multiple MADDPG agents"""
    states = env.reset()
    episode_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
    episode_steps = 0
    
    while episode_steps < config.episode_length:
        # Select actions
        actions = {}
        for agent_id, state in states.items():
            if agent_id in agents:
                agent = agents[agent_id]
                action = agent.select_action(state, noise_scale=0.1)
                actions[agent_id] = action
        
        # Execute actions
        new_states, rewards, done = env.step(actions)
        
        # Store experience for each agent
        for agent_id in agents.keys():
            if agent_id in states and agent_id in new_states:
                agent = agents[agent_id]
                
                # Get other agents' states and actions
                other_states = []
                other_actions = []
                other_next_states = []
                
                for other_id in agents.keys():
                    if other_id != agent_id:
                        if other_id in states and other_id in new_states:
                            other_states.append(states[other_id])
                            other_actions.append(actions.get(other_id, [0, 0]))
                            other_next_states.append(new_states[other_id])
                
                # Store experience
                agent.remember(
                    states[agent_id], actions[agent_id], rewards[agent_id],
                    new_states[agent_id], done, other_states, other_actions, other_next_states
                )
                
                episode_rewards[agent_id] += rewards[agent_id]
        
        states = new_states
        episode_steps += 1
        
        if done:
            break
    
    # Update agents
    for agent_id, agent in agents.items():
        other_agents = [a for a_id, a in agents.items() if a_id != agent_id]
        agent.update(batch_size=config.batch_size, other_agents=other_agents)
    
    # Calculate final coverage from current UAV positions
    drone_positions = [uav.pos for uav in env.uavs]
    total_coverage = 0.0
    
    for zone in env.disaster_zones:
        coverage = zone.update_coverage(drone_positions)
        total_coverage += coverage * zone.severity
    
    final_coverage = total_coverage / len(env.disaster_zones)
    
    return episode_rewards, final_coverage, episode_steps

def plot_results(results, title="Experiment Results"):
    """Plot experiment results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot coverage over episodes
    ax1.plot(results['episode_coverage'])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Progress')
    ax1.grid(True)
    
    # Plot rewards over episodes
    episode_rewards = [sum(r.values()) for r in results['episode_rewards']]
    ax2.plot(episode_rewards)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Reward Progress')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(num_uavs=3, num_episodes=500, episode_length=100)
    
    # Run federated experiment
    print("Running Federated MADDPG Experiment...")
    federated_results = run_experiment(config, federated=True)
    
    # Plot results
    plot_results(federated_results, "Federated MADDPG Results")
    
    print(f"Final Coverage: {federated_results['final_coverage']:.3f}")
    print(f"Federated Exchanges: {federated_results['federated_exchanges']}") 