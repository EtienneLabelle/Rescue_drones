#!/usr/bin/env python3
"""
Example usage of the modular RL and FL framework for drone federated learning research.

This script demonstrates how to:
1. Switch between different RL algorithms (DQN, PPO, A2C)
2. Switch between different FL algorithms (FedAvg, FedProx, FedNova, FedAdam)
3. Run comparison experiments
4. Analyze results
"""

import torch
import numpy as np
from experiment_framework import ExperimentConfig, FederatedRLExperiment, run_comparison_experiment
import matplotlib.pyplot as plt
import json

def create_experiment_configs():
    """Create different experiment configurations for comparison"""
    
    # Base configuration
    base_config = ExperimentConfig()
    base_config.simulation_steps = 1000  # Shorter for faster testing
    base_config.federated_rounds = 5
    base_config.clients_per_round = 2
    
    # Configuration 1: PPO + FedAvg
    config_ppo_fedavg = ExperimentConfig()
    config_ppo_fedavg.rl_algorithm = 'ppo'
    config_ppo_fedavg.fl_algorithm = 'fedavg'
    config_ppo_fedavg.simulation_steps = 1000
    config_ppo_fedavg.federated_rounds = 5
    config_ppo_fedavg.clients_per_round = 2
    config_ppo_fedavg.experiment_name = "PPO_FedAvg_Comparison"
    
    # Configuration 2: DQN + FedProx
    config_dqn_fedprox = ExperimentConfig()
    config_dqn_fedprox.rl_algorithm = 'dqn'
    config_dqn_fedprox.fl_algorithm = 'fedprox'
    config_dqn_fedprox.simulation_steps = 1000
    config_dqn_fedprox.federated_rounds = 5
    config_dqn_fedprox.clients_per_round = 2
    config_dqn_fedprox.experiment_name = "DQN_FedProx_Comparison"
    
    # Configuration 3: A2C + FedNova
    config_a2c_fednova = ExperimentConfig()
    config_a2c_fednova.rl_algorithm = 'a2c'
    config_a2c_fednova.fl_algorithm = 'fednova'
    config_a2c_fednova.simulation_steps = 1000
    config_a2c_fednova.federated_rounds = 5
    config_a2c_fednova.clients_per_round = 2
    config_a2c_fednova.experiment_name = "A2C_FedNova_Comparison"
    
    # Configuration 4: PPO + FedAdam
    config_ppo_fedadam = ExperimentConfig()
    config_ppo_fedadam.rl_algorithm = 'ppo'
    config_ppo_fedadam.fl_algorithm = 'fedadam'
    config_ppo_fedadam.simulation_steps = 1000
    config_ppo_fedadam.federated_rounds = 5
    config_ppo_fedadam.clients_per_round = 2
    config_ppo_fedadam.experiment_name = "PPO_FedAdam_Comparison"
    
    return {
        "PPO_FedAvg": config_ppo_fedavg,
        "DQN_FedProx": config_dqn_fedprox,
        "A2C_FedNova": config_a2c_fednova,
        "PPO_FedAdam": config_ppo_fedadam
    }

def run_single_experiment():
    """Run a single experiment with PPO and FedAvg"""
    print("Running single experiment: PPO + FedAvg")
    
    config = ExperimentConfig()
    config.rl_algorithm = 'ppo'
    config.fl_algorithm = 'fedavg'
    config.simulation_steps = 500  # Shorter for demo
    config.federated_rounds = 3
    config.clients_per_round = 2
    
    experiment = FederatedRLExperiment(config)
    results = experiment.run_experiment()
    
    print("Single experiment completed!")
    print(f"Total RL episodes: {len(results['rl_episodes'])}")
    print(f"Total FL rounds: {len(results['fl_rounds'])}")
    
    return results

def run_comparison_experiments():
    """Run comparison experiments with different algorithm combinations"""
    print("Running comparison experiments...")
    
    configs = create_experiment_configs()
    results = run_comparison_experiment(configs)
    
    return results

def analyze_results(results):
    """Analyze and visualize experiment results"""
    print("\n" + "="*60)
    print("EXPERIMENT ANALYSIS")
    print("="*60)
    
    # Calculate metrics for each configuration
    analysis = {}
    
    for config_name, result in results.items():
        # Average reward per episode
        episode_rewards = [episode['total_reward'] for episode in result['rl_episodes']]
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # FL round participation
        fl_rounds = len(result['fl_rounds'])
        total_participants = sum(round_info['total_clients'] for round_info in result['fl_rounds'])
        
        analysis[config_name] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'fl_rounds': fl_rounds,
            'total_participants': total_participants,
            'episode_rewards': episode_rewards
        }
        
        print(f"\n{config_name}:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  FL Rounds: {fl_rounds}")
        print(f"  Total Participants: {total_participants}")
    
    # Find best performing configuration
    best_config = max(analysis.keys(), key=lambda x: analysis[x]['avg_reward'])
    print(f"\nBest performing configuration: {best_config}")
    print(f"Average reward: {analysis[best_config]['avg_reward']:.2f}")
    
    return analysis

def plot_results(analysis):
    """Create plots to visualize results"""
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average rewards comparison
        configs = list(analysis.keys())
        avg_rewards = [analysis[config]['avg_reward'] for config in configs]
        std_rewards = [analysis[config]['std_reward'] for config in configs]
        
        bars = ax1.bar(configs, avg_rewards, yerr=std_rewards, capsize=5)
        ax1.set_title('Average Rewards by Configuration')
        ax1.set_ylabel('Average Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, avg_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{reward:.2f}', ha='center', va='bottom')
        
        # Plot 2: Learning curves (first few episodes)
        for config_name, data in analysis.items():
            rewards = data['episode_rewards'][:20]  # First 20 episodes
            episodes = range(1, len(rewards) + 1)
            ax2.plot(episodes, rewards, label=config_name, marker='o')
        
        ax2.set_title('Learning Curves (First 20 Episodes)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
        print("Results plot saved as 'experiment_results.png'")
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def save_analysis(analysis, filename='experiment_analysis.json'):
    """Save analysis results to file"""
    # Convert numpy arrays to lists for JSON serialization
    analysis_serializable = {}
    for config_name, data in analysis.items():
        analysis_serializable[config_name] = {
            'avg_reward': float(data['avg_reward']),
            'std_reward': float(data['std_reward']),
            'fl_rounds': data['fl_rounds'],
            'total_participants': data['total_participants'],
            'episode_rewards': [float(r) for r in data['episode_rewards']]
        }
    
    with open(filename, 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    
    print(f"Analysis saved to {filename}")

def main():
    """Main function to run experiments"""
    print("="*60)
    print("DRONE FEDERATED LEARNING RESEARCH FRAMEWORK")
    print("="*60)
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch is required but not installed.")
        print("Please install PyTorch: pip install torch")
        return
    
    # Run experiments
    print("\n1. Running comparison experiments...")
    results = run_comparison_experiments()
    
    # Analyze results
    print("\n2. Analyzing results...")
    analysis = analyze_results(results)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    plot_results(analysis)
    
    # Save analysis
    print("\n4. Saving analysis...")
    save_analysis(analysis)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check 'experiment_results.png' for visualizations")
    print("2. Check 'experiment_analysis.json' for detailed results")
    print("3. Check 'experiments/' directory for saved models")
    print("4. Modify configurations in 'create_experiment_configs()' for different experiments")

if __name__ == "__main__":
    main() 