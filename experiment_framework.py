import torch
import torch.nn as nn
import numpy as np
from RL import create_rl_agent
from FL import create_fl_algorithm, FederatedTrainer
from env import Simulation
from drones import Drone
import json
import os
from datetime import datetime

class DroneNetwork(nn.Module):
    """Neural network for drone decision making"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(DroneNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ExperimentConfig:
    """Configuration class for experiments"""
    
    def __init__(self):
        # RL Configuration
        self.rl_algorithm = 'ppo'  # 'dqn', 'ppo', 'a2c'
        self.state_dim = 10  # Adjust based on your state representation
        self.action_dim = 4   # Adjust based on your action space
        self.rl_learning_rate = 0.001
        self.rl_gamma = 0.99
        
        # FL Configuration
        self.fl_algorithm = 'fedavg'  # 'fedavg', 'fedprox', 'fednova', 'fedadam'
        self.fl_learning_rate = 0.001
        self.local_epochs = 1
        self.federated_rounds = 10
        self.clients_per_round = 3
        
        # Simulation Configuration
        self.simulation_steps = 2000
        self.obstacle_count = 50
        self.obstacle_size = 1000
        self.deployment_distance = 1000
        
        # Communication Configuration
        self.frequency = 2.4e9
        self.bandwidth = 20e6
        self.noise_power = -90
        
        # Experiment Configuration
        self.experiment_name = f"drone_fl_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_models = True
        self.log_metrics = True
    
    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save(self, path):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load configuration from file"""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

class DroneEnvironment:
    """Environment wrapper for RL training"""
    
    def __init__(self, config):
        self.config = config
        self.sim = Simulation(config.bandwidth, config.frequency, config.noise_power)
        self.sim.create_obstacles(config.obstacle_count, config.obstacle_size)
        
        # Initialize drones
        self.operator = Drone(id="operator", position=[0, 0])
        self.sim.drones.append(self.operator)
        self.sim.deploy_relay(id="video_drone")
        
        self.step_count = 0
        self.max_steps = config.simulation_steps
    
    def get_state(self, drone_id):
        """Get state representation for a specific drone"""
        drone = None
        for d in self.sim.drones:
            if d.id == drone_id:
                drone = d
                break
        
        if drone is None:
            return None
        
        # Create state vector
        state = []
        
        # Drone position
        state.extend(drone.pos)
        
        # Distance to nearest drone
        min_distance = float('inf')
        for other_drone in self.sim.drones:
            if other_drone.id != drone_id:
                dist = np.sqrt((drone.pos[0] - other_drone.pos[0])**2 + 
                             (drone.pos[1] - other_drone.pos[1])**2)
                min_distance = min(min_distance, dist)
        state.append(min_distance)
        
        # Communication quality (if applicable)
        if hasattr(self.sim, 'links') and self.sim.links:
            avg_capacity = sum(link.capacity_bps for link in self.sim.links) / len(self.sim.links)
            state.append(avg_capacity)
        else:
            state.append(0.0)
        
        # Battery level
        state.append(drone.battery_level)
        
        # Obstacle proximity
        min_obstacle_dist = float('inf')
        for obstacle in self.sim.obstacles:
            dist = abs(drone.pos[0] - obstacle.center_pos[0])
            min_obstacle_dist = min(min_obstacle_dist, dist)
        state.append(min_obstacle_dist)
        
        # Pad state to fixed size
        while len(state) < self.config.state_dim:
            state.append(0.0)
        
        return np.array(state[:self.config.state_dim])
    
    def step(self, actions):
        """Execute actions and return new state, reward, done"""
        rewards = {}
        
        # Execute actions for each drone
        for drone_id, action in actions.items():
            drone = None
            for d in self.sim.drones:
                if d.id == drone_id:
                    drone = d
                    break
            
            if drone is None:
                continue
            
            # Convert action to movement
            if action == 0:  # Move up
                drone.move([0, 10])
            elif action == 1:  # Move down
                drone.move([0, -10])
            elif action == 2:  # Move right
                drone.move([10, 0])
            elif action == 3:  # Move left
                drone.move([-10, 0])
            
            # Calculate reward
            reward = self._calculate_reward(drone)
            rewards[drone_id] = reward
        
        # Update simulation
        self.sim.update_links()
        
        # Check if relay deployment is needed
        if len(self.sim.drones) >= 2:
            distance = np.sqrt((self.sim.drones[-1].pos[0] - self.sim.drones[0].pos[0])**2 + 
                             (self.sim.drones[-1].pos[1] - self.sim.drones[0].pos[1])**2)
            
            if distance > self.config.deployment_distance:
                self.sim.deploy_relay(id=f"relay_drone_{len(self.sim.drones)-1}")
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Get new states
        new_states = {}
        for drone in self.sim.drones:
            new_states[drone.id] = self.get_state(drone.id)
        
        return new_states, rewards, done
    
    def _calculate_reward(self, drone):
        """Calculate reward for a drone"""
        reward = 0.0
        
        # Reward for maintaining communication
        if hasattr(self.sim, 'links') and self.sim.links:
            avg_capacity = sum(link.capacity_bps for link in self.sim.links) / len(self.sim.links)
            reward += avg_capacity / 1e6  # Normalize capacity
        
        # Penalty for low battery
        if drone.battery_level < 20:
            reward -= 10.0
        
        # Penalty for being too close to obstacles
        for obstacle in self.sim.obstacles:
            dist = abs(drone.pos[0] - obstacle.center_pos[0])
            if dist < 100:  # Too close to obstacle
                reward -= 5.0
        
        return reward
    
    def reset(self):
        """Reset environment"""
        self.sim = Simulation(self.config.bandwidth, self.config.frequency, self.config.noise_power)
        self.sim.create_obstacles(self.config.obstacle_count, self.config.obstacle_size)
        
        self.operator = Drone(id="operator", position=[0, 0])
        self.sim.drones.append(self.operator)
        self.sim.deploy_relay(id="video_drone")
        
        self.step_count = 0
        
        # Return initial states
        initial_states = {}
        for drone in self.sim.drones:
            initial_states[drone.id] = self.get_state(drone.id)
        
        return initial_states

class FederatedRLExperiment:
    """Main experiment class combining FL and RL"""
    
    def __init__(self, config):
        self.config = config
        self.env = DroneEnvironment(config)
        
        # Initialize global model
        self.global_model = DroneNetwork(config.state_dim, 128, config.action_dim)
        
        # Initialize FL trainer
        self.fl_trainer = FederatedTrainer(
            config.fl_algorithm, 
            self.global_model,
            learning_rate=config.fl_learning_rate
        )
        
        # Initialize RL agents for each drone
        self.rl_agents = {}
        self._initialize_rl_agents()
        
        # Experiment tracking
        self.results = {
            'fl_rounds': [],
            'rl_episodes': [],
            'communication_metrics': [],
            'deployment_metrics': []
        }
    
    def _initialize_rl_agents(self):
        """Initialize RL agents for each drone"""
        for drone in self.env.sim.drones:
            agent = create_rl_agent(
                self.config.rl_algorithm,
                self.config.state_dim,
                self.config.action_dim,
                learning_rate=self.config.rl_learning_rate,
                gamma=self.config.rl_gamma
            )
            self.rl_agents[drone.id] = agent
    
    def run_federated_round(self):
        """Run one federated learning round"""
        # Select participating clients
        available_clients = list(self.rl_agents.keys())
        participating_clients = np.random.choice(
            available_clients, 
            min(self.config.clients_per_round, len(available_clients)),
            replace=False
        )
        
        # Prepare client data (simulated training data)
        client_data = {}
        for client_id in participating_clients:
            # Generate synthetic training data based on drone's experience
            # In practice, this would be real training data from the drone
            client_data[client_id] = self._generate_training_data(client_id)
        
        # Add clients to FL trainer if not already added
        for client_id in participating_clients:
            if client_id not in self.fl_trainer.clients:
                model = DroneNetwork(self.config.state_dim, 128, self.config.action_dim)
                self.fl_trainer.add_client(
                    client_id, 
                    model, 
                    local_epochs=self.config.local_epochs,
                    learning_rate=self.config.fl_learning_rate
                )
        
        # Run federated training round
        self.fl_trainer.train_round(participating_clients, client_data)
        
        # Update RL agents with new global model
        global_state = self.fl_trainer.get_global_model().state_dict()
        for client_id in participating_clients:
            if client_id in self.rl_agents:
                # Update the agent's model (this would need to be adapted based on your RL implementation)
                pass
        
        # Record metrics
        self.results['fl_rounds'].append({
            'round': self.fl_trainer.algorithm.round,
            'participating_clients': participating_clients,
            'total_clients': len(self.rl_agents)
        })
    
    def _generate_training_data(self, client_id):
        """Generate synthetic training data for a client"""
        # This is a placeholder - in practice, this would be real training data
        # collected from the drone's experience
        num_samples = np.random.randint(10, 100)
        return [(np.random.randn(self.config.state_dim), 
                np.random.randint(0, self.config.action_dim)) 
                for _ in range(num_samples)]
    
    def run_rl_episode(self):
        """Run one RL training episode"""
        states = self.env.reset()
        episode_rewards = {drone_id: 0.0 for drone_id in states.keys()}
        
        for step in range(self.config.simulation_steps):
            # Select actions for each drone
            actions = {}
            for drone_id, state in states.items():
                if drone_id in self.rl_agents:
                    agent = self.rl_agents[drone_id]
                    action = agent.select_action(state)
                    actions[drone_id] = action
            
            # Execute actions
            new_states, rewards, done = self.env.step(actions)
            
            # Store experience for RL agents
            for drone_id in states.keys():
                if drone_id in self.rl_agents:
                    agent = self.rl_agents[drone_id]
                    if drone_id in new_states and new_states[drone_id] is not None:
                        # Store experience (adapt based on your RL algorithm)
                        if hasattr(agent, 'remember'):
                            agent.remember(
                                states[drone_id], 
                                actions.get(drone_id, 0), 
                                rewards.get(drone_id, 0), 
                                new_states[drone_id], 
                                done
                            )
                    
                    episode_rewards[drone_id] += rewards.get(drone_id, 0)
            
            states = new_states
            
            if done:
                break
        
        # Update RL agents
        for drone_id, agent in self.rl_agents.items():
            if hasattr(agent, 'update'):
                agent.update()
        
        # Record metrics
        self.results['rl_episodes'].append({
            'episode': len(self.results['rl_episodes']) + 1,
            'rewards': episode_rewards,
            'total_reward': sum(episode_rewards.values())
        })
    
    def run_experiment(self, fl_rounds=None, rl_episodes=None):
        """Run the complete experiment"""
        if fl_rounds is None:
            fl_rounds = self.config.federated_rounds
        if rl_episodes is None:
            rl_episodes = fl_rounds * 5  # 5 RL episodes per FL round
        
        print(f"Starting experiment: {self.config.experiment_name}")
        print(f"FL Algorithm: {self.config.fl_algorithm}")
        print(f"RL Algorithm: {self.config.rl_algorithm}")
        print(f"FL Rounds: {fl_rounds}, RL Episodes: {rl_episodes}")
        
        for fl_round in range(fl_rounds):
            print(f"FL Round {fl_round + 1}/{fl_rounds}")
            self.run_federated_round()
            
            # Run multiple RL episodes between FL rounds
            episodes_per_round = rl_episodes // fl_rounds
            for episode in range(episodes_per_round):
                self.run_rl_episode()
        
        # Save results
        if self.config.save_models:
            self.save_experiment_results()
        
        print("Experiment completed!")
        return self.results
    
    def save_experiment_results(self):
        """Save experiment results and models"""
        # Create experiment directory
        exp_dir = f"experiments/{self.config.experiment_name}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration
        self.config.save(f"{exp_dir}/config.json")
        
        # Save results
        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save models
        if self.config.save_models:
            # Save global FL model
            self.fl_trainer.save_global_model(f"{exp_dir}/global_model.pth")
            
            # Save RL agent models
            for drone_id, agent in self.rl_agents.items():
                if hasattr(agent, 'save_model'):
                    agent.save_model(f"{exp_dir}/rl_agent_{drone_id}.pth")
        
        print(f"Results saved to {exp_dir}")

def run_comparison_experiment(configs):
    """Run comparison experiments with different configurations"""
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Running experiment: {config_name}")
        print(f"{'='*50}")
        
        experiment = FederatedRLExperiment(config)
        results[config_name] = experiment.run_experiment()
    
    # Compare results
    print("\n" + "="*50)
    print("EXPERIMENT COMPARISON")
    print("="*50)
    
    for config_name, result in results.items():
        avg_fl_reward = np.mean([r['total_reward'] for r in result['rl_episodes']])
        print(f"{config_name}: Average Reward = {avg_fl_reward:.2f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Create different configurations for comparison
    configs = {
        "PPO_FedAvg": ExperimentConfig(),
        "DQN_FedProx": ExperimentConfig(),
        "A2C_FedNova": ExperimentConfig()
    }
    
    # Modify configurations
    configs["PPO_FedAvg"].rl_algorithm = 'ppo'
    configs["PPO_FedAvg"].fl_algorithm = 'fedavg'
    
    configs["DQN_FedProx"].rl_algorithm = 'dqn'
    configs["DQN_FedProx"].fl_algorithm = 'fedprox'
    
    configs["A2C_FedNova"].rl_algorithm = 'a2c'
    configs["A2C_FedNova"].fl_algorithm = 'fednova'
    
    # Run comparison
    results = run_comparison_experiment(configs) 