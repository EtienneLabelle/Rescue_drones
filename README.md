# Drone Federated Learning Research Framework

A modular framework for researching federated learning applied to drone networks with reinforcement learning. This project allows easy switching between different RL and FL algorithms to test their effectiveness in drone communication scenarios.

## 🚁 Overview

This framework combines:
- **Reinforcement Learning (RL)**: DQN, PPO, A2C algorithms for drone decision making
- **Federated Learning (FL)**: FedAvg, FedProx, FedNova, FedAdam for collaborative learning
- **Drone Communication Simulation**: Realistic channel modeling with obstacles and interference

## 📁 Project Structure

```
Rescue_Drone_v1/
├── RL.py                    # Modular RL algorithms (DQN, PPO, A2C)
├── FL.py                    # Modular FL algorithms (FedAvg, FedProx, FedNova, FedAdam)
├── experiment_framework.py   # Unified experiment framework
├── example_usage.py         # Example usage and comparison experiments
├── main.py                  # Original simulation (unchanged)
├── env.py                   # Simulation environment
├── drones.py                # Drone classes
├── coms.py                  # Communication modeling
├── display.py               # Visualization
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Example Experiments

```bash
python example_usage.py
```

This will run comparison experiments with different RL+FL combinations and generate:
- Performance comparisons
- Learning curves
- Saved models and results

### 3. Custom Experiments

```python
from experiment_framework import ExperimentConfig, FederatedRLExperiment

# Create custom configuration
config = ExperimentConfig()
config.rl_algorithm = 'ppo'      # 'dqn', 'ppo', 'a2c'
config.fl_algorithm = 'fedavg'    # 'fedavg', 'fedprox', 'fednova', 'fedadam'
config.simulation_steps = 1000
config.federated_rounds = 10

# Run experiment
experiment = FederatedRLExperiment(config)
results = experiment.run_experiment()
```

## 🔧 Modular Architecture

### RL Algorithms (`RL.py`)

**Available Algorithms:**
- **DQN**: Deep Q-Network with experience replay
- **PPO**: Proximal Policy Optimization with clipping
- **A2C**: Advantage Actor-Critic

**Usage:**
```python
from RL import create_rl_agent

agent = create_rl_agent('ppo', state_dim=10, action_dim=4)
action = agent.select_action(state)
agent.update(batch)
```

### FL Algorithms (`FL.py`)

**Available Algorithms:**
- **FedAvg**: Federated Averaging (McMahan et al.)
- **FedProx**: Federated Proximal (Li et al.)
- **FedNova**: Federated Normalized Averaging (Wang et al.)
- **FedAdam**: Federated Adam (Reddi et al.)

**Usage:**
```python
from FL import FederatedTrainer

trainer = FederatedTrainer('fedavg', global_model)
trainer.add_client(client_id, model)
trainer.train_round(client_ids, client_data)
```

## 🧪 Experiment Framework

### Key Features

1. **Modular Design**: Easy switching between algorithms
2. **Comprehensive Metrics**: Track RL episodes, FL rounds, communication quality
3. **Visualization**: Automatic plotting of results
4. **Reproducibility**: Save/load configurations and models
5. **Comparison Tools**: Built-in A/B testing framework

### Configuration Options

```python
config = ExperimentConfig()

# RL Settings
config.rl_algorithm = 'ppo'
config.state_dim = 10
config.action_dim = 4
config.rl_learning_rate = 0.001

# FL Settings
config.fl_algorithm = 'fedavg'
config.federated_rounds = 10
config.clients_per_round = 3
config.local_epochs = 1

# Simulation Settings
config.simulation_steps = 2000
config.obstacle_count = 50
config.deployment_distance = 1000
```

## 📊 Research Contributions

### Potential Research Directions

1. **Bandwidth-Constrained FL**: How to perform federated learning with limited communication resources
2. **Mobility-Aware FL**: Adapting FL algorithms to handle moving drones
3. **Energy-Efficient FL**: Balancing learning performance with drone battery constraints
4. **Dynamic Network FL**: Handling changing network topologies as drones move

### Experiment Ideas

1. **Algorithm Comparison**: Test different RL+FL combinations
2. **Communication Constraints**: Study impact of bandwidth limitations
3. **Network Topology**: Analyze effect of different drone network configurations
4. **Scalability**: Test with varying numbers of drones

## 📈 Results Analysis

The framework automatically generates:

- **Performance Metrics**: Average rewards, learning curves
- **Communication Metrics**: Link quality, capacity, SINR
- **Deployment Metrics**: Relay placement, network topology
- **Visualizations**: Comparative plots, learning curves

## 🔬 Advanced Usage

### Custom RL Algorithms

```python
from RL import BaseRLAgent

class CustomRLAgent(BaseRLAgent):
    def select_action(self, state):
        # Your custom action selection logic
        pass
    
    def update(self, batch):
        # Your custom update logic
        pass
```

### Custom FL Algorithms

```python
from FL import BaseFederatedAlgorithm

class CustomFLAlgorithm(BaseFederatedAlgorithm):
    def aggregate_models(self, client_models, client_weights=None):
        # Your custom aggregation logic
        pass
```

### Custom Environment

```python
from experiment_framework import DroneEnvironment

class CustomDroneEnvironment(DroneEnvironment):
    def get_state(self, drone_id):
        # Your custom state representation
        pass
    
    def _calculate_reward(self, drone):
        # Your custom reward function
        pass
```

## 🛠️ Troubleshooting

### Common Issues

1. **PyTorch Import Error**: Install PyTorch with `pip install torch`
2. **Memory Issues**: Reduce `simulation_steps` or `federated_rounds`
3. **Slow Training**: Use smaller networks or fewer clients per round

### Performance Tips

1. **GPU Acceleration**: Use CUDA if available
2. **Batch Processing**: Increase batch sizes for faster training
3. **Parallel Processing**: Run multiple experiments in parallel

## 📚 References

- **FedAvg**: McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data"
- **FedProx**: Li, T., et al. "Federated optimization in heterogeneous networks"
- **FedNova**: Wang, J., et al. "Tackling the objective inconsistency problem in heterogeneous federated optimization"
- **PPO**: Schulman, J., et al. "Proximal policy optimization algorithms"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your custom algorithms or improvements
4. Submit a pull request

## 📄 License

This project is for research purposes. Please cite if used in academic work.

---

**Happy Researching! 🚁📊** 