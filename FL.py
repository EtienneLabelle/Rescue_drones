import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
import torch.nn.functional as F

class BaseFederatedAlgorithm(ABC):
    """Abstract base class for federated learning algorithms"""
    
    def __init__(self, global_model, learning_rate=0.001):
        self.global_model = global_model
        self.learning_rate = learning_rate
        self.round = 0
        
    @abstractmethod
    def aggregate_models(self, client_models, client_weights=None):
        """Aggregate client models to update global model"""
        pass
    
    @abstractmethod
    def distribute_model(self, client_id):
        """Distribute global model to client"""
        pass
    
    def get_global_model_state(self):
        """Get current global model state"""
        return deepcopy(self.global_model.state_dict())
    
    def set_global_model_state(self, state_dict):
        """Set global model state"""
        self.global_model.load_state_dict(state_dict)

class FedAvg(BaseFederatedAlgorithm):
    """Federated Averaging (FedAvg) implementation"""
    
    def __init__(self, global_model, learning_rate=0.001):
        super().__init__(global_model, learning_rate)
    
    def aggregate_models(self, client_models, client_weights=None):
        """
        Aggregate client models using weighted averaging
        
        Args:
            client_models: List of (model_state_dict, num_samples) tuples
            client_weights: Optional list of weights for each client
        """
        if not client_models:
            return
        
        # If no weights provided, use equal weights
        if client_weights is None:
            total_samples = sum(num_samples for _, num_samples in client_models)
            client_weights = [num_samples / total_samples for _, num_samples in client_models]
        
        # Initialize aggregated model with zeros
        aggregated_state = {}
        for key in client_models[0][0].keys():
            aggregated_state[key] = torch.zeros_like(client_models[0][0][key])
        
        # Weighted averaging
        for i, (model_state, _) in enumerate(client_models):
            weight = client_weights[i]
            for key in aggregated_state:
                aggregated_state[key] += weight * model_state[key]
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
    
    def distribute_model(self, client_id):
        """Distribute global model to client"""
        return deepcopy(self.global_model.state_dict())

class FedProx(BaseFederatedAlgorithm):
    """Federated Proximal (FedProx) implementation"""
    
    def __init__(self, global_model, learning_rate=0.001, mu=0.01):
        super().__init__(global_model, learning_rate)
        self.mu = mu  # Proximal term coefficient
    
    def aggregate_models(self, client_models, client_weights=None):
        """Same aggregation as FedAvg"""
        if not client_models:
            return
        
        if client_weights is None:
            total_samples = sum(num_samples for _, num_samples in client_models)
            client_weights = [num_samples / total_samples for _, num_samples in client_models]
        
        aggregated_state = {}
        for key in client_models[0][0].keys():
            aggregated_state[key] = torch.zeros_like(client_models[0][0][key])
        
        for i, (model_state, _) in enumerate(client_models):
            weight = client_weights[i]
            for key in aggregated_state:
                aggregated_state[key] += weight * model_state[key]
        
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
    
    def distribute_model(self, client_id):
        """Distribute global model to client with proximal term info"""
        return {
            'model_state': deepcopy(self.global_model.state_dict()),
            'mu': self.mu
        }

class FedNova(BaseFederatedAlgorithm):
    """Federated Normalized Averaging (FedNova) implementation"""
    
    def __init__(self, global_model, learning_rate=0.001):
        super().__init__(global_model, learning_rate)
        self.client_epochs = {}  # Track epochs per client
    
    def aggregate_models(self, client_models, client_weights=None, client_epochs=None):
        """
        Aggregate with normalization for different local epochs
        
        Args:
            client_models: List of (model_state_dict, num_samples) tuples
            client_weights: Optional list of weights for each client
            client_epochs: List of local epochs for each client
        """
        if not client_models:
            return
        
        if client_weights is None:
            total_samples = sum(num_samples for _, num_samples in client_models)
            client_weights = [num_samples / total_samples for _, num_samples in client_models]
        
        if client_epochs is None:
            client_epochs = [1] * len(client_models)  # Default to 1 epoch
        
        # Normalize by local epochs
        normalized_weights = []
        for i, (_, num_samples) in enumerate(client_models):
            weight = client_weights[i]
            epochs = client_epochs[i]
            normalized_weight = weight / epochs
            normalized_weights.append(normalized_weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(normalized_weights)
        normalized_weights = [w / total_weight for w in normalized_weights]
        
        # Aggregate with normalized weights
        aggregated_state = {}
        for key in client_models[0][0].keys():
            aggregated_state[key] = torch.zeros_like(client_models[0][0][key])
        
        for i, (model_state, _) in enumerate(client_models):
            weight = normalized_weights[i]
            for key in aggregated_state:
                aggregated_state[key] += weight * model_state[key]
        
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
    
    def distribute_model(self, client_id):
        """Distribute global model to client"""
        return deepcopy(self.global_model.state_dict())

class FedAdam(BaseFederatedAlgorithm):
    """Federated Adam implementation"""
    
    def __init__(self, global_model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(global_model, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def aggregate_models(self, client_models, client_weights=None):
        """Aggregate and apply Adam update"""
        if not client_models:
            return
        
        if client_weights is None:
            total_samples = sum(num_samples for _, num_samples in client_models)
            client_weights = [num_samples / total_samples for _, num_samples in client_models]
        
        # Calculate weighted average of model updates
        current_state = self.global_model.state_dict()
        update_state = {}
        
        for key in current_state.keys():
            update_state[key] = torch.zeros_like(current_state[key])
        
        for i, (model_state, _) in enumerate(client_models):
            weight = client_weights[i]
            for key in update_state:
                update_state[key] += weight * (model_state[key] - current_state[key])
        
        # Initialize Adam moments if first round
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in update_state:
                self.m[key] = torch.zeros_like(update_state[key])
                self.v[key] = torch.zeros_like(update_state[key])
        
        # Apply Adam update
        self.t += 1
        for key in update_state:
            g = update_state[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            current_state[key] += self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        self.global_model.load_state_dict(current_state)
        self.round += 1
    
    def distribute_model(self, client_id):
        """Distribute global model to client"""
        return deepcopy(self.global_model.state_dict())

class Client:
    """Federated learning client"""
    
    def __init__(self, client_id, model, local_epochs=1, learning_rate=0.001):
        self.client_id = client_id
        self.model = model
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.data_size = 0
    
    def train(self, train_data, global_model_state=None, algorithm_info=None):
        """
        Train the client model
        
        Args:
            train_data: Training data for this client
            global_model_state: Global model state to start from
            algorithm_info: Additional info from FL algorithm (e.g., mu for FedProx)
        """
        if global_model_state is not None:
            self.model.load_state_dict(global_model_state)
        
        self.data_size = len(train_data)
        
        for epoch in range(self.local_epochs):
            self.model.train()
            for batch in train_data:
                self.optimizer.zero_grad()
                
                if algorithm_info and 'mu' in algorithm_info:
                    # FedProx proximal term
                    mu = algorithm_info['mu']
                    proximal_term = 0
                    for param, global_param in zip(self.model.parameters(), 
                                                 torch.load(global_model_state).values()):
                        proximal_term += (mu / 2) * torch.norm(param - global_param) ** 2
                    
                    # Add proximal term to loss (assuming loss is computed elsewhere)
                    # This is a simplified version - in practice, you'd integrate with your loss function
                    pass
                
                # Your training loop here
                # loss.backward()
                # self.optimizer.step()
    
    def get_model_state(self):
        """Get current model state and data size"""
        return self.model.state_dict(), self.data_size

# Factory function to create FL algorithms
def create_fl_algorithm(algorithm_type, global_model, **kwargs):
    """Factory function to create different FL algorithms"""
    algorithms = {
        'fedavg': FedAvg,
        'fedprox': FedProx,
        'fednova': FedNova,
        'fedadam': FedAdam
    }
    
    if algorithm_type not in algorithms:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available: {list(algorithms.keys())}")
    
    return algorithms[algorithm_type](global_model, **kwargs)

class FederatedTrainer:
    """Main federated learning trainer"""
    
    def __init__(self, algorithm_type, global_model, **algorithm_kwargs):
        self.algorithm = create_fl_algorithm(algorithm_type, global_model, **algorithm_kwargs)
        self.clients = {}
        self.round_history = []
    
    def add_client(self, client_id, model, local_epochs=1, learning_rate=0.001):
        """Add a client to the federation"""
        self.clients[client_id] = Client(client_id, model, local_epochs, learning_rate)
    
    def train_round(self, client_ids, client_data):
        """
        Perform one round of federated training
        
        Args:
            client_ids: List of client IDs to participate in this round
            client_data: Dict mapping client_id to training data
        """
        # Distribute global model to clients
        client_models = []
        client_weights = []
        client_epochs = []
        
        for client_id in client_ids:
            if client_id not in self.clients:
                raise ValueError(f"Client {client_id} not found")
            
            client = self.clients[client_id]
            global_state = self.algorithm.distribute_model(client_id)
            
            # Get algorithm-specific info
            algorithm_info = None
            if isinstance(self.algorithm, FedProx):
                algorithm_info = {'mu': self.algorithm.mu}
            
            # Train client
            client.train(client_data[client_id], global_state, algorithm_info)
            
            # Get updated model
            model_state, data_size = client.get_model_state()
            client_models.append((model_state, data_size))
            client_weights.append(data_size)
            client_epochs.append(client.local_epochs)
        
        # Aggregate models
        if isinstance(self.algorithm, FedNova):
            self.algorithm.aggregate_models(client_models, client_weights, client_epochs)
        else:
            self.algorithm.aggregate_models(client_models, client_weights)
        
        # Record round info
        self.round_history.append({
            'round': self.algorithm.round,
            'participating_clients': client_ids,
            'total_samples': sum(client_weights)
        })
    
    def get_global_model(self):
        """Get current global model"""
        return self.algorithm.global_model
    
    def save_global_model(self, path):
        """Save global model"""
        torch.save(self.algorithm.global_model.state_dict(), path)
    
    def load_global_model(self, path):
        """Load global model"""
        self.algorithm.global_model.load_state_dict(torch.load(path)) 