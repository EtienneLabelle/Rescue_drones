import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from abc import ABC, abstractmethod

class BaseRLAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
    @abstractmethod
    def select_action(self, state):
        pass
    
    @abstractmethod
    def update(self, batch):
        pass
    
    @abstractmethod
    def save_model(self, path):
        pass
    
    @abstractmethod
    def load_model(self, path):
        pass

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseRLAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000):
        super().__init__(state_dim, action_dim, learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent(BaseRLAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, gamma=0.99, 
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__(state_dim, action_dim, learning_rate)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, batch_size=None):
        if len(self.memory) == 0:
            return
        
        states = torch.FloatTensor([e[0] for e in self.memory])
        actions = torch.LongTensor([e[1] for e in self.memory])
        rewards = torch.FloatTensor([e[2] for e in self.memory])
        next_states = torch.FloatTensor([e[3] for e in self.memory])
        dones = torch.BoolTensor([e[4] for e in self.memory])
        old_log_probs = torch.FloatTensor([e[5] for e in self.memory])
        
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate values and advantages
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        
        # PPO update
        for _ in range(10):
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            entropy_loss = -dist.entropy().mean()
            
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        self.memory = []
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class A2CAgent(BaseRLAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate)
        self.gamma = gamma
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, batch_size=None):
        if len(self.memory) == 0:
            return
        
        states = torch.FloatTensor([e[0] for e in self.memory])
        actions = torch.LongTensor([e[1] for e in self.memory])
        rewards = torch.FloatTensor([e[2] for e in self.memory])
        next_states = torch.FloatTensor([e[3] for e in self.memory])
        dones = torch.BoolTensor([e[4] for e in self.memory])
        log_probs = torch.FloatTensor([e[5] for e in self.memory])
        
        # Calculate returns and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        returns = []
        for i, (reward, done, next_value) in enumerate(zip(rewards, dones, next_values)):
            if done:
                returns.append(reward)
            else:
                returns.append(reward + self.gamma * next_value)
        
        returns = torch.FloatTensor(returns)
        advantages = returns - values.detach()
        
        # Actor loss
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        actor_loss = -(new_log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values, returns)
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        self.memory = []
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class MADDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights and biases properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)  # Zero bias for output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Continuous actions in [-1, 1]

class MADDPGCritic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CentralizedReplayBuffer:
    """Centralized replay buffer for multi-agent MADDPG"""
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, joint_state, joint_action, joint_reward, joint_next_state, joint_done):
        """Store joint experience for all agents"""
        self.buffer.append((joint_state, joint_action, joint_reward, joint_next_state, joint_done))
    
    def sample(self, batch_size):
        """Sample batch of joint experiences"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        joint_states, joint_actions, joint_rewards, joint_next_states, joint_dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(joint_states)),
            torch.FloatTensor(np.array(joint_actions)),
            torch.FloatTensor(np.array(joint_rewards)),
            torch.FloatTensor(np.array(joint_next_states)),
            torch.BoolTensor(np.array(joint_dones))
        )
    
    def __len__(self):
        return len(self.buffer)

class MADDPGAgent(BaseRLAgent):
    def __init__(self, agent_id, state_dim, action_dim, num_agents,
                 learning_rate=0.001, gamma=0.99, tau=0.01, shared_buffer=None):
        super().__init__(state_dim, action_dim, learning_rate)
        
        self.agent_id = agent_id
        self.num_agents = num_agents
        
        # Compute dimensions dynamically
        self.total_state_dim = state_dim * num_agents
        self.total_action_dim = action_dim * num_agents
        
        self.gamma = gamma
        self.tau = tau
        
        # Shared centralized replay buffer
        self.shared_buffer = shared_buffer
        
        # Networks
        self.actor = MADDPGActor(state_dim, action_dim)
        self.critic = MADDPGCritic(self.total_state_dim, self.total_action_dim)
        self.target_actor = MADDPGActor(state_dim, action_dim)
        self.target_critic = MADDPGCritic(self.total_state_dim, self.total_action_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize target networks
        self.update_target_networks(tau=1.0)
    
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def remember(self, joint_state, joint_action, joint_reward, joint_next_state, joint_done):
        """Store experience in shared buffer - called by environment"""
        if self.shared_buffer is not None:
            self.shared_buffer.push(joint_state, joint_action, joint_reward, joint_next_state, joint_done)
    
    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        
        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update(self, batch_size=32, return_losses=False):
        """Update using centralized replay buffer"""
        if self.shared_buffer is None or len(self.shared_buffer) < batch_size:
            return None if return_losses else None
        
        # Sample from shared buffer
        batch = self.shared_buffer.sample(batch_size)
        if batch is None:
            return None if return_losses else None
        
        joint_states, joint_actions, joint_rewards, joint_next_states, joint_dones = batch
        
        # Extract this agent's data from joint tensors
        # agent_id format is "uav_1", "uav_2", etc. (1-indexed)
        # Convert to 0-indexed for array access
        agent_idx = int(self.agent_id.split('_')[1]) - 1  # uav_1 -> 0, uav_2 -> 1, etc.
        state_dim = self.state_dim
        action_dim = self.action_dim
        
        # Consistent ordering: agent_0, agent_1, agent_2, etc.
        start_state_idx = agent_idx * state_dim
        end_state_idx = start_state_idx + state_dim
        start_action_idx = agent_idx * action_dim
        end_action_idx = start_action_idx + action_dim
        
        # Extract this agent's states and actions
        states = joint_states[:, start_state_idx:end_state_idx]
        actions = joint_actions[:, start_action_idx:end_action_idx]
        rewards = joint_rewards[:, agent_idx]  # This agent's reward
        next_states = joint_next_states[:, start_state_idx:end_state_idx]
        dones = joint_dones[:, agent_idx]  # This agent's done flag
        
        # Critic update
        current_q_values = self.critic(joint_states, joint_actions)
        
        with torch.no_grad():
            # Compute next actions for all agents
            next_actions = []
            for i in range(self.num_agents):
                start_idx = i * state_dim
                end_idx = start_idx + state_dim
                agent_next_states = joint_next_states[:, start_idx:end_idx]
                next_actions.append(self.target_actor(agent_next_states))
            
            # Concatenate all next actions in consistent order
            next_joint_actions = torch.cat(next_actions, dim=1)
            next_q_values = self.target_critic(joint_next_states, next_joint_actions)
            
            # Compute target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        self.actor_optimizer.zero_grad()
        
        # Compute new actions for this agent
        new_actions = self.actor(states)
        
        # Create joint actions with new action for this agent, old actions for others
        new_joint_actions = joint_actions.clone()
        new_joint_actions[:, start_action_idx:end_action_idx] = new_actions
        
        # Compute actor loss (no action bias penalties - let critic shape behavior)
        actor_q_values = self.critic(joint_states, new_joint_actions)
        actor_loss = -actor_q_values.mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        if return_losses:
            return (critic_loss.item() + actor_loss.item()) / 2  # Return average loss
        return None
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'agent_id': self.agent_id
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.agent_id = checkpoint['agent_id']

    def reset_critic(self):
        """Reset critic network to fix backwards learning"""
        self.critic = MADDPGCritic(self.total_state_dim, self.total_action_dim).to(self.device)
        self.target_critic = MADDPGCritic(self.total_state_dim, self.total_action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.update_target_networks(tau=1.0)  # Copy weights immediately

# Factory function
def create_rl_agent(agent_type, state_dim, action_dim, **kwargs):
    """Factory function to create RL agents"""
    agents = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'a2c': A2CAgent,
        'maddpg': MADDPGAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    # Handle MADDPG special case
    if agent_type == 'maddpg':
        if 'agent_id' not in kwargs or 'num_agents' not in kwargs:
            raise ValueError("MADDPG requires 'agent_id' and 'num_agents' parameters")
        return agents[agent_type](kwargs['agent_id'], state_dim, action_dim, kwargs['num_agents'], **kwargs)
    
    return agents[agent_type](state_dim, action_dim, **kwargs)
