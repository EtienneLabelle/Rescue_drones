import os
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
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
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
        
        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        # CRITICAL FIX: Proper shape handling for DQN targets
        rewards = rewards.unsqueeze(1)  # (B,) -> (B,1)
        dones = dones.float().unsqueeze(1)  # (B,) -> (B,1)
        next_q_values = next_q_values.unsqueeze(1)  # (B,) -> (B,1)
        
        target_q_values = rewards + (self.gamma * next_q_values * (1.0 - dones))
        
        # CRITICAL FIX: Consistent shapes for loss computation
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
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
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
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
        
        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, batch_size=None):
        if len(self.memory) == 0:
            return
        
        states = torch.FloatTensor([e[0] for e in self.memory]).to(self.device)
        actions = torch.LongTensor([e[1] for e in self.memory]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in self.memory]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in self.memory]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([e[5] for e in self.memory]).to(self.device)
        
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate values and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        returns = []
        for i, (reward, done, next_value) in enumerate(zip(rewards, dones, next_values)):
            if done:
                returns.append(reward)
            else:
                returns.append(reward + self.gamma * next_value)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # CRITICAL FIX: Ensure consistent shapes for A2C
        values = values.unsqueeze(1)  # (B,) -> (B,1) to match critic output
        returns = returns.unsqueeze(1)  # (B,) -> (B,1)
        
        advantages = returns - values.detach()
        
        # Actor loss
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        actor_loss = -(new_log_probs * advantages.squeeze()).mean()  # Squeeze advantages for actor
        critic_loss = nn.MSELoss()(values, returns)  # Both (B,1)
        
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

class A2CActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class A2CCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class A2CAgent(BaseRLAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        super().__init__(state_dim, action_dim, learning_rate)
        self.gamma = gamma
        
        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = A2CActor(state_dim, action_dim).to(self.device)
        self.critic = A2CCritic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, batch_size=None):
        if len(self.memory) == 0:
            return
        
        states = torch.FloatTensor([e[0] for e in self.memory]).to(self.device)
        actions = torch.LongTensor([e[1] for e in self.memory]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in self.memory]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in self.memory]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in self.memory]).to(self.device)
        log_probs = torch.FloatTensor([e[5] for e in self.memory]).to(self.device)
        
        # Calculate returns and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        returns = []
        for i, (reward, done, next_value) in enumerate(zip(rewards, dones, next_values)):
            if done:
                returns.append(reward)
            else:
                returns.append(reward + self.gamma * next_value)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # CRITICAL FIX: Ensure consistent shapes for A2C
        values = values.unsqueeze(1)  # (B,) -> (B,1) to match critic output
        returns = returns.unsqueeze(1)  # (B,) -> (B,1)
        
        advantages = returns - values.detach()
        
        # Actor loss
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        actor_loss = -(new_log_probs * advantages.squeeze()).mean()  # Squeeze advantages for actor
        critic_loss = nn.MSELoss()(values, returns)  # Both (B,1)
        
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

class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))  # Continuous actions in [-1, 1]

class DDPGCritic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        for layer in (self.fc1, self.fc2, self.fc3):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

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

class PrivateReplayBuffer:
    """Per-agent replay buffer — each UAV stores only its own experience."""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(np.array(dones)),
        )

    def __len__(self):
        return len(self.buffer)


def collect_step(agents: dict, prev_states: dict, actions: dict,
                 rewards: dict, next_states: dict, done: bool):
    """Push one environment step of experience into each agent's buffer.

    Distributed mode: each agent receives its own (s, a, r, s').
    Centralized mode: builds joint vectors and pushes once to the shared buffer
                      (all agents reference the same CentralizedReplayBuffer).
    """
    first = next(iter(agents.values()))

    if first.distributed:
        for aid, agent in agents.items():
            if aid in prev_states and aid in next_states:
                agent.remember(
                    np.array(prev_states[aid]),
                    np.array(actions.get(aid, [0] * agent.action_dim)),
                    rewards.get(aid, 0.0),
                    np.array(next_states[aid]),
                    done,
                )
    else:
        joint_state, joint_action = [], []
        joint_reward, joint_next_state, joint_done = [], [], []
        for aid in sorted(agents.keys()):
            agent = agents[aid]
            if aid in prev_states and aid in next_states:
                joint_state.extend(prev_states[aid])
                joint_action.extend(actions.get(aid, [0] * agent.action_dim))
                joint_reward.append(rewards.get(aid, 0.0))
                joint_next_state.extend(next_states[aid])
            else:
                joint_state.extend([0] * agent.state_dim)
                joint_action.extend([0] * agent.action_dim)
                joint_reward.append(0.0)
                joint_next_state.extend([0] * agent.state_dim)
            joint_done.append(done)
        first.remember(
            np.array(joint_state),  np.array(joint_action),
            np.array(joint_reward), np.array(joint_next_state),
            np.array(joint_done),
        )


class DDPGAgent(BaseRLAgent):
    """Base DDPG agent: actor + target actor, select_action, soft_update, save/load.

    Subclasses supply the critic, replay buffer, remember(), and update().
    """

    # Set by each subclass so collect_step can branch without isinstance checks.
    distributed: bool
    shared_buffer = None

    def __init__(self, agent_id, state_dim, action_dim,
                 learning_rate=0.001, actor_lr=None, critic_lr=None,
                 gamma=0.99, tau=0.01):
        super().__init__(state_dim, action_dim, learning_rate)
        self.agent_id = agent_id
        self.gamma    = gamma
        self.tau      = tau
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _actor_lr = actor_lr if actor_lr is not None else learning_rate
        self._critic_lr = critic_lr if critic_lr is not None else learning_rate

        self.actor        = DDPGActor(state_dim, action_dim).to(self.device)
        self.target_actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=_actor_lr)
        self.soft_update(self.actor, self.target_actor, tau=1.0)

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state, explore=False, noise_scale=0.0):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).squeeze(0).detach().cpu().numpy()
        if explore and noise_scale > 0:
            action = np.clip(action + np.random.normal(0, noise_scale, action.shape), -1, 1)
        return action

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            self.agent_id: {
                'actor':            self.actor.state_dict(),
                'critic':           self.critic.state_dict(),
                'target_actor':     self.target_actor.state_dict(),
                'target_critic':    self.target_critic.state_dict(),
                'actor_optimizer':  self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }
        }, path)

    def load_model(self, path):
        raw = torch.load(path, map_location=self.device)
        states = raw[self.agent_id] if self.agent_id in raw else raw
        self.actor.load_state_dict(states['actor'])
        self.critic.load_state_dict(states['critic'])
        self.target_actor.load_state_dict(states['target_actor'])
        self.target_critic.load_state_dict(states['target_critic'])
        if 'actor_optimizer' in states:
            self.actor_optimizer.load_state_dict(states['actor_optimizer'])
        if 'critic_optimizer' in states:
            self.critic_optimizer.load_state_dict(states['critic_optimizer'])


class IDDPGAgent(DDPGAgent):
    """Independent DDPG + gossip FL.

    Each agent trains independently with its own private replay buffer and a
    local critic (own obs + action only).  Actor weights are periodically
    synchronised with neighbours via gossip, which is the only coordination
    mechanism — there is no shared/centralised critic.
    """

    distributed  = True
    shared_buffer = None

    def __init__(self, agent_id, state_dim, action_dim,
                 learning_rate=0.001, actor_lr=None, critic_lr=None,
                 gamma=0.99, tau=0.01, buffer_size=100000):
        super().__init__(agent_id, state_dim, action_dim,
                         learning_rate, actor_lr, critic_lr, gamma, tau)
        self.private_buffer = PrivateReplayBuffer(max_size=buffer_size)
        self.critic        = DDPGCritic(state_dim, action_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._critic_lr)
        self.soft_update(self.critic, self.target_critic, tau=1.0)
        # kept for model_io / checkpoint compatibility
        self.total_state_dim  = state_dim
        self.total_action_dim = action_dim

    def remember(self, state, action, reward, next_state, done):
        self.private_buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size=32, return_losses=False, **_):
        if len(self.private_buffer) < batch_size:
            return None
        batch = self.private_buffer.sample(batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones       = dones.float().to(self.device).unsqueeze(1)

        with torch.no_grad():
            target_q = rewards + self.gamma * (1.0 - dones) * \
                       self.target_critic(next_states, self.target_actor(next_states))
        critic_loss = nn.MSELoss()(self.critic(states, actions), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.soft_update(self.actor,  self.target_actor,  self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)
        return float(critic_loss.item()) if return_losses else None

    def reset_critic(self):
        self.critic        = DDPGCritic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = DDPGCritic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._critic_lr)
        self.soft_update(self.critic, self.target_critic, tau=1.0)


class MADDPGAgent(DDPGAgent):
    """Multi-Agent DDPG with centralised critic.

    All agents share a single CentralizedReplayBuffer.  Each agent's critic
    receives the full joint observation and joint action vectors, giving it
    global visibility during training while the actor remains decentralised.
    """

    distributed = False

    def __init__(self, agent_id, state_dim, action_dim, num_agents,
                 learning_rate=0.001, actor_lr=None, critic_lr=None,
                 gamma=0.99, tau=0.01, shared_buffer=None):
        super().__init__(agent_id, state_dim, action_dim,
                         learning_rate, actor_lr, critic_lr, gamma, tau)
        self.num_agents    = num_agents
        self.shared_buffer = shared_buffer
        self.private_buffer = None

        critic_state_dim  = state_dim  * num_agents
        critic_action_dim = action_dim * num_agents
        self.total_state_dim  = critic_state_dim
        self.total_action_dim = critic_action_dim

        self.critic        = DDPGCritic(critic_state_dim, critic_action_dim).to(self.device)
        self.target_critic = DDPGCritic(critic_state_dim, critic_action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._critic_lr)
        self.soft_update(self.critic, self.target_critic, tau=1.0)

    def remember(self, joint_state, joint_action, reward, joint_next_state, done):
        if self.shared_buffer is not None:
            self.shared_buffer.push(joint_state, joint_action, reward, joint_next_state, done)

    def update(self, batch_size=32, all_target_actors=None, return_losses=False):
        if self.shared_buffer is None or len(self.shared_buffer) < batch_size:
            return None
        batch = self.shared_buffer.sample(batch_size)
        if batch is None:
            return None

        joint_states, joint_actions, joint_rewards, joint_next_states, joint_dones = batch
        joint_states      = joint_states.to(self.device)
        joint_actions     = joint_actions.to(self.device)
        joint_rewards     = joint_rewards.to(self.device)
        joint_next_states = joint_next_states.to(self.device)
        joint_dones       = joint_dones.to(self.device)

        agent_idx        = int(self.agent_id.split('_')[1]) - 1
        sd, ad           = self.state_dim, self.action_dim
        s0, s1           = agent_idx * sd,  agent_idx * sd  + sd
        a0, a1           = agent_idx * ad,  agent_idx * ad  + ad

        states  = joint_states[:, s0:s1]
        rewards = joint_rewards[:, agent_idx].unsqueeze(1)
        dones   = joint_dones[:, agent_idx].float().unsqueeze(1)

        with torch.no_grad():
            if all_target_actors is None:
                all_target_actors = [self.target_actor] * self.num_agents
            next_acts = torch.cat([
                all_target_actors[i](joint_next_states[:, i*sd:(i+1)*sd])
                for i in range(self.num_agents)
            ], dim=1)
            target_q = rewards + self.gamma * (1.0 - dones) * \
                       self.target_critic(joint_next_states, next_acts)

        critic_loss = nn.MSELoss()(self.critic(joint_states, joint_actions), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        new_joint_actions = joint_actions.clone()
        new_joint_actions[:, a0:a1] = self.actor(states)
        actor_loss = -self.critic(joint_states, new_joint_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.soft_update(self.actor,  self.target_actor,  self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)
        return float(critic_loss.item()) if return_losses else None

    def reset_critic(self):
        self.critic        = DDPGCritic(self.total_state_dim, self.total_action_dim).to(self.device)
        self.target_critic = DDPGCritic(self.total_state_dim, self.total_action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._critic_lr)
        self.soft_update(self.critic, self.target_critic, tau=1.0)

# Factory function
def create_rl_agent(agent_type, state_dim, action_dim, **kwargs):
    """Factory function to create RL agents.

    agent_type options:
        'dqn'    — DQNAgent
        'ppo'    — PPOAgent
        'a2c'    — A2CAgent
        'iddpg'  — IDDPGAgent  (independent DDPG + gossip FL; requires agent_id)
        'maddpg' — MADDPGAgent (centralised critic; requires agent_id, num_agents)
    """
    registry = {
        'dqn':    DQNAgent,
        'ppo':    PPOAgent,
        'a2c':    A2CAgent,
        'iddpg':  IDDPGAgent,
        'maddpg': MADDPGAgent,
    }

    if agent_type not in registry:
        raise ValueError(f"Unknown agent type: {agent_type!r}. Available: {list(registry)}")

    if agent_type == 'iddpg':
        if 'agent_id' not in kwargs:
            raise ValueError("IDDPGAgent requires 'agent_id'")
        return IDDPGAgent(kwargs.pop('agent_id'), state_dim, action_dim, **kwargs)

    if agent_type == 'maddpg':
        if 'agent_id' not in kwargs or 'num_agents' not in kwargs:
            raise ValueError("MADDPGAgent requires 'agent_id' and 'num_agents'")
        return MADDPGAgent(kwargs.pop('agent_id'), state_dim, action_dim,
                           kwargs.pop('num_agents'), **kwargs)

    if agent_type == 'hybrid_ppo':
        if 'n_clients' not in kwargs:
            raise ValueError("HybridPPOAgent requires 'n_clients'")
        return HybridPPOAgent(state_dim, kwargs.pop('n_clients'), **kwargs)

    return registry[agent_type](state_dim, action_dim, **kwargs)


# ===========================================================================
# RL-for-FL: multi-head PPO for the UAV relay problem
# ===========================================================================

class HybridPPOActor(nn.Module):
    """Multi-head actor: displacement (Gaussian+tanh), selection (Bernoulli),
    bandwidth (Gaussian logits; env applies softmax).

    Action stored in trajectory:
        [dx, dy (tanh-squashed),  sel_0..N-1 ({0,1}),  bw_0..N-1 (Gaussian raw)]

    NOTE: client selection is a RELAXED continuous approximation of the true
    discrete combinatorial head.  The actor outputs Bernoulli logits; the env
    thresholds at sigmoid > 0.5.  Full discrete selection would require
    REINFORCE-over-subsets, deferred to a future pass.
    """

    def __init__(self, state_dim: int, n_clients: int, hidden_dim: int = 256):
        super().__init__()
        self.n_clients = n_clients

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Displacement head — Gaussian mean, tanh-squashed to [-1, 1]
        self.disp_mean    = nn.Linear(hidden_dim, 2)
        self.disp_log_std = nn.Parameter(torch.zeros(2) - 0.5)

        # Selection head — Bernoulli logits → sample {0, 1}
        self.sel_head = nn.Linear(hidden_dim, n_clients)

        # Bandwidth head — Gaussian logits; softmax applied in env
        self.bw_mean    = nn.Linear(hidden_dim, n_clients)
        self.bw_log_std = nn.Parameter(torch.zeros(n_clients) - 0.5)

        # Orthogonal init for trunk; small gain for output layers
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for layer in (self.disp_mean, self.sel_head, self.bw_mean):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def get_action_and_logprob(self, x: torch.Tensor):
        """Sample action and return (action_tensor, log_prob_scalar)."""
        h = self._trunk(x)

        # Displacement — Normal + tanh squash
        d_mean = self.disp_mean(h)
        d_std  = self.disp_log_std.clamp(-4, 2).exp()
        d_dist = torch.distributions.Normal(d_mean, d_std)
        d_raw  = d_dist.rsample()
        d_act  = torch.tanh(d_raw)
        lp_d   = (d_dist.log_prob(d_raw)
                  - torch.log(1.0 - d_act.pow(2) + 1e-6)).sum(-1)

        # Selection — Bernoulli
        sel_logits = self.sel_head(h)
        sel_dist   = torch.distributions.Bernoulli(logits=sel_logits)
        sel_act    = sel_dist.sample()
        lp_s       = sel_dist.log_prob(sel_act).sum(-1)

        # Bandwidth — Gaussian logits (softmax handled in env)
        b_mean = self.bw_mean(h)
        b_std  = self.bw_log_std.clamp(-4, 2).exp()
        b_dist = torch.distributions.Normal(b_mean, b_std)
        b_raw  = b_dist.rsample()
        lp_b   = b_dist.log_prob(b_raw).sum(-1)

        log_prob = lp_d + lp_s + lp_b
        action   = torch.cat([d_act, sel_act, b_raw], dim=-1)
        return action, log_prob

    def log_prob_of(self, x: torch.Tensor, action: torch.Tensor):
        """Recompute log_prob + entropy for a stored (state, action) batch."""
        N = self.n_clients
        h = self._trunk(x)

        d_act   = action[:, :2]
        sel_act = action[:, 2:2 + N]
        b_raw   = action[:, 2 + N:]

        # Displacement
        d_mean  = self.disp_mean(h)
        d_std   = self.disp_log_std.clamp(-4, 2).exp()
        d_dist  = torch.distributions.Normal(d_mean, d_std)
        d_raw   = torch.atanh(d_act.clamp(-0.9999, 0.9999))
        lp_d    = (d_dist.log_prob(d_raw)
                   - torch.log(1.0 - d_act.pow(2) + 1e-6)).sum(-1)
        ent_d   = d_dist.entropy().sum(-1)

        # Selection
        sel_logits = self.sel_head(h)
        sel_dist   = torch.distributions.Bernoulli(logits=sel_logits)
        lp_s       = sel_dist.log_prob(sel_act).sum(-1)
        ent_s      = sel_dist.entropy().sum(-1)

        # Bandwidth
        b_mean = self.bw_mean(h)
        b_std  = self.bw_log_std.clamp(-4, 2).exp()
        b_dist = torch.distributions.Normal(b_mean, b_std)
        lp_b   = b_dist.log_prob(b_raw).sum(-1)
        ent_b  = b_dist.entropy().sum(-1)

        return lp_d + lp_s + lp_b, ent_d + ent_s + ent_b


class HybridPPOAgent(BaseRLAgent):
    """Single-agent PPO for UAV-relay FL optimisation.

    Uses GAE for advantage estimation and clipped surrogate objective.
    Trajectory is collected for a full episode then updated in mini-batches.
    """

    def __init__(self, state_dim: int, n_clients: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, ppo_epochs: int = 4,
                 batch_size: int = 64, hidden_dim: int = 256):
        action_dim = 2 + 2 * n_clients
        super().__init__(state_dim, action_dim, learning_rate)
        self.n_clients   = n_clients
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_ratio  = clip_ratio
        self.value_coef  = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs  = ppo_epochs
        self.batch_size  = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor  = HybridPPOActor(state_dim, n_clients, hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )

        self.trajectory: list = []

    # ------ action selection ------

    def select_action(self, state: np.ndarray):
        """Returns (action_np, log_prob_float, value_float)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, lp_t = self.actor.get_action_and_logprob(s)
            value_t        = self.critic(s)
        return (
            action_t.squeeze(0).cpu().numpy(),
            float(lp_t.item()),
            float(value_t.item()),
        )

    def store(self, state, action, log_prob, reward, done, value):
        self.trajectory.append((state, action, log_prob, reward, done, value))

    # ------ PPO update ------

    def update(self) -> dict:
        if not self.trajectory:
            return {}

        states   = torch.FloatTensor(np.array([t[0] for t in self.trajectory])).to(self.device)
        actions  = torch.FloatTensor(np.array([t[1] for t in self.trajectory])).to(self.device)
        old_lps  = torch.FloatTensor([t[2] for t in self.trajectory]).to(self.device)
        rewards  = [t[3] for t in self.trajectory]
        dones    = [t[4] for t in self.trajectory]
        values   = [t[5] for t in self.trajectory]

        # ---- GAE advantage estimation ----
        advantages, returns = [], []
        gae = 0.0
        next_val = 0.0
        for i in reversed(range(len(rewards))):
            mask     = 0.0 if dones[i] else 1.0
            delta    = rewards[i] + self.gamma * next_val * mask - values[i]
            gae      = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_val = values[i]

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO mini-batch updates ----
        T = len(self.trajectory)
        logs: dict[str, list] = dict(actor_loss=[], critic_loss=[], entropy=[])

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                mb      = idx[start:start + self.batch_size]
                new_lps, entropy = self.actor.log_prob_of(states[mb], actions[mb])
                ratio   = torch.exp(new_lps - old_lps[mb])

                adv_mb  = advantages[mb]
                surr1   = ratio * adv_mb
                surr2   = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_mb
                a_loss  = -torch.min(surr1, surr2).mean()

                v_pred  = self.critic(states[mb]).squeeze(-1)
                c_loss  = nn.MSELoss()(v_pred, returns[mb])

                loss = a_loss + self.value_coef * c_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()

                logs['actor_loss'].append(float(a_loss.item()))
                logs['critic_loss'].append(float(c_loss.item()))
                logs['entropy'].append(float(entropy.mean().item()))

        self.trajectory = []
        return {k: float(np.mean(v)) for k, v in logs.items()}

    # ------ BaseRLAgent abstract methods ------

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'actor':     self.actor.state_dict(),
            'critic':    self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_clients': self.n_clients,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
