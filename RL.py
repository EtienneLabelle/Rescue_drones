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

    return registry[agent_type](state_dim, action_dim, **kwargs)
