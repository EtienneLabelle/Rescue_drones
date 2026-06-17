"""Save and load agent checkpoints.

Usage (standalone, skip retraining):
    from model_io import load_agents
    agents = load_agents("models/FL_best.pt")               # IDDPG (default)
    agents = load_agents("models/maddpg_best.pt", distributed=False)  # MADDPG
"""

import os
import torch
from RL import IDDPGAgent, MADDPGAgent, CentralizedReplayBuffer
from config import Config

MODELS_DIR = "models"


def save_agents(agents: dict, path: str):
    """Write all agents' network weights and optimizer states to a single checkpoint file."""
    os.makedirs(os.path.dirname(path) or MODELS_DIR, exist_ok=True)
    torch.save(
        {aid: {
            'actor':            agent.actor.state_dict(),
            'critic':           agent.critic.state_dict(),
            'target_actor':     agent.target_actor.state_dict(),
            'target_critic':    agent.target_critic.state_dict(),
            'actor_optimizer':  agent.actor_optimizer.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
        } for aid, agent in agents.items()},
        path,
    )


def load_agents(path: str, distributed: bool = True) -> dict:
    """Reconstruct agents from a checkpoint produced by save_agents.

    Args:
        distributed: True  → IDDPGAgent (private buffer, local critic).
                     False → MADDPGAgent (shared centralised buffer, joint critic).
    """
    checkpoint = torch.load(path, map_location='cpu')

    shared_buffer = (
        None if distributed
        else CentralizedReplayBuffer(max_size=Config.BUFFER_SIZE)
    )

    agents = {}
    for aid, states in checkpoint.items():
        if distributed:
            agent = IDDPGAgent(
                agent_id=aid,
                state_dim=Config.STATE_DIM,
                action_dim=Config.ACTION_DIM,
                actor_lr=Config.ACTOR_LR,
                critic_lr=Config.CRITIC_LR,
                buffer_size=Config.BUFFER_SIZE,
            )
        else:
            agent = MADDPGAgent(
                agent_id=aid,
                state_dim=Config.STATE_DIM,
                action_dim=Config.ACTION_DIM,
                num_agents=Config.NUM_UAVS,
                actor_lr=Config.ACTOR_LR,
                critic_lr=Config.CRITIC_LR,
                shared_buffer=shared_buffer,
            )
        agent.actor.load_state_dict(states['actor'])
        agent.critic.load_state_dict(states['critic'])
        agent.target_actor.load_state_dict(states['target_actor'])
        agent.target_critic.load_state_dict(states['target_critic'])
        if 'actor_optimizer' in states:
            agent.actor_optimizer.load_state_dict(states['actor_optimizer'])
        if 'critic_optimizer' in states:
            agent.critic_optimizer.load_state_dict(states['critic_optimizer'])
        agents[aid] = agent
    return agents
