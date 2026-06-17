"""Decentralised gossip aggregation for UAV federated learning.

Each UAV maintains its own model and averages only with neighbours within
R_comm metres — no central server required.

Drop-in alternative to FedAvg in the simulation loop:
  FedAvg path  → distribute_model / train / aggregate_models
  Gossip path  → train_local / gossip_round
Both expose global_loss() for evaluation.
"""

from __future__ import annotations
import numpy as np
import torch
from copy import deepcopy

from comms.channel import DelayLossChannel


def actor_gossip_step(agents: dict, uav_positions: list, r_comm: float) -> int:
    """Gossip-average actor network weights of agents within r_comm.

    Synchronous: all actor weight snapshots are taken before any update, so
    every agent exchanges the same generation of weights.
    Target actors are intentionally left unchanged — normal soft-updates (tau)
    will bring them in line gradually without causing instability.

    Args:
        agents:        dict {agent_id: IDDPGAgent}, sorted order assumed to
                       match uav_positions index
        uav_positions: list of [x, y] positions, same order as sorted(agents)
        r_comm:        communication radius in metres

    Returns:
        number of agents that actually exchanged weights with at least one peer
    """
    agent_ids = sorted(agents.keys())
    n = len(agent_ids)

    # Snapshot before any write
    snapshots = {
        aid: {k: v.clone() for k, v in agents[aid].actor.state_dict().items()}
        for aid in agent_ids
    }

    exchanged = 0
    for i, aid in enumerate(agent_ids):
        xi, yi = uav_positions[i]
        nbrs = [
            j for j in range(n)
            if j != i and
            np.hypot(uav_positions[j][0] - xi, uav_positions[j][1] - yi) <= r_comm
        ]
        if not nbrs:
            continue
        peers = [snapshots[aid]] + [snapshots[agent_ids[j]] for j in nbrs]
        avg = {
            key: torch.stack([s[key].float() for s in peers]).mean(0)
            for key in peers[0]
        }
        agents[aid].actor.load_state_dict(avg)
        exchanged += 1

    return exchanged
