from __future__ import annotations
import numpy as np
import torch
import flwr as fl

from energy.meter import EnergyMeter
from comms.channel import DelayLossChannel
from datasets import seed_for_client, maybe_apply_drift

from config import Config
from RL import MADDPGAgent, CentralizedReplayBuffer
from env import DisasterCoverageEnvironment
from features.obs import build_obs_dict

def _actor_params_to_numpy(agents):
    arrays = []
    for aid in sorted(agents.keys()):
        sd = agents[aid].actor.state_dict()
        for _, v in sd.items():
            arrays.append(v.detach().cpu().numpy())
    return arrays

def _load_actor_params_from_numpy(agents, arrays):
    idx = 0
    for aid in sorted(agents.keys()):
        sd = agents[aid].actor.state_dict()
        new_sd = {}
        for k, v in sd.items():
            new_sd[k] = torch.tensor(arrays[idx]).to(v.device).type(v.dtype)
            idx += 1
        agents[aid].actor.load_state_dict(new_sd, strict=True)

class RLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, steps_per_round: int = 200, noniid_alpha: float = 0.3, drift_steps: int = 0):
        self.client_id = client_id
        self.steps_per_round = steps_per_round
        self.drift_steps = drift_steps
        seed_for_client(client_id, noniid_alpha)

        self.env = DisasterCoverageEnvironment(Config)
        self.buffer = CentralizedReplayBuffer(max_size=getattr(Config, "BUFFER_SIZE", 10000))
        self.num_agents = Config.NUM_UAVS
        self.state_dim = Config.STATE_DIM
        self.action_dim = Config.ACTION_DIM
        self.agents = {f"uav_{i+1}": MADDPGAgent(
            agent_id=f"uav_{i+1}",
            state_dim=self.state_dim, action_dim=self.action_dim,
            num_agents=self.num_agents, learning_rate=1e-3,
            shared_buffer=self.buffer,
        ) for i in range(self.num_agents)}

        self.energy = EnergyMeter()
        self.channel = DelayLossChannel()

    def get_parameters(self, config):
        return _actor_params_to_numpy(self.agents)

    def fit(self, parameters, config):
        rnd = int(config.get("round", 0))
        if parameters:
            _load_actor_params_from_numpy(self.agents, parameters)

        states = self.env.reset()
        states = build_obs_dict(states, Config)
        first = next(iter(states.values()))
        assert first.shape[0] == Config.STATE_DIM
        total_samples = 0
        for step in range(self.steps_per_round):
            maybe_apply_drift(step, self.drift_steps)
            actions = {}
            for aid, agent in self.agents.items():
                if aid in states:
                    actions[aid] = agent.select_action(states[aid], explore=True, noise_scale=0.1)
            new_states, rewards, done = self.env.step(actions)
            states = build_obs_dict(new_states, Config)
            total_samples += 1

            # cheap compute-energy accounting
            self.energy.charge_compute(1e6 * self.num_agents)

            if done:
                states = self.env.reset()

            if step % 5 == 0:
                targets = [self.agents[f"uav_{i+1}"].target_actor for i in range(self.num_agents)]
                for agent in self.agents.values():
                    agent.update(batch_size=64, all_target_actors=targets, return_losses=False)
                if step % 20 == 0:
                    tau = 0.005
                    for agent in self.agents.values():
                        agent.soft_update(agent.actor, agent.target_actor, tau)
                        agent.soft_update(agent.critic, agent.target_critic, tau)

        outbound = _actor_params_to_numpy(self.agents)
        size_bytes = sum(arr.nbytes for arr in outbound)
        dropped = self.channel.will_drop()
        delay_ms = self.channel.tx_delay_ms(size_bytes)
        self.energy.charge_tx(size_bytes)

        metrics = {
            "client_round": rnd,
            "delay_ms": float(delay_ms),
            "num_samples": int(total_samples),
            "energy_j": float(self.energy.total()),
            "dropped": bool(dropped),
        }
        # Note: if dropped=True, server may ignore this update (handled in strategy)
        return outbound, total_samples, metrics

    def evaluate(self, parameters, config):
        # optional: return 0 for now
        return 0.0, 0, {}
