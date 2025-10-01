# Minimal PettingZoo Parallel wrapper around your existing env
from typing import Dict
import numpy as np
from env import DisasterCoverageEnvironment
from config import Config
from features.obs import build_obs_dict
try:
    from pettingzoo.utils.env import ParallelEnv
except ImportError:
    from pettingzoo.utils import ParallelEnv

class PZDisasterCoverage(ParallelEnv):
    metadata = {"name": "RescueDrones-v0", "is_parallelizable": True}

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else Config
        self._env = DisasterCoverageEnvironment(self.cfg)

        # Private sizes (avoid PettingZoo-reserved names)
        self._n_agents = getattr(self.cfg, "NUM_UAVS", getattr(self._env, "num_uavs", 2))
        self._obs_dim  = getattr(self.cfg, "STATE_DIM", getattr(self._env, "state_dim", 8))
        self._act_dim  = getattr(self.cfg, "ACTION_DIM", getattr(self._env, "action_dim", 2))

        # Build agents and spaces
        self.agents = [f"uav_{i+1}" for i in range(self._n_agents)]
        self.possible_agents = self.agents[:]
        from gymnasium import spaces
        import numpy as np
        self._obs_spaces = {a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float32) for a in self.agents}
        self._act_spaces = {a: spaces.Box(-1.0, 1.0, shape=(self._act_dim,), dtype=np.float32) for a in self.agents}

    @property
    def observation_spaces(self): return self._obs_spaces
    @property
    def action_spaces(self): return self._act_spaces

    def reset(self, seed: int | None = None, options: Dict | None = None):
        raw = self._env.reset()
        vec = build_obs_dict(raw, Config)
        first = next(iter(vec.values()))
        assert first.shape[0] == Config.STATE_DIM
        infos = {a: {} for a in self.agents}
        return vec, infos

    def step(self, actions: Dict[str, np.ndarray]):
        new_states, rewards, done = self._env.step(actions)
        vec_next = build_obs_dict(new_states, Config)
        terminations = {a: bool(done) for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos        = {a: {} for a in self.agents}
        return vec_next, rewards, terminations, truncations, infos

    def render(self): 
        # delegate to your existing visualization if desired
        return

    def close(self): 
        pass
