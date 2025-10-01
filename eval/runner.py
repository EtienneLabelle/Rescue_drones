import numpy as np, random, torch
from env import DisasterCoverageEnvironment
from RL import MADDPGAgent, CentralizedReplayBuffer

def run_eval(seed=42, episodes=1):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    env = DisasterCoverageEnvironment()
    num_agents = getattr(env, "num_uavs", 2)
    state_dim  = getattr(env, "state_dim", 8)
    action_dim = getattr(env, "action_dim", 2)
    buf = CentralizedReplayBuffer(max_size=getattr(env, "buffer_size", 10000))
    agents = {f"uav_{i+1}": MADDPGAgent(f"uav_{i+1}", state_dim, action_dim, num_agents, 1e-3, shared_buffer=buf) for i in range(num_agents)}
    # Prefer your existing eval method if present
    if hasattr(env, "run_validation_eval"):
        rew, cov = env.run_validation_eval(agents, seed, f"Eval Seed {seed}")
        print(f"reward={rew:.3f}, coverage={cov:.3f}")
        return rew, cov
    # Fallback: 1 episode rollout
    states = env.reset()
    total_r = 0.0
    done = False
    while not done:
        actions = {aid: agents[aid].select_action(states[aid], explore=False) for aid in agents}
        states, rewards, done = env.step(actions)
        total_r += np.mean([r for _, r in rewards.items()]) if isinstance(rewards, dict) else float(rewards)
    print(f"reward={total_r:.3f}")
    return total_r, np.nan
