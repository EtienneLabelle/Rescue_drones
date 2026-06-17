"""Eval (+ optional visualization) for a saved checkpoint.

Programmatic use:
    from eval.runner import run_eval
    reward, sr = run_eval(model="FL", seed=64553)

CLI use (run from project root):
    python eval/runner.py                        # FL_best, with viz
    python eval/runner.py --model NoFL
    python eval/runner.py --model models/my.pt
    python eval/runner.py --seed 1234
    python eval/runner.py --no-viz               # metrics only, no animation
"""
import numpy as np, random, torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DisasterCoverageEnvironment
from config import Config
from model_io import load_agents, MODELS_DIR

EVAL_SEED = 64553

_SHORTCUTS = {
    "FL":         "FL_best.pt",
    "NoFL":       "NoFL_best.pt",
    "FL_best":    "FL_best.pt",
    "NoFL_best":  "NoFL_best.pt",
    "FL_final":   "FL_final.pt",
    "NoFL_final": "NoFL_final.pt",
}


def _resolve(model: str) -> str:
    name = _SHORTCUTS.get(model, model)
    if not os.path.isabs(name) and not name.startswith(MODELS_DIR):
        name = os.path.join(MODELS_DIR, name)
    return name


def run_eval(model="FL", seed=EVAL_SEED, viz=False):
    """Load a checkpoint and run one validation episode.

    Args:
        model: 'FL', 'NoFL', or a path to a .pt file (default: 'FL')
        seed:  random seed (default: 64553)
        viz:   launch the tkinter animation after eval (default: False)
    Returns:
        (reward, sr_mbps)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    path = _resolve(model)
    agents = load_agents(path)
    env    = DisasterCoverageEnvironment(Config)
    label  = os.path.splitext(os.path.basename(path))[0]

    if hasattr(env, "run_validation_eval"):
        rew, sr = env.run_validation_eval(agents, seed, f"Eval [{label}]")
        print(f"reward={rew:.3f}  SR={sr:.2f} Mbps")
    else:
        # Fallback: 1-episode rollout
        states = env.reset()
        rew, done = 0.0, False
        while not done:
            actions = {aid: agents[aid].select_action(states[aid], explore=False) for aid in agents}
            states, rewards, done = env.step(actions)
            rew += np.mean(list(rewards.values())) if isinstance(rewards, dict) else float(rewards)
        sr = float("nan")
        print(f"reward={rew:.3f}")

    if viz:
        from display import run_episode_viz
        print("\nLaunching animation ...")
        run_episode_viz(agents, env, seed, label=label)

    return rew, sr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval + visualize a saved checkpoint")
    parser.add_argument("--model",   default="FL",
                        help="FL / NoFL / path to .pt  (default: FL → models/FL_best.pt)")
    parser.add_argument("--seed",    type=int, default=EVAL_SEED,
                        help=f"Random seed (default: {EVAL_SEED})")
    parser.add_argument("--no-viz",  action="store_true",
                        help="Print metrics only, skip animation")
    args = parser.parse_args()

    path = _resolve(args.model)
    if not os.path.exists(path):
        sys.exit(f"Checkpoint not found: {path}")

    run_eval(model=args.model, seed=args.seed, viz=not args.no_viz)
