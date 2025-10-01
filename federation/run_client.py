import argparse
import flwr as fl
from federation.client_rl import RLClient

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--drift", type=int, default=0)
    args = ap.parse_args()
    fl.client.start_numpy_client(client=RLClient(args.id, steps_per_round=args.steps, noniid_alpha=args.alpha, drift_steps=args.drift))
