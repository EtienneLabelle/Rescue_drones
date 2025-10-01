#!/usr/bin/env python3
"""
Start an FRL experiment locally: 1 Flower server + N clients.
Verbose, explicit ports, and hard timeouts so you see *something* or it fails.
"""
import time
import multiprocessing as mp
import argparse
import os, sys, socket

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SERVER_ADDR = "[::]:8081"          # server bind addr (all interfaces, port 8081)
CLIENT_ADDR = "127.0.0.1:8081"     # clients connect here

def _server(num_rounds: int):
    print(f"[server] booting, rounds={num_rounds}, bind={SERVER_ADDR}", flush=True)
    import flwr as fl
    from federation.server import DelayAwareFedAvg
    cfg = fl.server.ServerConfig(num_rounds=num_rounds)
    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=cfg,
        strategy=DelayAwareFedAvg(),
    )
    print("[server] done", flush=True)

def _client(cid: int, steps: int, alpha: float, drift: int):
    print(f"[client {cid}] booting, steps={steps}, connect={CLIENT_ADDR}", flush=True)
    import flwr as fl
    from federation.client_rl import RLClient
    client = RLClient(client_id=cid, steps_per_round=steps, noniid_alpha=alpha, drift_steps=drift)
    fl.client.start_numpy_client(server_address=CLIENT_ADDR, client=client)
    print(f"[client {cid}] done", flush=True)

def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) != 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--drift", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180, help="hard stop in seconds")
    args = ap.parse_args()

    # sanity: port must be free
    host, port = "127.0.0.1", int(CLIENT_ADDR.split(":")[-1])
    if not _port_free(host, port):
        print(f"[launcher] FAIL: port {port} already in use. Try a different port.", flush=True)
        sys.exit(1)

    print(f"[launcher] start server@{SERVER_ADDR}, then {args.clients} clients → rounds={args.rounds}, steps={args.steps}", flush=True)

    srv = mp.Process(target=_server, args=(args.rounds,), daemon=True)
    srv.start()
    time.sleep(1.0)  # give server time to bind

    # quick bind probe
    if _port_free(host, port):
        print("[launcher] WARN: server port not bound yet; waiting a bit more…", flush=True)
        time.sleep(2.0)
        if _port_free(host, port):
            print("[launcher] FAIL: server did not bind to port; aborting.", flush=True)
            srv.terminate()
            sys.exit(1)

    clients = []
    for cid in range(args.clients):
        p = mp.Process(target=_client, kwargs={"cid": cid, "steps": args.steps, "alpha": args.alpha, "drift": args.drift}, daemon=True)
        p.start()
        clients.append(p)
        time.sleep(0.2)

    # wait with timeout
    deadline = time.time() + args.timeout
    while time.time() < deadline and srv.is_alive():
        time.sleep(0.5)

    if srv.is_alive():
        print("[launcher] FAIL: server timeout", flush=True)
        srv.terminate()
        for p in clients: p.terminate()
        sys.exit(1)

    # Let clients finish
    for p in clients:
        p.join(timeout=10)

    bad = [p.exitcode for p in clients if p.exitcode not in (0, None)]
    if bad:
        print(f"[launcher] FAIL: client exit codes: {bad}", flush=True)
        sys.exit(1)

    print("[launcher] ✅ FRL run completed", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    # macOS default is 'spawn' — keep it explicit
    mp.set_start_method("spawn", force=True)
    main()
