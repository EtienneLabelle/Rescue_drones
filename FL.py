"""
FL.py — Real FedAvg workload for the RL-for-FL simulation.

Three classes:
  FLClient          — ground node: holds non-IID local dataset, trains a TinyMLP.
  FedAvgAggregator  — base station: standard FedAvg weight averaging.
  FLWorkload        — orchestrates clients + aggregator, exposes per-round pipeline.

Two round modes:
  run_round_analytical  — uses cost models from coms.py (no gradient descent).
                          Used in the inner RL loop (fast, differentiable-free).
  run_round_real        — actually runs local SGD + FedAvg.
                          Optional; call from outside the RL loop for validation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from coms import (
    air_to_ground_path_loss,
    uav_to_user_rate,
    rician_fading,
    calculate_shannon_capacity,
    dBm_to_linear,
    fspl_db,
)
from energy.meter import EnergyMeter
from utils import calculate_distance


# ---------------------------------------------------------------------------
# Tiny local model — demand / load prediction (4-in, 16-hidden, 1-out MLP)
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return self.param_count() * 4  # float32 = 4 bytes


# ---------------------------------------------------------------------------
# FL Client
# ---------------------------------------------------------------------------

class FLClient:
    """Ground-node FL participant.

    Holds a synthetic non-IID time-series dataset (sine wave with a
    client-specific phase) and a local copy of the shared TinyMLP.
    """

    def __init__(self, client_id: int, position: list,
                 n_samples: int = 200, phase_shift: float = None):
        self.id  = client_id
        self.pos = list(position)
        self.n_samples = n_samples
        self.phase = phase_shift if phase_shift is not None else np.random.uniform(0, 2 * math.pi)
        self._make_dataset()
        self.model = TinyMLP()

    def _make_dataset(self):
        t  = np.linspace(0, 4 * math.pi, self.n_samples, dtype=np.float32)
        X  = np.stack([
            np.sin(t + self.phase),
            np.cos(t + self.phase),
            t / (4 * math.pi),
            np.full_like(t, self.id / 10.0),
        ], axis=1)
        y  = np.sin(t + self.phase + 0.5).reshape(-1, 1).astype(np.float32)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def estimated_flops(self, epochs: int = 1) -> float:
        return 2.0 * self.model.param_count() * self.n_samples * epochs

    def set_global_model(self, global_state_dict: dict):
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def train_local(self, epochs: int = 5, lr: float = 0.01) -> dict:
        """Run local SGD; return weight delta Δw = w_local − w_global."""
        global_state = copy.deepcopy(self.model.state_dict())
        optim   = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            optim.zero_grad()
            loss_fn(self.model(self.X), self.y).backward()
            optim.step()
        return {
            k: self.model.state_dict()[k].clone() - global_state[k]
            for k in global_state
        }

    def local_loss(self) -> float:
        self.model.eval()
        with torch.no_grad():
            return float(nn.MSELoss()(self.model(self.X), self.y).item())


# ---------------------------------------------------------------------------
# FedAvg Aggregator (base station)
# ---------------------------------------------------------------------------

class FedAvgAggregator:
    def __init__(self):
        self.global_model = TinyMLP()

    def state_dict(self) -> dict:
        return self.global_model.state_dict()

    def aggregate(self, deltas: list[dict], weights: list[float] = None):
        """Weighted FedAvg: w_global += Σ weight_i · Δw_i  (normalised)."""
        if not deltas:
            return
        if weights is None:
            weights = [1.0 / len(deltas)] * len(deltas)
        total = sum(weights)
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                update = sum(w * d[name] for w, d in zip(weights, deltas)) / total
                param.data.add_(update)

    def global_loss(self, clients: list[FLClient]) -> float:
        self.global_model.eval()
        losses = []
        with torch.no_grad():
            for c in clients:
                pred = self.global_model(c.X)
                losses.append(float(nn.MSELoss()(pred, c.y).item()))
        return float(np.mean(losses)) if losses else 1.0

    def reset(self):
        self.global_model = TinyMLP()


# ---------------------------------------------------------------------------
# FL Workload
# ---------------------------------------------------------------------------

class FLWorkload:
    """Orchestrates clients + aggregator; exposes per-round pipeline.

    One RL step = one FL round.
    """

    def __init__(self, config):
        self.config    = config
        self.n_clients = getattr(config, 'N_CLIENTS', 10)
        self.clients:  list[FLClient] = []
        self.aggregator = FedAvgAggregator()
        self.round     = 0
        self.loss_history: list[float] = []
        self.staleness  = np.zeros(self.n_clients, dtype=np.float32)
        self._init_clients()

    # ------------------------------------------------------------------
    # Initialisation / reset
    # ------------------------------------------------------------------

    def _init_clients(self, seed: int = None):
        margin   = getattr(self.config, 'SPAWN_MARGIN', 100)
        W, H     = self.config.ENV_WIDTH, self.config.ENV_HEIGHT
        rng      = np.random.default_rng(seed)
        positions = [
            [float(rng.uniform(margin, W - margin)),
             float(rng.uniform(margin, H - margin))]
            for _ in range(self.n_clients)
        ]
        phases   = rng.uniform(0, 2 * math.pi, self.n_clients)
        n_samp   = getattr(self.config, 'FL_LOCAL_SAMPLES', 200)
        self.clients = [
            FLClient(i, positions[i], n_samples=n_samp, phase_shift=float(phases[i]))
            for i in range(self.n_clients)
        ]
        for c in self.clients:
            c.set_global_model(self.aggregator.state_dict())

    def reset(self, seed: int = None):
        self.aggregator.reset()
        self.round        = 0
        self.loss_history = []
        self.staleness    = np.zeros(self.n_clients, dtype=np.float32)
        self._init_clients(seed)

    def client_positions(self) -> list[list[float]]:
        return [c.pos for c in self.clients]

    # ------------------------------------------------------------------
    # Convergence bound
    # ------------------------------------------------------------------

    def rounds_to_eps(self, eps: float = None, theta: float = None) -> float:
        """K(eps, theta) = xi · log(1/eps) / (1 − theta) — upper bound on rounds."""
        xi    = getattr(self.config, 'FL_XI',    2.0)
        eps   = eps   or getattr(self.config, 'FL_TARGET_EPS', 0.05)
        theta = theta or getattr(self.config, 'FL_THETA',      0.5)
        if eps <= 0:
            return float('inf')
        return xi * math.log(1.0 / eps) / max(1.0 - theta, 1e-6)

    # ------------------------------------------------------------------
    # Analytical round (inner RL loop — no gradient descent)
    # ------------------------------------------------------------------

    def run_round_analytical(
        self,
        selected_ids: list[int],
        bw_alloc: np.ndarray,
        uav_pos:  list[float],
        bs_pos:   list[float],
        config,
    ) -> dict:
        """Compute per-round cost using channel + energy models only.

        Safe to call millions of times — does NOT run neural-net training.

        Returns dict: energy_J, latency_s, accuracy_gain, plus debug fields.
        """
        if not selected_ids:
            self._tick_staleness(selected_ids)
            self.round += 1
            return dict(energy_J=0.0, latency_s=0.0, accuracy_gain=0.0,
                        selected_ids=[], uplink_rates=[], backhaul_rate=0.0)

        h_uav        = getattr(config, 'UAV_RELAY_H', getattr(config, 'UAV_HEIGHT', 100.0))
        freq         = config.FREQUENCY
        P_tx         = config.TRANSMIT_POWER
        N_0          = config.NOISE_POWER
        model_bytes  = getattr(config, 'CLIENT_MODEL_SIZE_BYTES', 4096)
        local_epochs = getattr(config, 'FL_LOCAL_EPOCHS', 5)
        bw_total     = getattr(config, 'BANDWIDTH_PER_UAV', 1e6)

        # Normalise bandwidth allocation to selected clients
        sel_bw = np.array([bw_alloc[i] for i in selected_ids], dtype=float)
        if sel_bw.sum() < 1e-9:
            sel_bw = np.ones(len(selected_ids))
        sel_bw = sel_bw / sel_bw.sum()

        # ---- Uplink: client → UAV ----------------------------------------
        uplink_rates = []
        for idx, cid in enumerate(selected_ids):
            d2d  = calculate_distance(self.clients[cid].pos, uav_pos)
            bw_i = float(sel_bw[idx]) * bw_total
            rate = uav_to_user_rate(d2d, h_uav, freq, bw_i, P_tx, N_0)
            uplink_rates.append(max(rate, 1.0))

        # Latency limited by slowest client
        uplink_time = max((model_bytes * 8) / r for r in uplink_rates)

        # ---- Backhaul: UAV ↔ BS (Rician fading) -------------------------
        d_bs     = calculate_distance(uav_pos, bs_pos)
        K        = getattr(config, 'BACKHAUL_RICIAN_K', 3.0)
        fading   = float(rician_fading(K))
        pl_bs    = fspl_db(max(d_bs, 1.0), freq)
        rx_dBm   = P_tx - pl_bs + 10 * math.log10(max(fading, 1e-9))
        snr_lin  = dBm_to_linear(rx_dBm - N_0)
        bh_rate  = max(calculate_shannon_capacity(bw_total, snr_lin), 1.0)

        # Upload all client deltas; download global model
        backhaul_up_time   = (model_bytes * len(selected_ids) * 8) / bh_rate
        backhaul_down_time = (model_bytes * 8) / bh_rate

        # ---- Downlink: UAV → clients (weakest link) ----------------------
        dl_rates = []
        bw_dl    = bw_total / max(len(selected_ids), 1)
        for cid in selected_ids:
            d2d  = calculate_distance(self.clients[cid].pos, uav_pos)
            rate = uav_to_user_rate(d2d, h_uav, freq, bw_dl, P_tx, N_0)
            dl_rates.append(max(rate, 1.0))
        downlink_time = (model_bytes * 8) / min(dl_rates)

        # ---- Compute time -----------------------------------------------
        compute_flops = sum(
            self.clients[cid].estimated_flops(local_epochs)
            for cid in selected_ids
        )
        compute_time = compute_flops / getattr(config, 'CLIENT_FLOPS_PER_SEC', 1e9)

        latency_s = compute_time + uplink_time + backhaul_up_time \
                    + backhaul_down_time + downlink_time

        # ---- Energy -------------------------------------------------------
        nj_tx  = getattr(config, 'ENERGY_W_PER_BIT_TX', 5e-9) * 1e9
        nj_cpu = getattr(config, 'ENERGY_W_PER_FLOP',   5e-12) * 1e9
        meter  = EnergyMeter(tx_nj_per_bit=nj_tx, compute_nj_per_flop=nj_cpu)

        for _ in selected_ids:             # client uplink TX
            meter.charge_tx(model_bytes)
        meter.charge_rx(model_bytes * len(selected_ids))    # UAV relay RX
        meter.charge_tx(model_bytes * len(selected_ids))    # UAV relay → BS
        meter.charge_tx(model_bytes)                        # UAV downlink TX
        meter.charge_compute(compute_flops)                 # client compute

        energy_comms = meter.total()
        energy_prop  = self._propulsion_energy(latency_s, config)
        energy_J     = energy_comms + energy_prop

        # ---- Accuracy gain estimate (heuristic) --------------------------
        K_total       = max(self.rounds_to_eps(), 1.0)
        k_rem         = max(K_total - self.round, 1.0)
        frac_part     = len(selected_ids) / self.n_clients
        accuracy_gain = frac_part / k_rem

        # ---- Staleness update + round counter ----------------------------
        self._tick_staleness(selected_ids)
        self.round += 1

        return dict(
            energy_J=energy_J,
            latency_s=latency_s,
            accuracy_gain=accuracy_gain,
            selected_ids=selected_ids,
            uplink_rates=uplink_rates,
            backhaul_rate=bh_rate,
        )

    def _propulsion_energy(self, duration_s: float, config) -> float:
        """Hover-mode rotary-wing propulsion energy: (P0 + Pi) × t."""
        P0 = getattr(config, 'PROPULSION_P0', 100.0)
        Pi = getattr(config, 'PROPULSION_Pi', 120.0)
        return (P0 + Pi) * duration_s

    def _tick_staleness(self, selected_ids: list[int]):
        for i in range(self.n_clients):
            if i in selected_ids:
                self.staleness[i] = 0.0
            else:
                self.staleness[i] += 1.0

    # ------------------------------------------------------------------
    # Real round — optional validation (actual FedAvg, not for RL loop)
    # ------------------------------------------------------------------

    def run_round_real(
        self,
        selected_ids: list[int],
        bw_alloc: np.ndarray = None,
    ) -> dict:
        """Run actual local SGD + FedAvg.  NOT used in the inner RL loop."""
        if not selected_ids:
            return dict(delta_loss=0.0, loss=self._last_loss())

        global_state = self.aggregator.state_dict()
        epochs       = getattr(self.config, 'FL_LOCAL_EPOCHS', 5)
        lr           = getattr(self.config, 'FL_LEARNING_RATE', 0.01)

        deltas = []
        for cid in selected_ids:
            self.clients[cid].set_global_model(global_state)
            deltas.append(self.clients[cid].train_local(epochs=epochs, lr=lr))

        self.aggregator.aggregate(deltas)
        new_state = self.aggregator.state_dict()
        for cid in selected_ids:
            self.clients[cid].set_global_model(new_state)

        loss_before = self._last_loss()
        loss_after  = self.aggregator.global_loss(self.clients)
        self.loss_history.append(loss_after)
        self._tick_staleness(selected_ids)
        self.round += 1

        return dict(delta_loss=loss_before - loss_after, loss=loss_after)

    def _last_loss(self) -> float:
        if self.loss_history:
            return self.loss_history[-1]
        return self.aggregator.global_loss(self.clients)
