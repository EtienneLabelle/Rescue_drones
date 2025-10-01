import flwr as fl
import numpy as np

class DelayAwareFedAvg(fl.server.strategy.FedAvg):
    """Weights client updates by num_samples / (1 + staleness). Drops 'dropped' clients."""
    def aggregate_fit(self, rnd, results, failures):
        ok_params, weights = [], []
        for _, fitres in results:
            if fitres is None: 
                continue
            metrics = fitres.metrics or {}
            if metrics.get("dropped", False):
                continue
            staleness = max(0, int(rnd) - int(metrics.get("client_round", rnd)))
            num = float(metrics.get("num_samples", 1))
            weights.append(num / (1.0 + staleness))
            ok_params.append(fitres.parameters.tensors)

        if not ok_params:
            return None, {}

        ws = np.array(weights, dtype=np.float64)
        ws /= ws.sum()
        agg = []
        for p_idx in range(len(ok_params[0])):
            stacked = np.stack([np.array(client_params[p_idx], dtype=np.float64) for client_params in ok_params])
            agg.append((stacked * ws[:, None]).sum(axis=0).astype(stacked.dtype))
        return fl.common.Parameters(tensors=agg, tensor_type="numpy"), {}

def start_server(num_rounds=5):
    fl.server.start_server(strategy=DelayAwareFedAvg(), config=fl.server.ServerConfig(num_rounds=num_rounds))
