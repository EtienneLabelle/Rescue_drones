from dataclasses import dataclass
import numpy as np

@dataclass
class DelayLossChannel:
    delay_ms_mean: float = 40.0
    jitter_ms: float = 10.0
    drop_prob: float = 0.02
    throughput_bps: float = 2e6  # fallback cap

    def tx_delay_ms(self, size_bytes: int) -> float:
        base = np.random.normal(self.delay_ms_mean, self.jitter_ms)
        tx = (size_bytes * 8) / max(self.throughput_bps, 1e-9) * 1000.0
        return float(max(0.0, base + tx))

    def will_drop(self) -> bool:
        return bool(np.random.rand() < self.drop_prob)
