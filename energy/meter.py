from dataclasses import dataclass

@dataclass
class EnergyMeter:
    tx_nj_per_bit: float = 1.2   # nanojoules/bit
    rx_nj_per_bit: float = 0.8
    compute_nj_per_flop: float = 0.4

    def __post_init__(self):
        self.joules = 0.0

    def charge_tx(self, bytes_count: int):
        self.joules += (bytes_count * 8) * self.tx_nj_per_bit * 1e-9

    def charge_rx(self, bytes_count: int):
        self.joules += (bytes_count * 8) * self.rx_nj_per_bit * 1e-9

    def charge_compute(self, flop_count: float):
        self.joules += flop_count * self.compute_nj_per_flop * 1e-9

    def total(self) -> float:
        return float(self.joules)

    def reset(self):
        self.joules = 0.0
