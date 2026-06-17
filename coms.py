import math
from utils import calculate_distance
import numpy as np

# Constants
FREQUENCY = 2.4e9  # 2.4 GHz
BANDWIDTH = 20e6   # 20 MHz
TRANSMIT_POWER = 30  # Transmit power in dBm (decibel milliwatts)
MIN_RECEIVE_POWER = -80  # Minimum power in dBm for a stable link
BOLTZMANN_CONSTANT = 1.38e-23  # Boltzmann's constant in Joules/Kelvin
TEMPERATURE = 290  # Room temperature in Kelvin
C = 3e8  # meters/second
WAVELENGHT = C / FREQUENCY


def fspl_db(distance, frequency):
    """Free-space path loss (dB)."""
    return 20 * math.log10(max(distance, 1.0)) + 20 * math.log10(frequency) - 147.55


class Link:
    def __init__(self, drone1, drone2, bandwidth, frequency, noise_power_dBm,
                 enable_delay=False, base_latency_ms=5.0, jitter_ms=2.0,
                 packet_loss_prob=0.0, bandwidth_cap_bps=None):
        self.drone1 = drone1
        self.drone2 = drone2
        self.distance = self.calculate_distance()
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.noise_power_dBm = noise_power_dBm
        self.sinr_dB = None
        self.capacity_bps = None
        self.isBlocked = False
        # Optional realism parameters (non-breaking)
        self.enable_delay = enable_delay
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.packet_loss_prob = packet_loss_prob
        self.bandwidth_cap_bps = bandwidth_cap_bps

    def __str__(self):
        # This string will be returned whenever print(link) is called
        return (f"Link from Drone {self.drone1.id}, {self.drone1.pos}, to Drone {self.drone2.id}, {self.drone2.pos}\n"
                f"  SINR: {self.sinr_dB:.2f} dB, Capacity: {self.capacity_bps:.2f} bps\n"
                f"  Distance: {self.distance:.2f} meters, Blocked?: {self.isBlocked}")
    
    def calculate_distance(self):
        distance = math.sqrt((self.drone1.pos[0] - self.drone2.pos[0])**2 + 
                         (self.drone1.pos[1] - self.drone2.pos[1])**2)
        return distance

    def calculate_fspl(self):
        return fspl_db(self.distance, self.frequency)

    def calculate_sinr(self):
        fspl = self.calculate_fspl()
        received_power_dBm = TRANSMIT_POWER - fspl
        self.sinr_dB = received_power_dBm - self.noise_power_dBm
        return self.sinr_dB
    
    def calculate_capacity(self):
        if self.sinr_dB is None:
            self.calculate_sinr()
        sinr_linear = 10 ** (self.sinr_dB / 10)
        theoretical_capacity = self.bandwidth * math.log2(1 + sinr_linear)
        if self.bandwidth_cap_bps is not None:
            self.capacity_bps = min(theoretical_capacity, self.bandwidth_cap_bps)
        else:
            self.capacity_bps = theoretical_capacity
        return self.capacity_bps

    def sample_latency_ms(self):
        if not self.enable_delay:
            return 0.0
        # Simple jitter model: Gaussian clipped at zero
        jitter = np.random.normal(loc=0.0, scale=self.jitter_ms)
        latency = max(0.0, self.base_latency_ms + jitter)
        return latency

    def will_drop_packet(self):
        if self.packet_loss_prob <= 0.0:
            return False
        return np.random.rand() < self.packet_loss_prob

    def estimate_tx_time_ms(self, num_bytes):
        """Estimate transmission time in milliseconds for a payload.
        Uses current capacity and accounts for bandwidth cap if set.
        Returns inf if capacity is zero or packet is dropped (synthetic loss).
        """
        if self.will_drop_packet():
            return float('inf')
        capacity = self.calculate_capacity()
        if capacity <= 0:
            return float('inf')
        tx_seconds = (num_bytes * 8.0) / capacity
        return 1000.0 * tx_seconds + self.sample_latency_ms()

    def obstacle_detection(self, obstacles):
        """
        Check if there is an obstacle between drone1 and drone2.
        Sets self.isBlocked to True if an obstacle blocks the path, otherwise False.
        """
        # Coordinates of drone1 and drone2
        x1, y1 = self.drone1.pos
        x2, y2 = self.drone2.pos

        # Loop through each obstacle in the environment
        for obstacle in obstacles:
            obstacle_x = obstacle.center_pos[0]
            obstacle_y_range = [edge[1] for edge in obstacle.edges_pos]  # Y-range of the obstacle

            # Check if the obstacle's x-coordinate is between drone1 and drone2
            if min(x1, x2) <= obstacle_x <= max(x1, x2):
                # Calculate expected y at obstacle's x-position based on line from (x1, y1) to (x2, y2)
                if x1 != x2:  # Avoid division by zero for vertical lines
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    obstacle_y_at_x = slope * obstacle_x + intercept

                    # Check if the y-coordinate of the obstacle is close to the calculated line y
                    if min(obstacle_y_range) <= obstacle_y_at_x <= max(obstacle_y_range):
                        self.isBlocked = True
                        break
                else:  # Vertical line case
                    if min(y1, y2) <= obstacle.center_pos[1] <= max(y1, y2):
                        self.isBlocked = True
                        break

            self.isBlocked = False

def calculate_received_power(distance):
    """
    Calculate the received power in dBm using the Friis transmission equation in dBm directly.
    distance: in meters
    """
    if distance == 0:
        return TRANSMIT_POWER  # If distance is zero, received power equals transmit power

    # Friis transmission equation in dBm 
    received_power_dBm = TRANSMIT_POWER + 20 * math.log10(WAVELENGHT) - 20 * math.log10(4 * math.pi) - 20 * math.log10(distance)
    
    return received_power_dBm

def calculate_noise_power(bandwidth):
    """
    Calculate the noise power in dBm using thermal noise formula.
    
    bandwidth: The system bandwidth in Hz.
    
    Returns the noise power in dBm.
    """
    # Calculate thermal noise power in watts
    noise_power_watts = BOLTZMANN_CONSTANT * TEMPERATURE * bandwidth
    
    # Convert noise power from watts to dBm
    noise_power_dBm = 10 * math.log10(noise_power_watts * 1000)  # Convert watts to milliwatts (mW)
    
    return noise_power_dBm

def calculate_interference_power(interference_sources):
    """Total interference power (dBm) from a list of received powers (dBm)."""
    interference_powers = [10 ** (p / 10) for p in interference_sources]
    return 10 * np.log10(sum(interference_powers))


# ---------------------------------------------------------------------------
# SNR / SINR primitives — all dBm↔linear conversions go through here
# ---------------------------------------------------------------------------

def dBm_to_linear(power_dBm):
    return 10 ** (power_dBm / 10)

def linear_to_dBm(power_linear):
    return 10 * math.log10(max(power_linear, 1e-30))

def compute_snr_linear(transmit_power_dBm, path_loss_dB, noise_power_dBm):
    """SNR as a linear ratio (no interference)."""
    return dBm_to_linear(transmit_power_dBm - path_loss_dB - noise_power_dBm)

def compute_sinr_linear(transmit_power_dBm, path_loss_dB, noise_power_dBm,
                        interferers_received_dBm):
    """SINR as a linear ratio.

    interferers_received_dBm: list of received powers (dBm) at the user from each
                              interfering UAV, i.e. P_tx - PathLoss(interferer→user).
    """
    signal = dBm_to_linear(transmit_power_dBm - path_loss_dB)
    noise  = dBm_to_linear(noise_power_dBm)
    inter  = dBm_to_linear(calculate_interference_power(interferers_received_dBm)) \
             if interferers_received_dBm else 0.0
    return signal / (noise + inter)

def rayleigh_fading():
    """
    Simulate Rayleigh fading (NLoS).
    The result is a fading factor that multiplies the received signal.
    """
    return np.random.rayleigh()

def rician_fading(K=3):
    """
    Simulate Rician fading (LoS) with a Rician factor K.
    K > 0 means there is a strong LoS component.
    """
    return np.random.rayleigh(scale=np.sqrt(K)) + np.random.rayleigh(scale=np.sqrt(1/(2*(K + 1))))

def calculate_sinr_with_fading(positions,fading_type='rician', interference_sources=None):
    """
    Calculate the SNR for a communication link that includes AWGN, fading, and interference.
    
    positions: List of positions representing the chain of communication.
    transmit_power_dBm: Transmit power in dBm.
    bandwidth: System bandwidth in Hz.
    fading_type: Choose between 'rayleigh' or 'rician' fading.
    interference_sources: A list of interference powers (in dBm) from external sources.
    
    Returns the end-to-end SNR in dB.
    """
    total_inverse_snr = 0  # For harmonic mean of SNRs
    
    # Calculate noise power from thermal noise
    noise_power_dBm = calculate_noise_power(BANDWIDTH)
    
    # Add interference power to noise power if any interference is present
    if interference_sources:
        interference_power_dBm = calculate_interference_power(interference_sources)
        # Combine noise power and interference power (in dBm)
        total_noise_power_dBm = 10 * np.log10(10 ** (noise_power_dBm / 10) + 10 ** (interference_power_dBm / 10))
    else:
        total_noise_power_dBm = noise_power_dBm
    
    for i in range(len(positions) - 1):
        # Calculate the distance between two positions
        distance = calculate_distance(positions[i+1], positions[i])
        
        # Calculate FSPL for this link
        fspl = 20 * math.log10(distance) + 20 * math.log10(FREQUENCY) - 147.55
        
        # Calculate received power in dBm
        received_power_dBm = TRANSMIT_POWER - fspl
        
        # Apply fading to the received power
        if fading_type == 'rayleigh':
            fading_factor = rayleigh_fading()
        elif fading_type == 'rician':
            fading_factor = rician_fading(K=3)  # Adjust K for stronger or weaker LoS component
        
        # Multiply received power by fading factor (convert fading factor to dB)
        received_power_dBm += 10 * np.log10(fading_factor)
        
        # Calculate SNR (SNR = received power - total noise power)
        snr_dB = received_power_dBm - total_noise_power_dBm
        
        # Convert SNR from dB to linear scale for harmonic mean calculation
        snr_linear = 10 ** (snr_dB / 10)
        total_inverse_snr += 1 / snr_linear
    
    # Calculate the end-to-end SNR
    end_to_end_snr_linear = 1 / total_inverse_snr
    end_to_end_snr_dB = 10 * math.log10(end_to_end_snr_linear)
    
    return end_to_end_snr_dB

def calculate_shannon_capacity(bandwidth, linear_snr):
    # Shannon capacity formula
    return bandwidth * math.log2(1 + linear_snr)


# ---------------------------------------------------------------------------
# Air-to-ground probabilistic LoS model (Al-Hourani et al., 2014)
# ---------------------------------------------------------------------------

_ATG_ENV_PARAMS = {
    'urban':       (9.61,  0.16),
    'suburban':    (4.88,  0.43),
    'dense_urban': (12.08, 0.11),
    'rural':       (0.01,  0.23),
}

def prob_los(d_2d, h_uav, environment='urban'):
    """Probability of LoS for an air-to-ground link."""
    a, b = _ATG_ENV_PARAMS.get(environment, _ATG_ENV_PARAMS['urban'])
    theta = np.degrees(np.arctan2(h_uav, max(d_2d, 1e-6)))
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def air_to_ground_path_loss(d_2d, h_uav, frequency, environment='urban',
                             eta_los=1.0, eta_nlos=20.0):
    """Mean air-to-ground path loss (dB) with probabilistic LoS.

    eta_los / eta_nlos: excess attenuation (dB) for LoS / NLoS conditions.
    """
    d_3d = math.sqrt(d_2d ** 2 + h_uav ** 2)
    p = prob_los(d_2d, h_uav, environment)
    fspl = fspl_db(d_3d, frequency)
    return p * (fspl + eta_los) + (1 - p) * (fspl + eta_nlos)


def uav_to_user_rate(d_2d, h_uav, frequency, bandwidth,
                     transmit_power_dBm=30, noise_power_dBm=-90,
                     interferers_received_dBm=None,
                     environment='urban'):
    """Shannon rate (bps) for one UAV-to-ground-user link.

    bandwidth should already be the per-user share (B_users / users_per_uav).
    Pass interferers_received_dBm (list of P_tx - PathLoss per interferer) for
    SINR-based rate; omit for SNR-only (interference-free upper bound).
    """
    pl = air_to_ground_path_loss(d_2d, h_uav, frequency, environment)
    if interferers_received_dBm:
        ratio = compute_sinr_linear(transmit_power_dBm, pl, noise_power_dBm,
                                    interferers_received_dBm)
    else:
        ratio = compute_snr_linear(transmit_power_dBm, pl, noise_power_dBm)
    return calculate_shannon_capacity(bandwidth, ratio)


def compute_global_sum_rate_mbps(uavs, assignment, config):
    """Global SINR-based sum rate (Mbps) across all served users.

    assignment: dict {uav.id: [GroundUser, ...]}
    """
    bw_per_uav = getattr(config, 'BANDWIDTH_PER_UAV', 1e6)
    h_uav      = getattr(config, 'UAV_HEIGHT', 100.0)
    total = 0.0
    for uav in uavs:
        users = assignment.get(uav.id, [])
        if not users:
            continue
        bw = bw_per_uav / len(users)
        for user in users:
            d = math.sqrt((user.pos[0] - uav.pos[0]) ** 2 + (user.pos[1] - uav.pos[1]) ** 2)
            interferers_dBm = [
                config.TRANSMIT_POWER - air_to_ground_path_loss(
                    math.sqrt((user.pos[0] - o.pos[0]) ** 2 + (user.pos[1] - o.pos[1]) ** 2),
                    h_uav, config.FREQUENCY
                )
                for o in uavs if o.id != uav.id
            ]
            total += uav_to_user_rate(d, h_uav, config.FREQUENCY, bw,
                                      config.TRANSMIT_POWER, config.NOISE_POWER,
                                      interferers_received_dBm=interferers_dBm)
    return total / 1e6

def calculate_end_to_end_path_loss(positions):
    """
    Calculate the end-to-end path loss for the entire communication chain.
    
    positions: A list of positions in meters representing the operator, relays, and video drone.
               For example: [operator_position, relay1_position, ..., video_drone_position]
    frequency: Frequency in Hz (e.g., 2.4e9 for 2.4 GHz)
    
    Returns the total path loss in dB.
    """
    total_path_loss = 0
    # Iterate through the chain and calculate the path loss for each link
    for i in range(len(positions) - 1):
        distance = calculate_distance(positions[i+1],positions[i])     
        # Free-space path loss formula
        fspl = 20 * math.log10(distance) + 20 * math.log10(FREQUENCY) - 147.55
        print(f"For link {i} to link {i+1}, FSPL is {fspl:.2f} dB")

    #for last drone to operator or video drone to operator if no other positions    
    distance = calculate_distance(positions[-1], [0,0])    
    fspl = 20 * math.log10(distance) + 20 * math.log10(FREQUENCY) - 147.55
    total_path_loss += fspl 
    print(f"For lthe last link {i+1}, FSPL is {fspl:.2f} dB")