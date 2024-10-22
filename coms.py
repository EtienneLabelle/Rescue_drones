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
# Speed of light
C = 3e8  # meters/second
# Wavelength of the signal (lambda = c / f)
WAVELENGHT = C / FREQUENCY
# Constants


class Coms_Metrics:

    def __init__(self):
        self.end_to_end_PL = None  # Path Loss
        self.link_SNRs = []  # List to store SNR values for each drone-to-drone link
        self.drones_RP = []  # Received Power between drones
        self.data_rate_Mbps = None  # Data rate in Mbps (Shannon Capacity)
        self.latency = None  # Total end-to-end delay
        self.packet_loss_rate = None  # Packet loss rate
        self.throughput_Mbps = None  # Actual data rate (throughput)


""" kept this cause nice implementation but nothing in it work for now
    def update_end_to_end_PL(self, positions):
        # Update end-to-end path loss based on positions
        self.end_to_end_PL = calculate_end_to_end_path_loss(positions)

    def update_drones_SNR(self, positions):
        # Update SNR based on positions
        self.drones_SNR = calculate_snr_linear(positions)

    def update_drones_RP(self, positions):
        # Update received power based on positions
        self.drones_RP = calculate_received_power(positions)
"""

# Function collecting all metrics with the intention of displaying and debugging
def calculate_coms_metrics(positions):
    # Instantiate the metrics object
    coms_metrics = Coms_Metrics()
    
    # Update the metrics by calling the appropriate methods
    #coms_metrics.update_end_to_end_PL(positions)  ### not very useful for the system can be a good debbuging tool
    #coms_metrics.update_drones_SNR(positions)
    #coms_metrics.update_drones_RP(positions)
    
    # Return or display the collected metrics
    return coms_metrics


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
    """
    Calculate the total interference power from nearby sources (in dBm).
    interference_sources: A list of powers (in dBm) from different interference sources.
    """
    # Convert interference sources from dBm to linear scale (watts)
    interference_powers = [10 ** (interference / 10) for interference in interference_sources]
    
    # Sum all interference powers (in linear scale) and convert back to dBm
    total_interference_power_watts = sum(interference_powers)
    
    # Convert total interference power back to dBm
    total_interference_power_dBm = 10 * np.log10(total_interference_power_watts)
    
    return total_interference_power_dBm


#You can apply these fading models to the received power as a multiplicative factor.
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