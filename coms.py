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


class Link:
    def __init__(self, drone1, drone2, bandwidth, frequency, noise_power_dBm):
        self.drone1 = drone1
        self.drone2 = drone2
        self.distance = self.calculate_distance()
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.noise_power_dBm = noise_power_dBm
        self.sinr_dB = None
        self.capacity_bps = None

    def calculate_distance(self):
        return math.sqrt((self.drone1.pos[0] - self.drone2.pos[0])**2 + 
                         (self.drone1.pos[1] - self.drone2.pos[1])**2)

    def calculate_fspl(self):
        fspl = 20 * math.log10(self.distance) + 20 * math.log10(self.frequency) - 147.55
        return fspl

    def calculate_sinr(self):
        fspl = self.calculate_fspl()
        received_power_dBm = TRANSMIT_POWER - fspl
        self.sinr_dB = received_power_dBm - self.noise_power_dBm
        return self.sinr_dB
    
    def calculate_capacity(self):
        if self.sinr_dB is None:
            self.calculate_sinr()
        sinr_linear = 10 ** (self.sinr_dB / 10)
        self.capacity_bps = self.bandwidth * math.log2(1 + sinr_linear)
        return self.capacity_bps

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