import math

# Constants
FREQUENCY = 2.4e9  # 2.4 GHz
BANDWIDTH = 20e6   # 20 MHz
TRANSMIT_POWER = 30  # Transmit power in dBm (decibel milliwatts)
MIN_RECEIVE_POWER = -80  # Minimum power in dBm for a stable link
# Speed of light
C = 3e8  # meters/second
# Wavelength of the signal (lambda = c / f)
WAVELENGHT = C / FREQUENCY

class Coms_Metrics:

    def __init__(self):
        self.end_to_end_PL = None
        self.drones_SNR = None
        self.drones_RP = None


    def update_end_to_end_PL(self, positions):
        # Update end-to-end path loss based on positions
        self.end_to_end_PL = calculate_end_to_end_path_loss(positions)

    def update_drones_SNR(self, positions):
        # Update SNR based on positions
        self.drones_SNR = calculate_snr(positions)

    def update_drones_RP(self, positions):
        # Update received power based on positions
        self.drones_RP = calculate_received_power(positions)

# Function collecting all metrics with the intention of displaying and debugging
def calculate_coms_metrics(positions):
    # Instantiate the metrics object
    coms_metrics = Coms_Metrics()
    
    # Update the metrics by calling the appropriate methods
    coms_metrics.update_end_to_end_PL(positions)
    coms_metrics.update_drones_SNR(positions)
    coms_metrics.update_drones_RP(positions)
    
    # Return or display the collected metrics
    return coms_metrics
    

def calculate_fspl(distance):
    """
    Calculate the free-space path loss (FSPL) in dB.
    distance: in meters
    frequency: in Hz
    """
    if distance == 0:
        return 0  # No loss if distance is zero
    
    fspl_db = 20 * math.log10(distance) + 20 * math.log10(FREQUENCY) - 147.55
    return fspl_db

def calculate_received_power(distance):
    """
    Calculate the received power using the Friis transmission equation.
    transmit_power: in dBm
    distance: in meters
    frequency: in Hz
    """
    # Convert transmit power to milliwatts
    transmit_power_mW = 10 ** (TRANSMIT_POWER / 10)
     
    # Friis transmission equation: Pr = Pt * (lambda / (4 * pi * d))^2
    if distance == 0:
        return TRANSMIT_POWER  # If distance is zero, received power is equal to transmit power
    
    received_power_mW = transmit_power_mW * (WAVELENGHT / (4 * math.pi * distance)) ** 2
    
    # Convert received power back to dBm
    received_power_dBm = 10 * math.log10(received_power_mW)
    
    return received_power_dBm

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
        distance = abs(positions[i+1] - positions[i])
        if distance == 0:
            continue  # No path loss if there's no distance
        
        # Free-space path loss formula
        fspl = 20 * math.log10(distance) + 20 * math.log10(FREQUENCY) - 147.55
        total_path_loss += fspl
    
    return total_path_loss

# Function to calculate SNR (dB) given signal and noise power in dBm
def calculate_snr_dB(signal_power_dBm, noise_power_dBm):
    """
    Calculate the SNR in dB given the signal and noise powers in dBm.
    SNR (dB) = Signal Power (dBm) - Noise Power (dBm)
    """
    return signal_power_dBm - noise_power_dBm

# Function to calculate linear SNR given signal and noise power in watts
def calculate_snr_linear(signal_power_watts, noise_power_watts):
    """
    Calculate the SNR in linear form given the signal and noise powers in watts.
    SNR (linear) = Signal Power / Noise Power
    """
    return signal_power_watts / noise_power_watts

