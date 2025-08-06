"""
Configuration file for the multi-agent drone simulation
"""

class Config:
    # Environment settings
    NUM_UAVS = 6
    EPISODE_LENGTH = 400
    TRAINING_EPISODES = 1000  # Balanced training time for gradual penalty learning
    
    # Learning parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
    
    # Exploration parameters
    INITIAL_NOISE = 0.5
    FINAL_NOISE = 0.01
    NOISE_DECAY = 0.001
    
    # Reward parameters
    ZONE_REWARD = 500.0
    PROXIMITY_REWARD = 100.0
    BOUNDARY_PENALTY = 50.0  # Much milder penalty - reduced from 200.0
    BOUNDARY_MARGIN = 500  # Much smaller margin - reduced from 1500
    
    # Environment dimensions
    ENV_WIDTH = 20000
    ENV_HEIGHT = 20000
    SPAWN_MARGIN = 1000
    
    # Communication settings
    FREQUENCY = 2.4e9  # 2.4 GHz
    BANDWIDTH = 20e6   # 20 MHz
    TRANSMIT_POWER = 30  # dBm
    MIN_RECEIVE_POWER = -80  # dBm
    NOISE_POWER = -90  # dBm 