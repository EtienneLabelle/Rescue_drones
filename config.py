"""
Configuration file for the multi-agent drone simulation
"""

class Config:
    # Environment settings
    NUM_UAVS = 6
    EPISODE_LENGTH = 400
    TRAINING_EPISODES = 500 
    
    # Learning parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
    
    # Exploration parameters
    INITIAL_NOISE = 0.5
    FINAL_NOISE = 0.01
    NOISE_DECAY = 0.001
    
    # State and action dimensions
    STATE_DIM = 13  
    ACTION_DIM = 2
    
    # Reward parameters
    ZONE_REWARD = 500.0
    PROXIMITY_REWARD = 100.0
    BOUNDARY_PENALTY = 50.0  
    BOUNDARY_MARGIN = 500  
    
    # Reward normalization settings
    NORMALIZE_REWARDS = True
    NORM_WARMUP = 10
    NORM_CLIP = 3.0
    NORM_EMA_BETA = 0.99  # EMA decay factor for non-stationarity handling
    NORM_RAMP_STEPS = 20  # Gradual ramp-up steps after warmup
    RAW_REWARD_CLIP = 1000.0  # Clip raw rewards before normalization
    
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
    
    # Comms realism toggles (non-breaking; used by extended Link)
    COMMS_ENABLE_DELAY = False
    COMMS_BASE_LATENCY_MS = 5.0
    COMMS_JITTER_MS = 2.0
    COMMS_PACKET_LOSS_PROB = 0.0
    COMMS_BANDWIDTH_CAP_BPS = None  # None = use Shannon capacity
    
    # Logging settings
    VERBOSE_LOGGING = False  # Enable debug output (impacts performance)
    
    # Reproducibility settings
    RANDOM_SEED = None  # Set to integer for reproducible results 

    # Federated RL toggles (additive path; main training unaffected)
    FRL_ENABLED = False
    FRL_NUM_CLIENTS = 3
    FRL_DIRICHLET_ALPHA = 0.5  # non-IID; lower = more skew
    FRL_ASYNC = True  # async FedAvg variant
    FRL_STALENESS_BETA = 0.5  # weight by recency
    FRL_POISONING_RATE = 0.0  # fraction of adversarial clients
    FRL_ROBUST_AGG = "none"  # options: none|median|trimmed
    
    # Energy accounting toggles
    ENERGY_ENABLE = False
    ENERGY_W_PER_FLOP = 5e-12
    ENERGY_W_PER_BIT_TX = 5e-9