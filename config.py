"""
Configuration file for the multi-agent drone simulation
"""

class Config:
    # Environment settings
    NUM_UAVS = 5
    # Ground users
    NUM_GROUND_USERS = 20
    # State and action dimensions — STATE_DIM is derived; change the primitives, not the formula
    # 2 (own pos) + NUM_NEIGHBORS*2 (nearest UAV rel. pos) + 3 (avg_rate, n_users, avg_SINR) + NUM_USER_NEIGHBORS*2 (nearest user rel. pos)
    NUM_NEIGHBORS = 4
    NUM_USER_NEIGHBORS = 3  # k nearest ground users represented as (dx, dy) pairs in state
    STATE_DIM = 2 + NUM_NEIGHBORS * 2 + 3 + NUM_USER_NEIGHBORS * 2  # = 19
    ACTION_DIM = 2

    GROUND_USERS_MOBILE = False          # If True, users move each step
    GROUND_USERS_DISTRIBUTION = "even"  # "even": equal count per zone (round-robin), then random within zone
                                         # "normal": zone assignment drawn from a Gaussian over zone indices, then random within zone

    NUM_ZONES = NUM_UAVS
    EPISODE_LENGTH = 200
    TRAINING_EPISODES = 1000
    
    # Learning parameters
    LEARNING_RATE = 0.001       # used by DQN / PPO / A2C agents
    ACTOR_LR = 1e-4             # MADDPG actor (paper Table II)
    CRITIC_LR = 3e-4            # MADDPG critic — lowered from 1e-3 to stabilise centralized critic
    BATCH_SIZE = 128
    BUFFER_SIZE = 1000000
    
    # Exploration parameters
    INITIAL_NOISE = 0.5
    FINAL_NOISE = 0.01
    NOISE_DECAY = (INITIAL_NOISE - FINAL_NOISE) / TRAINING_EPISODES

    # Reward parameters  rᵢ = local_rate_i - β·interference + collision_penalty + boundary_penalty
    INTERFERENCE_BETA = 0.1     # co-channel interference weight; tune up if UAVs cluster
    COLLISION_DISTANCE = 100.0  # metres — pairs closer than this trigger penalty
    COLLISION_PENALTY = 5.0     # Mbps deducted per colliding neighbour (reward is in Mbps)
    BOUNDARY_PENALTY = 5.0      # Mbps deducted when within BOUNDARY_MARGIN of edge
    BOUNDARY_MARGIN = 50

    # Environment dimensions
    ENV_WIDTH = 2000
    ENV_HEIGHT = 2000
    SPAWN_MARGIN = 100
    UAV_MAX_STEP = 20
    
    # Communication settings
    FREQUENCY = 2.4e9  # 2.4 GHz
    BANDWIDTH_PER_UAV = 1e6   # 1 MHz per UAV (used for sum-rate reporting)
    SYS_BANDWIDTH = BANDWIDTH_PER_UAV * NUM_UAVS  # 5 MHz total system bandwidth
    TRANSMIT_POWER = 30  # dBm
    MIN_RECEIVE_POWER = -80  # dBm
    NOISE_POWER = -90  # dBm
    
    SINR_MIN_DB = 5.0              # users with best-UAV SINR below this threshold are left unassigned

    # Comms toggles
    COMMS_ENABLED = True          # False = skip SINR/capacity link calculations each step (faster training)
    COMMS_ENABLE_DELAY = False     # True = add latency/jitter/packet-loss model on top of capacity calc
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

    # Minimal FL + throughput simulation (main.py)
    UAV_HEIGHT = 100.0           # UAV altitude in metres
    UAV_INIT_POSITIONS = "random"  # "random": uniform random spawn (RL training); "grid": uniform grid (FL sim)
    B_USERS_FRACTION = 0.9       # fraction of BANDWIDTH for user service; remainder goes to FL comms
    FL_AGGREGATION = "gossip"    # "fedavg": central FedAvg (FL.py); "gossip": decentralised gossip (federation/)
    R_COMM = 800.0               # gossip communication radius in metres
    FL_INTERVAL = 20             # gossip every N episodes in the MADDPG training loop
    SIM_ROUNDS = 50              # number of simulation rounds T
    FL_LOCAL_EPOCHS = 5          # local SGD epochs per FL round
    FL_LEARNING_RATE = 0.01      # SGD learning rate for FL clients
    FL_LOCAL_SAMPLES = 200       # local training samples per UAV/client
    LINEAR_REGRESSION_DIM = 8   # feature dimension for the FL linear-regression task (fedavg path)

    # -----------------------------------------------------------------------
    # RL-for-FL: UAV relay optimises a federated learning workload (main.py)
    # -----------------------------------------------------------------------

    # FL workload
    N_CLIENTS               = 10
    FL_TARGET_EPS           = 0.05      # loss threshold that ends an episode
    FL_THETA                = 0.5       # local accuracy param for convergence bound
    FL_XI                   = 2.0       # K(eps,theta) = xi * log(1/eps) / (1-theta)
    FL_MAX_ROUNDS           = 100       # hard episode cap (rounds)
    CLIENT_MODEL_SIZE_BYTES = 4096      # bytes per model update (TinyMLP @ float32)
    CLIENT_FLOPS_PER_SEC    = 1e9       # client compute throughput (FLOP/s)

    # Base station (FL aggregator) — default placed at right edge, mid-height
    BS_POSITION             = None      # None → [ENV_WIDTH, ENV_HEIGHT/2] at runtime

    # UAV relay
    UAV_RELAY_H             = 100.0     # relay altitude (m) — same as UAV_HEIGHT
    UAV_ENERGY_BUDGET       = 5000.0    # J per episode

    # Backhaul channel
    BACKHAUL_RICIAN_K       = 3.0       # Rician K-factor for UAV↔BS link

    # Rotary-wing propulsion (hover model)
    PROPULSION_P0           = 100.0     # W  blade profile power
    PROPULSION_Pi           = 120.0     # W  induced power

    # Reward shaping
    RHO                     = 0.5       # weight: (1-rho)*energy + rho*latency
    ACCURACY_GAIN_W         = 1.0       # weight on accuracy gain term
    STALENESS_PENALTY_W     = 0.01      # penalise high staleness variance
    FL_BOUNDARY_PENALTY     = 0.1       # penalty when UAV near map edge
    UNREACHABLE_PENALTY     = 0.05      # penalty per unselected client fraction
    REWARD_MAX_ENERGY       = 500.0     # J  — normalisation reference
    REWARD_MAX_LATENCY      = 60.0      # s  — normalisation reference

    # PPO hyperparameters
    PPO_LR                  = 3e-4
    PPO_CLIP                = 0.2
    PPO_GAMMA               = 0.99
    PPO_GAE_LAMBDA          = 0.95
    PPO_ENTROPY_COEF        = 0.01
    PPO_EPOCHS              = 4
    PPO_BATCH_SIZE          = 64
    PPO_HIDDEN_DIM          = 256

    # Training loop (main.py)
    FL_RL_TRAIN_EPISODES    = 500
    FL_RL_EVAL_EVERY        = 50