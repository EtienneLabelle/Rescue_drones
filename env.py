# env.py       
import random
import numpy as np
from coms import Link
from drones import Drone
from config import Config
import torch

class Simulation:
    def __init__(self,bandwidth,frequency,noise_power_dBm):
        self.drones = []  # List of drones in the simulation : index 0 is video drone last is operator
        self.obstacles = [] # List of obstacles
        self.links = []   # List of links between drones
        self.bandwidth = bandwidth
        self.frequency = frequency
        self.noise_power_dBm = noise_power_dBm
 
    def get_all_positions(self):
        """
        Return a list of all drone positions.
        """
        return [drone.pos for drone in self.drones]    

    def create_links(self):
        self.links = []
        for i in range(len(self.drones) - 1):
            link = Link(self.drones[i], self.drones[i + 1], self.bandwidth, self.frequency, self.noise_power_dBm)
            self.links.append(link)

    def update_links(self):
        # Compute SINR and capacity for each link
        for link in self.links:
            link.calculate_distance()
            link.calculate_sinr()
            link.calculate_capacity()
            link.obstacle_detection(self.obstacles)
            #print(link)
   
    def deploy_relay(self,id):
        new_relay = Drone(id,position=[10,10]) 
        self.drones.insert(-1, new_relay)
        self.create_links()
    
    def create_obstacles(self, number_of_obstacles, size=5):
        
        self.obstacles = []  
        
        for i in range(number_of_obstacles):
            center_x = random.randint(5, 19000) # will need to change this at some point
            center_y = random.randint(5, 19000)        
            obstacle = Obstacle(center_position=(center_x, center_y))
            obstacle.calculate_edges(size)
            self.obstacles.append(obstacle)

class Obstacle:  #self.shape? tj ligne en ce moment
    def __init__(self, center_position,):
        self.center_pos = center_position
        self.edges_pos = []
    
    def calculate_edges(self, size):
        x = self.center_pos[0]  # x coordinate remains constant for a vertical line
        start_y = self.center_pos[1] - size
        end_y = self.center_pos[1] + size
        # Generate all the points from (x, start_y) to (x, end_y)
        self.edges_pos = [(x, y) for y in range(start_y, end_y + 1)]
        # return self.edges_pos

class DisasterZone:
    """Represents a disaster zone that needs coverage"""
    def __init__(self, center, radius, severity=1.0):
        self.center = center
        self.radius = radius
        self.severity = severity  # 0-1, higher means more critical
        self.coverage_status = 0.0  # 0-1, how well covered this zone is
        
    def update_coverage(self, drone_positions):
        """Update coverage based on drone positions"""
        total_coverage = 0.0
        for drone_pos in drone_positions:
            distance = np.sqrt((drone_pos[0] - self.center[0])**2 + 
                             (drone_pos[1] - self.center[1])**2)
            if distance <= self.radius:
                # Coverage decreases with distance
                coverage = max(0, 1 - distance / self.radius)
                total_coverage += coverage  # Use cumulative coverage instead of max
        
        # Cap coverage at 1.0 (100% coverage)
        self.coverage_status = min(1.0, total_coverage)
        return self.coverage_status

class DisasterCoverageEnvironment:
    """Environment for disaster area coverage with 3 UAVs using MADDPG"""
    
    def __init__(self, config):
        self.config = config
        self.sim = Simulation(config.BANDWIDTH, config.FREQUENCY, config.NOISE_POWER)
        
        # Create disaster zones (random for training)
        self.disaster_zones = self._create_disaster_zones(deterministic=False)
        
        # Initialize UAVs
        self.uavs = []
        self._initialize_uavs()
        
        # Coverage tracking
        self.total_coverage = 0.0
        self.episode_steps = 0
        self.max_steps = config.EPISODE_LENGTH
        
        # Communication links between UAVs
        self.sim.drones = self.uavs
        self.sim.create_links()
        
        # Reward normalization infrastructure - PER-AGENT stats
        self.reward_stats = {}  # Dict keyed by uav.id for per-agent stats
        self.normalize_rewards = getattr(config, 'NORMALIZE_REWARDS', True)
        self.norm_warmup = getattr(config, 'NORM_WARMUP', 10)
        self.norm_clip = getattr(config, 'NORM_CLIP', 3.0)
        self.ema_beta = getattr(config, 'NORM_EMA_BETA', 0.99)  # EMA decay factor
        
        # Initialize per-agent stats
        self._init_per_agent_stats()
    
    def _init_per_agent_stats(self):
        """Initialize reward statistics for each UAV agent"""
        for uav in self.uavs:
            self.reward_stats[uav.id] = {
                'count': 0,
                'mean': 0.0,
                'std': 1.0,
                'ema_mean': 0.0,
                'ema_var': 1.0,
                'm2': 0.0  # For Welford's algorithm
            }
    
    def _create_disaster_zones(self, deterministic=False, seed=None):
        """Create disaster zones in the environment
        
        Args:
            deterministic: If True, use fixed zone positions for evaluation
            seed: Seed for deterministic zone generation
        """
        zones = []
        num_uavs = getattr(self.config, 'NUM_UAVS', 3)
        
        if deterministic and seed is not None:
            # Deterministic zones for evaluation
            np.random.seed(seed)
            random.seed(seed)
            
            # Fixed grid-based approach for evaluation
            grid_size = int(np.ceil(np.sqrt(num_uavs)))
            spacing = 20000 // (grid_size + 1)
            
            zone_configs = []
            for i in range(num_uavs):
                row = i // grid_size
                col = i % grid_size
                x = spacing * (col + 1)
                y = spacing * (row + 1)
                radius = 800 + (i % 3) * 200
                severity = 0.6 + (i % 4) * 0.1
                zone_configs.append(((x, y), radius, severity))
        else:
            # Random zones for training
            zone_configs = []
            for i in range(num_uavs):
                x = np.random.uniform(1000, 19000)
                y = np.random.uniform(1000, 19000)
                radius = np.random.uniform(600, 1200)
                severity = np.random.uniform(0.5, 1.0)
                zone_configs.append(((x, y), radius, severity))
        
        for center, radius, severity in zone_configs:
            zone = DisasterZone(center, radius, severity)
            zones.append(zone)
        
        return zones
    
    def _initialize_uavs(self):
        """Initialize UAVs at random positions within bounds"""
        num_uavs = getattr(self.config, 'NUM_UAVS', 3)
        
        # Generate random spawn positions
        start_positions = []
        for _ in range(num_uavs):
            x = np.random.uniform(1000, 19000)
            y = np.random.uniform(1000, 19000)
            start_positions.append([x, y])
        
        for i, pos in enumerate(start_positions):
            uav = Drone(id=f"uav_{i+1}", position=pos.copy())
            uav.battery_level = 100.0
            self.uavs.append(uav)
    
    def get_state(self, uav_id):
        """Get state representation for a specific UAV"""
        uav = None
        for u in self.uavs:
            if u.id == uav_id:
                uav = u
                break
        
        if uav is None:
            return None
        
        state = []
        
        # Find nearest disaster zone
        nearest_zone = None
        min_dist = float('inf')
        for zone in self.disaster_zones:
            dist = np.sqrt((uav.pos[0] - zone.center[0])**2 + 
                         (uav.pos[1] - zone.center[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_zone = zone
        
        if nearest_zone:
            # 1. Relative position to nearest zone (delta_x, delta_y)
            delta_x = (nearest_zone.center[0] - uav.pos[0]) / 20000.0  # Normalize
            delta_y = (nearest_zone.center[1] - uav.pos[1]) / 20000.0  # Normalize
            state.extend([delta_x, delta_y])
            
            # 2. Distance to nearest zone
            state.append(min_dist / 20000.0)  # Normalize
            
            # 3. Zone radius and severity
            state.append(nearest_zone.radius / 20000.0)  # Normalize
            state.append(nearest_zone.severity)
            
            # 4. Coverage status of nearest zone
            state.append(nearest_zone.coverage_status)
            
            # IMPLEMENT FIX 3: Enhanced state representation with explicit directional signals
            # Add explicit directional guidance (validated in tests)
            state.append(1.0 if delta_x > 0.01 else -1.0 if delta_x < -0.01 else 0.0)  # X direction signal
            state.append(1.0 if delta_y > 0.01 else -1.0 if delta_y < -0.01 else 0.0)  # Y direction signal
        else:
            # If no zones, use zeros
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Added 2 more zeros for direction signals
        
        # 5. UAV battery level
        state.append(uav.battery_level / 100.0)  # Normalize to [0,1]
        
        # 6. Relative distances to other UAVs
        for other_uav in self.uavs:
            if other_uav.id != uav_id:
                dist = np.sqrt((uav.pos[0] - other_uav.pos[0])**2 + 
                             (uav.pos[1] - other_uav.pos[1])**2)
                state.append(dist / 20000.0)  # Normalize
            else:
                state.append(0.0)  # Self distance is 0
        
        # 7. Communication quality (if applicable)
        min_capacity = self._get_minimum_link_capacity()
        state.append(min_capacity)
        
        # Pad state to fixed size
        while len(state) < self.config.STATE_DIM:
            state.append(0.0)
        
        return np.array(state[:self.config.STATE_DIM])
    
    def _get_minimum_link_capacity(self):
        """Get minimum link capacity in Mbps, representing the bottleneck link"""
        if not hasattr(self.sim, 'links') or not self.sim.links:
            return 0.0
        
        min_capacity = float('inf')
        valid_links = 0
        
        for link in self.sim.links:
            # Calculate capacity if not already calculated
            if link.capacity_bps is None:
                link.calculate_capacity()
            
            # Track minimum capacity if valid
            if link.capacity_bps is not None:
                min_capacity = min(min_capacity, link.capacity_bps)
                valid_links += 1
        
        # Return minimum capacity in Mbps
        if valid_links > 0:
            return min_capacity / 1e6
        else:
            return 0.0
    
    def step(self, actions):
        """Execute actions for all UAVs and return new states, rewards, done"""
        rewards = {}
        
        # Execute actions for each UAV
        for i, uav in enumerate(self.uavs):
            if uav.id in actions:
                action = actions[uav.id]
                
                # Convert continuous action [-1,1] to movement
                # Scale action to reasonable movement range (further reduced to prevent corner rushing)
                max_movement = 25  # Further reduced from 50 to prevent systematic bias
                
                # Simple direct action interpretation for easier learning
                movement_x = action[0] * max_movement
                movement_y = action[1] * max_movement
                
                # Clamp movements to prevent extreme actions
                movement_x = np.clip(movement_x, -max_movement, max_movement)
                movement_y = np.clip(movement_y, -max_movement, max_movement)
                
                # Update position
                uav.move([movement_x, movement_y])
                
                # Store movement magnitude for reward calculation
                movement_magnitude = np.sqrt(movement_x**2 + movement_y**2)
                uav.last_movement_magnitude = movement_magnitude
            
                
                # Decrease battery
                uav.battery_level = max(0, uav.battery_level - 0.1)
        
        # Update disaster zone coverage
        drone_positions = [uav.pos for uav in self.uavs]
        total_coverage = 0.0
        
        for zone in self.disaster_zones:
            coverage = zone.update_coverage(drone_positions)
            total_coverage += coverage * zone.severity
        
        self.total_coverage = total_coverage / len(self.disaster_zones)
        
        # Update communication links
        self.sim.update_links()
        
        # Calculate rewards
        for uav in self.uavs:
            # Get movement magnitude for this UAV
            movement_magnitude = getattr(uav, 'last_movement_magnitude', 0)
            raw_reward = self._calculate_reward(uav, movement_magnitude)
            
            # Always update stats, but only normalize if enabled
            if self.normalize_rewards:
                normalized_reward = self._normalize_reward(uav.id, raw_reward)
                rewards[uav.id] = normalized_reward
            else:
                # Just update stats without normalization
                self._update_reward_stats(uav.id, raw_reward)
                rewards[uav.id] = raw_reward
            
            # Debug: Log reward values (only when verbose logging is enabled)
            if self.episode_steps % 50 == 0 and getattr(self.config, 'VERBOSE_LOGGING', False):
                if self.normalize_rewards:
                    # Use proper logging instead of print for performance
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"UAV {uav.id}: Raw={raw_reward:.2f}, Norm={rewards[uav.id]:.2f}, Pos={uav.pos}, Move={movement_magnitude:.2f}")
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"UAV {uav.id}: Raw={raw_reward:.2f} (no norm), Pos={uav.pos}, Move={movement_magnitude:.2f}")
        
        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps
        
        # Get new states
        new_states = {}
        for uav in self.uavs:
            new_states[uav.id] = self.get_state(uav.id)
        
        return new_states, rewards, done
    
    def _get_nearest_zone(self, uav_pos):
        """Get the nearest disaster zone and distance to it"""
        nearest_zone = None
        min_distance = float('inf')
        
        for zone in self.disaster_zones:
            distance = np.sqrt((uav_pos[0] - zone.center[0])**2 + 
                             (uav_pos[1] - zone.center[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_zone = zone
        
        return nearest_zone, min_distance

    def _calculate_reward(self, uav, movement_magnitude=0):
        """Calculate reward for a UAV - progress-based approach"""
        reward = 0.0
        
        # Get nearest zone and current distance
        nearest_zone, current_distance = self._get_nearest_zone(uav.pos)
        
        if nearest_zone:
            # 1. Strong reward for being IN a disaster zone
            if current_distance <= nearest_zone.radius:
                reward += self.config.ZONE_REWARD * nearest_zone.severity * 2.0  # Doubled zone reward
            else:
                # 2. Progress-based reward for getting closer to the nearest zone (ONLY when NOT in zone)
                if uav.prev_distance_to_zone is not None:
                    delta = uav.prev_distance_to_zone - current_distance
                    if delta > 0:
                        reward += delta * 10.0  # Reward for moving toward zone
                    elif delta < 0:
                        reward += delta * 8.0   # Stronger penalty for moving away from zone
            
            # 3. Update tracker for next step
            uav.prev_distance_to_zone = current_distance
        
        return reward
    
    def _update_reward_stats(self, uav_id, reward):
        """Update reward statistics using Welford's algorithm and EMA for stability"""
        if uav_id not in self.reward_stats:
            self.reward_stats[uav_id] = {
                'count': 0,
                'mean': 0.0,
                'std': 1.0,
                'ema_mean': 0.0,
                'ema_var': 1.0,
                'm2': 0.0  # For Welford's algorithm
            }

        stats = self.reward_stats[uav_id]
        stats['count'] += 1
        
        # Welford's algorithm for numerical stability
        delta = reward - stats['mean']
        stats['mean'] += delta / stats['count']
        delta2 = reward - stats['mean']
        stats['m2'] += delta * delta2
        
        # Update standard deviation
        if stats['count'] > 1:
            stats['std'] = max(np.sqrt(stats['m2'] / (stats['count'] - 1)), 1e-8)
        
        # EMA updates for non-stationarity handling
        if stats['count'] == 1:
            stats['ema_mean'] = reward
            stats['ema_var'] = 1.0
        else:
            # EMA mean update
            stats['ema_mean'] = self.ema_beta * stats['ema_mean'] + (1 - self.ema_beta) * reward
            
            # EMA variance update (using squared error)
            squared_error = (reward - stats['ema_mean']) ** 2
            stats['ema_var'] = self.ema_beta * stats['ema_var'] + (1 - self.ema_beta) * squared_error
    
    def _normalize_reward(self, uav_id, reward):
        """Normalize reward using EMA-based statistics with warm-up and clipping"""
        # Always update stats first
        self._update_reward_stats(uav_id, reward)
        
        # Warm-up: until we have enough samples, return raw reward
        if (not self.normalize_rewards) or (self.reward_stats[uav_id]['count'] < self.norm_warmup):
            return reward
        
        # Clip raw reward before normalization to handle heavy tails
        raw_clip_threshold = getattr(self.config, 'RAW_REWARD_CLIP', 1000.0)
        clipped_reward = np.clip(reward, -raw_clip_threshold, raw_clip_threshold)
        
        # Use EMA statistics for normalization (more adaptive to non-stationarity)
        stats = self.reward_stats[uav_id]
        ema_std = max(np.sqrt(stats['ema_var']), 1e-8)
        
        # Gradual ramp-up to prevent volatile z-scores after warmup
        ramp_steps = getattr(self.config, 'NORM_RAMP_STEPS', 20)
        if stats['count'] < self.norm_warmup + ramp_steps:
            # Linear interpolation from raw reward to normalized reward
            alpha = (stats['count'] - self.norm_warmup) / ramp_steps
            raw_normalized = (clipped_reward - stats['ema_mean']) / ema_std
            normalized = alpha * raw_normalized + (1 - alpha) * clipped_reward
        else:
            # Full normalization after ramp-up
            normalized = (clipped_reward - stats['ema_mean']) / ema_std
        
        # Clip to prevent extreme values
        return float(np.clip(normalized, -self.norm_clip, self.norm_clip))
    
    def reset(self):
        """Reset environment"""
        num_uavs = len(self.uavs)
        
        # Generate random spawn positions
        start_positions = []
        for _ in range(num_uavs):
            x = np.random.uniform(1000, 19000)
            y = np.random.uniform(1000, 19000)
            start_positions.append([x, y])
        
        for i, uav in enumerate(self.uavs):
            uav.pos = start_positions[i].copy()
            uav.battery_level = 100.0
            # Initialize prev_distance_to_zone for progress-based rewards
            nearest_zone, current_distance = self._get_nearest_zone(uav.pos)
            uav.prev_distance_to_zone = current_distance if nearest_zone else None
        
        # Recreate disaster zones for training (random)
        self.disaster_zones = self._create_disaster_zones(deterministic=False)
        
        # Reset tracking variables
        self.total_coverage = 0.0
        self.episode_steps = 0
        
        # Recreate communication links
        self.sim.drones = self.uavs
        self.sim.create_links()
        
        # Reset reward normalization stats for new episode
        self._init_per_agent_stats()
        
        # Return initial states
        initial_states = {}
        for uav in self.uavs:
            initial_states[uav.id] = self.get_state(uav.id)
        
        return initial_states
    
    def get_coverage_metrics(self):
        """Get current coverage metrics"""
        metrics = {
            'total_coverage': self.total_coverage,
            'zone_coverage': [zone.coverage_status for zone in self.disaster_zones],
            'uav_positions': [uav.pos for uav in self.uavs],
            'battery_levels': [uav.battery_level for uav in self.uavs]
        }
        return metrics
    
    def render(self, save_path=None):
        """Render the current state of the environment"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot disaster zones
        for zone in self.disaster_zones:
            circle = plt.Circle(zone.center, zone.radius, 
                              alpha=0.3, color='red' if zone.severity > 0.7 else 'orange')
            ax.add_patch(circle)
            
            # Add severity text
            ax.text(zone.center[0], zone.center[1], f'S:{zone.severity:.1f}\nC:{zone.coverage_status:.2f}',
                   ha='center', va='center', fontsize=8)
        
        # Plot UAVs
        colors = ['blue', 'green', 'purple']
        for i, uav in enumerate(self.uavs):
            ax.scatter(uav.pos[0], uav.pos[1], c=colors[i], s=100, 
                      label=f'UAV {i+1} (B:{uav.battery_level:.1f}%)')
        
        # Plot communication links
        if hasattr(self.sim, 'links') and self.sim.links:
            for link in self.sim.links:
                pos1 = link.drone1.pos
                pos2 = link.drone2.pos
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k--', alpha=0.5)
        
        ax.set_xlim(0, 20000)
        ax.set_ylim(0, 20000)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Disaster Coverage Environment\nTotal Coverage: {self.total_coverage:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def reset(self, deterministic=False, fixed_positions=None):
        """Reset environment
        
        Args:
            deterministic: If True, use fixed spawn positions for reproducible evaluation
            fixed_positions: List of fixed positions to use when deterministic=True
        """
        num_uavs = len(self.uavs)
        
        if deterministic and fixed_positions is not None:
            # Use fixed positions for deterministic evaluation
            start_positions = fixed_positions[:num_uavs]
            # Pad with random positions if needed
            while len(start_positions) < num_uavs:
                x = np.random.uniform(1000, 19000)
                y = np.random.uniform(1000, 19000)
                start_positions.append([x, y])
        else:
            # Generate random spawn positions
            start_positions = []
            for _ in range(num_uavs):
                x = np.random.uniform(1000, 19000)
                y = np.random.uniform(1000, 19000)
                start_positions.append([x, y])
        
        for i, uav in enumerate(self.uavs):
            uav.pos = start_positions[i].copy()
            uav.battery_level = 100.0
            # Initialize prev_distance_to_zone for progress-based rewards
            nearest_zone, current_distance = self._get_nearest_zone(uav.pos)
            uav.prev_distance_to_zone = current_distance if nearest_zone else None
        
        # Reset disaster zones
        for zone in self.disaster_zones:
            zone.coverage_status = 0.0
        
        # Reset tracking variables
        self.total_coverage = 0.0
        self.episode_steps = 0
        
        # Reset reward normalization stats for new episode
        self._init_per_agent_stats()
        
        # Recreate communication links
        self.sim.drones = self.uavs
        self.sim.create_links()
        
        # Return initial states
        initial_states = {}
        for uav in self.uavs:
            initial_states[uav.id] = self.get_state(uav.id)
        
        return initial_states
    
    def get_coverage_metrics(self):
        """Get current coverage metrics"""
        metrics = {
            'total_coverage': self.total_coverage,
            'zone_coverage': [zone.coverage_status for zone in self.disaster_zones],
            'uav_positions': [uav.pos for uav in self.uavs],
            'battery_levels': [uav.battery_level for uav in self.uavs]
        }
        return metrics
    
    def render(self, save_path=None):
        """Render the current state of the environment"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot disaster zones
        for zone in self.disaster_zones:
            circle = plt.Circle(zone.center, zone.radius, 
                              alpha=0.3, color='red' if zone.severity > 0.7 else 'orange')
            ax.add_patch(circle)
            
            # Add severity text
            ax.text(zone.center[0], zone.center[1], f'S:{zone.severity:.1f}\nC:{zone.coverage_status:.2f}',
                   ha='center', va='center', fontsize=8)
        
        # Plot UAVs
        colors = ['blue', 'green', 'purple']
        for i, uav in enumerate(self.uavs):
            ax.scatter(uav.pos[0], uav.pos[1], c=colors[i], s=100, 
                      label=f'UAV {i+1} (B:{uav.battery_level:.1f}%)')
        
        # Plot communication links
        if hasattr(self.sim, 'links') and self.sim.links:
            for link in self.sim.links:
                pos1 = link.drone1.pos
                pos2 = link.drone2.pos
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k--', alpha=0.5)
        
        ax.set_xlim(0, 20000)
        ax.set_ylim(0, 20000)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Disaster Coverage Environment\nTotal Coverage: {self.total_coverage:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def run_validation_eval(self, agents, seed, eval_name):
        """Run a deterministic evaluation with fixed seed"""
        print("\n" + "="*60)
        print("HONEST EVALUATION: Running validation tests...")
        print("="*60)       
        print(f"\nRunning {eval_name} with seed {seed}...")
        
        # Set seeds for deterministic behavior
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Recreate disaster zones deterministically for evaluation
        self.disaster_zones = self._create_disaster_zones(deterministic=True, seed=seed)
        
        # Generate fixed spawn positions for deterministic evaluation
        num_uavs = self.config.NUM_UAVS
        fixed_positions = []
        for i in range(num_uavs):
            # Use seed-based deterministic positioning
            np.random.seed(seed + i)  # Different seed for each UAV
            x = np.random.uniform(1000, 19000)
            y = np.random.uniform(1000, 19000)
            fixed_positions.append([x, y])
        
        # Reset with deterministic positioning
        states = self.reset(deterministic=True, fixed_positions=fixed_positions)
        total_reward = 0.0
        total_coverage = 0.0
        
        for step in range(self.config.EPISODE_LENGTH):
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in states:
                    # CORRECT: Use explore=False for honest evaluation
                    action = agent.select_action(states[agent_id], explore=False, noise_scale=0.0)  # NO NOISE
                    actions[agent_id] = action
            
            new_states, rewards, done = self.step(actions)
            
            # Calculate coverage
            drone_positions = [uav.pos for uav in self.uavs]
            total_coverage_step = 0.0
            for zone in self.disaster_zones:
                coverage = zone.update_coverage(drone_positions)
                total_coverage_step += coverage * zone.severity
            avg_coverage_step = total_coverage_step / len(self.disaster_zones) if self.disaster_zones else 0.0
            
            total_reward += np.mean(list(rewards.values()))
            total_coverage += avg_coverage_step
            states = new_states
        
        avg_reward = total_reward / self.config.EPISODE_LENGTH
        avg_coverage = total_coverage / self.config.EPISODE_LENGTH
        
        print(f"{eval_name} Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Coverage: {avg_coverage:.3f}")
        
        return avg_reward, avg_coverage