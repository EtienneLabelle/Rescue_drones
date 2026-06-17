# env.py       
import random
import numpy as np
from coms import (Link, prob_los, air_to_ground_path_loss, uav_to_user_rate,
                  calculate_shannon_capacity, compute_global_sum_rate_mbps,
                  compute_snr_linear, compute_sinr_linear, linear_to_dBm)
from utils import calculate_distance
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
            center_x = random.randint(5, Config.ENV_WIDTH - 5)
            center_y = random.randint(5, Config.ENV_HEIGHT - 5)
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

class GroundUser:
    """A person on the ground that UAVs may need to locate or serve."""

    def __init__(self, id, position, mobile=False):
        self.id = id
        self.pos = list(position)
        self.mobile = mobile

    def update(self):
        """Advance the user's position by one step. Only called when mobile=True."""
        if not self.mobile:
            return
        # Movement logic to be defined
        pass


class DisasterZone:
    """Represents a disaster zone that needs coverage"""
    def __init__(self, center, radius, severity=1.0):
        self.center = center
        self.radius = radius
        self.severity = severity  # 0-1, higher means more critical
        self.coverage_status = 0.0  # 0-1, how well covered this zone is
        
    def update_coverage(self, drone_positions):
        """Update coverage based on drone positions"""
        covered = any(
            np.sqrt((dp[0] - self.center[0])**2 + (dp[1] - self.center[1])**2) <= self.radius
            for dp in drone_positions
        )
        self.coverage_status = 1.0 if covered else 0.0
        return self.coverage_status

class DisasterCoverageEnvironment:
    """Environment for disaster area coverage with 3 UAVs using MADDPG"""
    
    def __init__(self, config):
        self.config = config
        self.sim = Simulation(config.BANDWIDTH_PER_UAV, config.FREQUENCY, config.NOISE_POWER)
        
        # Create disaster zones (random for training)
        self.disaster_zones = self._create_disaster_zones(deterministic=False)
        
        # Initialize UAVs
        self.uavs = []
        self._initialize_uavs()

        # Initialize ground users
        self.ground_users = []
        self._initialize_ground_users()

        # Coverage tracking
        self.total_coverage = 0.0
        self.episode_steps = 0
        self.max_steps = config.EPISODE_LENGTH
        self.current_assignment = {uav.id: [] for uav in self.uavs}
        
        # Communication links between UAVs
        self.sim.drones = self.uavs
        self.sim.create_links()
        
    
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
            spacing = self.config.ENV_WIDTH // (grid_size + 1)

            zone_configs = []
            for i in range(num_uavs):
                row = i // grid_size
                col = i % grid_size
                x = spacing * (col + 1)
                y = spacing * (row + 1)
                radius = int(0.04 * self.config.ENV_WIDTH)
                zone_configs.append(((x, y), radius, 1.0))
        else:
            # Random zones for training
            zone_configs = []
            for i in range(num_uavs):
                margin = getattr(self.config, 'SPAWN_MARGIN', 100)
                x = np.random.uniform(margin, self.config.ENV_WIDTH  - margin)
                y = np.random.uniform(margin, self.config.ENV_HEIGHT - margin)
                radius = np.random.uniform(0.03 * self.config.ENV_WIDTH,
                                           0.06 * self.config.ENV_WIDTH)
                severity = np.random.uniform(0.5, 1.0)
                zone_configs.append(((x, y), radius, severity))
        
        for center, radius, severity in zone_configs:
            zone = DisasterZone(center, radius, severity)
            zones.append(zone)
        
        return zones
    
    def _initialize_uavs(self):
        """Initialize UAVs. Position mode set by config.UAV_INIT_POSITIONS:
          'random' (default) — uniform random spawn used by RL training.
          'grid'             — uniform grid used by the FL simulation.
        """
        num_uavs = getattr(self.config, 'NUM_UAVS', 3)
        mode = getattr(self.config, 'UAV_INIT_POSITIONS', 'random')

        if mode == 'grid':
            cols = int(np.ceil(np.sqrt(num_uavs)))
            rows = int(np.ceil(num_uavs / cols))
            sx = self.config.ENV_WIDTH  / (cols + 1)
            sy = self.config.ENV_HEIGHT / (rows + 1)
            start_positions = [
                [sx * (i % cols + 1), sy * (i // cols + 1)]
                for i in range(num_uavs)
            ]
        else:
            start_positions = [
                [np.random.uniform(getattr(self.config, 'SPAWN_MARGIN', 100),
                                   self.config.ENV_WIDTH  - getattr(self.config, 'SPAWN_MARGIN', 100)),
                 np.random.uniform(getattr(self.config, 'SPAWN_MARGIN', 100),
                                   self.config.ENV_HEIGHT - getattr(self.config, 'SPAWN_MARGIN', 100))]
                for _ in range(num_uavs)
            ]

        for i, pos in enumerate(start_positions):
            uav = Drone(id=f"uav_{i+1}", position=pos.copy())
            uav.battery_level = 100.0
            self.uavs.append(uav)
    
    def _uniform_pos_in_zone(self, zone):
        """Uniform random point inside a zone disk (kept for coverage/render logic)."""
        r = zone.radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        x = np.clip(zone.center[0] + r * np.cos(theta), 0, self.config.ENV_WIDTH)
        y = np.clip(zone.center[1] + r * np.sin(theta), 0, self.config.ENV_HEIGHT)
        return [x, y]

    def _sample_user_positions(self, num_users):
        """Return num_users positions respecting GROUND_USERS_DISTRIBUTION.

        'uniform' (default): uniform random inside the boundary margin.
        'even': users distributed round-robin across a grid of NUM_ZONES cells,
                then uniform random within each cell — prevents clustering.
        """
        margin = getattr(self.config, 'BOUNDARY_MARGIN', 50)
        dist   = getattr(self.config, 'GROUND_USERS_DISTRIBUTION', 'uniform')

        if dist == 'even':
            num_zones = getattr(self.config, 'NUM_ZONES', 1)
            cols   = int(np.ceil(np.sqrt(num_zones)))
            rows   = int(np.ceil(num_zones / cols))
            cell_w = (self.config.ENV_WIDTH  - 2 * margin) / cols
            cell_h = (self.config.ENV_HEIGHT - 2 * margin) / rows
            positions = []
            for i in range(num_users):
                zone = i % num_zones
                col  = zone % cols
                row  = zone // cols
                x = margin + col * cell_w + np.random.uniform(0, cell_w)
                y = margin + row * cell_h + np.random.uniform(0, cell_h)
                positions.append([x, y])
            return positions

        return [
            [np.random.uniform(margin, self.config.ENV_WIDTH  - margin),
             np.random.uniform(margin, self.config.ENV_HEIGHT - margin)]
            for _ in range(num_users)
        ]

    def _initialize_ground_users(self):
        """Spawn ground users inside the disaster zones."""
        num_users = getattr(self.config, 'NUM_GROUND_USERS', 0)
        mobile = getattr(self.config, 'GROUND_USERS_MOBILE', False)
        for i, pos in enumerate(self._sample_user_positions(num_users)):
            user = GroundUser(id=f"user_{i+1}", position=pos, mobile=mobile)
            self.ground_users.append(user)

    def get_state_RL(self, uav_id):
        """Get STATE_DIM-dim state for one UAV.

        [own_x, own_y,  dx1,dy1, ..., dx_K,dy_K (UAV neighbors),  avg_rate, n_users, avg_sinr,  dx1,dy1, ..., dx_k,dy_k (nearest users)]
        Dimensions: 2 + NUM_NEIGHBORS*2 + 3 + NUM_USER_NEIGHBORS*2
        """
        uav = next((u for u in self.uavs if u.id == uav_id), None)
        if uav is None:
            return None

        state = []

        # Normalise positions by max traversable distance per episode so that
        # one action unit (UAV_MAX_STEP metres) maps to 1/EPISODE_LENGTH obs units.
        pos_scale = self.config.UAV_MAX_STEP * self.config.EPISODE_LENGTH

        state.append(uav.pos[0] / pos_scale)
        state.append(uav.pos[1] / pos_scale)

        # Relative positions to 4 nearest neighbours (dx, dy) — padded with zeros if fewer
        n_neighbors = getattr(self.config, 'NUM_NEIGHBORS', 4)
        others = sorted(
            [u for u in self.uavs if u.id != uav_id],
            key=lambda u: calculate_distance(u.pos, uav.pos)
        )
        for nb in others[:n_neighbors]:
            state.append((nb.pos[0] - uav.pos[0]) / pos_scale)
            state.append((nb.pos[1] - uav.pos[1]) / pos_scale)
        for _ in range(max(0, n_neighbors - len(others))):
            state.extend([0.0, 0.0])

        # User-related features from the most recent assignment
        assigned = self.current_assignment.get(uav_id, [])
        h_uav = getattr(self.config, 'UAV_HEIGHT', 100.0)
        n_total = max(len(self.ground_users), 1)

        if assigned:
            bw_per_user = self.config.BANDWIDTH_PER_UAV / len(assigned)
            rates, sinrs_dB = [], []
            for user in assigned:
                d_2d = calculate_distance(user.pos, uav.pos)
                pl   = air_to_ground_path_loss(d_2d, h_uav, self.config.FREQUENCY)
                inter_sources_dBm = [
                    self.config.TRANSMIT_POWER - air_to_ground_path_loss(
                        calculate_distance(user.pos, o.pos), h_uav, self.config.FREQUENCY
                    )
                    for o in self.uavs if o.id != uav_id
                ]
                rates.append(uav_to_user_rate(
                    d_2d, h_uav, self.config.FREQUENCY, bw_per_user,
                    self.config.TRANSMIT_POWER, self.config.NOISE_POWER,
                    interferers_received_dBm=inter_sources_dBm,
                ))
                sinrs_dB.append(linear_to_dBm(compute_sinr_linear(
                    self.config.TRANSMIT_POWER, pl, self.config.NOISE_POWER, inter_sources_dBm
                )))
            avg_rate_mbps = np.mean(rates) / 1e6
            avg_sinr_norm = np.mean(sinrs_dB) / 30.0
        else:
            avg_rate_mbps = 0.0
            avg_sinr_norm = 0.0

        state.append(avg_rate_mbps)
        state.append(len(assigned) / n_total)   # fraction of users served
        state.append(avg_sinr_norm)

        # Relative positions to k nearest ground users (dx, dy) — padded with zeros if fewer
        k = getattr(self.config, 'NUM_USER_NEIGHBORS', 3)
        nearest_users = sorted(self.ground_users, key=lambda u: calculate_distance(u.pos, uav.pos))
        for u in nearest_users[:k]:
            state.append((u.pos[0] - uav.pos[0]) / pos_scale)
            state.append((u.pos[1] - uav.pos[1]) / pos_scale)
        for _ in range(max(0, k - len(nearest_users))):
            state.extend([0.0, 0.0])

        return np.array(state, dtype=np.float32)
    
    def _get_minimum_link_capacity(self):
        """Get minimum link capacity in Mbps, representing the bottleneck link"""
        if not getattr(self.config, 'COMMS_ENABLED', True):
            return 0.0

        if not hasattr(self.sim, 'links') or not self.sim.links:
            return 0.0

        min_capacity = float('inf')
        valid_links = 0

        for link in self.sim.links:
            if link.capacity_bps is not None:
                min_capacity = min(min_capacity, link.capacity_bps)
                valid_links += 1

        return (min_capacity / 1e6) if valid_links > 0 else 0.0
    
    def step(self, actions):
        """Execute actions for all UAVs and return new states, rewards, done"""
        rewards = {}
        
        # Execute actions for each UAV
        for i, uav in enumerate(self.uavs):
            if uav.id in actions:
                action = actions[uav.id]
                
                # Convert continuous action [-1,1] to movement
                max_movement = getattr(self.config, 'UAV_MAX_STEP', 20)

                movement_x = action[0] * max_movement
                movement_y = action[1] * max_movement

                # Update position and clamp to map bounds
                uav.move([movement_x, movement_y])
                uav.pos[0] = np.clip(uav.pos[0], 0, self.config.ENV_WIDTH)
                uav.pos[1] = np.clip(uav.pos[1], 0, self.config.ENV_HEIGHT)
                
                # Store movement magnitude for reward calculation
                movement_magnitude = np.sqrt(movement_x**2 + movement_y**2)
                uav.last_movement_magnitude = movement_magnitude
            
                
                # Decrease battery
                uav.battery_level = max(0, uav.battery_level - 0.1)
        
        # Update ground users
        for user in self.ground_users:
            user.update()

        # Update disaster zone coverage
        drone_positions = [uav.pos for uav in self.uavs]
        self.total_coverage = sum(
            zone.update_coverage(drone_positions) for zone in self.disaster_zones
        ) / len(self.disaster_zones)
        
        # Update communication links (skipped when COMMS_ENABLED=False for speed)
        if getattr(self.config, 'COMMS_ENABLED', True):
            self.sim.update_links()
        
        # Assign each user to the nearest UAV only if that UAV's SINR >= SINR_MIN_DB
        uav_assignment = {uav.id: [] for uav in self.uavs}
        h_uav      = getattr(self.config, 'UAV_HEIGHT', 100.0)
        sinr_min   = getattr(self.config, 'SINR_MIN_DB', 5.0)
        for user in self.ground_users:
            dists    = [calculate_distance(user.pos, u.pos) for u in self.uavs]
            best_i   = int(np.argmin(dists))
            best_uav = self.uavs[best_i]
            pl_best  = air_to_ground_path_loss(dists[best_i], h_uav, self.config.FREQUENCY)
            inter_dBm = [
                self.config.TRANSMIT_POWER - air_to_ground_path_loss(dists[j], h_uav, self.config.FREQUENCY)
                for j in range(len(self.uavs)) if j != best_i
            ]
            sinr_dB = linear_to_dBm(compute_sinr_linear(
                self.config.TRANSMIT_POWER, pl_best, self.config.NOISE_POWER, inter_dBm
            ))
            if sinr_dB >= sinr_min:
                uav_assignment[best_uav.id].append(user)
        self.current_assignment = uav_assignment
        for uav in self.uavs:
            rewards[uav.id] = self._calculate_reward(uav, uav_assignment[uav.id])
        
        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps
        
        # Get new states
        new_states = {}
        for uav in self.uavs:
            new_states[uav.id] = self.get_state_RL(uav.id)
        
        return new_states, rewards, done

    def _calculate_reward(self, uav, assigned_users):
        """rᵢ = Σ Shannon(SINR, userⱼ) / n_total  [Mbps] - collision - boundary

        Normalised by n_total so all terms stay in the same [0, ~0.5] Mbps scale,
        keeping the critic stable. Penalties are scaled the same way.
        """
        n_total = max(len(self.ground_users), 1)
        h_uav   = getattr(self.config, 'UAV_HEIGHT', 100.0)

        sinr_rate = 0.0
        if assigned_users:
            bw_per_user = self.config.BANDWIDTH_PER_UAV / len(assigned_users)
            for user in assigned_users:
                d_self  = calculate_distance(user.pos, uav.pos)
                pl_self = air_to_ground_path_loss(d_self, h_uav, self.config.FREQUENCY)
                inter_sources_dBm = [
                    self.config.TRANSMIT_POWER - air_to_ground_path_loss(
                        calculate_distance(user.pos, o.pos), h_uav, self.config.FREQUENCY,
                    )
                    for o in self.uavs if o.id != uav.id
                ]
                sinr_lin   = compute_sinr_linear(self.config.TRANSMIT_POWER, pl_self,
                                                 self.config.NOISE_POWER, inter_sources_dBm)
                sinr_rate += calculate_shannon_capacity(bw_per_user, sinr_lin)
        sinr_rate /= (n_total * 1e6)

        collision_dist = getattr(self.config, 'COLLISION_DISTANCE', 100.0)
        collision_pen  = getattr(self.config, 'COLLISION_PENALTY', 5.0) / n_total
        collision_penalty = -collision_pen * sum(
            1 for o in self.uavs if o.id != uav.id
            and calculate_distance(o.pos, uav.pos) < collision_dist
        )

        margin       = getattr(self.config, 'BOUNDARY_MARGIN', 50)
        boundary_pen = getattr(self.config, 'BOUNDARY_PENALTY', 2.0) / n_total
        near_edge = (
            uav.pos[0] < margin or uav.pos[0] > self.config.ENV_WIDTH  - margin or
            uav.pos[1] < margin or uav.pos[1] > self.config.ENV_HEIGHT - margin
        )
        boundary_penalty = -boundary_pen if near_edge else 0.0

        return sinr_rate + collision_penalty + boundary_penalty

    @property
    def global_sum_rate_mbps(self):
        return compute_global_sum_rate_mbps(self.uavs, self.current_assignment, self.config)

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
            margin = getattr(self.config, 'SPAWN_MARGIN', 100)
            while len(start_positions) < num_uavs:
                start_positions.append([
                    np.random.uniform(margin, self.config.ENV_WIDTH  - margin),
                    np.random.uniform(margin, self.config.ENV_HEIGHT - margin),
                ])
        else:
            margin = getattr(self.config, 'SPAWN_MARGIN', 100)
            start_positions = [
                [np.random.uniform(margin, self.config.ENV_WIDTH  - margin),
                 np.random.uniform(margin, self.config.ENV_HEIGHT - margin)]
                for _ in range(num_uavs)
            ]
        
        for i, uav in enumerate(self.uavs):
            uav.pos = start_positions[i].copy()
            uav.battery_level = 100.  
        # Re-randomize zone layout each episode (used for coverage tracking / rendering)
        if not deterministic:
            self.disaster_zones = self._create_disaster_zones(deterministic=False)

        # Re-spawn ground users uniformly across the full environment each episode
        mobile = getattr(self.config, 'GROUND_USERS_MOBILE', False)
        for user, pos in zip(self.ground_users, self._sample_user_positions(len(self.ground_users))):
            user.pos = pos
            user.mobile = mobile

        # Reset disaster zones
        for zone in self.disaster_zones:
            zone.coverage_status = 0.0
        
        # Reset tracking variables
        self.total_coverage = 0.0
        self.episode_steps = 0
        self.current_assignment = {uav.id: [] for uav in self.uavs}
        
        # Recreate communication links
        self.sim.drones = self.uavs
        self.sim.create_links()
        
        # Recompute baseline coverage from current positions before returning states
        drone_positions = [uav.pos for uav in self.uavs]
        self.total_coverage = sum(
            zone.update_coverage(drone_positions) for zone in self.disaster_zones
        ) / len(self.disaster_zones) if self.disaster_zones else 0.0

        # Return initial states
        initial_states = {}
        for uav in self.uavs:
            initial_states[uav.id] = self.get_state_RL(uav.id)
        
        return initial_states
    
    def get_coverage_metrics(self):
        """Get current coverage metrics"""
        metrics = {
            'total_coverage': self.total_coverage,
            'zone_coverage': [zone.coverage_status for zone in self.disaster_zones],
            'uav_positions': [uav.pos for uav in self.uavs],
            'battery_levels': [uav.battery_level for uav in self.uavs],
            'ground_user_positions': [user.pos for user in self.ground_users],
        }
        return metrics
    
    def get_coms_metrics(self, uav_users, fl_model_bytes=None):
        """Per-UAV and global comms metrics for the FL simulation task.

        Args:
            uav_users: dict {uav_index: [GroundUser, ...]} — current user assignment
            fl_model_bytes: optional model size in bytes; when provided, includes
                            an estimate of FL uplink transmission time per UAV

        Returns a dict with:
          'per_uav'  — one entry per UAV with channel quality and bandwidth info
          'global_*' — aggregates across all UAVs / users useful as training context
        """
        B_total = self.config.BANDWIDTH_PER_UAV
        alpha   = getattr(self.config, 'B_USERS_FRACTION', 0.7)
        B_FL    = (1 - alpha) * B_total
        h_uav   = getattr(self.config, 'UAV_HEIGHT', 100.0)
        freq    = self.config.FREQUENCY
        P_tx    = self.config.TRANSMIT_POWER
        N_0     = self.config.NOISE_POWER

        all_rates = []
        all_plos  = []
        per_uav   = []

        for i, uav in enumerate(self.uavs):
            served      = uav_users.get(i, [])
            n_users     = len(served)
            per_user_bw = (alpha * B_total / n_users) if n_users > 0 else 0.0

            user_rates, sinrs_db, plos_vals = [], [], []

            for user in served:
                d_2d = calculate_distance(user.pos, uav.pos)
                pl   = air_to_ground_path_loss(d_2d, h_uav, freq)
                p    = prob_los(d_2d, h_uav)
                sinr = P_tx - pl - N_0          # dB
                rate = uav_to_user_rate(d_2d, h_uav, freq, per_user_bw, P_tx, N_0)
                user_rates.append(rate)
                sinrs_db.append(sinr)
                plos_vals.append(p)

            all_rates.extend(user_rates)
            all_plos.extend(plos_vals)

            entry = {
                'uav_id':            uav.id,
                'uav_pos':           list(uav.pos),
                'battery':           uav.battery_level,
                'n_users':           n_users,
                'B_users':           per_user_bw * n_users,
                'B_FL':              B_FL,
                'user_rates_bps':    user_rates,
                'avg_user_rate_bps': float(np.mean(user_rates)) if user_rates else 0.0,
                'sinr_dB':           sinrs_db,
                'avg_sinr_dB':       float(np.mean(sinrs_db)) if sinrs_db else 0.0,
                'plos':              plos_vals,
                'avg_plos':          float(np.mean(plos_vals)) if plos_vals else 0.0,
            }

            if fl_model_bytes is not None:
                entry['fl_tx_time_s'] = (fl_model_bytes * 8) / B_FL if B_FL > 0 else float('inf')

            per_uav.append(entry)

        return {
            'per_uav':             per_uav,
            'global_avg_rate_bps': float(np.mean(all_rates)) if all_rates else 0.0,
            'global_avg_plos':     float(np.mean(all_plos))  if all_plos  else 0.0,
            'global_throughput_bps': float(np.sum(all_rates)),
            'zone_coverage':       [z.coverage_status for z in self.disaster_zones],
            'total_coverage':      self.total_coverage,
            'B_FL_per_uav':        B_FL,
            'B_users_total':       alpha * B_total,
        }

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
        
        # Plot ground users
        if self.ground_users:
            gx = [u.pos[0] for u in self.ground_users]
            gy = [u.pos[1] for u in self.ground_users]
            ax.scatter(gx, gy, c='black', s=30, marker='^', label='Ground users', zorder=3)

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
        
        ax.set_xlim(0, self.config.ENV_WIDTH)
        ax.set_ylim(0, self.config.ENV_HEIGHT)
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
            margin = getattr(self.config, 'SPAWN_MARGIN', 100)
            x = np.random.uniform(margin, self.config.ENV_WIDTH  - margin)
            y = np.random.uniform(margin, self.config.ENV_HEIGHT - margin)
            fixed_positions.append([x, y])
        
        # Reset with deterministic positioning
        states = self.reset(deterministic=True, fixed_positions=fixed_positions)
        total_reward   = 0.0
        total_sum_rate = 0.0

        for step in range(self.config.EPISODE_LENGTH):
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in states:
                    action = agent.select_action(states[agent_id], explore=False, noise_scale=0.0)
                    actions[agent_id] = action

            new_states, rewards, done = self.step(actions)
            total_reward   += np.mean(list(rewards.values()))
            total_sum_rate += self.global_sum_rate_mbps
            states = new_states

        avg_reward   = total_reward   / self.config.EPISODE_LENGTH
        avg_sum_rate = total_sum_rate / self.config.EPISODE_LENGTH

        print(f"{eval_name} Results:")
        print(f"  Average Reward:   {avg_reward:.2f}")
        print(f"  Avg Sum Rate:     {avg_sum_rate:.2f} Mbps")

        return avg_reward, avg_sum_rate