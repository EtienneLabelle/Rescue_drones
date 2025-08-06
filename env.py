# env/grid.py       
import random
import numpy as np
from coms import Link
from drones import Drone
from config import Config
        

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
        self.sim = Simulation(config.bandwidth, config.frequency, config.noise_power)
        
        # Create disaster zones
        self.disaster_zones = self._create_disaster_zones()
        
        # Initialize 3 UAVs
        self.uavs = []
        self._initialize_uavs()
        
        # Coverage tracking
        self.total_coverage = 0.0
        self.episode_steps = 0
        self.max_steps = config.simulation_steps
        
        # Communication links between UAVs
        self.sim.drones = self.uavs
        self.sim.create_links()
    
    def _create_disaster_zones(self):
        """Create disaster zones in the environment"""
        zones = []
        num_uavs = getattr(self.config, 'num_uavs', 3)
        
        if num_uavs == 5:
            # Create 5 disaster zones with varying severity
            zone_configs = [
                ((5000, 5000), 1000, 0.9),   # High severity
                ((15000, 5000), 800, 0.7),    # Medium severity
                ((10000, 10000), 1200, 0.8),  # Medium-high severity
                ((5000, 15000), 600, 0.6),    # Medium severity
                ((15000, 15000), 900, 0.85)   # High severity
            ]
        elif num_uavs == 7:
            # Create 7 disaster zones for 7 UAVs
            zone_configs = [
                ((3000, 3000), 800, 0.9),     # Zone 1
                ((12000, 3000), 700, 0.8),    # Zone 2
                ((20000, 3000), 600, 0.7),    # Zone 3
                ((3000, 10000), 900, 0.85),   # Zone 4
                ((12000, 10000), 1000, 0.9),  # Zone 5
                ((3000, 17000), 700, 0.75),   # Zone 6
                ((12000, 17000), 800, 0.8)    # Zone 7
            ]
        else:
            # Generic zones for other numbers
            grid_size = int(np.ceil(np.sqrt(num_uavs)))
            spacing = 20000 // (grid_size + 1)
            
            zone_configs = []
            for i in range(num_uavs):
                row = i // grid_size
                col = i % grid_size
                x = spacing * (col + 1)
                y = spacing * (row + 1)
                radius = 800 + (i % 3) * 200  # Vary radius
                severity = 0.6 + (i % 4) * 0.1  # Vary severity
                zone_configs.append(((x, y), radius, severity))
        
        for center, radius, severity in zone_configs:
            zone = DisasterZone(center, radius, severity)
            zones.append(zone)
        
        return zones
    
    def _initialize_uavs(self):
        """Initialize UAVs at random positions within bounds"""
        num_uavs = getattr(self.config, 'num_uavs', 3)
        
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
        while len(state) < self.config.state_dim:
            state.append(0.0)
        
        return np.array(state[:self.config.state_dim])
    
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
            reward = self._calculate_reward(uav, movement_magnitude)
            rewards[uav.id] = reward
            
            # Debug: Print reward values to see if they're reasonable
            if self.episode_steps % 50 == 0:  # Print every 50 steps
                print(f"UAV {uav.id}: Reward = {reward:.2f}, Pos = {uav.pos}, Movement = {movement_magnitude:.2f}")
        
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
                reward += Config.ZONE_REWARD * nearest_zone.severity * 2.0  # Doubled zone reward
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
        
        # Reset disaster zones
        for zone in self.disaster_zones:
            zone.coverage_status = 0.0
        
        # Reset tracking variables
        self.total_coverage = 0.0
        self.episode_steps = 0
        
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

