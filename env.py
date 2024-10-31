# env/grid.py       
import random
from coms import Link
from drones import Drone
        

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

