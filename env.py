# env/grid.py
import random
import numpy as np
from coms import Link
from drones import Drone
        

class Simulation:
    def __init__(self, bandwidth, frequency, noise_power_dBm):
        self.drones = []  # List of drones in the simulation : index 0 is video drone last is operator
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

    def run_simulation(self):
        # Compute SINR and capacity for each link
        for link in self.links:
            sinr = link.calculate_sinr()
            capacity = link.calculate_capacity()
            print(f"Link from Drone {self.drones.index(link.drone1)} to Drone {self.drones.index(link.drone2)}")
            print(f"  SINR: {sinr:.2f} dB, Capacity: {capacity:.2f} bps")
    
    def deploy_relay(self):
        new_relay = Drone(position=[10,10]) 
        self.drones.insert(-1, new_relay)
        self.create_links()


