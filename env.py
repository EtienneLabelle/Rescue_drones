# env/grid.py
import random
import numpy as np
import coms

class GridEnvironment:
    def __init__(self, width=10, height=10, num_interference_sources=5):
        """
        Initialize the grid environment.

        :param width: Width of the grid
        :param height: Height of the grid
        :param num_obstacles: Number of obstacles on the grid
        """
        self.width = width
        self.height = height
        # Create the grid: 0 = empty, 2 = obstacle, 3 = goal
        self.grid = np.zeros((height, width), dtype=int)
        self.interference_sources = self.create_interference_sources(num_interference_sources)

    def create_interference_sources(self, num_sources):
        """
        Randomly place interference sources on the grid.
        :param num_sources: Number of interference sources to generate.
        :return: List of interference source positions.
        """
        interference_sources = []
        for _ in range(num_sources):
            while True:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                if self.grid[y, x] == 0:  # Make sure the spot is empty
                    self.grid[y, x] = 99  # Mark the grid with a value for interference (e.g., 3)
                    interference_sources.append((x, y))
                    break
        return interference_sources


    def render(self):
        """Prints the grid to the console."""
        render_grid = self.grid.copy()
        print(render_grid)

        
""" This could be good not sure yet
class Simulation:
    def __init__(self):
        self.drones = []  # List of all drone objects
        self.positions = []  # List of all drone positions (updated regularly)

    def add_drone(self, drone):
        self.drones.append(drone)
        self.positions.append(drone.get_position())  # Initialize with starting position

    def update_positions(self):
        # Update the positions list based on drone movements
        self.positions = [drone.get_position() for drone in self.drones]

    def simulate_step(self):
        # Move each drone (this is just an example, you'd have specific logic)
        for drone in self.drones:
            drone.move((1, 1))  # Example movement
        self.update_positions()  # Update the position list after moving
"""