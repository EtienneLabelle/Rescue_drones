import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimpleAnimator:
    def __init__(self, interval=100, xlim=(0, 20000), ylim=(0, 20000)):
        self.interval = interval
        self.simulation_data = []  # Store positions at intervals
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.drones_plot, = self.ax.plot([], [], 'bo', label="Drones")  # Drones as blue dots
        self.obstacle_edges = []  # Store static obstacle edges

    def set_obstacles(self, obstacles):
        """Store and plot obstacle edges as vertical lines once."""
        for obstacle in obstacles:
            # Extract the x-coordinate and y-coordinates for the obstacle line
            x = obstacle.center_pos[0]  # Use the center x-coordinate
            y_start, y_end = obstacle.edges_pos[0][1], obstacle.edges_pos[-1][1]  # Start and end y-coordinates
            # Plot the obstacle as a vertical line from y_start to y_end at x
            self.ax.plot([x, x], [y_start, y_end], 'r-')  # Draws a vertical red line


    def record_positions(self, drones, step):
        """Capture drone positions at intervals."""
        if step % self.interval == 0:
            drone_positions = [(drone.pos[0], drone.pos[1]) for drone in drones]
            self.simulation_data.append(drone_positions)

    def init_plot(self):
        """Initialize plot with static obstacles."""
        self.drones_plot.set_data([], [])
        return self.drones_plot,

    def update_plot(self, frame):
        """Update the plot with drone positions for each frame in the animation."""
        drone_positions = self.simulation_data[frame]
        drone_x, drone_y = zip(*drone_positions)
        self.drones_plot.set_data(drone_x, drone_y)
        return self.drones_plot,

    def animate(self):
        """Run the animation after the simulation completes."""
        ani = FuncAnimation(self.fig, self.update_plot, frames=len(self.simulation_data), 
                            init_func=self.init_plot, blit=True, repeat=False)
        plt.legend()
        plt.show()
