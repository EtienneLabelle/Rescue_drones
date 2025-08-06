import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

class SimpleAnimator:
    def __init__(self, root, obstacles, xlim=(0, 20000), ylim=(0, 20000), disaster_zones=None):
        self.simulation_data = []  # Store drone positions at intervals
        self.obstacles = obstacles
        self.disaster_zones = disaster_zones or []  # Add disaster zones support
        self.current_frame = 0  # Track the current frame
        self.playing = False  # Track if animation is play

        # Set up the main frame to hold both the plot and metrics side by side
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for the graph
        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set up the figure and axis for the plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        
        # Embed the matplotlib figure in Tkinter on the left side
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add the toolbar for zooming and panning functionality
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)  # Pack the toolbar below the canvas

        # Right frame for displaying metrics with scrolling
        self.metrics_frame = tk.Frame(main_frame, width=200, bg="lightgray")
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add a Text widget with a scrollbar for the metrics display
        self.metrics_display = scrolledtext.ScrolledText(self.metrics_frame, wrap=tk.WORD, width=70, height=20)
        self.metrics_display.pack(fill=tk.BOTH, expand=True)

        # Buttons for Play and Stop functionality
        self.play_button = tk.Button(root, text="Play", command=self.play_animation)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_animation)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Add step and reset buttons for manual control
        self.step_button = tk.Button(root, text="Step", command=self.step_animation)
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_animation)
        self.reset_button.pack(side=tk.LEFT, padx=5)

    def set_metrics_text(self, link_metrics):
        """Format and display link metrics as a table in the metrics display."""
        # Check if link_metrics is a dictionary (summary stats) or list (individual links)
        if isinstance(link_metrics, dict):
            # Display summary statistics
            summary_text = f"Total Links: {link_metrics.get('total_links', 0)}\n"
            summary_text += f"Active Links: {link_metrics.get('active_links', 0)}\n"
            summary_text += f"Average SNR: {link_metrics.get('avg_snr', 0):.2f} dB\n"
            
            self.metrics_display.delete(1.0, tk.END)  # Clear previous text
            self.metrics_display.insert(tk.END, summary_text)  # Insert new text
            self.metrics_display.see(tk.END)  # Scroll to the bottom if text overflows
        else:
            # Handle list of Link objects (legacy support)
            headers = f"{'Link ID':<10} {'SINR (dB)':<12} {'Capacity (bps)':<15} {'Distance (m)':<15} {'Blocked?':<10}\n"
            separator = "-" * 60 + "\n"
            
            # Format each link's metrics into a table row
            table_rows = [headers, separator]
            for i, link in enumerate(link_metrics):
                # Access attributes of the Link object directly
                row = (f"{i:<10} {link.sinr_dB:<12.2f} {link.capacity_bps:<15.2f} "
                    f"{link.distance:<15.2f} {str(link.isBlocked):<10}\n")
                table_rows.append(row)
            
            # Join all rows into a single text block
            table_text = "".join(table_rows)
            
            # Display the formatted table in the metrics display
            self.metrics_display.delete(1.0, tk.END)  # Clear previous text
            self.metrics_display.insert(tk.END, table_text)  # Insert new table text
            self.metrics_display.see(tk.END)  # Scroll to the bottom if text overflows

    def record_positions(self, drones, links):
        """Record drone positions and links for animation"""
        # Record drone positions - ensure they're in the correct format
        drone_positions = []
        for drone in drones:
            if hasattr(drone, 'pos') and drone.pos is not None:
                # Ensure position is a list/array with at least 2 elements
                if len(drone.pos) >= 2:
                    drone_positions.append([float(drone.pos[0]), float(drone.pos[1])])
                else:
                    drone_positions.append([0.0, 0.0])  # Default position if invalid
            else:
                drone_positions.append([0.0, 0.0])  # Default position if no pos attribute
        
        # For now, we'll skip recording links since we're focusing on drones and zones
        links_pos = []
        link_metrics = {
            'total_links': 0,
            'active_links': 0,
            'avg_snr': 0.0
        }
        
        self.simulation_data.append((drone_positions, links_pos, link_metrics))
        
    def set_links(self, links_pos):
        """Plot links between drones for the current frame."""
        for start, end in links_pos:
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', label="Link")  # Green line for links

    def set_disaster_zones(self):
        """Plot disaster zones if they exist."""
        if self.disaster_zones:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
            for i, zone in enumerate(self.disaster_zones):
                color = colors[i % len(colors)]
                circle = plt.Circle(zone.center, zone.radius, 
                                  alpha=0.3, color=color, edgecolor='black', linewidth=2)
                self.ax.add_patch(circle)
                
                # Add zone info
                self.ax.text(zone.center[0], zone.center[1], 
                            f'Zone {i+1}\nS:{zone.severity:.1f}\nC:{zone.coverage_status:.2f}',
                            ha='center', va='center', fontsize=8, weight='bold')

    def set_obstacles(self):
        """Store and plot obstacle edges as vertical lines once."""
        for obstacle in self.obstacles:
            # Assuming obstacles have 'center_pos' (x-coordinate) and 'edges_pos' (y-coordinates for start and end)
            x = obstacle.center_pos[0]
            y_start, y_end = obstacle.edges_pos[0][1], obstacle.edges_pos[-1][1]
            self.ax.plot([x, x], [y_start, y_end], 'r-', linewidth=2, label="Obstacle")  # Red line for obstacles

    def start_animation(self):
        """Advance one step in the animation each time the button is clicked."""
        if self.current_frame < len(self.simulation_data):
            # Clear previous positions
            self.ax.cla()
            self.set_disaster_zones()  # Add disaster zones

            # Plot the current frame's positions
            drone_positions, links_pos, link_metrics = self.simulation_data[self.current_frame]
            
            # Plot drones - handle the position format correctly
            if drone_positions:
                drone_x = [pos[0] for pos in drone_positions]
                drone_y = [pos[1] for pos in drone_positions]
                self.ax.scatter(drone_x, drone_y, c='blue', s=100, label="Drones", zorder=5)
                
                # Add drone labels
                for i, (x, y) in enumerate(zip(drone_x, drone_y)):
                    self.ax.annotate(f'Drone {i+1}', (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8, weight='bold')
            
            # Set axis labels and title
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_title(f'Simulation Step {self.current_frame + 1}')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)

            self.set_metrics_text(link_metrics) 
            # Update the canvas and advance the frame
            self.canvas.draw()
            self.current_frame += 1

            if self.playing:
                self.canvas.get_tk_widget().after(1, self.start_animation)

    def play_animation(self):
        """Start the animation."""
        if not self.playing:
            self.playing = True
            self.start_animation()

    def stop_animation(self):
        """Stop the animation."""
        self.playing = False
    
    def step_animation(self):
        """Step the animation forward manually"""
        if hasattr(self, 'env'):
            # Record current state
            drones = getattr(self.env, 'uavs', [])
            links = getattr(self.env.sim, 'links', []) if hasattr(self.env, 'sim') else []
            self.record_positions(drones, links)
        
        # Advance animation if there's data
        if self.simulation_data:
            self.start_animation()
    
    def reset_animation(self):
        """Reset the animation to step 1"""
        self.current_frame = 0
        
        # Reset the environment if available
        if hasattr(self, 'env'):
            self.env.reset()
        
        # Clear the display but keep the data
        self.ax.cla()
        self.set_disaster_zones()
        self.canvas.draw()
        
        # Clear metrics display
        if hasattr(self, 'metrics_display'):
            self.metrics_display.delete(1.0, tk.END)
        
        print("Animation reset to step 1!")




