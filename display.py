import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

class SimpleAnimator:
    def __init__(self, root, obstacles, xlim=(0, 20000), ylim=(0, 20000)):
        self.simulation_data = []  # Store drone positions at intervals
        self.obstacles = obstacles
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

    def set_metrics_text(self, link_metrics):
        """Format and display link metrics as a table in the metrics display."""
        # Define headers for the table
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
        """Capture drone positions and link positions at intervals."""

        drone_positions = [(drone.pos[0], drone.pos[1]) for drone in drones]
        links_pos = [((link.drone1.pos[0], link.drone1.pos[1]), (link.drone2.pos[0], link.drone2.pos[1])) for link in links]
        link_metrics = links
        self.simulation_data.append((drone_positions, links_pos, link_metrics))  # Store drones and links together

    def set_links(self, links_pos):
        """Plot links between drones for the current frame."""
        for start, end in links_pos:
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', label="Link")  # Green line for links


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
            self.set_obstacles()


            # Plot the current frame's positions
            drone_positions, links_pos, link_metrics = self.simulation_data[self.current_frame]
            drone_x, drone_y = zip(*drone_positions)
            self.ax.plot(drone_x, drone_y, 'bo', label="Drones")  # Blue dots for drones
            self.set_links(links_pos)  # Green lines for links    
            

            self.set_metrics_text(link_metrics) 
            # Update the canvas and advance the frame
            self.canvas.draw()
            self.current_frame += 1

            if self.playing:
                self.canvas.get_tk_widget().after(1,self.start_animation)

    def play_animation(self):
        """Start the animation."""
        if not self.playing:
            self.playing = True
            self.start_animation()

    def stop_animation(self):
        """Stop the animation."""
        self.playing = False





