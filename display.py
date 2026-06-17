import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

class SimpleAnimator:
    def __init__(self, root, obstacles, xlim=(0, 2000), ylim=(0, 2000), disaster_zones=None):
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
        
        self.xlim = xlim
        self.ylim = ylim

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

    def set_metrics_text(self, link_metrics, step_rewards=None):
        """Format and display per-UAV throughput metrics."""
        if not isinstance(link_metrics, dict) or 'per_uav' not in link_metrics:
            return

        lines = []
        if step_rewards:
            mean_r = float(np.mean(list(step_rewards.values())))
            lines.append(f"Mean reward  : {mean_r:.4f}\n")
            lines.append("\n")
        per_uav_sums = []

        for e in link_metrics['per_uav']:
            uav_label = e['uav_id'].replace('uav_', 'UAV ')
            n         = e['n_users']
            sum_rate  = sum(e['user_rates_bps']) / 1e6
            avg_sinr  = e['avg_sinr_dB']
            plos      = e['avg_plos']

            per_uav_sums.append(sum_rate)

            lines.append(f"{uav_label}  ({n} user{'s' if n != 1 else ''})\n")
            lines.append(f"  Sum rate : {sum_rate:.2f} Mbps\n")
            lines.append(f"  Avg SINR : {avg_sinr:.1f} dB\n")
            lines.append(f"  P_LoS    : {plos:.3f}\n")
            if step_rewards:
                r = step_rewards.get(e['uav_id'])
                if r is not None:
                    lines.append(f"  Reward   : {r:.4f}\n")
            lines.append("\n")

        avg_sum_rate  = float(np.mean(per_uav_sums)) if per_uav_sums else 0.0
        total_throughput = link_metrics['global_throughput_bps'] / 1e6

        lines.append("-" * 30 + "\n")
        lines.append(f"Avg sum rate : {avg_sum_rate:.2f} Mbps\n")
        lines.append(f"Total        : {total_throughput:.2f} Mbps\n")
        lines.append(f"Avg P_LoS    : {link_metrics['global_avg_plos']:.3f}\n")

        self.metrics_display.delete(1.0, tk.END)
        self.metrics_display.insert(tk.END, "".join(lines))
        self.metrics_display.see(tk.END)

    def record_positions(self, drones, links, ground_users=None, uav_users=None, coms_metrics=None, step_rewards=None):
        """Record drone positions and links for animation.

        Args:
            ground_users: list of GroundUser objects
            uav_users:    dict {uav_index: [GroundUser, ...]} for drawing assignment lines
            coms_metrics: output of env.get_coms_metrics(), shown in the metrics panel
        """
        drone_positions = []
        for drone in drones:
            if hasattr(drone, 'pos') and drone.pos is not None:
                if len(drone.pos) >= 2:
                    drone_positions.append([float(drone.pos[0]), float(drone.pos[1])])
                else:
                    drone_positions.append([0.0, 0.0])
            else:
                drone_positions.append([0.0, 0.0])

        links_pos = []
        link_metrics = {'total_links': 0, 'active_links': 0, 'avg_snr': 0.0}

        # Snapshot user positions now so playback isn't affected by later mutations
        user_positions = [[float(u.pos[0]), float(u.pos[1])] for u in (ground_users or [])]

        # Snapshot UAV→user assignment: {uav_index: [[x, y], ...]}
        assignment_snapshot = {}
        if uav_users:
            for k, users in uav_users.items():
                if isinstance(k, int):
                    idx = k
                else:
                    try:
                        idx = int(str(k).split('_')[-1]) - 1
                    except (ValueError, IndexError):
                        continue
                assignment_snapshot[idx] = [[float(u.pos[0]), float(u.pos[1])] for u in users]

        extras = {
            'user_positions':  user_positions,
            'coms_metrics':    coms_metrics,
            'uav_assignment':  assignment_snapshot,
            'step_rewards':    step_rewards,
        }

        self.simulation_data.append((drone_positions, links_pos, link_metrics, extras))
        
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

    _UAV_COLORS = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red',
                   'tab:orange', 'tab:cyan', 'tab:pink']

    def start_animation(self):
        """Advance one step in the animation each time the button is clicked."""
        if self.current_frame < len(self.simulation_data):
            self.ax.cla()

            frame = self.simulation_data[self.current_frame]
            drone_positions, links_pos, link_metrics = frame[0], frame[1], frame[2]
            extras = frame[3] if len(frame) > 3 else {}

            coms_metrics   = extras.get('coms_metrics')
            user_positions = extras.get('user_positions', [])
            uav_assignment = extras.get('uav_assignment', {})
            step_rewards   = extras.get('step_rewards')

            # Ground users — black stars
            if user_positions:
                xs = [p[0] for p in user_positions]
                ys = [p[1] for p in user_positions]
                self.ax.scatter(xs, ys, c='black', s=60, marker='*', zorder=4, label='Ground users')

            # UAVs + assignment lines to their served users
            for i, (x, y) in enumerate(drone_positions):
                color = self._UAV_COLORS[i % len(self._UAV_COLORS)]
                # assignment lines first (behind the UAV marker)
                for up in uav_assignment.get(i, []):
                    self.ax.plot([x, up[0]], [y, up[1]], color=color,
                                 alpha=0.35, linewidth=0.9, zorder=3)
                self.ax.scatter(x, y, c=color, s=150, zorder=5,
                                edgecolors='black', linewidths=1)
                self.ax.annotate(f'UAV {i+1}', (x, y), xytext=(5, 5),
                                 textcoords='offset points', fontsize=8, weight='bold')

            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            self.ax.set_xlabel('X Position (m)')
            self.ax.set_ylabel('Y Position (m)')
            self.ax.set_title(f'Simulation — step {self.current_frame + 1}')
            self.ax.grid(True, alpha=0.3)

            self.set_metrics_text(coms_metrics if coms_metrics is not None else link_metrics, step_rewards=step_rewards)
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
        self.canvas.draw()
        
        # Clear metrics display
        if hasattr(self, 'metrics_display'):
            self.metrics_display.delete(1.0, tk.END)
        
        print("Animation reset to step 1!")


def run_episode_viz(agents, env, seed, label=""):
    """Run one deterministic episode and launch the tkinter animator.

    Spawns UAVs at positions derived from `seed`, rolls out with no exploration
    noise, records every step, then hands off to the interactive SimpleAnimator.
    """
    import sys
    import tkinter as tk
    import numpy as np
    import random
    import torch
    from config import Config

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    fixed_positions = []
    for i in range(Config.NUM_UAVS):
        np.random.seed(seed + i)
        margin = Config.SPAWN_MARGIN
        fixed_positions.append([
            np.random.uniform(margin, Config.ENV_WIDTH  - margin),
            np.random.uniform(margin, Config.ENV_HEIGHT - margin),
        ])

    states = env.reset(deterministic=True, fixed_positions=fixed_positions)

    root = tk.Tk()
    if label:
        root.title(f"Simulation — {label}")
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), sys.exit(0)))
    animator = SimpleAnimator(root, env.sim.obstacles)
    animator.env = env

    for step in range(Config.EPISODE_LENGTH):
        actions = {
            aid: agent.select_action(states[aid], explore=False, noise_scale=0.0)
            for aid, agent in agents.items() if aid in states
        }
        new_states, rewards, done = env.step(actions)

        if Config.COMMS_ENABLED:
            env.sim.update_links()

        uav_users_int = {
            i: env.current_assignment.get(f"uav_{i+1}", [])
            for i in range(len(env.uavs))
        }
        coms = env.get_coms_metrics(uav_users_int)

        animator.record_positions(
            env.uavs, env.sim.links,
            ground_users=env.ground_users,
            uav_users=getattr(env, "current_assignment", {}),
            coms_metrics=coms,
            step_rewards=rewards,
        )

        if step % 50 == 0:
            r  = float(np.mean(list(rewards.values())))
            sr = env.global_sum_rate_mbps
            print(f"  Step {step:3d}: reward={r:.3f}  SR={sr:.2f} Mbps")

        states = new_states

    root.mainloop()

