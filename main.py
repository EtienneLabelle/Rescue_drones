from coms import calculate_sinr_with_fading
from drones import Drone, Operator, deploy_relay
from utils import calculate_distance

FREQUENCY = 2.4e9  # 2.4 GHz
BANDWIDTH = 20e6   # 20 MHz
TRANSMIT_POWER = 30  # Transmit power in dBm (decibel milliwatts)
MIN_RECEIVE_POWER = -80  # Minimum power in dBm for a stable link

# Setup initial positions in 2D
operator = Operator([0, 0])  # Operator starts at position (0, 0)
video_drone = Drone([0, 0])  # Video drone starts at the same position
# List to store relay drones
relay_drones = []
positions = []


# Simulation loop
for step in range(1, 2000):  

    video_drone.move([10, 10])  
    if len(positions) == 0:  # If the list is empty
        positions.append(video_drone.pos)  # Initialize the list with the first position
    else:
        positions[0] = video_drone.pos  # Replace the first position if it exists

    for i, drone in enumerate(relay_drones):
        drone.move([10,10])
        positions[i+1] = drone.pos  # Update existing position

    last_point = video_drone.pos if not relay_drones else relay_drones[-1].pos
    distance = calculate_distance(operator.pos,last_point)

    if distance> 1000:
        print("step #",step)
        deploy_relay(relay_drones)
        positions.append(relay_drones[-1].pos)  # Add new position for the relay drone
        #print("all positions", positions)
        #coms_metrics = calculate_coms_metrics(positions)
        end_to_end_sinr_DB = calculate_sinr_with_fading(positions)
        print(end_to_end_sinr_DB)




