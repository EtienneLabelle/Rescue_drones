import coms
from drones import Drone, Operator, deploy_relay

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
for step in range(1, 2000):  # Simulate for longer distances
    # Move the video drone by 10 units in x and 10 units in y
    video_drone.move([10, 10])  

    # Determine the last communication point (either the operator or the last relay)
    last_position = operator.pos if not relay_drones else relay_drones[-1].pos
    
    # Calculate the distance between the video drone and the last communication point (in 2D)
    distance = coms.calculate_distance(video_drone.pos, last_position)
    received_power = coms.calculate_received_power(distance)

    # Check if the signal strength drops below the threshold
    if received_power < MIN_RECEIVE_POWER:
        # Deploy a new relay to restore communication
        deploy_relay(video_drone, relay_drones)

    previous_position = operator.pos
    for relay_drone in relay_drones:
        relay_distance = coms.calculate_distance(relay_drone.pos, previous_position)
        received_power = coms.calculate_received_power(relay_distance)
        
        if received_power < MIN_RECEIVE_POWER:
            print(f"Communication failed at relay {relay_drone.pos:.2f} meters")
            link_ok = False
            break
        
        previous_position = relay_drone.pos

    # Final hop to the video drone
    last_relay_position = operator.pos if not relay_drones else relay_drones[-1].pos
    last_relay_distance = coms.calculate_distance(video_drone.pos, last_relay_position)
    received_power_last = coms.calculate_received_power(last_relay_distance)

    if step % 100 == 0:
        print(f"# of relays: {len(relay_drones)}, Signal strength: {received_power_last:.2f}, Video drone position: {video_drone.pos}")
