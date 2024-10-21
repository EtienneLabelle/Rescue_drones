import coms
from drones import RelayDrone, VideoDrone, Operator, deploy_relay

FREQUENCY = 2.4e9  # 2.4 GHz
BANDWIDTH = 20e6   # 20 MHz
TRANSMIT_POWER = 30  # Transmit power in dBm (decibel milliwatts)
MIN_RECEIVE_POWER = -80  # Minimum power in dBm for a stable link

# Setup initial positions
operator = Operator(0)  # Operator is at position 0
video_drone = VideoDrone(0)  # Video drone starts at the same position
# List to store relay drones
relay_drones = []
positions=[]

# Simulation loop
for step in range(1, 2000):  # Simulate for longer distances
    video_drone.move(10)  # Move 1 meter per step

    # Determine the last communication point (either the operator or the last relay)
    last_position = operator.position if not relay_drones else relay_drones[-1].position
    
    # Calculate the distance between the video drone and the last communication point
    distance = abs(video_drone.position - last_position)
    received_power = coms.calculate_received_power(TRANSMIT_POWER, distance)

    # Check if the signal strength drops below the threshold
    if received_power < MIN_RECEIVE_POWER:     
        # Deploy a new relay to restore communication
        deploy_relay(relay_drones)

    previous_position = operator.position
    for relay_drone in relay_drones:
        relay_distance = abs(relay_drone.position - previous_position) 
        received_power = coms.calculate_received_power(TRANSMIT_POWER, relay_distance)
        
        if received_power < MIN_RECEIVE_POWER:
            print(f"Communication failed at relay {relay_drone.position:.2f} meters")
            link_ok = False
            break
        
        previous_position = relay_drone.position

    if relay_drones:
        last_relay_distance = abs(video_drone.position - relay_drones[-1].position)
    else:
        last_relay_distance = abs(video_drone.position - operator.position)

    received_power_last = coms.calculate_received_power(TRANSMIT_POWER, last_relay_distance)
    
    if step%100 == 0:
        print(f"# of relays {len(relay_drones)} signal strenght {received_power_last} video drone position {video_drone.position}")



        

