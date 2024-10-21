class VideoDrone:
    def __init__(self, start_position):
        self.position = start_position  # Starting position of the drone
    
    def move(self, step):
        self.position += step  # Moves the drone by the given step size


class RelayDrone:
    def __init__(self, position=None):
        self.position = position  # Relay drone's position will be set dynamically

    def set_position(self, position):
        self.position = position  # Set the relay drone's position

    def reset_position(self):
        self.position = None  # Reset when not needed
    
    def move(self, step):
        self.position += step  # Moves the drone by the given step size



class Operator:
    def __init__(self, position):
        self.position = position  # Operator is stationary



def deploy_relay(existing_relays):
    """
    Deploy relay drones that follow the video drone at the maximum allowable distance.
    The first relay will follow the video drone.
    Subsequent relays will start from the operator's position plus 10 meters.
    """    
    # Deploy the new relay
    new_relay = RelayDrone()
    existing_relays.append(new_relay) # appemd means new drone is relay_drones[-1]
    print(f"New Relay Drone deployed at position {new_relay.position:.2f} meters, following the last relay")