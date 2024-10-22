class Drone:
    def __init__(self, position, battery_level=100):
        self.pos = position
        self.battery_level = battery_level

    def move(self, step):
        # Update the position with a step in both x and y directions
        self.pos[0] += step[0]
        self.pos[1] += step[1]

    def get_state(self):
        # Return the position as a tuple along with the battery level
        return (tuple(self.pos), self.battery_level)




class Operator:
    def __init__(self, position=[0,0]):
        self.pos = position  # Operator is stationary



def deploy_relay(video_drone, relay_drones):
    """
    Deploy a new relay drone, positioning it between the video drone and the last relay/operator.
    """
    last_point = [0,0] if not relay_drones else relay_drones[-1].pos
    # Calculate the new relay position as the midpoint between the video drone and the last point
    relay_position = [(video_drone.pos[0] + last_point[0]) / 2, 
                      (video_drone.pos[1] + last_point[1]) / 2]

    new_relay = Drone(position=relay_position)  # Initialize the new relay with a position
    relay_drones.append(new_relay)
    print(f"New Relay Drone deployed at position {relay_position}")