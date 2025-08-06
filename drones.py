class Drone:
    def __init__(self, id, position, battery_level=100):
        self.id = id
        self.pos = position
        self.last_pos = position
        self.battery_level = battery_level
        self.prev_distance_to_zone = None  # Track previous distance to nearest zone

    def move(self, step):
        # Update the position with a step in both x and y directions
        self.pos[0] += step[0]
        self.pos[1] += step[1]
        

class Operator:
    def __init__(self, position=[0,0]):
        self.pos = position  # Operator is stationary
        self.received_power=0



