class Drone:
    def __init__(self, position, battery_level=100):
        self.pos = position
        self.battery_level = battery_level

    def move(self, step):
        # Update the position with a step in both x and y directions
        self.pos[0] += step[0]
        self.pos[1] += step[1]

class Operator:
    def __init__(self, position=[0,0]):
        self.pos = position  # Operator is stationary
        self.received_power=0



