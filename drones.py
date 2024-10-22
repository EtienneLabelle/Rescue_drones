class Drone:
    def __init__(self, position, battery_level=100):
        self.pos = position
        self.battery_level = battery_level
        self.id = 0 # id 0 is video drone 

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
        self.received_power=0



def deploy_relay(relay_drones):
    new_relay = Drone(position=[10,10]) 
    new_relay.id = len(relay_drones)+1
    relay_drones.append(new_relay)
    #print(f"New Relay Drone deployed Id:{new_relay.id}")