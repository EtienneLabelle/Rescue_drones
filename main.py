from coms import calculate_sinr_with_fading
from drones import Drone
from utils import calculate_distance
from env import Simulation
from display import SimpleAnimator

FREQUENCY = 2.4e9  # 2.4 GHz
BANDWIDTH = 20e6   # 20 MHz
TRANSMIT_POWER = 30  # Transmit power in dBm (decibel milliwatts)
MIN_RECEIVE_POWER = -80  # Minimum power in dBm for a stable link
NOISE_POWER = -90


sim = Simulation(BANDWIDTH, FREQUENCY, NOISE_POWER)
sim.create_obstacles(50,1000)
animator = SimpleAnimator(interval=100)
animator.set_obstacles(sim.obstacles)  # Set static obstacle edges once

operator = Drone(id = "operator", position=[0,0])  # Operator starts at position (0, 0)
sim.drones.append(operator) #this works because deploy drone goes insert -1
sim.deploy_relay(id= "video drone") # this is video drone
    
# Simulation loop
for step in range(1, 2000):  

    for i, drone in enumerate(sim.drones[:-1]):
        drone.move([10,10])
        
    distance = calculate_distance(sim.drones[-1].pos,sim.drones[0].pos if len(sim.drones) == 2 else sim.drones[-2].pos)

    if distance> 1000:
        print("step #",step)
        sim.deploy_relay(id = "relay drone" + str((len(sim.drones)-1)))
        sim.update_links()

    animator.record_positions(sim.drones, step)

animator.animate()




