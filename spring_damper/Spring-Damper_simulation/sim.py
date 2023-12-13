from visualizer import Visualizer # Import Visualizer class
import json

# Opening JSON file
f = open('spring_damper\configs.json')
 
data = json.load(f)


dt = 0.05 # ΔT (sampling period) seconds

# Initial values
position = data['initial_values']['position']
velocity = data['initial_values']['velocity']
acceleration = data['initial_values']['acceleration']

# Constants
mass = data['constants']['mass']
k = data['constants']['k'] # spring coefficient
b = data['constants']['b'] # damping coefficient

sample_prob = data['sample_prob']

# Callback Function
def set(arg):
    global dt, position, velocity, acceleration, mass, k, b # Get global variables

    spring_force = k * position # Fs = k * x
    damper_force = b * velocity # Fb = b * x'

    # If we leave the acceleration alone in equation
    # acceleration = - ((b * velocity) + (k * position)) / mass
    acceleration = - (spring_force + damper_force) / mass
    velocity += (acceleration * dt) # Integral(a) = v
    position += (velocity * dt) # Integral(v) = x

    return (position, 0) # Return position

filename  = 'spring_damper\DATA\data'  + '+' + str(mass) + '+' +  str(k) + '+' + str(b) +  '.csv'
# Start simulation
Visualizer(callback=set, interval=dt * 1000, simulation_time=30, 
           initial=(position, 0, velocity, 0, acceleration, 0),
             sample_prob=sample_prob, filename=filename)
