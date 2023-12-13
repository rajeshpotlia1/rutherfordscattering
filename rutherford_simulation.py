import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import time

# Define physical constants
epsilon_0 = const.eps0.value  # Vacuum permittivity
q_alpha = 2 * const.e.si.value  # Charge of alpha particle
m_alpha = 4 * const.m_p.si.value  # Mass of alpha particle

# Set up simulation parameters
num_particles = 10  # Number of alpha particles
x_range = 1e-14  # Symmetrical x-range around gold nuclei
y_range = 5e-11  # Y-axis range
dt = 1e-24  # Time step

# Initialize particle positions and velocities
x_particles = np.random.uniform(-x_range, x_range, num_particles)
y_particles = np.random.uniform(-y_range, y_range, num_particles)
v_x_particles = np.zeros(num_particles)
v_y_particles = np.full(num_particles, 1e7)  # Initial y-velocity: 10^7 m/s

# Initialize gold nuclei positions (assumed to be at the origin)
x_gold_nuclei = 0.0
y_gold_nuclei = 0.0

# Create a figure for visualization
fig, ax = plt.subplots()

# Measure the start time
start_time = time.time()

# Main simulation loop
while np.any(y_particles >= -y_range):

    # Calculate distances between particles and gold nuclei
    r = np.sqrt((x_particles - x_gold_nuclei)**2 + (y_particles - y_gold_nuclei)**2)

    # Calculate forces using Coulomb's law
    F = (q_alpha**2) / (4 * np.pi * epsilon_0 * r**2)

    # Calculate accelerations using Newton's second law
    a_x_particles = -F * (x_particles - x_gold_nuclei) / r / m_alpha
    a_y_particles = -F * (y_particles - y_gold_nuclei) / r / m_alpha

    # Update positions and velocities
    x_particles += v_x_particles * dt
    y_particles += v_y_particles * dt
    v_x_particles += a_x_particles * dt
    v_y_particles += a_y_particles * dt

    # Update the plot for visualization
    ax.clear()
    ax.scatter(x_particles, y_particles)
    
    # Add a red dot for the nucleus
    ax.plot(x_gold_nuclei, y_gold_nuclei, 'ro', markersize=10)
    
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-y_range, y_range)
    plt.pause(0.1)

    # Check if 10 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= 10:
        break  # Exit the loop after 10 seconds

print("Simulation completed after 10 seconds.")