import numpy as np
import random

from md_config import KB_J_K, AMU_TO_KG
from md_energy import kinetic_energy, kinetic_initial

def box_muller():
    """Generate two standard normal random numbers (Box-Muller transform)."""
    rn1 = random.random()
    rn2 = random.random()

    while rn1 == 0:  # Avoid log(0)
        rn1 = random.random()

    rho = np.sqrt(-2.0 * np.log(rn1))
    theta = 2.0 * np.pi * rn2

    return rho * np.cos(theta), rho * np.sin(theta)

def generate_maxwell_velocities(n_atoms, mass_amu, temperature):
    """Generate Maxwell-Boltzmann velocities at target temperature."""
    mass = mass_amu * AMU_TO_KG
    sigma = np.sqrt(temperature * KB_J_K / mass)

    velocities = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(3):
            velocities[i, j] = sigma * box_muller()[0]

    return velocities

def remove_com_velocity(velocities):
    """Remove velocity of center of mass"""
    #Calculate mean along columns (x, y, z components)
    com_velocity = np.mean(velocities, axis=0)
    return velocities - com_velocity

def rescale_velocities(velocities, initial_kinetic):
    kinetic_bar = kinetic_energy(velocities)
    if kinetic_bar == 0: 
        return velocities
    factor = np.sqrt(initial_kinetic / kinetic_bar)
    return velocities * factor

def initialize_velocities(n_atoms, mass_amu, temperature):
    velocites = generate_maxwell_velocities(n_atoms, mass_amu, temperature)
    velocites = remove_com_velocity(velocites)
    initial_kinetic = kinetic_initial(n_atoms, temperature)
    return rescale_velocities(velocites, initial_kinetic)