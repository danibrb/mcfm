import numpy as np

from md_util import distance_two_points

from md_config import AMU_TO_KG, KB_EV_K, mass_amu

def kinetic_enegy(velocities):
    mass = mass_amu * AMU_TO_KG
    return (mass * velocities **2).sum()/2 #!!

def lennard_jones(epsilon, sigma, distance):
    sigma_dist = sigma / distance
    return 4 * epsilon * (sigma_dist ** 12 - sigma_dist ** 6)

def potential_energy(epsilon, sigma, n_atoms, positions=np.zeros(0)):

    potential = 0.0
    for i in range(0, n_atoms):
        for j in range(0, n_atoms):
            if i == j :
                continue
            distance = distance_two_points(positions[i], positions[j])
            potential += lennard_jones(epsilon, sigma, distance)
    
    return potential / 2.0

def kinetic_intial(initial_temperature, initial_potential):
    energy = initial_temperature * KB_EV_K
    return energy - initial_potential
