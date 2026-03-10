import numpy as np

from md_config import AMU_TO_KG, KB_EV_K, mass_amu

def kinetic_energy(velocities):
    """Calculates total kinetic energy."""
    mass = mass_amu * AMU_TO_KG
    # Sum over all atoms and dimensions
    return 0.5 * mass * np.sum(velocities ** 2)

def lennard_jones(epsilon, sigma, distance):
    """Calculates LJ potential for a single pair."""
    if distance == 0:
        return 0.0 # Avoid division by zero
    sigma_dist = sigma / distance
    return 4.0 * epsilon * (sigma_dist ** 12 - sigma_dist ** 6)

def potential_energy(epsilon, sigma, n_atoms, positions=np.zeros(0)):
    """Calculates total potential energy."""
    potential = 0.0
    # Iterate j from i+1 to count each pair once
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Vector from j to i
            rij = positions[i] - positions[j]
            
            # Calculate distance
            r = np.linalg.norm(rij)
            potential += lennard_jones(epsilon, sigma, r)
    
    return potential

def kinetic_initial(n_atoms, initial_temperature):
    """
    Calculates the target kinetic energy for a specific temperature
    using the equipartition theorem for a monatomic gas.
    """
    # Degrees of freedom: 3 dimensions per atom, minus 3 for fixed Center of Mass motion
    dof = 3 * n_atoms - 3
    
    # Target Kinetic Energy
    k_target = 0.5 * dof * KB_EV_K * initial_temperature
    
    return k_target

def forces_jn(epsilon, sigma, n_atoms, positions=np.zeros(0)):
    """Calculates forces on all atoms using Newton's 3rd Law."""
    forces = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        # Loop only over j > i
        for j in range(i + 1, n_atoms):
            # Vector from j to i
            rij = positions[i] - positions[j]
            
            # Calculate distance
            r = np.linalg.norm(rij)
            if r == 0: 
                continue

            # Force magnitude scalar: F = 24 * eps * (2(sigma/r)^12 - (sigma/r)^6) / r
            # F = -dU/dr.
            
            sigma_r = sigma / r
            # Force magnitude derived from derivative of LJ potential
            force_magnitude = 24.0 * epsilon * (2.0 * (sigma_r ** 12) - (sigma_r ** 6)) / r
            
            # Force vector on atom i due to atom j
            # rij points from j to i. Force is repulsive along this direction.
            f_vec = force_magnitude * (rij / r)
            
            # Apply Newton's 3rd Law
            forces[i] += f_vec
            forces[j] -= f_vec # Reaction force on j

    return forces