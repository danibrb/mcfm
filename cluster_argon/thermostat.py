"""
Thermostat implementation for NVT simulations.
"""


import numpy as np

from initialization import maxwell_boltzmann_velocities


def andersen(velocities:        np.ndarray,
             mass_amu:          float,
             target_temp_k:     float,
             collision_freq:    float,
             dt_fs:             float,
             rng:               np.random.Generator) -> np.ndarray:
    
    """
    Andersen thermostat 
    """

    n_atoms = velocities.shape[0]
    prob_collision = collision_freq * dt_fs

    # collision mask for n atoms
    collide = rng.random(n_atoms) < prob_collision

    # generates new velocities at target temperature
    new_velocities = maxwell_boltzmann_velocities(n_atoms, mass_amu, target_temp_k, rng)
    # applying mask to assign new velocities to selected particles 
    result = velocities.copy()
    result[collide] = new_velocities[collide]     

    return result


