"""
Thermostat implementations for NVT simulations.

Two Andersen variants are provided:

andersen()     — pure Python, uses numpy.random.Generator.
                 Correct canonical ensemble; used by nvt.py and
                 analysis_andersen.py where reproducibility with a
                 numpy seed is required.

andersen_jit() — @njit, uses Numba's internal Mersenne Twister RNG.
                 Functionally identical physics; eliminates Python function
                 call overhead and numpy Generator dispatch overhead
                 (~10-20 µs/step for N=38).  Used by heating_ramp.py.
                 Seed via seed_numba_rng(seed) before the simulation loop.

Reference: Andersen, J. Chem. Phys. 72, 2384 (1980).
"""

import numpy as np
from numba import njit

from constants     import KB_EV, AMU_ANG2_FS2_TO_EV
from initialization import maxwell_boltzmann_velocities


# Python (numpy Generator) version — used by nvt.py / analysis_andersen.py

def andersen(velocities:        np.ndarray,
             mass_amu:          float,
             target_temp_k:     float,
             collision_freq:    float,
             dt_fs:             float,
             rng:               np.random.Generator) -> np.ndarray:
    """
    Andersen thermostat using numpy.random.Generator.
    """
    n_atoms        = velocities.shape[0]
    prob_collision = collision_freq * dt_fs
    collide        = rng.random(n_atoms) < prob_collision
    new_velocities = maxwell_boltzmann_velocities(n_atoms, mass_amu,
                                                   target_temp_k, rng)
    result          = velocities.copy()
    result[collide] = new_velocities[collide]
    return result


# Numba JIT version — used by heating_ramp.py

@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """
    Seed Numba's internal Mersenne Twister RNG.

    Must be called once before the simulation loop.  Numba's RNG is
    independent of numpy's Generator; the same integer seed will produce
    a different but fully reproducible sequence.
    """
    np.random.seed(seed)


@njit(cache=True)
def andersen_jit(velocities:    np.ndarray,
                 mass_amu:      float,
                 target_temp_k: float,
                 collision_freq: float,
                 dt_fs:         float) -> np.ndarray:
    """
    Andersen thermostat compiled by Numba.

    Uses Numba's internal np.random (MT19937) — call seed_numba_rng(seed)
    once before the loop.  Physics identical to andersen(); avoids all
    Python/numpy dispatch overhead.

    Velocity components are drawn one at a time (scalar np.random.normal)
    which is the form supported across all Numba versions.
    """
    n_atoms  = velocities.shape[0]
    prob     = collision_freq * dt_fs
    sigma_v  = np.sqrt(KB_EV * target_temp_k / (mass_amu * AMU_ANG2_FS2_TO_EV))
    result   = velocities.copy()
    for i in range(n_atoms):
        if np.random.random() < prob:
            for k in range(3):
                result[i, k] = np.random.normal(0.0, sigma_v)
    return result