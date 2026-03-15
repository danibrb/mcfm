"""
Thermostat implementations for NVT molecular dynamics.
"""

from typing import Callable

import numpy as np

from initialization import maxwell_boltzmann_velocities


def andersen(velocities:     np.ndarray,
             mass_amu:       float,
             target_temp_k:  float,
             collision_freq: float,
             dt_fs:          float,
             rng:            np.random.Generator) -> np.ndarray:
    """
    Andersen thermostat: stochastic collision with a heat bath.

    Each atom is independently selected for a collision with probability:

        p = collision_freq * dt_fs

    Selected atoms have their velocity resampled from the Maxwell-Boltzmann
    distribution at target_temp_k. Unselected atoms are unchanged.

    Parameters
    ----------
    velocities     : np.ndarray, shape (n_atoms, 3)  [Å/fs]
    mass_amu       : float                           [amu]
    target_temp_k  : float   Target temperature      [K]
    collision_freq : float   Collision frequency     [1/fs]
    dt_fs          : float   Timestep                [fs]
    rng            : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)  [Å/fs]
    """
    n_atoms     = len(velocities)
    p_collision = collision_freq * dt_fs
    collide     = rng.random(n_atoms) < p_collision

    if np.any(collide):
        n_collide                = int(np.sum(collide))
        new_velocities           = velocities.copy()
        new_velocities[collide]  = maxwell_boltzmann_velocities(
            n_collide, mass_amu, target_temp_k, rng
        )
        return new_velocities

    return velocities


def get_thermostat(name:           str,
                   mass_amu:       float,
                   target_temp_k:  float,
                   dt_fs:          float,
                   rng:            np.random.Generator,
                   collision_freq: float = 0.005) -> Callable:
    """
    Return a thermostat callable bound to the given parameters.

    The returned function has the signature:
        velocities = thermostat_fn(velocities)

    and can be passed directly to run_nvt().

    Parameters
    ----------
    name           : str
    mass_amu       : float  [amu]
    target_temp_k  : float  [K]
    dt_fs          : float  [fs]
    rng            : np.random.Generator
    collision_freq : float  [1/fs]  used only by "andersen"

    Raises
    ------
    ValueError if name is not recognised
    """
    if name == "andersen":
        def thermostat_fn(velocities: np.ndarray) -> np.ndarray:
            return andersen(velocities, mass_amu, target_temp_k,
                            collision_freq, dt_fs, rng)
        return thermostat_fn

    # --- add new thermostats here ---
    # if name == "nose_hoover":
    #     def thermostat_fn(velocities):
    #         return nose_hoover(velocities, ...)
    #     return thermostat_fn

    available = ["andersen"]
    raise ValueError(f"Unknown thermostat '{name}'. Available: {available}")