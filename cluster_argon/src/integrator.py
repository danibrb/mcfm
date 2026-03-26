"""
Velocity Verlet integrator for classical molecular dynamics.
"""

import numpy as np

from constants    import FORCE_CONV
from lj_potential import compute_forces_and_potential


def velocity_verlet_step(positions:  np.ndarray,
                         velocities: np.ndarray,
                         forces:     np.ndarray,
                         mass_amu:   float,
                         dt_fs:      float,
                         epsilon_ev: float,
                         sigma_ang:  float):
    """
    Advance the system by one Velocity Verlet step.
    """
    # Convert forces to accelerations:  a [Å/fs²] = F [eV/Å] / m [amu] * FORCE_CONV
    acc = forces * (FORCE_CONV / mass_amu)                           

    # Step 1: Update positions
    # r(t+dt) = r(t) + v(t)·dt + ½·a(t)·dt²
    new_positions = positions + velocities * dt_fs + 0.5 * acc * dt_fs**2

    # Step 2: Evaluate forces and potential at updated positions
    new_forces, potential = compute_forces_and_potential(epsilon_ev, sigma_ang, new_positions)
    new_acc    = new_forces * (FORCE_CONV / mass_amu)                 

    # Step 3: Update velocities using the averaged acceleration
    # v(t+dt) = v(t) + ½·[a(t) + a(t+dt)]·dt
    new_velocities = velocities + 0.5 * (acc + new_acc) * dt_fs

    return new_positions, new_velocities, new_forces, potential
