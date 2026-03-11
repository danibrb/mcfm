"""
Microcanonical (NVE) ensemble simulation,
conserving the total energy E = K + U per step.
"""

import numpy as np

from integrator    import velocity_verlet_step
from initialization import kinetic_energy
from observables   import temperature
from lj_potential  import compute_forces_and_potential


def run_nve(positions:   np.ndarray,
            velocities:  np.ndarray,
            mass_amu:    float,
            epsilon_ev:  float,
            sigma_ang:   float,
            dt_fs:       float,
            n_steps:     int,
            save_interval: int = 10) -> dict:
    """
    Run an NVE simulation and return saved data.
    """
    times     = []
    pos_saved  = []
    vel_saved  = []
    kin_saved  = []
    pot_saved  = []
    temp_saved = []

    # Initial state
    forces, U = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    K = kinetic_energy(velocities, mass_amu)
    T = temperature(velocities, mass_amu)

    times.append(0.0)
    pos_saved.append(positions.copy())
    vel_saved.append(velocities.copy())
    kin_saved.append(K)
    pot_saved.append(U)
    temp_saved.append(T)

    # Main MD loop
    for step in range(1, n_steps + 1):
        positions, velocities, forces, U = velocity_verlet_step(
            positions, velocities, forces,
            mass_amu, dt_fs, epsilon_ev, sigma_ang,
        )

        if step % save_interval == 0:
            K = kinetic_energy(velocities, mass_amu)
            T = temperature(velocities, mass_amu)

            times.append(step * dt_fs)
            pos_saved.append(positions.copy())
            vel_saved.append(velocities.copy())
            kin_saved.append(K)
            pot_saved.append(U)
            temp_saved.append(T)

    kin_arr = np.array(kin_saved)
    pot_arr = np.array(pot_saved)

    return {
        'times':            np.array(times),
        'positions':        np.array(pos_saved),
        'velocities':       np.array(vel_saved),
        'kinetic_energy':   kin_arr,
        'potential_energy': pot_arr,
        'total_energy':     kin_arr + pot_arr,
        'temperature':      np.array(temp_saved),
    }