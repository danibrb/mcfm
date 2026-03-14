"""
Microcanonical (NVE) ensemble simulation.

At each step the Velocity Verlet algorithm advances positions and velocities,
conserving the total energy E = K + U to O(dt^2) per step.
"""

import time

import numpy as np
from tqdm import tqdm

from integrator     import velocity_verlet_step
from initialization import kinetic_energy
from observables    import temperature
from lj_potential   import compute_forces_and_potential


def run_nve(positions:     np.ndarray,
            velocities:    np.ndarray,
            mass_amu:      float,
            epsilon_ev:    float,
            sigma_ang:     float,
            dt_fs:         float,
            n_steps:       int,
            save_interval: int = 10) -> dict:
    """
    Run an NVE simulation and return trajectory data.
    """
    times           = []
    pos_trajectory  = []
    vel_trajectory  = []
    kin_trajectory  = []
    pot_trajectory  = []
    temp_trajectory = []

    # Initial state
    forces, U = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    K = kinetic_energy(velocities, mass_amu)
    T = temperature(velocities, mass_amu)

    times.append(0.0)
    pos_trajectory.append(positions.copy())
    vel_trajectory.append(velocities.copy())
    kin_trajectory.append(K)
    pot_trajectory.append(U)
    temp_trajectory.append(T)

    # Main MD loop
    t_start = time.perf_counter()

    with tqdm(total=n_steps, desc="NVE", unit="step") as pbar:
        for step in range(1, n_steps + 1):
            positions, velocities, forces, U = velocity_verlet_step(
                positions, velocities, forces,
                mass_amu, dt_fs, epsilon_ev, sigma_ang,
            )

            if step % save_interval == 0:
                K = kinetic_energy(velocities, mass_amu)
                T = temperature(velocities, mass_amu)

                times.append(step * dt_fs)
                pos_trajectory.append(positions.copy())
                vel_trajectory.append(velocities.copy())
                kin_trajectory.append(K)
                pot_trajectory.append(U)
                temp_trajectory.append(T)

                #pbar.set_postfix(T=f"{T:.2f} K", E=f"{U + K:.4e} eV")

            pbar.update(1)

    elapsed = time.perf_counter() - t_start
    print(f"Elapsed: {elapsed:.2f} s  ({elapsed / n_steps * 1000:.2f} ms/step)")

    kin_arr = np.array(kin_trajectory)
    pot_arr = np.array(pot_trajectory)

    return {
        'times':            np.array(times),
        'positions':        np.array(pos_trajectory),
        'velocities':       np.array(vel_trajectory),
        'kinetic_energy':   kin_arr,
        'potential_energy': pot_arr,
        'total_energy':     kin_arr + pot_arr,
        'temperature':      np.array(temp_trajectory),
    }