"""
Canonical (NVT) ensemble simulation.

The thermostat is applied after each Velocity Verlet step.
"""

import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from integrator     import velocity_verlet_step
from initialization import kinetic_energy
from observables    import temperature
from lj_potential   import compute_forces_and_potential


def run_nvt(positions:     np.ndarray,
            velocities:    np.ndarray,
            mass_amu:      float,
            epsilon_ev:    float,
            sigma_ang:     float,
            dt_fs:         float,
            n_steps:       int,
            thermostat_fn: Callable,
            save_interval: int = 10) -> dict:
    """
    Run an NVT simulation with a thermostat.
    """
    time_trajectory        = []
    pos_trajectory         = []
    vel_trajectory         = []
    kinetic_trajectory     = []
    potential_trajectory   = []
    temperature_trajectory = []

    # Initial state
    forces, potential = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    kinetic           = kinetic_energy(velocities, mass_amu)
    temp              = temperature(velocities, mass_amu)

    time_trajectory.append(0.0)
    pos_trajectory.append(positions.copy())
    vel_trajectory.append(velocities.copy())
    kinetic_trajectory.append(kinetic)
    potential_trajectory.append(potential)
    temperature_trajectory.append(temp)

    t_start = time.perf_counter()

    with tqdm(total=n_steps, desc="NVT", unit="step") as pbar:
        for step in range(1, n_steps + 1):

            # 1. Velocity Verlet step
            positions, velocities, forces, potential = velocity_verlet_step(
                positions, velocities, forces,
                mass_amu, dt_fs, epsilon_ev, sigma_ang,
            )

            # 2. Thermostat step
            velocities = thermostat_fn(velocities)

            if step % save_interval == 0:
                kinetic = kinetic_energy(velocities, mass_amu)
                temp    = temperature(velocities, mass_amu)

                time_trajectory.append(step * dt_fs)
                pos_trajectory.append(positions.copy())
                vel_trajectory.append(velocities.copy())
                kinetic_trajectory.append(kinetic)
                potential_trajectory.append(potential)
                temperature_trajectory.append(temp)

                pbar.set_postfix(T=f"{temp:.2f} K",
                                 E=f"{potential + kinetic:.4e} eV")

            pbar.update(1)

    elapsed = time.perf_counter() - t_start
    print(f"Elapsed: {elapsed:.2f} s  ({elapsed / n_steps * 1000:.2f} ms/step)")

    kinetic_arr   = np.array(kinetic_trajectory)
    potential_arr = np.array(potential_trajectory)

    return {
        'times':            np.array(time_trajectory),
        'positions':        np.array(pos_trajectory),
        'velocities':       np.array(vel_trajectory),
        'kinetic_energy':   kinetic_arr,
        'potential_energy': potential_arr,
        'total_energy':     kinetic_arr + potential_arr,
        'temperature':      np.array(temperature_trajectory),
    }
