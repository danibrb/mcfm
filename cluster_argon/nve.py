"""
Microcanonical (NVE) ensemble simulation.

At each step the Velocity Verlet algorithm advances positions and velocities,
conserving the total energy E = K + U to O(dt^2) per step (Swope et al., 1982).
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

    Parameters
    ----------
    positions     : np.ndarray, shape (n_atoms, 3)  [Å]
    velocities    : np.ndarray, shape (n_atoms, 3)  [Å/fs]
    mass_amu      : float                           [amu]
    epsilon_ev    : float   LJ well depth           [eV]
    sigma_ang     : float   LJ collision diameter   [Å]
    dt_fs         : float   Timestep                [fs]
    n_steps       : int     Number of MD steps
    save_interval : int     Save state every this many steps

    Returns
    -------
    dict with keys:
        times            : np.ndarray  [fs]
        positions        : np.ndarray, shape (n_saved, n_atoms, 3)  [Å]
        velocities       : np.ndarray, shape (n_saved, n_atoms, 3)  [Å/fs]
        kinetic_energy   : np.ndarray  [eV]
        potential_energy : np.ndarray  [eV]
        total_energy     : np.ndarray  [eV]
        temperature      : np.ndarray  [K]
    """
    times     = []
    pos_traj  = []
    vel_traj  = []
    kin_traj  = []
    pot_traj  = []
    temp_traj = []

    # Initial state
    forces, U = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    K = kinetic_energy(velocities, mass_amu)
    T = temperature(velocities, mass_amu)

    times.append(0.0)
    pos_traj.append(positions.copy())
    vel_traj.append(velocities.copy())
    kin_traj.append(K)
    pot_traj.append(U)
    temp_traj.append(T)

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
                pos_traj.append(positions.copy())
                vel_traj.append(velocities.copy())
                kin_traj.append(K)
                pot_traj.append(U)
                temp_traj.append(T)

                pbar.set_postfix(T=f"{T:.2f} K", E=f"{U + K:.4e} eV")

            pbar.update(1)

    elapsed = time.perf_counter() - t_start
    print(f"Elapsed: {elapsed:.2f} s  ({elapsed / n_steps * 1000:.2f} ms/step)")

    kin_arr = np.array(kin_traj)
    pot_arr = np.array(pot_traj)

    return {
        'times':            np.array(times),
        'positions':        np.array(pos_traj),
        'velocities':       np.array(vel_traj),
        'kinetic_energy':   kin_arr,
        'potential_energy': pot_arr,
        'total_energy':     kin_arr + pot_arr,
        'temperature':      np.array(temp_traj),
    }