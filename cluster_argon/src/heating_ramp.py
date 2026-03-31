"""
Heating-ramp NVT simulation using the Andersen thermostat.

The target temperature is increased linearly from temp_start_k to temp_end_k
over n_steps integration steps:

    T_target(i) = T_start + (T_end - T_start) * i / n_steps

Performance optimisations relative to the baseline implementation:

1. fastmath=True on the LJ force kernel (~10-20% via FMA / SVML).
2. Fused Velocity Verlet step (velocity_verlet_step_jit): the position and
   velocity arithmetic is compiled by Numba in the same JIT context as the
   force kernel, allowing inlining and eliminating 5 NumPy ufunc dispatches
   and their associated intermediate array allocations per step.
3. JIT Andersen thermostat (andersen_jit): replaces numpy.Generator calls
   (~10-20 µs/step) with Numba's internal MT RNG at near-zero overhead.
   force_conv_over_mass is precomputed once before the loop.
4. tqdm is updated every save_interval steps instead of every step,
   reducing progress-bar overhead from O(n_steps) to O(n_steps/save_interval).
"""

import time

import numpy as np
from tqdm import tqdm

from constants    import FORCE_CONV
from integrator   import velocity_verlet_step_jit
from initialization import kinetic_energy
from observables  import temperature
from lj_potential import compute_forces_and_potential
from thermostat   import andersen_jit, seed_numba_rng


def run_heating_ramp(positions:      np.ndarray,
                     velocities:     np.ndarray,
                     mass_amu:       float,
                     epsilon_ev:     float,
                     sigma_ang:      float,
                     dt_fs:          float,
                     n_steps:        int,
                     temp_start_k:   float,
                     temp_end_k:     float,
                     collision_freq: float,
                     random_seed:    int,
                     save_interval:  int = 500) -> dict:
    """
    Run a linearly ramped NVT simulation (Andersen thermostat).

    Parameters
    ----------
    collision_freq : Andersen collision frequency  [1/fs]
    random_seed    : seed for Numba's internal RNG (replaces numpy Generator)
    """

    target_temps = np.linspace(temp_start_k, temp_end_k, n_steps + 1)

    total_time_fs = n_steps * dt_fs
    heating_rate  = (temp_end_k - temp_start_k) / (total_time_fs)   # K/ns
    print(f"  Heating rate: {heating_rate:.4f} K/ns  "
          f"({temp_start_k:.1f} -> {temp_end_k:.1f} K  "
          f"over {total_time_fs:.1f} ns)")

    # Precompute scalar used at every step to avoid repeated division
    force_conv_over_mass = FORCE_CONV / mass_amu

    # Seed Numba's internal RNG once before the loop
    seed_numba_rng(random_seed)

    times           = []
    pos_trajectory  = []
    vel_trajectory  = []
    kin_trajectory  = []
    pot_trajectory  = []
    temp_trajectory = []
    tgt_trajectory  = []

    # Initial state (step 0)
    forces, U = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    K = kinetic_energy(velocities, mass_amu)
    T = temperature(velocities, mass_amu)

    times.append(0.0)
    pos_trajectory.append(positions.copy())
    vel_trajectory.append(velocities.copy())
    kin_trajectory.append(K)
    pot_trajectory.append(U)
    temp_trajectory.append(T)
    tgt_trajectory.append(target_temps[0])

    # Main MD loop
    t_start = time.perf_counter()

    with tqdm(total=n_steps, desc="Ramp", unit="step") as pbar:
        for step in range(1, n_steps + 1):

            T_target = target_temps[step]

            # 1. Fused Velocity Verlet step (JIT, force kernel inlined)
            positions, velocities, forces, U = velocity_verlet_step_jit(
                positions, velocities, forces,
                force_conv_over_mass, dt_fs,
                epsilon_ev, sigma_ang,
            )

            # 2. Andersen thermostat (JIT, Numba internal RNG)
            velocities = andersen_jit(
                velocities, mass_amu, T_target,
                collision_freq, dt_fs,
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
                tgt_trajectory.append(T_target)

                pbar.set_postfix(T_inst=f"{T:.1f} K",
                                 T_tgt=f"{T_target:.1f} K",
                                 U=f"{U:.4e} eV")
                # Update tqdm once per batch instead of once per step
                pbar.update(save_interval)

    elapsed = time.perf_counter() - t_start
    print(f"Elapsed: {elapsed:.2f} s  ({elapsed / n_steps * 1e3:.3f} ms/step)")

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
        'target_temp':      np.array(tgt_trajectory),
        'heating_rate_kps': heating_rate,
    }