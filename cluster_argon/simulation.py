"""
Main driver for the Lennard-Jones MD simulation of an Argon cluster.

Runs both NVE and NVT ensembles and saves results to separate output folders.

Unit system:
    Length   -> Angstrom (Å)
    Energy   -> electronvolt (eV)
    Mass     -> atomic mass unit (amu)
    Time     -> femtosecond (fs)
    Force    -> eV/Å
    Velocity -> Å/fs
"""

import os

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ,
                             OUTPUT_DIR_NVE, OUTPUT_DIR_NVT,
                             MASS_AMU, TEMP_INIT_K, TIMESTEP_FS,
                             N_STEPS, SAVE_INTERVAL, RANDOM_SEED,
                             THERMOSTAT, COLLISION_FREQ)
from io_handler     import read_xyz, read_lj_params, write_xyz_trajectory
from initialization import initialize_velocities
from lj_potential   import warmup_jit
from nve            import run_nve
from nvt            import run_nvt
from thermostat     import get_thermostat
from visualization  import plot_all


def make_run_label(ensemble: str, temp_k: float, n_steps: int) -> str:
    """Build a short label used in plot titles and filenames."""
    return f"{ensemble.upper()}  T={temp_k:.0f}K  {n_steps / 1000:.0f}k steps"


def main() -> None:

    # 1. Read structure and Lennard-Jones parameters
    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ_IN)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"LJ parameters: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} Å")

    # 2. Warm up Numba JIT
    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 3. Independent RNG streams
    rng_nve = np.random.default_rng(RANDOM_SEED)
    rng_nvt = np.random.default_rng(RANDOM_SEED + 1)

    # 4. Identical starting velocities for fair comparison
    velocities_nve = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_nve)
    velocities_nvt = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_nvt)

    # 5. Run NVE
    print(f"\nRunning NVE: {N_STEPS} steps,  dt = {TIMESTEP_FS} fs")
    traj_nve = run_nve(
        positions.copy(), velocities_nve,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS,
        save_interval=SAVE_INTERVAL,
    )

    # 6. Run NVT
    thermostat_fn = get_thermostat(
        THERMOSTAT, MASS_AMU, TEMP_INIT_K,
        TIMESTEP_FS, rng_nvt,
        collision_freq=COLLISION_FREQ,
    )
    print(f"\nRunning NVT ({THERMOSTAT}): {N_STEPS} steps,  dt = {TIMESTEP_FS} fs,"
          f"  T_target = {TEMP_INIT_K} K")
    traj_nvt = run_nvt(
        positions.copy(), velocities_nvt,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS,
        thermostat_fn=thermostat_fn,
        save_interval=SAVE_INTERVAL,
    )

    # 7. Create output directories
    os.makedirs(OUTPUT_DIR_NVE, exist_ok=True)
    os.makedirs(OUTPUT_DIR_NVT, exist_ok=True)

    # 8. Save VMD trajectories
    xyz_name = f"trajectory_T{TEMP_INIT_K:.0f}K_{N_STEPS / 1000:.0f}k_steps.xyz"
    write_xyz_trajectory(os.path.join(OUTPUT_DIR_NVE, xyz_name),
                         traj_nve['positions'], atom_names, traj_nve['times'])
    write_xyz_trajectory(os.path.join(OUTPUT_DIR_NVT, xyz_name),
                         traj_nvt['positions'], atom_names, traj_nvt['times'])

    # 9. Generate plots
    plot_all(traj_nve,
             label=make_run_label("nve", TEMP_INIT_K, N_STEPS),
             save_dir=OUTPUT_DIR_NVE)

    plot_all(traj_nvt,
             label=make_run_label("nvt", TEMP_INIT_K, N_STEPS),
             target_temp=TEMP_INIT_K,
             save_dir=OUTPUT_DIR_NVT)


if __name__ == "__main__":
    main()