"""
NVE and NVT comparison simulation for the Ar_38 Lennard-Jones cluster.

Unit system:  Å / eV / amu / fs
"""

import os

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ,
                             OUTPUT_DIR_NVE, OUTPUT_DIR_NVT,
                             MASS_AMU, TEMP_INIT_K, TIMESTEP_FS,
                             N_STEPS, SAVE_INTERVAL, RANDOM_SEED,
                             COLLISION_FREQ)
from io_handler     import (read_xyz, read_lj_params,
                             write_xyz_trajectory,
                             save_trajectory_with_metadata)
from initialization import initialize_velocities
from lj_potential   import warmup_jit
from nve            import run_nve
from nvt            import run_nvt
from visualization  import plot_all, make_label


def main() -> None:

    # 1. Read structure and LJ parameters
    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ_IN)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"LJ: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} A")

    # 2. Warm up Numba JIT
    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 3. Initialise velocities (same starting state for fair comparison)
    rng_nve        = np.random.default_rng(RANDOM_SEED)
    rng_nvt        = np.random.default_rng(RANDOM_SEED + 1)
    velocities_nve = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_nve)
    velocities_nvt = velocities_nve.copy()

    # 4. Create output directories before any save attempt
    for d in (OUTPUT_DIR_NVE, OUTPUT_DIR_NVT):
        os.makedirs(d, exist_ok=True)

    # 5. NVE run
    print(f"\nNVE: {N_STEPS} steps,  dt = {TIMESTEP_FS} fs")
    traj_nve = run_nve(
        positions.copy(), velocities_nve,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS,
        save_interval=SAVE_INTERVAL,
    )

    # 6. NVT run
    print(f"\nNVT (Andersen): {N_STEPS} steps,  dt = {TIMESTEP_FS} fs,"
          f"  T = {TEMP_INIT_K} K,  nu = {COLLISION_FREQ:.4e} fs^-1")
    traj_nvt = run_nvt(
        positions.copy(), velocities_nvt,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS,
        target_temp_k=TEMP_INIT_K,
        collision_freq=COLLISION_FREQ,
        rng=rng_nvt,
        save_interval=SAVE_INTERVAL,
    )

    # 7. Save trajectories
    base_nve = f"NVE_T{TEMP_INIT_K:.0f}K_{N_STEPS // 1000:.0f}ksteps"
    base_nvt = f"NVT_T{TEMP_INIT_K:.0f}K_{N_STEPS // 1000:.0f}ksteps"

    # XYZ (human-readable)
    write_xyz_trajectory(
        os.path.join(OUTPUT_DIR_NVE, base_nve + ".xyz"),
        traj_nve['positions'], atom_names, traj_nve['times'])
    write_xyz_trajectory(
        os.path.join(OUTPUT_DIR_NVT, base_nvt + ".xyz"),
        traj_nvt['positions'], atom_names, traj_nvt['times'])

    # DCD + PDB (compact binary)
    save_trajectory_with_metadata(
        dcd_filename = os.path.join(OUTPUT_DIR_NVE, base_nve + ".dcd"),
        pdb_filename = os.path.join(OUTPUT_DIR_NVE, base_nve + ".pdb"),
        positions    = traj_nve['positions'],
        atom_names   = atom_names,
        times        = traj_nve['times'],
    )
    save_trajectory_with_metadata(
        dcd_filename = os.path.join(OUTPUT_DIR_NVT, base_nvt + ".dcd"),
        pdb_filename = os.path.join(OUTPUT_DIR_NVT, base_nvt + ".pdb"),
        positions    = traj_nvt['positions'],
        atom_names   = atom_names,
        times        = traj_nvt['times'],
    )

    # 8. Plots
    label_nve = make_label("nve", TEMP_INIT_K, n_steps=N_STEPS)
    label_nvt = make_label("nvt", TEMP_INIT_K, n_steps=N_STEPS)

    plot_all(traj_nve, label=label_nve, save_dir=OUTPUT_DIR_NVE)
    plot_all(traj_nvt, label=label_nvt, target_temp_k=TEMP_INIT_K,
             save_dir=OUTPUT_DIR_NVT)

    print(f"\nNVE output: {OUTPUT_DIR_NVE}/")
    print(f"NVT output: {OUTPUT_DIR_NVT}/")


if __name__ == "__main__":
    main()