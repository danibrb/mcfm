"""
Heating-ramp NVT simulation of the Ar_38 Lennard-Jones cluster.

"""

import os

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ, OUTPUT_DIR_HR,
                             MASS_AMU, TEMP_HR_START_K, TEMP_HR_END_K,
                             TIMESTEP_FS, N_STEPS_HR, SAVE_INTERVAL_HR,
                             COLLISION_FREQ, RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params, save_trajectory_with_metadata
from initialization import initialize_velocities
from lj_potential   import warmup_jit
from heating_ramp   import run_heating_ramp
from visualization  import plot_ramp_all, make_label


def main() -> None:

    # 1. Read structure and Lennard-Jones parameters
    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ_IN)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"LJ parameters: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} Å")

    # 2. Warm up Numba JIT before the timed loop
    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 3. Initialise velocities at the starting temperature
    rng = np.random.default_rng(RANDOM_SEED)
    velocities = initialize_velocities(n_atoms, MASS_AMU, TEMP_HR_START_K, rng)

    # 4. Run the heating ramp
    total_time_ps = N_STEPS_HR * TIMESTEP_FS * 1e-3
    print("\nRunning heating ramp:")
    print(f"  From:  {TEMP_HR_START_K:.1f} K  -> To:  {TEMP_HR_END_K:.1f} K")
    print(f"  Total simulation time:  {N_STEPS_HR} steps  x  {TIMESTEP_FS} fs  =  {total_time_ps:.1f} ps")
    print(f"  Andersen collision frequency: {COLLISION_FREQ:.4e} fs^-1")
    print(f"  Save interval: every {SAVE_INTERVAL_HR} steps "
          f"({N_STEPS_HR // SAVE_INTERVAL_HR} frames)")

    traj = run_heating_ramp(
        positions.copy(), velocities,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS_HR,
        temp_start_k=TEMP_HR_START_K,
        temp_end_k=TEMP_HR_END_K,
        collision_freq=COLLISION_FREQ,
        rng=rng,
        save_interval=SAVE_INTERVAL_HR,
    )

    # 5. Create output directory and save results
    os.makedirs(OUTPUT_DIR_HR, exist_ok=True)

    label = make_label("ramp",
                       temp_start=TEMP_HR_START_K,
                       temp_end=TEMP_HR_END_K,
                       n_steps=N_STEPS_HR)

    # # XYZ trajectory for VMD
    # xyz_file = f"trajectory_{label}.xyz"
    # write_xyz_trajectory(
    #     os.path.join(OUTPUT_DIR_RAMP, xyz_file),
    #     traj['positions'], atom_names, traj['times'],
    # )

    #  trajectory for VMD
    pdb_file = f"trajectory_{label}.pdb"
    dcd_file = f"trajectory_{label}.dcd"
    save_trajectory_with_metadata(
        os.path.join(OUTPUT_DIR_HR, dcd_file),
        os.path.join(OUTPUT_DIR_HR, pdb_file),
        traj['positions'], atom_names, traj['times'],
    )

    # Diagnostic and caloric-curve plots
    plot_ramp_all(traj, label=label, save_dir=OUTPUT_DIR_HR)

    print(f"\nAll output saved to: {OUTPUT_DIR_HR}/")


if __name__ == "__main__":
    main()
