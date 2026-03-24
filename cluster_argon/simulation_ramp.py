"""
Heating-ramp NVT simulation of the Ar_38 Lennard-Jones cluster.

The system is heated from TEMP_RAMP_START_K to TEMP_RAMP_END_K using the
Andersen thermostat with a linearly increasing target temperature.  The
caloric curves E_tot(T) and E_pot(T) allow identification of the
solid-to-liquid phase transition temperature.

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
from config         import (FILENAME_XYZ_IN, FILENAME_LJ, OUTPUT_DIR_RAMP,
                             MASS_AMU, TEMP_RAMP_START_K, TEMP_RAMP_END_K,
                             TIMESTEP_FS, N_STEPS_RAMP, SAVE_INTERVAL_RAMP,
                             COLLISION_FREQ, RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params, write_xyz_trajectory, save_trajectory_with_metadata
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
    velocities = initialize_velocities(n_atoms, MASS_AMU, TEMP_RAMP_START_K, rng)

    # 4. Run the heating ramp
    total_time_ps = N_STEPS_RAMP * TIMESTEP_FS * 1e-3
    print(f"\nRunning heating ramp:")
    print(f"  {TEMP_RAMP_START_K:.1f} K  →  {TEMP_RAMP_END_K:.1f} K")
    print(f"  {N_STEPS_RAMP} steps  ×  {TIMESTEP_FS} fs  =  {total_time_ps:.1f} ps")
    print(f"  Andersen collision frequency: {COLLISION_FREQ:.4e} fs⁻¹")
    print(f"  Save interval: every {SAVE_INTERVAL_RAMP} steps "
          f"({N_STEPS_RAMP // SAVE_INTERVAL_RAMP} frames)")

    traj = run_heating_ramp(
        positions.copy(), velocities,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS_RAMP,
        temp_start_k=TEMP_RAMP_START_K,
        temp_end_k=TEMP_RAMP_END_K,
        collision_freq=COLLISION_FREQ,
        rng=rng,
        save_interval=SAVE_INTERVAL_RAMP,
    )

    # 5. Create output directory and save results
    os.makedirs(OUTPUT_DIR_RAMP, exist_ok=True)

    label = make_label("ramp",
                       temp_start=TEMP_RAMP_START_K,
                       temp_end=TEMP_RAMP_END_K,
                       n_steps=N_STEPS_RAMP)

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
        os.path.join(OUTPUT_DIR_RAMP, dcd_file),
        os.path.join(OUTPUT_DIR_RAMP, pdb_file),
        traj['positions'], atom_names, traj['times'],
    )

    # Diagnostic and caloric-curve plots
    plot_ramp_all(traj, label=label, save_dir=OUTPUT_DIR_RAMP)

    print(f"\nAll output saved to: {OUTPUT_DIR_RAMP}/")


if __name__ == "__main__":
    main()
