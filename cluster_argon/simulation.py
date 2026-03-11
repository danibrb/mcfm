"""
Main driver for the Lennard-Jones NVE molecular dynamics simulation of an Argon cluster.

Unit system:
    Length   -> Angstrom (Å)
    Energy   -> electronvolt (eV)
    Mass     -> atomic mass unit (amu)
    Time     -> femtosecond (fs)
    Force    -> eV/Å
    Velocity -> Å/fs
"""

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ, FILENAME_LJ, MASS_AMU,
                             TEMP_INIT_K, TIMESTEP_FS, N_STEPS, RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params
from initialization import initialize_velocities
from nve            import run_nve
from visualization import plot_all


def main() -> None:

    # 1. Read structure and Lennard-Jones parameters
    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"LJ parameters: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} Å")

    # 2. Initialise velocities
    rng = np.random.default_rng(RANDOM_SEED)
    velocities = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng)

    # 3. Run NVE simulation
    print(f"\nRunning NVE simulation: {N_STEPS} steps, dt = {TIMESTEP_FS} fs")
    trajectory = run_nve(
        positions, velocities,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, N_STEPS,
    )

    # 4. Print summary
    # print(f"\n{'t [fs]':>10}  {'K [eV]':>14}  {'U [eV]':>14}  {'E [eV]':>14}  {'T [K]':>8}")
    # print("-" * 68)
    # for i in range(len(trajectory['times'])):
    #     print(f"{trajectory['times'][i]:>10.1f}  "
    #           f"{trajectory['kinetic_energy'][i]:>+14.6e}  "
    #           f"{trajectory['potential_energy'][i]:>+14.6e}  "
    #           f"{trajectory['total_energy'][i]:>+14.6e}  "
    #           f"{trajectory['temperature'][i]:>8.3f}")
        
    # 5. Plot data
    plot_all(trajectory)


if __name__ == "__main__":
    main()