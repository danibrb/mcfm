"""
Main driver for the Lennard-Jones molecular dynamics simulation of Argon cluster.

Unit system:
    Length   -> Angstrom (Å)
    Energy   -> electronvolt (eV)
    Mass     -> atomic mass unit (amu)
    Time     -> femtosecond (fs)
    Force    -> eV/Å
    Velocity -> Å/fs

"""

import numpy as np

from constants     import KB_EV
from config        import (FILENAME_XYZ, FILENAME_LJ, MASS_AMU,
                            TEMP_INIT_K, TIMESTEP_FS, RANDOM_SEED)
from io_handler    import read_xyz, read_lj_params
from lj_potential  import compute_forces, potential_energy
from initialization import initialize_velocities, kinetic_energy
from integrator    import velocity_verlet_step


def print_energies(label: str, kin_e: float, pot_e: float) -> None:
    """Print the energy breakdown for a given simulation state."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Kinetic   energy : {kin_e:+.6e} eV")
    print(f"  Potential energy : {pot_e:+.6e} eV")
    print(f"  Total     energy : {kin_e + pot_e:+.6e} eV")


def main() -> None:

    # 1. Read structure and Lennard-Jones parameters

    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV    # Kelvin -> eV
    sigma_ang  = params['sigma_ang']             # already in Å
    print(f"LJ parameters: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} Å")

    # 2. Initialise velocities (Maxwell-Boltzmann at TEMP_INIT_K)

    rng = np.random.default_rng(RANDOM_SEED)
    velocities = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng)

    # 3. Compute initial state

    forces = compute_forces(epsilon_ev, sigma_ang, positions)
    kin_e  = kinetic_energy(velocities, MASS_AMU)
    pot_e  = potential_energy(epsilon_ev, sigma_ang, positions)
    print_energies("Initial state  (t = 0 fs)", kin_e, pot_e)

    # 4. Single Velocity Verlet step

    positions, velocities, forces = velocity_verlet_step(
        positions, velocities, forces,
        MASS_AMU, TIMESTEP_FS, epsilon_ev, sigma_ang,
    )

    kin_e = kinetic_energy(velocities, MASS_AMU)
    pot_e = potential_energy(epsilon_ev, sigma_ang, positions)
    print_energies(f"After Velocity Verlet step  (dt = {TIMESTEP_FS:.1f} fs)",
                    kin_e, pot_e)
    print()


if __name__ == "__main__":
    main()
