import numpy as np

from md_io import read_xyz, read_lj_param
from md_config import filename_xyz, filename_lj, temp_i, mass_amu, KB_EV_K
from md_velocities import initialize_velocities
from md_energy import kinetic_energy, potential_energy, forces_jn

def main():
    # 1. Read Structure
    n_atoms, positions, atom_names, comment = read_xyz(filename_xyz)
    positions = np.array(positions, dtype=np.float64)
    print(f"Read {n_atoms} atoms")

    # 2. Read Parameters
    param = read_lj_param(filename_lj)
    
    # Extract parameters
    epsi_lj_k = param['epsi_lj_k']
    sigma = param['sigma_lj']
    
    # Convert epsilon to eV
    epsilon = epsi_lj_k * KB_EV_K

    # 3. Initialize Velocities
    velocities = initialize_velocities(n_atoms, mass_amu, temp_i)

    # 4. Calculate Energy and Forces
    kin_e = kinetic_energy(velocities)
    pot_e = potential_energy(epsilon, sigma, n_atoms, positions)
    
    # Calculate forces
    forces = forces_jn(epsilon, sigma, n_atoms, positions)

    print(f"Intial Kinetic Energy: {kin_e:.6e} eV")
    print(f"Initial Potential Energy: {pot_e:.6e} eV")
    print(f"Initial Total Energy: {kin_e + pot_e:.6e} eV")
    print(f"Force on atom 0: {forces[0]}")

if __name__ == "__main__":
    main()