import numpy as np

from md_io import read_xyz, read_lj_param
from md_config import filename_xyz, filename_lj, temp_i
from md_velocities import generate_maxwell_velocities, remove_com_velocity
from md_energy import kinetic_enegy, potential_energy, kinetic_intial

def main():
    n_atoms, positions, atom_names, comment = read_xyz(filename_xyz)
    positions = np.array(positions, dtype=np.float64)
    print(f"Read {n_atoms} atoms")

    param = read_lj_param(filename_lj)

    epsi_lj_k=119.
    sigma=3.4
    cbol=8.62e-05

    # convert epsilon in eV
    epsilon = epsi_lj_k * cbol



if __name__ == "__main__":
    main()