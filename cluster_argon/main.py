import numpy as np

from md_io import read_xyz
from md_config import filename_xyz

def main():
    n_atoms, positions, atom_names, comment = read_xyz(filename_xyz)
    positions = np.array(positions, dtype=np.float64)
    print(f"Read {n_atoms} atoms")
    print(positions)
    print(atom_names)


if __name__ == "__main__":
    main()