import numpy as np

from md_io import read_xyz
from md_config import filename

def main():
    n_atoms, positions, atom_names, comment = read_xyz(filename)
    positions = np.array(positions, dtype=np.float64)
    print(f"Read {n_atoms} atoms")
    print(positions)
    print(atom_names)


if __name__ == "__main__":
    main()