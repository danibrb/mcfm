def read_xyz(filename):

    with open(filename) as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    comment = lines[1].strip()

    atom_names = []
    positions = []

    for i in range(2, 2 + n_atoms):
        atom, x, y, z = lines[i].split()
        atom_names.append(atom)
        positions.append([float(x), float(y), float(z)])

    return n_atoms, positions, atom_names, comment