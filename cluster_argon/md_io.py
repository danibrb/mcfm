def read_xyz(filename):

    with open(filename, 'r') as f:
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

def read_lj_param(filename):

    params = {}

    with open(filename, 'r') as f:
        for line in f:
            if '=' in line:
                key, rest = line.split('=',1)
                value = rest.split()[0]
                
                # change 'd' notation for 10^n
                value = value.replace('d','e')
                params[key.strip()] = float(value)

    return params
