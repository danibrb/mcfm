"""
Input/output utilities for molecular structure and force-field parameters.
"""

import numpy as np


def read_xyz(filename: str):
    """
    Parse an XYZ-format coordinate file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms    = int(lines[0])
    comment    = lines[1].strip()
    atom_names = []
    positions  = []

    for line in lines[2: 2 + n_atoms]:
        atom, x, y, z = line.split()
        atom_names.append(atom)
        positions.append([float(x), float(y), float(z)])

    return n_atoms, np.array(positions, dtype=np.float64), atom_names, comment


def read_lj_params(filename: str) -> dict:
    """
    Parse a Lennard-Jones parameter file.
    """
    raw = {}
    with open(filename, 'r') as f:
        for line in f:
            if '=' in line:
                key, rest = line.split('=', 1)
                value = rest.split()[0].replace('d', 'e')
                raw[key.strip()] = float(value)

    return {
        'epsilon_K': raw['epsi_lj_k'],   # [K]
        'sigma_ang': raw['sigma_lj'],     # [Å]
    }
