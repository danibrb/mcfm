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

def write_xyz_trajectory(filename: str,
                         positions: np.ndarray,
                         atom_names: list,
                         times: np.ndarray) -> None:
    """
    Write a XYZ trajectory file.
 
    The format repeats the standard XYZ block for each saved frame:
 
        <n_atoms>
        comment line (frame index and simulation time)
        <symbol>  <x>  <y>  <z>
    """
    n_frames, n_atoms, _ = positions.shape
 
    with open(filename, 'w') as f:
        for frame in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write(f"frame {frame}  t = {times[frame]:.4f} fs\n")
            for i in range(n_atoms):
                x, y, z = positions[frame, i]
                f.write(f"{atom_names[i]:4s}  {x:16.8f}  {y:16.8f}  {z:16.8f}\n")
 
    print(f"Saved: {filename}  ({n_frames} frames, {n_atoms} atoms)")
