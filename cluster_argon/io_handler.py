"""
Input/output utilities for molecular structure and force-field parameters.
"""

import numpy as np
import mdtraj as md


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


def write_pdb_file(filename: str, positions: np.ndarray, atom_names: list) -> None:
    """
    Write a PDB file with atom names and initial positions.
    """
    n_atoms = positions.shape[0]
    with open(filename, 'w') as f:
        for i in range(n_atoms):
            x, y, z = positions[i]
            f.write(f"ATOM  {i+1:5d}  {atom_names[i]:4s} MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
    print(f"Saved initial structure: {filename}")

def write_dcd_trajectory(filename: str, positions: np.ndarray, atom_names: list, times: np.ndarray) -> None:
    """
    Write a DCD trajectory file using MDTraj.
    Note: The DCD format does not store atom names or times, so this information will not be included in the DCD file.
    """
    n_frames, n_atoms, _ = positions.shape

    # Create a minimal topology
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue('MOL', chain)
    for atom_name in atom_names:
        element = md.element.get_by_symbol(atom_name)
        topology.add_atom(atom_name, element, residue)

    # Create a trajectory (positions should be in nanometers)
    positions_nm = positions / 10  # Convert angstroms to nanometers
    trajectory = md.Trajectory(positions_nm, topology)

    # Save the trajectory to a DCD file
    trajectory.save_dcd(filename)

    print(f"Saved: {filename}  ({n_frames} frames, {n_atoms} atoms)")

def save_trajectory_with_metadata(dcd_filename: str, pdb_filename: str, positions: np.ndarray, atom_names: list, times: np.ndarray) -> None:
    """
    Save the trajectory data along with atom names and initial positions.
    """
    write_pdb_file(pdb_filename, positions[0], atom_names)
    write_dcd_trajectory(dcd_filename, positions, atom_names, times)