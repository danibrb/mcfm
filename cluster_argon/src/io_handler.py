"""
Input/output utilities for molecular structure and force-field parameters.

Trajectory formats
------------------
XYZ  — plain text, human-readable, recommended for short simulations.
       Opened directly in VMD: File > New Molecule > XYZ.

DCD + PDB — binary DCD trajectory with a companion PDB topology file.
            Recommended for long simulations (much smaller file size).
            Load in VMD: File > New Molecule > PDB (topology), then
            File > Load Data Into Molecule > DCD (trajectory).

PDB notes
---------
The PDB format is fixed-width.  VMD uses the element field (columns 77-78)
to identify atom type and decide whether to draw bonds.  Without it, VMD
falls back to distance-based bonding, which produces the unwanted rods.

Each Ar atom is written as a HETATM record in its own residue so that
VMD does not connect atoms that happen to be close together.
"""

import numpy as np
import mdtraj as md


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def read_xyz(filename: str):
    """Parse an XYZ-format coordinate file."""
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
    """Parse a Lennard-Jones parameter file."""
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


# ---------------------------------------------------------------------------
# XYZ output  (short simulations)
# ---------------------------------------------------------------------------

def write_xyz_trajectory(filename:   str,
                         positions:  np.ndarray,
                         atom_names: list,
                         times:      np.ndarray) -> None:
    """
    Write a multi-frame XYZ trajectory file.

    Each frame is a standard XYZ block:
        <n_atoms>
        frame <i>  t = <time> fs
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


# ---------------------------------------------------------------------------
# PDB output  (topology companion for DCD)
# ---------------------------------------------------------------------------

def _pdb_hetatm_line(serial:    int,
                     atom_name: str,
                     resname:   str,
                     chain:     str,
                     resseq:    int,
                     x:         float,
                     y:         float,
                     z:         float,
                     element:   str) -> str:
    """
    Build one PDB HETATM record, strictly obeying the fixed-column layout.

    Column positions (1-indexed, inclusive):
        1-6   record type   "HETATM"
        7-11  serial number
        13-16 atom name (left-justified in field)
        18-20 residue name
        22    chain ID
        23-26 residue sequence number
        31-38 x coordinate (8.3f)
        39-46 y coordinate (8.3f)
        47-54 z coordinate (8.3f)
        55-60 occupancy
        61-66 temperature factor
        77-78 element symbol (right-justified)

    Using HETATM instead of ATOM and one residue per atom prevents VMD
    from drawing automatic distance-based bonds between Ar atoms.
    """
    # part1: cols 1-30  (30 chars before x)
    part1 = f"HETATM{serial:5d} {atom_name:<4s}{resname:>3s} {chain}{resseq:4d}     "
    # part2: cols 31-78  (48 chars: xyz + occ + bfac + pad + element)
    part2 = f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    return part1 + part2 + "\n"


def write_pdb_file(filename:   str,
                   positions:  np.ndarray,
                   atom_names: list) -> None:
    """
    Write a single-frame PDB topology file.

    One HETATM record per atom, each in its own residue (residue number =
    atom index + 1).  The element symbol is written in columns 77-78 so
    that VMD correctly identifies atom types and suppresses auto-bonding.
    """
    n_atoms = positions.shape[0]

    with open(filename, 'w') as f:
        f.write("REMARK  Ar38 cluster — topology for DCD trajectory\n")
        for i in range(n_atoms):
            x, y, z   = positions[i]
            atom_name  = atom_names[i].upper()
            resname    = atom_name[:3]
            element    = atom_names[i].capitalize()[:2]
            line = _pdb_hetatm_line(
                serial    = i + 1,
                atom_name = atom_name,
                resname   = resname,
                chain     = 'A',
                resseq    = i + 1,
                x=x, y=y, z=z,
                element   = element,
            )
            f.write(line)
        f.write("END\n")

    print(f"Saved: {filename}  ({n_atoms} atoms)")


# ---------------------------------------------------------------------------
# DCD output  (long simulations)
# ---------------------------------------------------------------------------

def _build_mdtraj_topology(atom_names: list) -> md.Topology:
    """
    Build an MDTraj topology with one residue per atom.

    This avoids any automatic bond inference inside MDTraj and ensures
    the companion PDB written by MDTraj carries no CONECT records.
    """
    topology = md.Topology()
    chain    = topology.add_chain()
    for i, name in enumerate(atom_names):
        resname = name.upper()[:3]
        residue = topology.add_residue(resname, chain)
        element = md.element.get_by_symbol(name.capitalize())
        topology.add_atom(name.upper(), element, residue)
    return topology


def write_dcd_trajectory(filename:   str,
                         positions:  np.ndarray,
                         atom_names: list,
                         times:      np.ndarray) -> None:
    """
    Write a binary DCD trajectory using MDTraj.

    Positions are converted from Angstrom to nanometres (MDTraj convention).
    The DCD format does not store atom names or simulation times; load the
    companion PDB file as the topology in VMD.
    """
    n_frames, n_atoms, _ = positions.shape
    topology     = _build_mdtraj_topology(atom_names)
    positions_nm = (positions / 10.0).astype(np.float32)  # Å -> nm
    trajectory   = md.Trajectory(positions_nm, topology)
    trajectory.save_dcd(filename)
    print(f"Saved: {filename}  ({n_frames} frames, {n_atoms} atoms)")


def save_trajectory_with_metadata(dcd_filename: str,
                                   pdb_filename: str,
                                   positions:    np.ndarray,
                                   atom_names:   list,
                                   times:        np.ndarray) -> None:
    """
    Save a DCD trajectory and its companion PDB topology file.

    Load in VMD:
        1. File > New Molecule
        2. Browse to <pdb_filename>, type PDB, click Load.
        3. File > Load Data Into Molecule
        4. Browse to <dcd_filename>, type DCD, click Load.
    Then set Drawing Method to VDW (sphere) to see 38 balls.
    """
    # Use the first frame as the reference topology for the PDB
    write_pdb_file(pdb_filename, positions[0], atom_names)
    write_dcd_trajectory(dcd_filename, positions, atom_names, times)