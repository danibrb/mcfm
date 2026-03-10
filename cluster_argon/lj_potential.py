"""
Lennard-Jones pair potential, total potential energy, and atomic forces.
"""

import numpy as np


def lj_pair_energy(epsilon_ev: float, sigma_ang: float, r: float) -> float:
    """
    Lennard-Jones potential for a single atom pair.
    """
    s_r = sigma_ang / r
    return 4.0 * epsilon_ev * (s_r**12 - s_r**6)


def potential_energy(epsilon_ev: float, sigma_ang: float,
                     positions: np.ndarray) -> float:
    """
    Total LJ potential energy summed over all unique atom pairs.
    """
    n_atoms = len(positions)
    energy  = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0.0:
                energy += lj_pair_energy(epsilon_ev, sigma_ang, r)
    return energy


def compute_forces(epsilon_ev: float, sigma_ang: float,
                   positions: np.ndarray) -> np.ndarray:
    """
    Compute LJ forces on all atoms, using Newton's third law.
    """
    n_atoms = len(positions)
    forces  = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]          # vector j->i   [Å]
            r   = np.linalg.norm(rij)
            if r == 0.0:
                continue

            s_r      = sigma_ang / r
            # Scalar prefactor: 24ε·[2(σ/r)¹²−(σ/r)⁶] / r²   [eV/Å²]
            f_scalar = 24.0 * epsilon_ev * (2.0 * s_r**12 - s_r**6) / r**2
            f_vec    = f_scalar * rij                  # force on atom i  [eV/Å]

            forces[i] += f_vec
            forces[j] -= f_vec                         # Newton's third law

    return forces
