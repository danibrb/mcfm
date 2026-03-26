"""
Calculation of thermodynamics quantities
"""
import numpy as np

from initialization import kinetic_energy
from constants import KB_EV


def temperature(velocities: np.ndarray, mass_amu: float) -> float:
    """
    Instantaneous temperature from the equipartition theorem.

    T = 2K / (dof * k_B)   with dof = 3N - 3
    """
    n_atoms = len(velocities)
    kin_e = kinetic_energy(velocities, mass_amu)
    dof = 3 * n_atoms - 3
    return 2 * kin_e / (dof * KB_EV)

def compute_vacf(vel_trajectory: np.ndarray) -> np.ndarray:
    """
    Compute the normalised velocity autocorrelation function (VACF).
 
        C(t) = <v(t) * v(0)> / <v(0) * v(0)>

    """
    n_frames, n_atoms, _ = vel_trajectory.shape
 
    # v0[i, alpha] = velocity of atom i at t=0 along component alpha
    v0 = vel_trajectory[0]                          # shape (n_atoms, 3)
 
    # normalisation: <v(0) * v(0)> averaged over all atoms
    norm = np.mean(np.sum(v0 * v0, axis=1))         # scalar
 
    vacf = np.zeros(n_frames)
    for frame in range(n_frames):
        vt = vel_trajectory[frame]                   # shape (n_atoms, 3)
        # dot product per atom, then average over atoms
        vacf[frame] = np.mean(np.sum(vt * v0, axis=1))
 
    return vacf / norm