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

        C(t) = <v(t) · v(0)> / <v(0) · v(0)>

    The original implementation iterated over frames in a Python for-loop.
    The vectorised form below computes all frames simultaneously via
    NumPy broadcasting, eliminating the Python loop entirely:

        dot_products[frame, atom] = sum_k  v[frame, atom, k] * v[0, atom, k]
        vacf[frame]               = mean over atoms of dot_products[frame, :]

    For VACF_STEPS=5000 this replaces 5000 Python iterations with a single
    (n_frames, n_atoms, 3) element-wise multiply followed by two reductions.
    """
    v0   = vel_trajectory[0]                                    # (n_atoms, 3)
    norm = np.mean(np.sum(v0 * v0, axis=1))                    # scalar

    # broadcast v0 over the frames axis: (n_frames, n_atoms, 3)
    dot_products = np.sum(vel_trajectory * v0[np.newaxis], axis=2)  # (n_frames, n_atoms)
    vacf = np.mean(dot_products, axis=1)                            # (n_frames,)

    return vacf / norm