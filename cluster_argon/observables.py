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