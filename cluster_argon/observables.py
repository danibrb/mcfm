"""
Calculation of thermodynamics quantities
"""
import numpy as np

from initialization import kinetic_energy
from constants import KB_EV


def temperature(velocities: np.ndarray) -> float:
    """
    Calculation of temperature from the equipartition theorem

    ⟨K⟩ = (3N - 3)/2 · k_B · T

    """
    n_atoms = len(velocities)
    kin_e = kinetic_energy(velocities)

    dof = 3 * n_atoms - 3

    return 2 *  kin_e / (dof * KB_EV)