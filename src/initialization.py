"""
Velocity initialisation for MD simulations.
"""

import numpy as np

from constants import KB_EV, AMU_ANG2_FS2_TO_EV


# Kinetic energy


def kinetic_energy(velocities: np.ndarray, mass_amu: float) -> float:
    """
    Total kinetic energy in eV.
    """
    return 0.5 * mass_amu * np.sum(velocities**2) * AMU_ANG2_FS2_TO_EV


def target_kinetic_energy(n_atoms: int, temperature_k: float) -> float:
    """
    Target kinetic energy from the equipartition theorem.

    <K> = (3N - 3)/2 · k_B · T
    """
    dof = 3 * n_atoms - 3       # degrees of freedom after COM removal
    return 0.5 * dof * KB_EV * temperature_k



# Initialisation steps


def maxwell_boltzmann_velocities(n_atoms: int, mass_amu: float,
                                   temperature_k: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """
    Sample velocities from the Maxwell-Boltzmann distribution.
    """
    sigma_v = np.sqrt(KB_EV * temperature_k / (mass_amu * AMU_ANG2_FS2_TO_EV))
    return rng.normal(loc=0.0, scale=sigma_v, size=(n_atoms, 3))


def remove_com_drift(velocities: np.ndarray) -> np.ndarray:
    """
    Remove centre-of-mass velocity (equal-mass assumption).
    """
    return velocities - velocities.mean(axis=0)


def rescale_to_temperature(velocities: np.ndarray, mass_amu: float,
                             n_atoms: int, temperature_k: float) -> np.ndarray:
    """
    Uniformly rescale velocities to match the equipartition target KE.
    """
    k_current = kinetic_energy(velocities, mass_amu)
    if k_current == 0.0:
        return velocities
    k_target = target_kinetic_energy(n_atoms, temperature_k)
    return velocities * np.sqrt(k_target / k_current)


def initialize_velocities(n_atoms: int, mass_amu: float,
                           temperature_k: float,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Full velocity initialisation: sample -> remove COM drift -> rescale.
    """
    v = maxwell_boltzmann_velocities(n_atoms, mass_amu, temperature_k, rng)
    v = remove_com_drift(v)
    v = rescale_to_temperature(v, mass_amu, n_atoms, temperature_k)
    return v
