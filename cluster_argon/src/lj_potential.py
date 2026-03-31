"""
Lennard-Jones pair potential, total potential energy, and atomic forces.
"""

import numpy as np


from numba import njit
 
@njit(cache=True, fastmath=True)
def compute_forces_and_potential(epsilon_ev: float, sigma_ang: float,
                                    positions: np.ndarray) -> tuple:
    """
    Compute LJ forces and potential energy in eV.
    """
    n_atoms   = len(positions)
    forces    = np.zeros((n_atoms, 3))
    potential = 0.0

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r2 = dx*dx + dy*dy + dz*dz

            # multiplication is faster than pow
            s2  = (sigma_ang * sigma_ang) / r2
            s6  = s2 * s2 * s2
            s12 = s6 * s6

            potential += 4.0 * epsilon_ev * (s12 - s6)

            # takeing out 1/r2 to get a simpler factor to calculate
            f_scalar = 24.0 * epsilon_ev * (2.0 * s12 - s6) / r2
            forces[i, 0] += f_scalar * dx
            forces[i, 1] += f_scalar * dy
            forces[i, 2] += f_scalar * dz
            forces[j, 0] -= f_scalar * dx
            forces[j, 1] -= f_scalar * dy
            forces[j, 2] -= f_scalar * dz

    return forces, potential


def potential_energy(epsilon_ev: float, sigma_ang: float,
                     positions: np.ndarray) -> float:
    """Convenience wrapper returning only the potential energy."""
    _, u = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    return u
 
 
def compute_forces(epsilon_ev: float, sigma_ang: float,
                   positions: np.ndarray) -> np.ndarray:
    """Convenience wrapper returning only the forces."""
    f, _ = compute_forces_and_potential(epsilon_ev, sigma_ang, positions)
    return f


def warmup_jit(epsilon_ev: float, sigma_ang: float,
                positions: np.ndarray) -> None:
    """
    Trigger Numba JIT compilation for all kernels before the timed loop.

    Warms up: LJ force kernel, fused Velocity Verlet step, and Andersen
    thermostat.  All subsequent calls hit the compiled cache.
    """
    from integrator import velocity_verlet_step_jit
    from thermostat import andersen_jit, seed_numba_rng
    from constants  import FORCE_CONV

    print("Warming up JIT kernels...", end=" ", flush=True)

    # Force kernel
    compute_forces_and_potential(epsilon_ev, sigma_ang, positions)

    # Fused VV step (inlines force kernel)
    dummy_vel    = np.zeros_like(positions)
    dummy_forces = np.zeros_like(positions)
    force_conv_over_mass = FORCE_CONV / 40.0
    velocity_verlet_step_jit(positions, dummy_vel, dummy_forces,
                              force_conv_over_mass, 1.0,
                              epsilon_ev, sigma_ang)

    # Andersen thermostat
    seed_numba_rng(0)
    andersen_jit(dummy_vel, 40.0, 20.0, 1e-3, 1.0)

    print("done.")


def lj_pair_energy(epsilon_ev: float, sigma_ang: float, r: float) -> float:
    """
    Lennard-Jones potential for a single atom pair.
    """
    s_r = sigma_ang / r
    return 4.0 * epsilon_ev * (s_r**12 - s_r**6)


def potential_energy_no(epsilon_ev: float, sigma_ang: float,
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


def compute_forces_no(epsilon_ev: float, sigma_ang: float,
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

def compute_forces_and_potential_no(epsilon_ev: float, sigma_ang: float,
                   positions: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute LJ forces on all atoms, using Newton's third law.
    Add potential calculations to optimize loop
    """
    n_atoms = len(positions)
    forces  = np.zeros((n_atoms, 3))
    potential = 0.0

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]          # vector j->i   [Å]
            r   = np.linalg.norm(rij)
            if r == 0.0:
                continue
            
            s_r      = sigma_ang / r
            # Compute LJ potential
            potential += 4.0 * epsilon_ev * (s_r**12 - s_r**6)
            
            # Scalar prefactor: 24ε·[2(σ/r)¹²−(σ/r)⁶] / r²   [eV/Å²]
            f_scalar = 24.0 * epsilon_ev * (2.0 * s_r**12 - s_r**6) / r**2
            f_vec    = f_scalar * rij                  # force on atom i  [eV/Å]

            forces[i] += f_vec
            forces[j] -= f_vec                         # Newton's third law

    return forces, potential