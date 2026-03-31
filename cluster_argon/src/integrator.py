"""
Velocity Verlet integrator for classical molecular dynamics.

The public interface (velocity_verlet_step) is unchanged.  Internally the
position/velocity arithmetic is compiled by Numba via velocity_verlet_step_jit,
which calls compute_forces_and_potential from within the same JIT context.
Numba inlines the force kernel into the VV step, eliminating:
  - the Python function call overhead for the force evaluation,
  - intermediate (N,3) array allocations for accelerations,
  - NumPy ufunc dispatch overhead for the 5 element-wise array operations.
"""

import numpy as np
from numba import njit

from constants    import FORCE_CONV
from lj_potential import compute_forces_and_potential


@njit(cache=True, fastmath=True)
def velocity_verlet_step_jit(positions:            np.ndarray,
                              velocities:           np.ndarray,
                              forces:               np.ndarray,
                              force_conv_over_mass: float,
                              dt_fs:                float,
                              epsilon_ev:           float,
                              sigma_ang:            float):
    """
    Fused Velocity Verlet step compiled by Numba.

    Calling compute_forces_and_potential (also @njit) from within this
    function allows Numba to inline the force kernel, eliminating the
    JIT-to-JIT dispatch boundary and all intermediate array allocations.

    Parameters
    ----------
    force_conv_over_mass : FORCE_CONV / mass_amu, precomputed once outside
                           the loop to avoid the division at every step.
    """
    # a(t) = F(t) * (FORCE_CONV / m)
    acc = forces * force_conv_over_mass

    # r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_positions = positions + velocities * dt_fs + 0.5 * acc * (dt_fs * dt_fs)

    # F(t+dt), U(t+dt)
    new_forces, potential = compute_forces_and_potential(epsilon_ev, sigma_ang,
                                                          new_positions)
    new_acc = new_forces * force_conv_over_mass

    # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    new_velocities = velocities + 0.5 * (acc + new_acc) * dt_fs

    return new_positions, new_velocities, new_forces, potential


def velocity_verlet_step(positions:  np.ndarray,
                         velocities: np.ndarray,
                         forces:     np.ndarray,
                         mass_amu:   float,
                         dt_fs:      float,
                         epsilon_ev: float,
                         sigma_ang:  float):
    """
    Public wrapper — identical signature to the original pure-NumPy version.

    Precomputes force_conv_over_mass once per call and dispatches to the
    compiled kernel.  nve.py, nvt.py call this unchanged.
    The hot loop in heating_ramp.py calls velocity_verlet_step_jit directly
    (passing the precomputed scalar) to avoid the division at every step.
    """
    return velocity_verlet_step_jit(positions, velocities, forces,
                                     FORCE_CONV / mass_amu, dt_fs,
                                     epsilon_ev, sigma_ang)