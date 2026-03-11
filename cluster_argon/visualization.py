"""
Visualization of NVE trajectory data.

Produces three figures:
    1. 3D trajectory of a selected particle
    2. Energy evolution (kinetic, potential, total)
    3. Temperature evolution

All plots are saved to disk as PNG files.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_particle_trajectory(times: np.ndarray,
                              positions: np.ndarray,
                              particle_index: int = 0,
                              save_path: str = "trajectory.png") -> None:
    """
    Plot the 3D trajectory of a single particle and its x, y, z
    components as a function of time.

    Parameters
    ----------
    times          : np.ndarray, shape (n_saved,)              [fs]
    positions      : np.ndarray, shape (n_saved, n_atoms, 3)   [Å]
    particle_index : int   Atom index to visualise
    save_path      : str
    """
    pos = positions[:, particle_index, :]   # shape (n_saved, 3)

    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory
    ax3d = fig.add_subplot(121, projection='3d')
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=times, cmap='viridis', s=30)
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
              color='gray', linewidth=0.8, alpha=0.5)
    fig.colorbar(sc, ax=ax3d, label='Time [fs]', shrink=0.6)
    ax3d.set_xlabel('x [Å]')
    ax3d.set_ylabel('y [Å]')
    ax3d.set_zlabel('z [Å]')
    ax3d.set_title(f'3D trajectory — atom {particle_index}')

    # x, y, z vs time
    ax2d = fig.add_subplot(122)
    for dim, label in enumerate(['x', 'y', 'z']):
        ax2d.plot(times, pos[:, dim], label=label)
    ax2d.set_xlabel('Time [fs]')
    ax2d.set_ylabel('Position [Å]')
    ax2d.set_title(f'Cartesian components — atom {particle_index}')
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_energy(times: np.ndarray,
                kinetic_energy: np.ndarray,
                potential_energy: np.ndarray,
                total_energy: np.ndarray,
                save_path: str = "energy.png") -> None:
    """
    Plot kinetic, potential, and total energy as a function of time.
    A well-converged NVE run shows a flat total energy line.

    Parameters
    ----------
    times            : np.ndarray  [fs]
    kinetic_energy   : np.ndarray  [eV]
    potential_energy : np.ndarray  [eV]
    total_energy     : np.ndarray  [eV]
    save_path        : str
    """
    # Energy drift as percentage of mean total energy
    e_mean  = np.mean(total_energy)
    e_drift = (total_energy[-1] - total_energy[0]) / abs(e_mean) * 100

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(times, kinetic_energy,   label='Kinetic',   color='tab:orange')
    ax.plot(times, potential_energy, label='Potential', color='tab:blue')
    ax.plot(times, total_energy,     label='Total',     color='tab:green',
            linewidth=2)
    ax.set_ylabel('Energy [eV]')
    ax.set_title('Energy evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total energy deviation from its mean — amplifies drift visually
    ax2 = axes[1]
    ax2.plot(times, total_energy - e_mean, color='tab:green')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Time [fs]')
    ax2.set_ylabel('E - <E> [eV]')
    ax2.set_title(f'Total energy drift  ({e_drift:+.4f} %)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_temperature(times: np.ndarray,
                     temperature: np.ndarray,
                     target_temp: float = None,
                     save_path: str = "temperature.png") -> None:
    """
    Plot the instantaneous temperature as a function of time.
    In NVE the temperature fluctuates; its mean should match
    the initial target temperature.

    Parameters
    ----------
    times       : np.ndarray  [fs]
    temperature : np.ndarray  [K]
    target_temp : float       Initial target temperature [K] (optional)
    save_path   : str
    """
    t_mean = np.mean(temperature)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, temperature, color='tab:red', label='T(t)')
    ax.axhline(t_mean, color='tab:red', linestyle='--',
               linewidth=0.9, label=f'<T> = {t_mean:.2f} K')
    if target_temp is not None:
        ax.axhline(target_temp, color='black', linestyle=':',
                   linewidth=0.9, label=f'T_init = {target_temp:.1f} K')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Temperature evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_x_component(times: np.ndarray,
                     positions: np.ndarray,
                     particle_index: int = 0,
                     save_path: str = "x_component.png") -> None:
    """
    Plot the x coordinate of a single particle as a function of time.

    Parameters
    ----------
    times          : np.ndarray, shape (n_saved,)             [fs]
    positions      : np.ndarray, shape (n_saved, n_atoms, 3)  [Å]
    particle_index : int
    save_path      : str
    """
    x = positions[:, particle_index, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, x, color='tab:blue')
    ax.axhline(np.mean(x), color='gray', linestyle='--',
               linewidth=0.9, label=f'<x> = {np.mean(x):.3f} Å')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('x [Å]')
    ax.set_title(f'x component — atom {particle_index}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_all(traj: dict,
             particle_index: int = 0,
             target_temp: float = None) -> None:
    """
    Convenience wrapper: generate all three plots from a trajectory dict
    as returned by run_nve().

    Parameters
    ----------
    traj           : dict   Output of run_nve()
    particle_index : int    Atom to use for the trajectory plot
    target_temp    : float  Initial temperature [K] for reference line
    """
    plot_particle_trajectory(traj['times'], traj['positions'],
                             particle_index=particle_index)
    plot_energy(traj['times'],
                traj['kinetic_energy'],
                traj['potential_energy'],
                traj['total_energy'])
    plot_temperature(traj['times'], traj['temperature'],
                     target_temp=target_temp)
    plot_x_component(traj['times'], traj['positions'],
                             particle_index=particle_index)


