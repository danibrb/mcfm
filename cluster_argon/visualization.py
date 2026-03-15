"""
Visualization of MD trajectory data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def save(fig: plt.Figure, save_dir: str, filename: str) -> None:
    """Save figure to save_dir/filename and close it."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_energy(times:            np.ndarray,
                kinetic_energy:   np.ndarray,
                potential_energy: np.ndarray,
                total_energy:     np.ndarray,
                label:            str = "",
                save_dir:         str = ".") -> None:
    """
    Plot kinetic, potential, and total energy as a function of time.
    A second panel shows E - <E> to amplify any drift.
    """
    e_mean  = np.mean(total_energy)
    e_drift = (total_energy[-1] - total_energy[0]) / abs(e_mean) * 100

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(times, kinetic_energy,   label='Kinetic',   color='tab:orange')
    ax.plot(times, potential_energy, label='Potential', color='tab:blue')
    ax.plot(times, total_energy,     label='Total',     color='tab:green', linewidth=2)
    ax.set_ylabel('Energy [eV]')
    ax.set_title(f'Energy evolution  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, total_energy - e_mean, color='tab:green')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Time [fs]')
    ax2.set_ylabel('E - <E> [eV]')
    ax2.set_title(f'Total energy drift  ({e_drift:+.4f} %)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, save_dir, f"energy_{label}.png")


def plot_temperature(times:       np.ndarray,
                     temperature:  np.ndarray,
                     label:        str   = "",
                     target_temp:  float = None,
                     save_dir:     str   = ".") -> None:
    """
    Plot instantaneous temperature as a function of time.
    """
    t_mean = np.mean(temperature)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, temperature, color='tab:red', label='T(t)')
    ax.axhline(t_mean, color='tab:red', linestyle='--',
               linewidth=0.9, label=f'<T> = {t_mean:.2f} K')
    if target_temp is not None:
        ax.axhline(target_temp, color='black', linestyle=':',
                   linewidth=0.9, label=f'T_target = {target_temp:.1f} K')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature evolution  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, save_dir, f"temperature_{label}.png")


def plot_x_component(times:          np.ndarray,
                     positions:       np.ndarray,
                     label:           str = "",
                     particle_index:  int = 0,
                     save_dir:        str = ".") -> None:
    """
    Plot the x coordinate of a single particle as a function of time.
    """
    x = positions[:, particle_index, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, x, color='tab:blue')
    ax.axhline(np.mean(x), color='gray', linestyle='--',
               linewidth=0.9, label=f'<x> = {np.mean(x):.3f} Å')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('x [Å]')
    ax.set_title(f'x component — atom {particle_index}  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, save_dir, f"x_coord_atom{particle_index}_{label}.png")


def plot_all(trajectory:     dict,
             label:          str   = "",
             target_temp:    float = None,
             particle_index: int   = 0,
             save_dir:       str   = ".") -> None:
    """
    Generate all plots for a trajectory dict as returned by run_nve/run_nvt.

    Parameters
    ----------
    trajectory     : dict   output of run_nve() or run_nvt()
    label          : str    short description shown in plot titles
    target_temp    : float  reference temperature line  [K]
    particle_index : int    atom index for x-component plot
    save_dir       : str    output directory
    """
    plot_energy(trajectory['times'],
                trajectory['kinetic_energy'],
                trajectory['potential_energy'],
                trajectory['total_energy'],
                label=label,
                save_dir=save_dir)

    plot_temperature(trajectory['times'],
                     trajectory['temperature'],
                     label=label,
                     target_temp=target_temp,
                     save_dir=save_dir)

    plot_x_component(trajectory['times'],
                     trajectory['positions'],
                     label=label,
                     particle_index=particle_index,
                     save_dir=save_dir)