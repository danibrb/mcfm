"""
Visualization of NVE trajectory data.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_energy(times: np.ndarray,
                kinetic_energy: np.ndarray,
                potential_energy: np.ndarray,
                total_energy: np.ndarray,
                save_path: str = "energy.png") -> None:
    """
    Plot kinetic, potential, and total energy as a function of time.
    """

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(times, kinetic_energy,   label='Kinetic',   color='tab:orange')
    ax.plot(times, potential_energy, label='Potential', color='tab:blue')
    ax.plot(times, total_energy,     label='Total',     color='tab:green',
            linewidth=2)
    ax.set_ylabel('Energy [eV]')
    ax.set_title('Energy evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

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


def plot_all(trajectory: dict,
             particle_index: int = 0,
             target_temp: float = None) -> None:
    """
    Convenience wrapper: generate all plots.
    """

    plot_energy(trajectory['times'],
                trajectory['kinetic_energy'],
                trajectory['potential_energy'],
                trajectory['total_energy'])
    plot_temperature(trajectory['times'], trajectory['temperature'],
                     target_temp=target_temp)
    plot_x_component(trajectory['times'], trajectory['positions'],
                             particle_index=particle_index)


