"""
Visualization of MD trajectory data.

All plot functions save a PNG to save_dir and close the figure to avoid
memory accumulation during long analysis sessions.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_label(ensemble: str, temp_start: float,
               temp_end: float = None, n_steps: int = 0) -> str:
    """Build a short label for plot titles and filenames."""
    steps_str = f"{n_steps / 1000:.0f}k"
    if temp_end is not None:
        return f"{ensemble.upper()}_{temp_start:.0f}-{temp_end:.0f}K_{steps_str}steps"
    return f"{ensemble.upper()}_T{temp_start:.0f}K_{steps_str}steps"


def save(fig: plt.Figure, save_dir: str, filename: str) -> None:
    """Save figure to save_dir/filename and close it."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode='same')


# ---------------------------------------------------------------------------
# NVE / NVT diagnostic plots
# ---------------------------------------------------------------------------

def plot_energy(times:            np.ndarray,
                kinetic_energy:   np.ndarray,
                potential_energy: np.ndarray,
                total_energy:     np.ndarray,
                label:            str = "",
                save_dir:         str = ".") -> None:
    """Plot kinetic, potential, and total energy vs time."""
    filename_label = label.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(times, kinetic_energy,   label='Kinetic',   color='tab:orange')
    ax.plot(times, potential_energy, label='Potential', color='tab:blue')
    ax.plot(times, total_energy,     label='Total',     color='tab:green', linewidth=2)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title(f'Energy evolution  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, save_dir, f"energy_{filename_label}.png")


def plot_temperature(times:           np.ndarray,
                     temperature:     np.ndarray,
                     label:           str   = "",
                     target_temp_k:   float = None,
                     save_dir:        str   = ".") -> None:
    """Plot instantaneous temperature vs time."""
    t_mean = np.mean(temperature)
    filename_label = label.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, temperature, color='tab:red', label='T(t)')
    ax.axhline(t_mean, color='tab:red', linestyle='--',
               linewidth=0.9, label=f'<T> = {t_mean:.2f} K')
    if target_temp_k is not None:
        ax.axhline(target_temp_k, color='black', linestyle=':',
                   linewidth=0.9, label=f'T_target = {target_temp_k:.1f} K')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature evolution  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, save_dir, f"temperature_{filename_label}.png")


def plot_x_component(times:         np.ndarray,
                     positions:     np.ndarray,
                     label:         str = "",
                     particle_index: int = 0,
                     save_dir:      str = ".") -> None:
    """Plot the x coordinate of a single particle vs time."""
    x = positions[:, particle_index, 0]
    filename_label = label.replace(" ", "_")

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
    save(fig, save_dir, f"x_component_atom{particle_index}_{filename_label}.png")


def plot_all(trajectory:     dict,
             label:          str   = "",
             target_temp_k:  float = None,
             particle_index: int   = 0,
             save_dir:       str   = ".") -> None:
    """Generate energy, temperature, and x-component plots."""
    plot_energy(trajectory['times'],
                trajectory['kinetic_energy'],
                trajectory['potential_energy'],
                trajectory['total_energy'],
                label=label, save_dir=save_dir)

    plot_temperature(trajectory['times'],
                     trajectory['temperature'],
                     label=label,
                     target_temp_k=target_temp_k,
                     save_dir=save_dir)

    plot_x_component(trajectory['times'],
                     trajectory['positions'],
                     label=label,
                     particle_index=particle_index,
                     save_dir=save_dir)


# ---------------------------------------------------------------------------
# Andersen analysis plots
# ---------------------------------------------------------------------------

def plot_vacf(times:    np.ndarray,
              vacf:     np.ndarray,
              label:    str = "",
              save_dir: str = ".") -> None:
    """Plot the normalised velocity autocorrelation function vs time."""
    filename_label = label.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, vacf, color='tab:purple', linewidth=1.0)
    ax.axhline(0,       color='black', linewidth=0.8, linestyle='--')
    ax.axhline(1/np.e,  color='red',   linewidth=0.8, linestyle='--',
               label='1/e threshold')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('C(t)')
    ax.set_title(f'Velocity autocorrelation function  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xlim(0, min(2000, times[-1]))
    fig.tight_layout()
    save(fig, save_dir, f"vacf_{filename_label}.png")


def plot_temperature_multi(times_list:  list,
                           temp_list:   list,
                           freq_list:   np.ndarray,
                           target_temp: float,
                           label:       str = "",
                           save_dir:    str = ".") -> None:
    """Overlay temperature traces from NVT runs at different collision frequencies."""
    filename_label = label.replace(" ", "_")
    cmap = plt.get_cmap("Set1", len(freq_list))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (times, temp, freq) in enumerate(zip(times_list, temp_list, freq_list)):
        t_mean = np.mean(temp)
        window = int(max(10, min(500, freq * 1e5)))
        window = min(window, len(temp) // 5)
        temp_smooth = moving_average(temp, window)

        ax.plot(times, temp,        color=cmap(i), alpha=0.1, linewidth=0.5)
        ax.plot(times, temp_smooth, color=cmap(i), linewidth=1.8,
                label=f'ν={freq:.1e} fs⁻¹ | <T>={t_mean:.1f} K')

    ax.axhline(target_temp, color='black', linewidth=1.0, linestyle=':',
               label=f'T_target = {target_temp:.0f} K')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature — Andersen thermostat at different frequencies  {label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, save_dir, f"temperature_multi_{filename_label}.png")


# ---------------------------------------------------------------------------
# Heating-ramp plots
# ---------------------------------------------------------------------------

def plot_ramp_temperature(times:       np.ndarray,
                          temperature: np.ndarray,
                          target_temp: np.ndarray,
                          label:       str = "",
                          save_dir:    str = ".") -> None:
    """
    Plot instantaneous and target temperature vs simulation time for
    the heating-ramp run.
    """
    filename_label = label.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times * 1e-3, temperature,  color='tab:red',  linewidth=0.8,
            label='T instantaneous')
    ax.plot(times * 1e-3, target_temp,  color='black',    linewidth=1.2,
            linestyle='--', label='T target (ramp)')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature vs time — heating ramp  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, save_dir, f"ramp_T_vs_t_{filename_label}.png")


def plot_caloric_curve(target_temp:      np.ndarray,
                       total_energy:     np.ndarray,
                       potential_energy: np.ndarray,
                       label:            str = "",
                       save_dir:         str = ".") -> None:
    """
    Plot the caloric curve E_tot(T) and E_pot(T) vs thermostat target
    temperature.

    Discontinuities or changes of slope in these curves mark phase
    transitions (solid → liquid, surface evaporation).
    """
    filename_label = label.replace(" ", "_")

    # Smooth with a moderate moving average to reduce noise
    window = max(10, len(target_temp) // 200)
    e_tot_smooth = moving_average(total_energy,     window)
    e_pot_smooth = moving_average(potential_energy, window)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Panel 1: total energy vs T_target
    ax = axes[0]
    ax.plot(target_temp, total_energy,  color='tab:green', alpha=0.2,
            linewidth=0.5)
    ax.plot(target_temp, e_tot_smooth,  color='tab:green', linewidth=1.8,
            label='E_tot (smoothed)')
    ax.set_ylabel('Total energy [eV]')
    ax.set_title(f'Caloric curve — heating ramp  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: potential energy vs T_target
    ax = axes[1]
    ax.plot(target_temp, potential_energy, color='tab:blue', alpha=0.2,
            linewidth=0.5)
    ax.plot(target_temp, e_pot_smooth,     color='tab:blue', linewidth=1.8,
            label='E_pot (smoothed)')
    ax.set_xlabel('T target [K]')
    ax.set_ylabel('Potential energy [eV]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, save_dir, f"caloric_curve_{filename_label}.png")


def plot_ramp_all(trajectory: dict,
                  label:      str = "",
                  save_dir:   str = ".") -> None:
    """
    Generate all standard plots for a heating-ramp trajectory dict.
    Calls: plot_ramp_temperature, plot_caloric_curve.
    """
    plot_ramp_temperature(trajectory['times'],
                          trajectory['temperature'],
                          trajectory['target_temp'],
                          label=label, save_dir=save_dir)

    plot_caloric_curve(trajectory['target_temp'],
                       trajectory['total_energy'],
                       trajectory['potential_energy'],
                       label=label, save_dir=save_dir)