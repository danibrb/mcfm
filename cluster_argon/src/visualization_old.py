"""
Visualization utilities for MD trajectory data.

All functions save a PNG to `save_dir` and close the figure.

"""

import os

import numpy as np
import matplotlib.pyplot as plt


# Smoothing helper


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """
    Centred moving average (mode='same').
    Used for time-series plots where the full-length output is required.
    """
    if window <= 1:
        return y.copy()
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1          # keep odd so the window is symmetric
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='same')


def _smooth_valid(y: np.ndarray,
                  window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Centred moving average without boundary artifacts.

    Uses np.convolve with mode='valid', which discards the floor(window/2)
    points at each edge where the kernel would extend beyond the data
    (Press et al., Numerical Recipes, §13.4).  The returned index array
    allows the caller to slice the corresponding x-axis consistently.

    Parameters
    ----------
    y : array of length n
    window : averaging window size (forced odd for symmetry)

    Returns
    -------
    smoothed : array of length  n - window + 1
    indices  : integer array such that  x[indices]  aligns with smoothed
    """
    if window <= 1:
        return y.copy(), np.arange(len(y))
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1
    kernel   = np.ones(w) / w
    smoothed = np.convolve(y, kernel, mode='valid')     # length: n - w + 1
    half     = (w - 1) // 2
    indices  = np.arange(half, half + len(smoothed))
    return smoothed, indices


def _window_pts(times: np.ndarray, window_fs: float) -> int:
    """Convert a smoothing window in fs to the nearest number of frames."""
    if len(times) < 2:
        return 1
    dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
    return max(3, int(window_fs / dt))



# Helpers


def make_label(ensemble: str, temp_start: float,
               temp_end: float = None, n_steps: int = 0) -> str:
    steps_str = f"{n_steps / 1000:.0f}k"
    if temp_end is not None:
        return f"{ensemble.upper()}_{temp_start:.0f}-{temp_end:.0f}K_{steps_str}steps"
    return f"{ensemble.upper()}_T{temp_start:.0f}K_{steps_str}steps"


def _save(fig: plt.Figure, save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")



# NVE / NVT diagnostic plots


def plot_energy(times, kinetic_energy, potential_energy, total_energy,
                label="", save_dir="."):
    fn = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, kinetic_energy,   label='Kinetic',   color='tab:orange')
    ax.plot(times, potential_energy, label='Potential', color='tab:blue')
    ax.plot(times, total_energy,     label='Total',     color='tab:green', lw=2)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title(f'Energy — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, f"energy_{fn}.png")


def plot_temperature(times, temperature, label="",
                     target_temp_k=None, save_dir="."):
    t_mean = float(np.mean(temperature))
    t_std  = float(np.std(temperature))
    fn = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, temperature, color='tab:red', lw=0.7, label='T(t)')
    ax.axhline(t_mean, color='tab:red', ls='--', lw=1.0,
               label=f'<T> = {t_mean:.2f} +/- {t_std:.2f} K')
    if target_temp_k is not None:
        ax.axhline(target_temp_k, color='black', ls=':', lw=1.0,
                   label=f'T_target = {target_temp_k:.1f} K')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, f"temperature_{fn}.png")


def plot_x_component(times, positions, label="",
                     particle_index=0, save_dir="."):
    x  = positions[:, particle_index, 0]
    fn = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, x, color='tab:blue', lw=0.7)
    ax.axhline(np.mean(x), color='gray', ls='--', lw=0.9,
               label=f'<x> = {np.mean(x):.3f} A')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('x [A]')
    ax.set_title(f'x component — atom {particle_index} — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, f"x_component_atom{particle_index}_{fn}.png")


def plot_all(trajectory, label="", target_temp_k=None,
             particle_index=0, save_dir="."):
    plot_energy(trajectory['times'], trajectory['kinetic_energy'],
                trajectory['potential_energy'], trajectory['total_energy'],
                label=label, save_dir=save_dir)
    plot_temperature(trajectory['times'], trajectory['temperature'],
                     label=label, target_temp_k=target_temp_k,
                     save_dir=save_dir)
    plot_x_component(trajectory['times'], trajectory['positions'],
                     label=label, particle_index=particle_index,
                     save_dir=save_dir)



# VACF


def plot_vacf(times:    np.ndarray,
              vacf:     np.ndarray,
              tau_v:    float = None,
              label:    str   = "",
              save_dir: str   = ".") -> None:
    filename_label = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, vacf, color='tab:purple', linewidth=1.0, label='C(t)')
    ax.axhline(0,       color='black', linewidth=0.8, linestyle='--')
    ax.axhline(1/np.e,  color='red',   linewidth=0.8, linestyle='--',
               label='1/e threshold')
    if tau_v is not None:
        ax.axvline(tau_v, color='red', linewidth=1.0, linestyle=':',
                   label=f'tau_v = {tau_v:.1f} fs')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('C(t)')
    ax.set_title(f'Velocity autocorrelation function — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(2000, float(times[-1])))
    fig.tight_layout()
    _save(fig, save_dir, f"vacf_{filename_label}.png")


# Multi-frequency thermostat plot


def plot_temperature_multi(times_list, temp_list, freq_list, target_temp,
                           eta_c_safe=None, label="", save_dir="."):

    fn     = label.replace(" ", "_")
    n_runs = len(freq_list)
    cmap   = plt.get_cmap("tab20", n_runs)

    means = np.array([float(np.mean(t)) for t in temp_list])
    stds  = np.array([float(np.std(t))  for t in temp_list])


    fig, ax = plt.subplots(3, 1, figsize=(11, 12))

    # Distribute the temperature traces across the 3 subplots (2 traces per subplot)
    for i in range(3):
        # Get the data for the two traces in this subplot
        times1, temp1, freq1 = times_list[2*i], temp_list[2*i], freq_list[2*i]
        times2, temp2, freq2 = times_list[2*i + 1], temp_list[2*i + 1], freq_list[2*i + 1]

        # Plot the first trace on the current subplot
        color1 = cmap(2*i)
        ax[i].scatter(times1, temp1, color=color1, alpha=0.5, s=0.5,
                      label=f'eta_c={freq1:.1e} fs^-1  <T>={means[2*i]:.1f} K  s={stds[2*i]:.1f} K')

        # Plot the second trace on the current subplot
        color2 = cmap(2*i + 1)
        ax[i].scatter(times2, temp2, color=color2, alpha=0.5, s=0.5,
                      label=f'eta_c={freq2:.1e} fs^-1  <T>={means[2*i + 1]:.1f} K  s={stds[2*i + 1]:.1f} K')

        # Add horizontal line for target temperature
        ax[i].axhline(target_temp, color='black', lw=1.0, ls=':',
                     label=f'T_target = {target_temp:.0f} K')

        # Set labels and grid for the current subplot
        ax[i].set_xlabel('Time [fs]')
        ax[i].set_ylabel('Temperature [K]')
        ax[i].legend(fontsize=7, loc='upper right')
        ax[i].grid(True, alpha=0.3)

    # Set the title for the entire figure
    fig.suptitle(f'Andersen thermostat — frequency sweep — {label}')

    fig.tight_layout()
    _save(fig, save_dir, f"temperature_multi_{fn}.png")



# Heating-ramp plots

def plot_ramp_temperature(times, temperature, target_temp, label="", save_dir="."):
    fn = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times * 1e-3, temperature, color='tab:red', lw=0.6, alpha=0.6,
            label='T instantaneous')
    ax.plot(times * 1e-3, target_temp, color='black',   lw=1.4, ls='--',
            label='T target (ramp)')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature vs time — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, f"ramp_T_vs_t_{fn}.png")


def plot_caloric_curve(target_temp, total_energy, potential_energy,
                       label="", save_dir=".", smooth_window=10):
    """
    Caloric curve: E_tot and E_pot vs thermostat target temperature.

    Raw data are shown as a faint scatter; the moving-average overlay uses
    _smooth_valid (mode='valid') to avoid the zero-padding artifacts that
    arise with mode='same' at the first and last floor(window/2) points
    (Press et al., Numerical Recipes, §13.4).

    Parameters
    ----------
    smooth_window : number of consecutive frames averaged (default 10).
                    The window is forced odd internally for symmetry.
    """
    fn = label.replace(" ", "_")

    # Edge-safe smoothing: returns arrays shorter by (window-1) points,
    # and the index array to slice target_temp consistently.
    smooth_E_tot, idx = _smooth_valid(total_energy,     smooth_window)
    smooth_E_pot, _   = _smooth_valid(potential_energy, smooth_window)
    T_smooth = target_temp[idx]

    fig, axes = plt.subplots(2, 1, figsize=(8, 11), sharex=True)

    ax1 = axes[0]
    ax1.scatter(target_temp, total_energy,
                color='tab:green', alpha=0.25, s=0.4, label='E_tot (raw)')
    ax1.plot(T_smooth, smooth_E_tot,
             color='darkgreen', lw=1.4,
             label=f'E_tot (moving avg, w={smooth_window})')
    ax1.set_ylabel('Total energy [eV]')
    ax1.set_title(f'Caloric curve — {label}')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='major', alpha=0.3)
    ax1.grid(True, which='minor', alpha=0.1)

    ax2 = axes[1]
    ax2.scatter(target_temp, potential_energy,
                color='tab:blue', alpha=0.25, s=0.4, label='E_pot (raw)')
    ax2.plot(T_smooth, smooth_E_pot,
             color='darkblue', lw=1.4,
             label=f'E_pot (moving avg, w={smooth_window})')
    ax2.set_ylabel('Potential energy [eV]')
    ax2.set_xlabel('Target temperature [K]')
    ax2.legend(loc='upper left')
    ax2.grid(True, which='major', alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.1)

    t_min = float(np.min(target_temp))
    t_max = float(np.max(target_temp))
    ax2.set_xticks(np.arange(t_min, t_max + 1, 2))           # major every 2 K
    ax2.set_xticks(np.arange(t_min, t_max + 1, 0.5), minor=True)  # minor every 0.5 K

    fig.tight_layout()
    _save(fig, save_dir, f"caloric_curve_{fn}.png")


def plot_ramp_all(trajectory, label="", save_dir="."):
    plot_ramp_temperature(trajectory['times'], trajectory['temperature'],
                          trajectory['target_temp'], label=label, save_dir=save_dir)
    plot_caloric_curve(trajectory['target_temp'],
                       trajectory['total_energy'],
                       trajectory['potential_energy'],
                       label=label, save_dir=save_dir)