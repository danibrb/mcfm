"""
Visualization utilities for MD trajectory data.

All functions save a PNG to `save_dir` and close the figure.

Smoothing
---------
A plain centred moving average (np.convolve, mode='same') is used
everywhere.  The window is always expressed in femtoseconds and converted
to the number of saved frames at runtime, so results are independent of
the save interval chosen.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """
    Centred moving average with a window of `window` points.
    Edge effects: the convolution uses mode='same', so the first and last
    window//2 points are averages over fewer samples (numpy pads with
    zeros implicitly — acceptable for visualisation purposes).
    """
    if window <= 1:
        return y.copy()
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1          # keep odd so the window is symmetric
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='same')


def _window_pts(times: np.ndarray, window_fs: float) -> int:
    """Convert a smoothing window in fs to the nearest number of frames."""
    if len(times) < 2:
        return 1
    dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
    return max(3, int(window_fs / dt))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NVE / NVT diagnostic plots
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# VACF
# ---------------------------------------------------------------------------

def plot_vacf(times, vacf, tau_v=None, label="", save_dir="."):
    fn = label.replace(" ", "_")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, vacf, color='tab:purple', lw=1.0, label='C(t)')
    ax.axhline(0,      color='black', lw=0.8, ls='--')
    ax.axhline(1/np.e, color='red',   lw=0.8, ls='--', label='1/e threshold')
    if tau_v is not None:
        ax.axvline(tau_v, color='red', lw=1.0, ls=':',
                   label=f'tau_v = {tau_v:.1f} fs')
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('C(t)')
    ax.set_title(f'Velocity autocorrelation — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(2000, float(times[-1])))
    fig.tight_layout()
    _save(fig, save_dir, f"vacf_{fn}.png")


# ---------------------------------------------------------------------------
# Multi-frequency thermostat plot
# ---------------------------------------------------------------------------

def plot_temperature_multi(times_list, temp_list, freq_list, target_temp,
                           nu_safe=None, label="", save_dir="."):
    """
    Two-panel figure.

    Top — temperature traces with a fixed 500 fs smoothing window so the
          comparison between frequencies is fair.  The raw signal (low
          opacity) shows the actual fluctuation amplitude, which is
          suppressed at high frequency (over-thermostatting).

    Bottom — <T> ± sigma(T) vs collision frequency on a log scale.
    """
    fn     = label.replace(" ", "_")
    n_runs = len(freq_list)
    cmap   = plt.get_cmap("tab10", n_runs)
    SMOOTH_FS = 500.0

    means = np.array([float(np.mean(t)) for t in temp_list])
    stds  = np.array([float(np.std(t))  for t in temp_list])

    fig, axes = plt.subplots(2, 1, figsize=(11, 9),
                             gridspec_kw={'height_ratios': [3, 2]})

    ax = axes[0]
    for i, (times, temp, freq) in enumerate(zip(times_list, temp_list, freq_list)):
        color = cmap(i)
        npts  = _window_pts(times, SMOOTH_FS)
        t_sm  = _smooth(temp, npts)
        ax.plot(times, temp,  color=color, alpha=0.12, lw=0.5)
        ax.plot(times, t_sm,  color=color, lw=1.7,
                label=f'nu={freq:.1e} fs^-1  <T>={means[i]:.1f} K  s={stds[i]:.1f} K')
    ax.axhline(target_temp, color='black', lw=1.0, ls=':',
               label=f'T_target = {target_temp:.0f} K')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Andersen thermostat — frequency sweep — {label}')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(freq_list, means, yerr=stds, fmt='o-',
                 color='tab:red', capsize=4, lw=1.4,
                 label='<T> +/- sigma(T)')
    ax2.axhline(target_temp, color='black', lw=0.9, ls=':',
                label=f'T_target = {target_temp:.0f} K')
    if nu_safe is not None:
        ax2.axvline(nu_safe, color='steelblue', lw=1.2, ls='--',
                    label=f'nu_safe = {nu_safe:.2e} fs^-1')
    ax2.set_xscale('log')
    ax2.set_xlabel('Collision frequency nu [fs^-1]')
    ax2.set_ylabel('Temperature [K]')
    ax2.set_title('Mean and std-dev of T vs nu')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    _save(fig, save_dir, f"temperature_multi_{fn}.png")


# ---------------------------------------------------------------------------
# Heating-ramp plots
# ---------------------------------------------------------------------------

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
                       label="", save_dir="."):
    """
    Three-panel caloric curve: E_tot, E_pot, and C_v = dE_tot/dT.

    Window for energy smoothing: 3% of the temperature range.
    Window for C_v: one-third of the energy window (narrower keeps
    the transition peak sharp).
    """
    from constants import KB_EV
    N_ATOMS = 38

    fn    = label.replace(" ", "_")
    n_pts = len(target_temp)

    # Energy smoothing: window = 3% of total frames
    win_e  = max(11, n_pts // 33)
    e_tot_sm = _smooth(total_energy,     win_e)
    e_pot_sm = _smooth(potential_energy, win_e)

    # Heat capacity C_v = dE_tot/dT, smoothed with narrower window
    dT = np.gradient(target_temp)
    dE = np.gradient(e_tot_sm)
    with np.errstate(invalid='ignore', divide='ignore'):
        cv_raw = np.where(np.abs(dT) > 1e-12, dE / dT, 0.0)
    win_cv      = max(5, win_e // 3)
    cv_sm       = _smooth(cv_raw, win_cv)
    cv_per_atom = cv_sm / (N_ATOMS * KB_EV)

    peak_idx = int(np.argmax(cv_per_atom))
    T_peak   = float(target_temp[peak_idx])
    cv_peak  = float(cv_per_atom[peak_idx])

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    ax = axes[0]
    ax.plot(target_temp, total_energy, color='tab:green', alpha=0.2, lw=0.5)
    ax.plot(target_temp, e_tot_sm,     color='tab:green', lw=1.8,
            label='E_tot (smoothed)')
    ax.set_ylabel('Total energy [eV]')
    ax.set_title(f'Caloric curve — {label}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(target_temp, potential_energy, color='tab:blue', alpha=0.2, lw=0.5)
    ax.plot(target_temp, e_pot_sm,         color='tab:blue', lw=1.8,
            label='E_pot (smoothed)')
    ax.set_ylabel('Potential energy [eV]')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(target_temp, cv_per_atom, color='tab:red', lw=1.8,
            label='C_v / (N k_B)')
    ax.axvline(T_peak, color='tab:red', lw=1.0, ls='--',
               label=f'T_transition ~ {T_peak:.1f} K')
    ax.axhline(0, color='black', lw=0.5, ls=':')
    ax.set_xlabel('T target [K]')
    ax.set_ylabel('C_v / (N k_B)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    span = float(target_temp[-1] - target_temp[0])
    ax.annotate(f'peak: {cv_peak:.1f}\nT ~ {T_peak:.1f} K',
                xy=(T_peak, cv_peak),
                xytext=(T_peak + span * 0.05, cv_peak * 0.85),
                arrowprops=dict(arrowstyle='->', color='tab:red'),
                fontsize=8, color='tab:red')

    fig.tight_layout()
    _save(fig, save_dir, f"caloric_curve_{fn}.png")
    print(f"  Transition peak: T ~ {T_peak:.1f} K  C_v/Nk_B = {cv_peak:.2f}")


def plot_ramp_all(trajectory, label="", save_dir="."):
    plot_ramp_temperature(trajectory['times'], trajectory['temperature'],
                          trajectory['target_temp'], label=label, save_dir=save_dir)
    plot_caloric_curve(trajectory['target_temp'],
                       trajectory['total_energy'],
                       trajectory['potential_energy'],
                       label=label, save_dir=save_dir)


# ---------------------------------------------------------------------------
# Structural observables: Lindemann and cluster size
# ---------------------------------------------------------------------------

def plot_lindemann(target_temp, ldi, label="", save_dir="."):
    """
    Lindemann index delta(T) vs temperature.
    Melting criterion: delta > 0.10-0.15.
    """
    fn = label.replace(" ", "_")

    valid = ~np.isnan(ldi)
    T_v   = target_temp[valid]
    ldi_v = ldi[valid]
    win   = max(11, len(ldi_v) // 30)
    ldi_sm = _smooth(ldi_v, win)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(T_v, ldi_v,  color='tab:purple', alpha=0.3, lw=0.6)
    ax.plot(T_v, ldi_sm, color='tab:purple', lw=2.0,
            label='Lindemann index delta')
    ax.axhspan(0.10, 0.15, color='gold', alpha=0.35,
               label='Melting criterion 0.10 - 0.15')
    ax.set_xlabel('T target [K]')
    ax.set_ylabel('Lindemann index delta')
    ax.set_title(f'Lindemann index — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, f"lindemann_{fn}.png")


def plot_cluster_size(target_temp, cluster_sizes, n_atoms, label="", save_dir="."):
    """
    Largest cluster size and bound fraction vs temperature.
    Evaporation onset visible as monotonic decrease.
    """
    fn       = label.replace(" ", "_")
    fraction = cluster_sizes / n_atoms
    win      = max(11, len(cluster_sizes) // 30)
    cs_sm    = _smooth(cluster_sizes.astype(float), win)
    fr_sm    = _smooth(fraction,                    win)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(target_temp, cluster_sizes, color='tab:teal', alpha=0.3, lw=0.6)
    ax.plot(target_temp, cs_sm,         color='tab:teal', lw=2.0,
            label='N_cluster (smoothed)')
    ax.axhline(n_atoms, color='gray', lw=0.8, ls='--',
               label=f'N_total = {n_atoms}')
    ax.set_ylabel('Largest cluster size')
    ax.set_title(f'Cluster size — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(target_temp, fraction, color='tab:olive', alpha=0.3, lw=0.6)
    ax.plot(target_temp, fr_sm,    color='tab:olive', lw=2.0,
            label='N_cluster / N_total')
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('T target [K]')
    ax.set_ylabel('Bound fraction')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, f"cluster_size_{fn}.png")


def plot_transition_summary(target_temp, potential_energy, total_energy,
                            ldi, cluster_sizes, n_atoms,
                            label="", save_dir="."):
    """
    Four-panel summary: E_pot, C_v, Lindemann index, bound fraction —
    all on a shared temperature axis.
    """
    from constants import KB_EV
    N_ATOMS = n_atoms

    fn    = label.replace(" ", "_")
    n_pts = len(target_temp)
    win_e = max(11, n_pts // 33)

    e_tot_sm = _smooth(total_energy,     win_e)
    e_pot_sm = _smooth(potential_energy, win_e)

    dT = np.gradient(target_temp)
    dE = np.gradient(e_tot_sm)
    with np.errstate(invalid='ignore', divide='ignore'):
        cv_raw = np.where(np.abs(dT) > 1e-12, dE / dT, 0.0)
    cv_sm       = _smooth(cv_raw, max(5, win_e // 3))
    cv_per_atom = cv_sm / (N_ATOMS * KB_EV)
    peak_idx    = int(np.argmax(cv_per_atom))
    T_peak      = float(target_temp[peak_idx])

    valid  = ~np.isnan(ldi)
    T_v    = target_temp[valid]
    ldi_sm = _smooth(ldi[valid], max(11, int(valid.sum()) // 30))

    fraction = cluster_sizes / N_ATOMS
    fr_sm    = _smooth(fraction.astype(float), max(11, n_pts // 30))

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)

    ax = axes[0]
    ax.plot(target_temp, potential_energy, color='tab:blue', alpha=0.2, lw=0.5)
    ax.plot(target_temp, e_pot_sm,         color='tab:blue', lw=2.0,
            label='E_pot (smoothed)')
    ax.axvline(T_peak, color='tab:red', lw=1.0, ls='--',
               label=f'T_peak = {T_peak:.1f} K')
    ax.set_ylabel('E_pot [eV]')
    ax.set_title(f'Phase transition summary — {label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(target_temp, cv_per_atom, color='tab:red', lw=2.0,
            label='C_v / (N k_B)')
    ax.axvline(T_peak, color='tab:red', lw=1.0, ls='--',
               label=f'T_peak = {T_peak:.1f} K')
    ax.axhline(0, color='black', lw=0.5, ls=':')
    ax.set_ylabel('C_v / (N k_B)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(T_v, ldi[valid],  color='tab:purple', alpha=0.2, lw=0.5)
    ax.plot(T_v, ldi_sm,      color='tab:purple', lw=2.0,
            label='Lindemann delta')
    ax.axhspan(0.10, 0.15, color='gold', alpha=0.35, label='Melting 0.10-0.15')
    ax.axvline(T_peak, color='tab:red', lw=1.0, ls='--')
    ax.set_ylabel('Lindemann delta')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(target_temp, fraction, color='tab:teal', alpha=0.2, lw=0.5)
    ax.plot(target_temp, fr_sm,    color='tab:teal', lw=2.0,
            label='N_cluster / N_total')
    ax.axvline(T_peak, color='tab:red', lw=1.0, ls='--')
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('T target [K]')
    ax.set_ylabel('Bound fraction')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, f"transition_summary_{fn}.png")
    print(f"  Summary saved.  T_peak(C_v) = {T_peak:.1f} K")