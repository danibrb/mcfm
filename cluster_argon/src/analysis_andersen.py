"""
Andersen thermostat analysis.

1. VACF analysis
   Runs a short NVE trajectory, computes C(t), and extracts the velocity
   decorrelation time tau_v from the 1/e decay.

2. Thermostat frequency sweep
   Runs NVT at six frequencies spanning [0.001, 100] * eta_c_safe 
"""

import os

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ, OUTPUT_DIR_AND,
                             MASS_AMU, TEMP_INIT_K, TIMESTEP_FS, RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params
from initialization import initialize_velocities
from lj_potential   import warmup_jit
from nve            import run_nve
from nvt            import run_nvt
from observables    import compute_vacf
from visualization  import plot_vacf, plot_temperature_multi


VACF_STEPS    = 5_000
VACF_INTERVAL = 1          # save every step for a smooth C(t)

FREQ_RATIO = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]   # multiples of eta_c_safe

NVT_STEPS    = 100_000
NVT_INTERVAL = 10          # dense save so raw fluctuations are visible


def _estimate_tau(times: np.ndarray, vacf: np.ndarray) -> float:
    """
    First time at which C(t) falls to 1/e.
    """
    threshold = 1.0 / np.e
    for k in range(1, len(vacf)):
        if vacf[k] <= threshold:
            t0, t1 = times[k - 1], times[k]
            c0, c1 = vacf[k - 1], vacf[k]
            return float(t0 + (threshold - c0) * (t1 - t0) / (c1 - c0))
    # C(t) never reached 1/e — return last time as lower bound
    return float(times[-1])


def main() -> None:
    os.makedirs(OUTPUT_DIR_AND, exist_ok=True)

    # 1. Load system
    n_atoms, positions, atom_names, _ = read_xyz(FILENAME_XYZ_IN)
    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"Loaded {n_atoms} atoms")
    print(f"LJ: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} A")

    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 2. Short NVE run for VACF
    rng_vacf = np.random.default_rng(RANDOM_SEED)
    vel_vacf = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_vacf)

    print(f"\nVACF run: {VACF_STEPS} NVE steps, save every step")
    traj_vacf = run_nve(
        positions.copy(), vel_vacf,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, VACF_STEPS,
        save_interval=VACF_INTERVAL,
    )

    vacf  = compute_vacf(traj_vacf['velocities'])
    times = traj_vacf['times']

    tau_v   = _estimate_tau(times, vacf)
    eta_c_safe = 0.1 / tau_v          # safe collision frequency  [1/fs]

    print(f"\nVelocity decorrelation time  tau_v   = {tau_v:.1f} fs")
    print(f"Safe Andersen frequency      eta_c_safe  = {eta_c_safe:.4e} fs^-1")
    print(f"  (collision period = {1/eta_c_safe:.0f} fs  =  {1/(eta_c_safe*tau_v):.0f} * tau_v)")

    label_vacf = f"NVE_T{TEMP_INIT_K:.0f}K_{VACF_STEPS // 1000:.0f}ksteps"
    plot_vacf(times, vacf, tau_v=tau_v,
              label=label_vacf, save_dir=OUTPUT_DIR_AND)

    # 3. NVT frequency sweep
    freq_list  = np.array(FREQ_RATIO) * eta_c_safe
    times_list = []
    temp_list  = []

    for freq in freq_list:
        ratio = freq / eta_c_safe
        print(f"\nNVT: nu = {freq:.3e} fs^-1  ({ratio:.3g} x eta_c_safe),  "
              f"{NVT_STEPS} steps")
        rng_nvt = np.random.default_rng(RANDOM_SEED + 10)
        vel_nvt = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_nvt)

        traj_nvt = run_nvt(
            positions.copy(), vel_nvt,
            MASS_AMU, epsilon_ev, sigma_ang,
            TIMESTEP_FS, NVT_STEPS,
            target_temp_k=TEMP_INIT_K,
            collision_freq=freq,
            rng=rng_nvt,
            save_interval=NVT_INTERVAL,
        )
        times_list.append(traj_nvt['times'])
        temp_list.append(traj_nvt['temperature'])

    label_multi = f"NVT_T{TEMP_INIT_K:.0f}K_{NVT_STEPS // 1000:.0f}ksteps"
    plot_temperature_multi(
        times_list, temp_list, freq_list,
        eta_c_safe=eta_c_safe,
        target_temp=TEMP_INIT_K,
        label=label_multi,
        save_dir=OUTPUT_DIR_AND,
    )

    print(f"\nAll output saved to: {OUTPUT_DIR_AND}/")
    print(f"\nRecommended COLLISION_FREQ for config.py: {eta_c_safe:.4e} fs^-1")


if __name__ == "__main__":
    main()