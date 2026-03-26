"""
Andersen thermostat analysis.

This script performs two tasks:

1. VACF analysis (velocity autocorrelation function)
   Runs a short NVE trajectory, computes C(t), and estimates the velocity
   decorrelation time tau from the first zero crossing. This gives a
   physical upper bound for the Andersen collision frequency:

       nu << 1/tau

   A typical safe choice is nu < 0.1/tau.

2. Thermostat frequency comparison
   Runs NVT at several collision frequencies spanning a range around
   the estimate from step 1 and overlays the temperature traces so the
   effect of coupling strength is immediately visible.

"""

import os

import numpy as np

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ, OUTPUT_DIR_AND,
                             MASS_AMU, TEMP_INIT_K, TIMESTEP_FS,
                             RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params
from initialization import initialize_velocities
from lj_potential   import warmup_jit
from nve            import run_nve
from nvt            import run_nvt
from observables    import compute_vacf
from visualization  import plot_vacf, plot_temperature_multi



# NVE run length for VACF — needs to be long enough to capture
# the full decay of C(t), but short NVE is fine since we only
# need the velocity decorrelation time
VACF_STEPS    = 5000
VACF_INTERVAL = 1      # save every step for a smooth VACF curve

# Collision frequencies to compare in the NVT runs
FREQ_RATIO = [0.01, 0.1, 1, 10, 100, 1000]

NVT_STEPS    = 20000
NVT_INTERVAL = 50


def main() -> None:
    os.makedirs(OUTPUT_DIR_AND, exist_ok=True)

    # 1. Read structure and LJ parameters
    n_atoms, positions, atom_names, _ = read_xyz(FILENAME_XYZ_IN)
    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"Loaded {n_atoms} atoms")
    print(f"LJ parameters: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} Å")

    # Warm up JIT once before all timed runs
    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 2. VACF — run short NVE, save every step
    rng_vacf   = np.random.default_rng(RANDOM_SEED)
    vel_vacf   = initialize_velocities(n_atoms, MASS_AMU, TEMP_INIT_K, rng_vacf)

    print(f"\nRunning NVE for VACF: {VACF_STEPS} steps, save every step")
    traj_vacf = run_nve(
        positions.copy(), vel_vacf,
        MASS_AMU, epsilon_ev, sigma_ang,
        TIMESTEP_FS, VACF_STEPS,
        save_interval=VACF_INTERVAL,
    )

    vacf  = compute_vacf(traj_vacf['velocities'])
    times = traj_vacf['times']

    # Estimate relaxation time (time to decay to 1/e)
    idx = np.where(vacf < 1/np.e)[0]
    relaxation_time = idx[0] * TIMESTEP_FS

    print(f"Estimated relaxation time: {relaxation_time} fs")

    # Set collision frequency
    collision_freq = 1 / (10 * relaxation_time)
    print(f"Suggested collision frequency: {collision_freq:.8f} fs^-1")

    label_vacf = f"NVE_T{TEMP_INIT_K:.0f}K_{VACF_STEPS//1000:.0f}ksteps"
    plot_vacf(times, vacf, label=label_vacf, save_dir=OUTPUT_DIR_AND)

    # 3. NVT runs at different collision frequencies
    times_list = []
    temp_list  = []

    freq_list = np.array(FREQ_RATIO) * collision_freq

    for freq in freq_list:
        print(f"\nRunning NVT: ν = {freq:.8f} fs⁻¹,  {NVT_STEPS} steps")
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

    label_multi = f"NVT_T{TEMP_INIT_K:.0f}K_{NVT_STEPS//1000:.0f}ksteps"
    plot_temperature_multi(
        times_list, temp_list, freq_list,
        target_temp=TEMP_INIT_K,
        label=label_multi,
        save_dir=OUTPUT_DIR_AND,
    )

    print(f"\nAll output saved to: {OUTPUT_DIR_AND}/")


if __name__ == "__main__":
    main()