"""
Long heating-ramp simulation with checkpoint save/resume and ETA display.

Usage
-----
First run (or after deleting the checkpoint):
    python simulation_long.py

Resume after interruption:
    python simulation_long.py          # automatically picks up the checkpoint

The simulation can be interrupted cleanly at any time with Ctrl-C; the
last written checkpoint is preserved and the partial trajectory is plotted.

Checkpoint format
-----------------
A single NumPy .npz file (CHECKPOINT_FILE) stores:
    step             : last completed step index
    positions        : current particle positions  [Å]
    velocities       : current particle velocities [Å/fs]
    forces           : forces at current positions [eV/Å]
    times            : saved time array so far     [fs]
    kinetic_energy   : saved K array so far        [eV]
    potential_energy : saved U array so far        [eV]
    temperature      : saved T array so far        [K]
    target_temp      : saved T_target array so far [K]
    positions_traj   : saved position frames       [Å]
    velocities_traj  : saved velocity frames       [Å/fs]

The checkpoint is written every CHECKPOINT_INTERVAL steps and at
KeyboardInterrupt.  On resume the RNG state is seeded deterministically
from the checkpoint step so results are reproducible.
"""

import os
import sys
import time

import numpy as np
from tqdm import tqdm

from constants      import KB_EV
from config         import (FILENAME_XYZ_IN, FILENAME_LJ, OUTPUT_DIR_RAMP,
                             MASS_AMU, TEMP_RAMP_START_K, TEMP_RAMP_END_K,
                             TIMESTEP_FS, N_STEPS_RAMP, SAVE_INTERVAL_RAMP,
                             COLLISION_FREQ, RANDOM_SEED)
from io_handler     import read_xyz, read_lj_params, write_xyz_trajectory
from initialization import initialize_velocities
from lj_potential   import warmup_jit, compute_forces_and_potential
from integrator     import velocity_verlet_step
from initialization import kinetic_energy
from observables    import temperature
from thermostat     import andersen
from visualization  import plot_ramp_all, make_label


# ---------------------------------------------------------------------------
# Configuration — override here or in config.py
# ---------------------------------------------------------------------------

# How often to write a checkpoint (in MD steps)
CHECKPOINT_INTERVAL = 50_000

# Checkpoint file path
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR_RAMP, "checkpoint.npz")


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(path:          str,
                    step:          int,
                    positions:     np.ndarray,
                    velocities:    np.ndarray,
                    forces:        np.ndarray,
                    acc_times:     list,
                    acc_kin:       list,
                    acc_pot:       list,
                    acc_temp:      list,
                    acc_tgt:       list,
                    acc_pos:       list,
                    acc_vel:       list) -> None:
    """Write a checkpoint .npz file atomically (write then rename)."""
    tmp = path + ".tmp"
    np.savez_compressed(
        tmp,
        step             = np.array(step),
        positions        = positions,
        velocities       = velocities,
        forces           = forces,
        times            = np.array(acc_times,  dtype=np.float64),
        kinetic_energy   = np.array(acc_kin,    dtype=np.float64),
        potential_energy = np.array(acc_pot,    dtype=np.float64),
        temperature      = np.array(acc_temp,   dtype=np.float64),
        target_temp      = np.array(acc_tgt,    dtype=np.float64),
        positions_traj   = np.array(acc_pos,    dtype=np.float32),
        velocities_traj  = np.array(acc_vel,    dtype=np.float32),
    )
    os.replace(tmp, path)   # atomic on all platforms


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint .npz and return a plain dict."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ramp_resumable(n_atoms:        int,
                       positions:      np.ndarray,
                       velocities:     np.ndarray,
                       forces:         np.ndarray,
                       mass_amu:       float,
                       epsilon_ev:     float,
                       sigma_ang:      float,
                       dt_fs:          float,
                       n_steps:        int,
                       temp_start_k:   float,
                       temp_end_k:     float,
                       collision_freq: float,
                       start_step:     int,
                       rng:            np.random.Generator,
                       save_interval:  int,
                       checkpoint_interval: int,
                       checkpoint_path: str,
                       acc_times:      list,
                       acc_kin:        list,
                       acc_pot:        list,
                       acc_temp:       list,
                       acc_tgt:        list,
                       acc_pos:        list,
                       acc_vel:        list) -> dict:
    """
    Inner loop for the heating ramp, supporting resume from an arbitrary
    start_step and incremental accumulation into pre-populated lists.

    Returns the same trajectory dict as run_heating_ramp.
    """
    target_temps = np.linspace(temp_start_k, temp_end_k, n_steps + 1)

    total_time_fs = n_steps * dt_fs
    heating_rate  = (temp_end_k - temp_start_k) / (total_time_fs * 1e-3)
    remaining     = n_steps - start_step

    print(f"  Resuming from step {start_step} / {n_steps}")
    print(f"  Remaining: {remaining} steps  "
          f"({remaining * dt_fs * 1e-3:.1f} ps)")
    print(f"  Heating rate: {heating_rate:.4f} K/ps")

    # ETA bookkeeping
    t_wall_start = time.perf_counter()
    steps_done   = 0

    try:
        with tqdm(total=remaining, desc="Ramp", unit="step",
                  initial=0) as pbar:
            for step in range(start_step + 1, n_steps + 1):
                T_target = float(target_temps[step])

                positions, velocities, forces, U = velocity_verlet_step(
                    positions, velocities, forces,
                    mass_amu, dt_fs, epsilon_ev, sigma_ang,
                )
                velocities = andersen(
                    velocities, mass_amu, T_target,
                    collision_freq, dt_fs, rng,
                )

                steps_done += 1

                if step % save_interval == 0:
                    K = kinetic_energy(velocities, mass_amu)
                    T = temperature(velocities, mass_amu)
                    acc_times.append(step * dt_fs)
                    acc_kin.append(K)
                    acc_pot.append(U)
                    acc_temp.append(T)
                    acc_tgt.append(T_target)
                    acc_pos.append(positions.copy())
                    acc_vel.append(velocities.copy())

                    # ETA after first save
                    elapsed = time.perf_counter() - t_wall_start
                    rate    = steps_done / elapsed if elapsed > 0 else 1.0
                    eta_s   = (remaining - steps_done) / rate
                    pbar.set_postfix(
                        T_inst=f"{T:.1f}",
                        T_tgt=f"{T_target:.1f}",
                        ETA=_fmt_time(eta_s),
                    )

                # Checkpoint
                if step % checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_path, step,
                        positions, velocities, forces,
                        acc_times, acc_kin, acc_pot,
                        acc_temp, acc_tgt, acc_pos, acc_vel,
                    )

                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — saving checkpoint and partial results...")
        save_checkpoint(
            checkpoint_path, step - 1,
            positions, velocities, forces,
            acc_times, acc_kin, acc_pot,
            acc_temp, acc_tgt, acc_pos, acc_vel,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    elapsed_total = time.perf_counter() - t_wall_start
    print(f"Elapsed: {_fmt_time(elapsed_total)}  "
          f"({elapsed_total / max(steps_done, 1) * 1e3:.3f} ms/step)")

    kin_arr = np.array(acc_kin,  dtype=np.float64)
    pot_arr = np.array(acc_pot,  dtype=np.float64)

    return {
        'times':            np.array(acc_times,  dtype=np.float64),
        'positions':        np.array(acc_pos,    dtype=np.float32),
        'velocities':       np.array(acc_vel,    dtype=np.float32),
        'kinetic_energy':   kin_arr,
        'potential_energy': pot_arr,
        'total_energy':     kin_arr + pot_arr,
        'temperature':      np.array(acc_temp,   dtype=np.float64),
        'target_temp':      np.array(acc_tgt,    dtype=np.float64),
        'heating_rate_kps': heating_rate,
    }


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds as h:mm:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR_RAMP, exist_ok=True)

    # 1. Read structure and LJ parameters
    n_atoms, positions, atom_names, comment = read_xyz(FILENAME_XYZ_IN)
    print(f"Loaded {n_atoms} atoms  ({comment})")

    params     = read_lj_params(FILENAME_LJ)
    epsilon_ev = params['epsilon_K'] * KB_EV
    sigma_ang  = params['sigma_ang']
    print(f"LJ: epsilon = {epsilon_ev:.6e} eV,  sigma = {sigma_ang:.4f} A")

    warmup_jit(epsilon_ev, sigma_ang, positions)

    # 2. Decide whether to resume from a checkpoint
    start_step   = 0
    acc_times    : list = []
    acc_kin      : list = []
    acc_pot      : list = []
    acc_temp     : list = []
    acc_tgt      : list = []
    acc_pos      : list = []
    acc_vel      : list = []

    if os.path.exists(CHECKPOINT_FILE):
        print(f"\nCheckpoint found: {CHECKPOINT_FILE}")
        ck = load_checkpoint(CHECKPOINT_FILE)
        start_step = int(ck['step'])
        positions  = ck['positions']
        velocities = ck['velocities']
        forces     = ck['forces']
        # Restore accumulated trajectory lists
        acc_times = list(ck['times'])
        acc_kin   = list(ck['kinetic_energy'])
        acc_pot   = list(ck['potential_energy'])
        acc_temp  = list(ck['temperature'])
        acc_tgt   = list(ck['target_temp'])
        acc_pos   = [ck['positions_traj'][i]
                     for i in range(len(ck['positions_traj']))]
        acc_vel   = [ck['velocities_traj'][i]
                     for i in range(len(ck['velocities_traj']))]
        # Seed RNG from the checkpoint step for reproducibility
        rng = np.random.default_rng(RANDOM_SEED + start_step)
        print(f"  Resuming from step {start_step} / {N_STEPS_RAMP}  "
              f"({start_step / N_STEPS_RAMP * 100:.1f}% complete)")
        print(f"  Trajectory frames already saved: {len(acc_times)}")

    else:
        print(f"\nNo checkpoint found — starting fresh run")
        rng        = np.random.default_rng(RANDOM_SEED)
        velocities = initialize_velocities(n_atoms, MASS_AMU,
                                           TEMP_RAMP_START_K, rng)
        forces, _  = compute_forces_and_potential(epsilon_ev, sigma_ang,
                                                  positions)

        # Estimate total wall-clock time from a short benchmark (200 steps)
        _estimate_runtime(positions, velocities, forces,
                          mass_amu=MASS_AMU,
                          epsilon_ev=epsilon_ev, sigma_ang=sigma_ang,
                          dt_fs=TIMESTEP_FS, n_steps=N_STEPS_RAMP,
                          collision_freq=COLLISION_FREQ,
                          temp_start_k=TEMP_RAMP_START_K,
                          temp_end_k=TEMP_RAMP_END_K,
                          rng=np.random.default_rng(RANDOM_SEED))

    # 3. Run (or resume) the heating ramp
    traj = run_ramp_resumable(
        n_atoms        = n_atoms,
        positions      = positions,
        velocities     = velocities,
        forces         = forces,
        mass_amu       = MASS_AMU,
        epsilon_ev     = epsilon_ev,
        sigma_ang      = sigma_ang,
        dt_fs          = TIMESTEP_FS,
        n_steps        = N_STEPS_RAMP,
        temp_start_k   = TEMP_RAMP_START_K,
        temp_end_k     = TEMP_RAMP_END_K,
        collision_freq = COLLISION_FREQ,
        start_step     = start_step,
        rng            = rng,
        save_interval  = SAVE_INTERVAL_RAMP,
        checkpoint_interval = CHECKPOINT_INTERVAL,
        checkpoint_path     = CHECKPOINT_FILE,
        acc_times = acc_times,
        acc_kin   = acc_kin,
        acc_pot   = acc_pot,
        acc_temp  = acc_temp,
        acc_tgt   = acc_tgt,
        acc_pos   = acc_pos,
        acc_vel   = acc_vel,
    )

    # 4. Save outputs
    label = make_label("ramp",
                       temp_start=TEMP_RAMP_START_K,
                       temp_end=TEMP_RAMP_END_K,
                       n_steps=N_STEPS_RAMP)

    xyz_file = f"trajectory_{label}.xyz"
    write_xyz_trajectory(
        os.path.join(OUTPUT_DIR_RAMP, xyz_file),
        traj['positions'], atom_names, traj['times'],
    )

    plot_ramp_all(traj, label=label, save_dir=OUTPUT_DIR_RAMP)

    # Remove checkpoint only when the run completed fully
    if len(traj['times']) > 0:
        final_step = N_STEPS_RAMP   # completed
        expected_frames = N_STEPS_RAMP // SAVE_INTERVAL_RAMP
        if len(traj['times']) >= expected_frames - 1:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                print("Checkpoint removed (run complete).")

    print(f"\nAll output saved to: {OUTPUT_DIR_RAMP}/")


# ---------------------------------------------------------------------------
# Runtime estimator
# ---------------------------------------------------------------------------

def _estimate_runtime(positions:     np.ndarray,
                      velocities:    np.ndarray,
                      forces:        np.ndarray,
                      mass_amu:      float,
                      epsilon_ev:    float,
                      sigma_ang:     float,
                      dt_fs:         float,
                      n_steps:       int,
                      collision_freq: float,
                      temp_start_k:  float,
                      temp_end_k:    float,
                      rng:           np.random.Generator,
                      bench_steps:   int = 200) -> None:
    """
    Run bench_steps steps, measure wall time, extrapolate to n_steps,
    and print an estimated total runtime before the main loop starts.
    """
    print(f"\nBenchmarking {bench_steps} steps to estimate runtime...")
    pos = positions.copy()
    vel = velocities.copy()
    frc = forces.copy()
    t0  = time.perf_counter()
    target_temps = np.linspace(temp_start_k, temp_end_k, bench_steps + 1)
    for i in range(1, bench_steps + 1):
        pos, vel, frc, _ = velocity_verlet_step(pos, vel, frc,
                                                mass_amu, dt_fs,
                                                epsilon_ev, sigma_ang)
        vel = andersen(vel, mass_amu, float(target_temps[i]),
                       collision_freq, dt_fs, rng)
    elapsed  = time.perf_counter() - t0
    rate     = bench_steps / elapsed
    eta_s    = n_steps / rate
    print(f"  Performance : {rate:.0f} steps/s  "
          f"({elapsed / bench_steps * 1e3:.3f} ms/step)")
    print(f"  Estimated total runtime: {_fmt_time(eta_s)}  "
          f"({eta_s / 3600:.2f} h)")
    print(f"  Total simulated time   : "
          f"{n_steps * dt_fs * 1e-3:.1f} ps\n")


if __name__ == "__main__":
    main()
