# Argon Cluster Molecular Dynamics

Classical molecular dynamics simulation of an Ar₃₈ cluster using the Lennard-Jones pair potential. Supports microcanonical (NVE) and canonical (NVT) ensembles with trajectory output compatible with VMD.

---

## Table of Contents

1. [Physics background](#physics-background)
2. [Unit system](#unit-system)
3. [Project structure](#project-structure)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Configuration](#configuration)
7. [Module reference](#module-reference)
   - [constants.py](#constantspy)
   - [config.py](#configpy)
   - [io_handler.py](#io_handlerpy)
   - [initialization.py](#initializationpy)
   - [lj_potential.py](#lj_potentialpy)
   - [integrator.py](#integratorpy)
   - [observables.py](#observablespy)
   - [nve.py](#nvepy)
   - [nvt.py](#nvtpy)
   - [thermostat.py](#thermostatpy)
   - [visualization.py](#visualizationpy)
   - [simulation.py](#simulationpy)
8. [Output files](#output-files)
9. [Adding a new thermostat](#adding-a-new-thermostat)
10. [References](#references)

---

## Physics background

The simulation integrates Newton's equations of motion for N atoms interacting via the Lennard-Jones 12-6 pair potential (Allen & Tildesley, 2017):

$$U(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$

where $\varepsilon$ is the well depth and $\sigma$ the collision diameter. For Argon the parameters are $\varepsilon/k_B = 119\,\text{K}$ and $\sigma = 3.4\,\text{Å}$ (Rahman, 1964).

Equations of motion are propagated with the **Velocity Verlet** algorithm (Swope et al., 1982), a symplectic second-order integrator that conserves energy to $O(\Delta t^2)$ per step.

Two statistical ensembles are implemented:

- **NVE** (microcanonical): constant number of atoms, volume, and energy. Temperature fluctuates freely.
- **NVT** (canonical): constant number of atoms, volume, and temperature. The **Andersen thermostat** (Andersen, 1980) maintains the target temperature by stochastically reassigning velocities from the Maxwell-Boltzmann distribution.

---

## Unit system

All internal quantities use the following consistent set of units:

| Quantity  | Unit                  | Symbol |
|-----------|-----------------------|--------|
| Length    | Angstrom              | Å      |
| Energy    | electronvolt          | eV     |
| Mass      | atomic mass unit      | amu    |
| Time      | femtosecond           | fs     |
| Force     | eV/Å                  |        |
| Velocity  | Å/fs                  |        |

Two conversion factors derived from these units are used throughout the code:

**Kinetic energy**:
$$KE\,[\text{eV}] = \frac{1}{2}\, m\,[\text{amu}] \cdot v^2\,[\text{Å}^2/\text{fs}^2] \cdot \underbrace{\frac{m_u \cdot (10^5\,\text{m/s})^2}{e}}_{\texttt{AMU\_ANG2\_FS2\_TO\_EV} \approx 103.64}$$

**Newton's second law**:
$$a\,[\text{Å/fs}^2] = \frac{F\,[\text{eV/Å}]}{m\,[\text{amu}]} \cdot \underbrace{\frac{1}{\texttt{AMU\_ANG2\_FS2\_TO\_EV}}}_{\texttt{FORCE\_CONV} \approx 9.65 \times 10^{-3}}$$

These two factors are exact inverses by dimensional consistency. No other unit conversions occur at runtime.

---

## Project structure

```
project/
├── simulation.py        # Main entry point
├── config.py            # All runtime parameters
├── constants.py         # Physical constants and unit conversions
├── io_handler.py        # Read/write XYZ and parameter files
├── initialization.py    # Velocity initialisation and kinetic energy
├── lj_potential.py      # Lennard-Jones potential and forces (Numba JIT)
├── integrator.py        # Velocity Verlet integrator
├── observables.py       # Thermodynamic observables (temperature)
├── nve.py               # Microcanonical ensemble runner
├── nvt.py               # Canonical ensemble runner
├── thermostat.py        # Thermostat implementations
├── visualization.py     # Matplotlib plotting functions
├── ar38_to.xyz          # Input structure (Ar₃₈ cluster)
├── parametri_lj.txt     # Lennard-Jones parameters
└── output/
    ├── nve/             # NVE results (plots + trajectory)
    └── nvt/             # NVT results (plots + trajectory)
```

---

## Installation

Python 3.10 or later is required.

```bash
pip install numpy matplotlib tqdm numba
```

Numba is strongly recommended. Without it the code falls back to a pure NumPy implementation which is roughly 600x slower (see [lj_potential.py](#lj_potentialpy)).

---

## Quick start

```bash
python simulation.py
```

This runs both NVE and NVT simulations with the parameters defined in `config.py` and writes all results to `output/nve/` and `output/nvt/`.

To change the simulation, edit `config.py` — no other file needs to be touched for routine use.

---

## Configuration

All runtime parameters live in `config.py`.

| Parameter       | Default          | Description                                      |
|-----------------|------------------|--------------------------------------------------|
| `FILENAME_XYZ_IN`  | `ar38_to.xyz`    | Input structure file                             |
| `FILENAME_LJ`      | `parametri_lj.txt` | Lennard-Jones parameter file                   |
| `OUTPUT_DIR_NVE`   | `output/nve`     | Output directory for NVE results                 |
| `OUTPUT_DIR_NVT`   | `output/nvt`     | Output directory for NVT results                 |
| `MASS_AMU`         | `40.0`           | Atomic mass [amu]                                |
| `TEMP_INIT_K`      | `20.0`           | Initial temperature [K]                          |
| `TIMESTEP_FS`      | `5.0`            | Integration timestep [fs]                        |
| `N_STEPS`          | `10000`          | Number of MD steps                               |
| `SAVE_INTERVAL`    | `100`            | Save trajectory frame every N steps              |
| `THERMOSTAT`       | `"andersen"`     | Thermostat for NVT (see thermostat.py)           |
| `COLLISION_FREQ`   | `0.005`          | Andersen collision frequency [1/fs]              |
| `RANDOM_SEED`      | `1234567890`     | Random seed for reproducibility                  |

---

## Module reference

### `constants.py`

Physical constants and unit conversion factors derived from CODATA 2018 values. All other modules import from here — no numeric constants are defined elsewhere.

| Name                  | Value         | Description                          |
|-----------------------|---------------|--------------------------------------|
| `KB_SI`               | 1.380649e-23  | Boltzmann constant [J/K]             |
| `EV_TO_J`             | 1.602176634e-19 | Electronvolt to Joule [J/eV]       |
| `AMU_TO_KG`           | 1.66053906660e-27 | Atomic mass unit [kg/amu]        |
| `KB_EV`               | 8.6173e-5     | Boltzmann constant [eV/K]            |
| `AMU_ANG2_FS2_TO_EV`  | 103.6427      | Kinetic energy conversion factor     |
| `FORCE_CONV`          | 9.6485e-3     | Force-to-acceleration conversion     |

---

### `config.py`

Central configuration file. See [Configuration](#configuration) for the full parameter table. This is the **only file that should be edited** for routine simulation runs.

---

### `io_handler.py`

Input/output utilities for reading structure files and force-field parameters, and for writing trajectory files.

#### `read_xyz(filename)`

Parses a standard XYZ-format coordinate file.

```
<n_atoms>
<comment>
<symbol>  <x>  <y>  <z>
...
```

Returns `(n_atoms, positions, atom_names, comment)` where `positions` is a `float64` array of shape `(n_atoms, 3)` in Angstrom.

#### `read_lj_params(filename)`

Parses the Lennard-Jones parameter file. Supports Fortran-style exponent notation (`3.4d0` → `3.4e0`).

Returns a dict with keys `epsilon_K` [K] and `sigma_ang` [Å].

#### `write_xyz_trajectory(filename, positions, atom_names, times)`

Writes a multi-frame XYZ trajectory file readable by VMD. Each frame is written as a standard XYZ block with a comment line containing the frame index and simulation time.

```
<n_atoms>
frame 0  t = 0.0000 fs
Ar    15.74721705    2.04898943   13.38305577
...
```

To load in VMD:
```bash
vmd trajectory.xyz
```

---

### `initialization.py`

Velocity initialisation pipeline and kinetic energy calculation.

#### `kinetic_energy(velocities, mass_amu)`

Total kinetic energy in eV:

$$KE = \frac{1}{2}\, m \sum_i v_i^2 \cdot \texttt{AMU\_ANG2\_FS2\_TO\_EV}$$

#### `target_kinetic_energy(n_atoms, temperature_k)`

Target kinetic energy from the equipartition theorem (Frenkel & Smit, 2002):

$$\langle K \rangle = \frac{3N-3}{2} k_B T$$

The $3N-3$ degrees of freedom account for the removal of centre-of-mass motion.

#### `maxwell_boltzmann_velocities(n_atoms, mass_amu, temperature_k, rng)`

Samples each Cartesian velocity component independently from a zero-mean Gaussian:

$$v_\alpha \sim \mathcal{N}(0,\, \sigma_v), \quad \sigma_v = \sqrt{\frac{k_B T}{m \cdot \texttt{AMU\_ANG2\_FS2\_TO\_EV}}}$$

Returns an array of shape `(n_atoms, 3)` in Å/fs.

#### `remove_com_drift(velocities)`

Subtracts the centre-of-mass velocity from all atoms, setting the total linear momentum exactly to zero. Valid for equal-mass systems.

#### `rescale_to_temperature(velocities, mass_amu, n_atoms, temperature_k)`

Uniformly scales all velocities by $\lambda = \sqrt{K_\text{target} / K_\text{current}}$ to match the equipartition target exactly.

#### `initialize_velocities(n_atoms, mass_amu, temperature_k, rng)`

Full initialisation pipeline: sample → remove COM drift → rescale. This is the function called by `simulation.py`.

---

### `lj_potential.py`

Lennard-Jones pair potential, forces, and a JIT warmup utility.

#### `compute_forces_and_potential(epsilon_ev, sigma_ang, positions)` ⚡ Numba JIT

The core computational kernel, compiled to machine code by Numba on first call (`cache=True` saves the binary to `__pycache__` for subsequent runs). Iterates over all unique pairs $(i, j)$ with $j > i$ and applies Newton's third law ($\mathbf{F}_j = -\mathbf{F}_i$), giving $O(N^2/2)$ pair evaluations.

For each pair at displacement $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$:

$$U_{ij} = 4\varepsilon \left[({\sigma}/{r})^{12} - ({\sigma}/{r})^6\right]$$

$$\mathbf{F}_i = \frac{24\varepsilon}{r^2} \left[2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right] \mathbf{r}_{ij}$$

Returns `(forces, potential)` where forces has shape `(n_atoms, 3)` [eV/Å] and potential is a scalar [eV].

**Performance** (Ar₃₈, Intel CPU):

| Implementation | ms/call | 50k steps |
|----------------|---------|-----------|
| Python loop    | ~5.0    | ~250 s    |
| NumPy fallback | ~0.14   | ~7 s      |
| Numba JIT      | ~0.02   | ~1 s      |

#### `potential_energy(epsilon_ev, sigma_ang, positions)`

Convenience wrapper returning only the potential energy scalar.

#### `compute_forces(epsilon_ev, sigma_ang, positions)`

Convenience wrapper returning only the force array.

#### `warmup_jit(epsilon_ev, sigma_ang, positions)`

Triggers JIT compilation explicitly before the timed simulation loop. Should be called once in `simulation.py` after reading the structure. The first call costs ~1–3 s; all subsequent calls use the cached binary.

---

### `integrator.py`

#### `velocity_verlet_step(positions, velocities, forces, mass_amu, dt_fs, epsilon_ev, sigma_ang)`

Advances the system by one step of the Velocity Verlet algorithm (Swope et al., 1982):

$$\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \tfrac{1}{2}\mathbf{a}(t)\Delta t^2$$

$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + \tfrac{1}{2}\left[\mathbf{a}(t) + \mathbf{a}(t+\Delta t)\right]\Delta t$$

where $\mathbf{a} = \mathbf{F}/m \cdot \texttt{FORCE\_CONV}$.

Forces and potential at $t+\Delta t$ are evaluated in a single call to `compute_forces_and_potential`, avoiding a redundant $O(N^2)$ loop.

Returns `(new_positions, new_velocities, new_forces, potential)`.

---

### `observables.py`

#### `temperature(velocities, mass_amu)`

Instantaneous temperature from the equipartition theorem:

$$T = \frac{2K}{(3N-3)\, k_B}$$

In NVE this fluctuates around a mean value. In NVT it is driven toward the target temperature by the thermostat.

---

### `nve.py`

#### `run_nve(positions, velocities, mass_amu, epsilon_ev, sigma_ang, dt_fs, n_steps, save_interval)`

Runs the microcanonical ensemble. At each step:
1. One Velocity Verlet step advances positions and velocities.
2. Observables are computed and stored every `save_interval` steps.

A `tqdm` progress bar shows elapsed time and live T and E values.

Returns a trajectory dict with keys `times`, `positions`, `velocities`, `kinetic_energy`, `potential_energy`, `total_energy`, `temperature`. All arrays are indexed as `[frame, ...]`.

Energy conservation check: total energy drift should be below $10^{-4}$ eV over $10^4$ steps at $\Delta t = 5\,\text{fs}$.

---

### `nvt.py`

#### `run_nvt(positions, velocities, mass_amu, epsilon_ev, sigma_ang, dt_fs, n_steps, thermostat_fn, save_interval)`

Thermostat-agnostic canonical ensemble runner. At each step:
1. One Velocity Verlet step advances positions and velocities.
2. `thermostat_fn(velocities)` is called to apply the thermostat.
3. Observables are stored every `save_interval` steps.

`thermostat_fn` must have the signature `velocities -> velocities` with all parameters pre-bound. Use `get_thermostat()` from `thermostat.py` to construct it.

Returns the same trajectory dict structure as `run_nve`.

In NVT the total energy fluctuates because the system exchanges energy with the heat bath. Temperature should remain close to `TEMP_INIT_K`.

---

### `thermostat.py`

#### `andersen(velocities, mass_amu, target_temp_k, collision_freq, dt_fs, rng)`

Andersen thermostat (Andersen, 1980). At each step, each atom $i$ is selected for a stochastic collision with probability:

$$p = \nu \cdot \Delta t$$

where $\nu$ is `collision_freq` [1/fs]. Selected atoms have their velocity vector resampled from the Maxwell-Boltzmann distribution at `target_temp_k`. Unselected atoms are unchanged.

Coupling strength guidelines (Frenkel & Smit, 2002):

| `collision_freq` | Behaviour |
|------------------|-----------|
| 0.001 fs⁻¹       | Weak coupling, slow thermalisation, dynamics preserved |
| 0.005 fs⁻¹       | Moderate coupling (default) |
| 0.05 fs⁻¹        | Strong coupling, fast thermalisation, suppressed dynamics |

#### `get_thermostat(name, mass_amu, target_temp_k, dt_fs, rng, collision_freq)`

Factory function. Returns a callable `thermostat_fn(velocities) -> velocities` with all parameters pre-bound, ready to be passed to `run_nvt`. Raises `ValueError` if `name` is not recognised.

Currently available names: `"andersen"`.

---

### `visualization.py`

All plot functions accept a `label` string (shown in plot titles) and a `save_dir` path (where the PNG is written). Directories are created automatically.

#### `plot_energy(times, kinetic_energy, potential_energy, total_energy, label, save_dir)`

Two-panel figure:
- Upper panel: K, U, and E vs time.
- Lower panel: $E - \langle E \rangle$ vs time. This amplifies any drift that would be invisible on the absolute scale. The percentage drift is shown in the title.

Saved as `energy.png`.

#### `plot_temperature(times, temperature, label, target_temp, save_dir)`

Temperature vs time with a dashed line at $\langle T \rangle$. If `target_temp` is provided a dotted reference line is added, useful for checking NVT thermalisation.

Saved as `temperature.png`.

#### `plot_x_component(times, positions, label, particle_index, save_dir)`

x coordinate of a single atom vs time, with a dashed line at $\langle x \rangle$.

Saved as `x_component_atom{particle_index}.png`.

#### `plot_all(trajectory, label, target_temp, particle_index, save_dir)`

Convenience wrapper that calls all three plot functions for a trajectory dict as returned by `run_nve` or `run_nvt`.

---

### `simulation.py`

Main entry point. Runs the full simulation pipeline in eight steps:

1. Read structure (`read_xyz`) and LJ parameters (`read_lj_params`).
2. Warm up Numba JIT (`warmup_jit`).
3. Create two independent RNG streams (`np.random.default_rng`).
4. Initialise velocities for both ensembles from the same target temperature.
5. Run NVE (`run_nve`).
6. Build thermostat callable (`get_thermostat`) and run NVT (`run_nvt`).
7. Save XYZ trajectories to `OUTPUT_DIR_NVE` and `OUTPUT_DIR_NVT`.
8. Generate and save all plots via `plot_all`.

Trajectory filenames encode the run parameters, e.g. `trajectory_T20K_10000steps.xyz`.

---

## Output files

After a run the output directory contains:

```
output/
├── nve/
│   ├── trajectory_T20K_10000steps.xyz   # VMD-compatible trajectory
│   ├── energy.png                        # K, U, E and drift
│   ├── temperature.png                   # T(t)
│   └── x_component_atom0.png            # x(t) for atom 0
└── nvt/
    ├── trajectory_T20K_10000steps.xyz
    ├── energy.png
    ├── temperature.png
    └── x_component_atom0.png
```

The XYZ trajectory file name updates automatically when `TEMP_INIT_K` or `N_STEPS` are changed in `config.py`.

---

## Adding a new thermostat

1. Implement the thermostat function in `thermostat.py`:

```python
def nose_hoover(velocities, mass_amu, target_temp_k, tau_fs, dt_fs):
    # ... implementation ...
    return new_velocities
```

2. Add a branch in `get_thermostat`:

```python
if name == "nose_hoover":
    def thermostat_fn(velocities):
        return nose_hoover(velocities, mass_amu, target_temp_k,
                           tau_fs, dt_fs)
    return thermostat_fn
```

3. Add any new parameters to `config.py`.

4. Set `THERMOSTAT = "nose_hoover"` in `config.py`.

No other files need to change.

---

## References

Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press. https://doi.org/10.1093/oso/9780198803195.001.0001

Andersen, H. C. (1980). Molecular dynamics simulations at constant pressure and/or temperature. *J. Chem. Phys.*, 72(4), 2384–2393. https://doi.org/10.1063/1.439486

Frenkel, D., & Smit, B. (2002). *Understanding Molecular Simulation* (2nd ed.). Academic Press.

Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *Proc. LLVM-HPC Workshop*. https://doi.org/10.1145/2833157.2833162

Rahman, A. (1964). Correlations in the motion of atoms in liquid argon. *Phys. Rev.*, 136(2A), A405–A411. https://doi.org/10.1103/PhysRev.136.A405

Swope, W. C., Andersen, H. C., Berens, P. H., & Wilson, K. R. (1982). A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules. *J. Chem. Phys.*, 76(1), 637–649. https://doi.org/10.1063/1.442716
