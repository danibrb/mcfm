"""
Simulation configuration.
"""

import os

# I/O

FILENAME_XYZ_IN  = "ar38_to.xyz"
FILENAME_LJ      = "parametri_lj.txt"

# Output directories
OUTPUT_DIR_NVE  = os.path.join("..", "output", "nve")
OUTPUT_DIR_NVT  = os.path.join("..", "output", "nvt")
OUTPUT_DIR_AND = os.path.join("..", "output", "andersen_analysis")

# Integrator sOUTPUT_DIR_RAMP = os.path.join("output", "heating_ramp")

# Integrator settings

TIMESTEP_FS   = 1           # Integration timestep  [fs]
N_STEPS       = 10_000     # Steps for NVE and NVT
SAVE_INTERVAL = 10          # Save trajectory frame every N steps

# Particle properties

MASS_AMU = 40.0             # Argon-40 atomic mass  [amu]

# Thermodynamic state

TEMP_INIT_K  = 20.0         # Initial temperature   [K]

# Thermostat

COLLISION_FREQ = 0.0004   # Andersen collision frequency  [1/fs]

# Heating ramp
 
# Temperature range
TEMP_RAMP_START_K = 15.0    # Starting temperature      [K]
TEMP_RAMP_END_K   = 90.0   # Final temperature         [K]
 
# Number of steps: governs the heating rate dT/dt = ΔT / (N * dt).
# With dt = 1 fs and N = 2_000_000 the total simulated time is 2 ns

N_STEPS_RAMP       = 1_000_000
SAVE_INTERVAL_RAMP = 100    

# Reproducibility

RANDOM_SEED = 1234567890
