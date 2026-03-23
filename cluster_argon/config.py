"""
Simulation configuration.
"""

import os

# I/O

FILENAME_XYZ_IN  = "ar38_to.xyz"
FILENAME_LJ      = "parametri_lj.txt"

# Output directories
OUTPUT_DIR_NVE  = os.path.join("output", "nve")
OUTPUT_DIR_NVT  = os.path.join("output", "nvt")
OUTPUT_DIR_AND = os.path.join("output", "andersen_analysis")

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

# Reproducibility

RANDOM_SEED = 1234567890
