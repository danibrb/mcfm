"""
Simulation configuration.
"""

import os

# I/O

FILENAME_XYZ_IN  = "ar38_to.xyz"
FILENAME_LJ      = "parametri_lj.txt"

# Output directories
OUTPUT_DIR_NVE = os.path.join("output", "nve")
OUTPUT_DIR_NVT = os.path.join("output", "nvt")

# Particle properties

MASS_AMU = 40.0          # Argon-40 atomic mass  [amu]

# Thermodynamic state

TEMP_INIT_K = 20.0       # Initial temperature   [K]

# Integrator settings

TIMESTEP_FS    = 5.0     # Integration timestep  [fs]
N_STEPS        = 500_000   # Number of MD steps
SAVE_INTERVAL  = 25     # Save trajectory frame every N steps

# Thermostat — used only for NVT
# Options: "andersen"
THERMOSTAT     = "andersen"
COLLISION_FREQ = 0.005   # Andersen collision frequency  [1/fs]

# Reproducibility

RANDOM_SEED = 1234567890