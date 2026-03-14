"""
Simulation configuration
"""

# I/O

FILENAME_XYZ_IN     = "ar38_to.xyz"
FILENAME_LJ         = "parametri_lj.txt"
FILENAME_XYZ_OUT    = "trajectory.xyz"


# Particle properties

MASS_AMU = 40.0          # Argon-40 atomic mass  [amu]


# Thermodynamic state

TEMP_INIT_K = 20.0       # Initial temperature   [K]


# Integrator settings

TIMESTEP_FS = 5.0        # Integration timestep  [fs]
N_STEPS     = 50000        # Number of MD steps


# Reproducibility

RANDOM_SEED = 1234567890
