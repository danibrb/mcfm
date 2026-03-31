"""
Simulation configuration.

Project layout:
    project/
        input/      ar38_to.xyz, parametri_lj.txt
        output/     generated automatically
        src/        all .py files

"""

import os

# Directory containing this file (src/)
_SRC = os.path.dirname(os.path.abspath(__file__))

# Project root (one level up from src/)
_ROOT = os.path.dirname(_SRC)

def _inp(*parts) -> str:
    return os.path.join(_ROOT, "input",  *parts)

def _out(*parts) -> str:
    return os.path.join(_ROOT, "output", *parts)


# I/O


FILENAME_XYZ_IN = _inp("ar38_to.xyz")
FILENAME_LJ     = _inp("parametri_lj.txt")

OUTPUT_DIR_NVE  = _out("nve")
OUTPUT_DIR_NVT  = _out("nvt")
OUTPUT_DIR_AND  = _out("andersen_analysis")
OUTPUT_DIR_HR   = _out("heating_ramp")
#OUTPUT_DIR_LG   = _out("transition_LG")
#OUTPUT_DIR_VAL   = _out("validation")


# Integrator


TIMESTEP_FS   = 1           # [fs]
N_STEPS       = 100_000     # steps for the NVE / NVT comparison run
SAVE_INTERVAL = 10          # save a frame every N steps


# Particle properties


MASS_AMU = 40.0             # Argon-40  [amu]


# NVE / NVT initial conditions


TEMP_INIT_K    = 20.0       # [K]
COLLISION_FREQ = 0.0004     # Andersen eta_c  [1/fs]


# Heating ramp


TEMP_HR_START_K  = 15.0
TEMP_HR_END_K    = 27.0
N_STEPS_HR       = 90_000_000
SAVE_INTERVAL_HR = 25


# Andersen collision frequency for ramp simulations

COLLISION_FREQ_RAMP = 5.0e-4    # [1/fs]


# Reproducibility

RANDOM_SEED = 1234567890