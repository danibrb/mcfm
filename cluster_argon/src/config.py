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


# Integrator


TIMESTEP_FS   = 1           # [fs]  — used by NVE / NVT / Andersen analysis
N_STEPS       = 100_000     # steps for the NVE / NVT comparison run
SAVE_INTERVAL = 10          # save a frame every N steps


# Particle properties


MASS_AMU = 40.0             # Argon-40  [amu]


# NVE / NVT initial conditions


TEMP_INIT_K    = 20.0       # [K]
COLLISION_FREQ = 0.0004     # Andersen eta_c  [1/fs]


# Heating ramp
#
# Target heating rate: 0.1 K/ns  (= 1 K / 10 ns)
# Temperature range:   15 -> 27 K  =>  delta_T = 12 K
# Required physical time:  12 K / 0.1 K/ns = 120 ns = 1.2e8 fs
#
# A 5 fs timestep is stable for Ar LJ at these temperatures and gives a
# 5x reduction in wall-time relative to 1 fs without measurable loss of
# energy conservation (Allen & Tildesley, 2017, §3.3).
#
#   N_STEPS_HR = 1.2e8 fs / 5 fs = 24_000_000  steps

TEMP_HR_START_K  = 15.0
TEMP_HR_END_K    = 49.0
TIMESTEP_FS_RAMP = 5.0          # [fs]  — dedicated ramp timestep
N_STEPS_HR       = 35_000_000   # 24 M steps  x  5 fs  =  120 ns
SAVE_INTERVAL_HR = 1000          # save a frame every N steps


# Andersen collision frequency for ramp simulations

COLLISION_FREQ_RAMP = 5.0e-4    # [1/fs]


# Reproducibility

RANDOM_SEED = 1234567890