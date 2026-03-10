"""
Physical constants and unit conversion factors.
"""

# Fundamental constants

KB_SI   = 1.380649e-23      # Boltzmann constant          [J/K]
EV_TO_J = 1.602176634e-19   # Electronvolt to Joule       [J/eV]
AMU_TO_KG = 1.66053906660e-27  # Unified atomic mass unit [kg/amu]

# Boltzmann constant in simulation energy units
KB_EV = KB_SI / EV_TO_J     #  8.6173e-5                 [eV/K]

# Unit-system conversion factors  (Å / eV / amu / fs)

# KE [eV] = 0.5 * m [amu] * v^2 [Å^2/fs^2] * AMU_ANG2_FS2_TO_EV
# 1 Å/fs = 1e-10 m / 1e-15 s = 1e5 m/s
AMU_ANG2_FS2_TO_EV = AMU_TO_KG * (1e5 ** 2) / EV_TO_J   #  103.6427

# a [Å/fs^2] = F [eV/Å] / m [amu] * FORCE_CONV
FORCE_CONV = 1.0 / AMU_ANG2_FS2_TO_EV                     #  9.6485e-3
