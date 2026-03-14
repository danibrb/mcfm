# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Added
- **Project Structure**: Created basic project structure.
- **File Reading**: Added support for reading `xyz` and parameters files.
- **File Writing**: Added support for saving trajectory to `xyz` file. 

### Features
- **Velocity Generation**:
  - ~~Implemented Box-Muller transform for random number generation.~~
  - Added Maxwell-Boltzmann distribution for velocity assignment.
  - Removed center-of-mass velocity.
  - Calculated kinetic energy and rescaled velocities to target temperature.
  - Integrated all steps into a cohesive workflow.

- **Lennard-Jones Force Calculation**:
  - Resolved unit inconsistencies.
  - Standardized units to electronvolts (eV) for consistency.
  - Adapted all calculations accordingly.

- **Observables**:
  - Added temperature calculation.

- **Simulation**:
  - Implemented NVE (microcanonical) ensemble simulations.

- **Visualization**:
  - Added plots for:
    - Kinetic, potential, and total energy.
    - Temperature.
    - X-coordinate of the first particle.
