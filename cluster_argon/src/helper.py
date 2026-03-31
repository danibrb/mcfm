from config import TEMP_HR_START_K, TEMP_HR_END_K, TIMESTEP_FS

# Desired heating rate: 1 K / 10 ns
desired_rate = 0.1  # 1 K/ns

# Total temperature change
delta_T = TEMP_HR_END_K - TEMP_HR_START_K

# Total time required to achieve delta_T at desired_rate
total_time_ns = delta_T / desired_rate

# Convert to femtoseconds
total_time_fs = total_time_ns * 1e6

# Number of steps
n_steps = total_time_fs / TIMESTEP_FS

print(f"Set n_steps to {n_steps:1.2e} to obtain a heating rate of {desired_rate:.1f} K/ns")
print(f"From {TEMP_HR_START_K} K to {TEMP_HR_END_K} K simulation time: {total_time_ns} ns")