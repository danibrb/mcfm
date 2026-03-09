import numpy as np

def distance_two_points(position_a, position_b):
    x = position_a[0] - position_b[0]
    y = position_a[1] - position_b[1]
    z = position_a[2] - position_b[2]
    return np.sqrt(x**2 + y**2 + z**2)