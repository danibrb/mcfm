import numpy as np
import random
import math

def box_muller():

    rn1 = random.random()
    rn2 = random.random()

    while rn1 == 0:  # Avoid log(0)
        rn1 = random.random()

    rho = math.sqrt(-2.0 * math.log(rn1))
    theta = 2.0 * math.pi * rn2

    return rho * math.cos(theta), rho * math.sin(theta)