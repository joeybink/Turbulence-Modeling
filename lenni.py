import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Constants
karman = 0.4
Re_tau = np.array([
    [180], 
    [395], 
    [590]
])

N = 10
H = 3
nu = 1e-6

# Part a)

y = np.linspace(1e-5, H, N, endpoint= True)

# Part b) (need to plot)
velocity_gradient_plus= np.zeros(N)

C = (karman * Re_tau)**2
C_1 = 4 * C / H**3
C_2 = 4 * C / H**2
C_3 = 2 * C / H**2

numerator = -1 + np.sqrt(1 - C_1 * y**3 + C_2 * y**2)
denominator = C_3 * y**2

velocity_gradient_plus = numerator / denominator

# Part c) + d)
velocity_plus = cumulative_trapezoid(velocity_gradient_plus, y, axis=1, initial=0)

