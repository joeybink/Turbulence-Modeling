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
u_tau = Re_tau * (nu / H)
rho  = 0.001

# Part a)

y = np.linspace(1e-5, H, N, endpoint= True)

# Part b) - d)
y_plus = y * (u_tau / nu)
velocity_gradient_plus= np.zeros(N)

C = np.square(karman * Re_tau)
H_squared = np.square(H)
C_1 = 4 * C / H_squared * H
C_2 = 4 * C / H_squared
C_3 = 2 * C / H_squared

numerator = -1 + np.sqrt(1 - C_1 * y**3 + C_2 * np.square(y))
denominator = C_3 * np.square(y)

velocity_gradient_plus = numerator / denominator

velocity_plus = cumulative_trapezoid(velocity_gradient_plus, y_plus, axis=1, initial=0) # initial=0 imposes no-slip condition
velocity_mean = velocity_plus * u_tau 
velocity_gradient = velocity_gradient_plus * (np.square(u_tau) / nu) # Chain rule



# Part e)
l_m = karman * y
viscous_shear = rho * nu * velocity_gradient
reynolds_shear =  rho * np.square(l_m) * np.square(velocity_gradient) 

tau_tot = viscous_shear + reynolds_shear
tau_w = rho * np.square(u_tau)

ratio_calculated = tau_tot / tau_w
ratio_expected = 1 - (y/H)

if not np.allclose(ratio_calculated, ratio_expected,rtol=1e-5, atol=1e-6):
    raise ValueError("Calculations failed: stress ratio not satisfied")


