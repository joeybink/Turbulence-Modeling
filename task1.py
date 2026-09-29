import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Save plots beside this script, regardless of where Python is launched.
output_folder = Path(__file__).resolve().parent

# Inputs
re_tau = 180.0
kappa = 0.41
n_points = 11

# Equally spaced positions from the wall to the centreline
y_plus = np.linspace(0.0, re_tau, n_points)

# Mixing length and prescribed total stress at every position
L = kappa * y_plus
T = 1.0 - y_plus / re_tau

# Mean velocity gradient at every position
g = 2.0 * T / (1.0 + np.sqrt(1.0 + 4.0 * L**2 * T))

# Mean velocity in wall units
U_plus = np.zeros(n_points)

# No-slip condition at the wall
U_plus[0] = 0.0

# Integrate the gradient from the wall toward the centreline
for i in range(1, n_points):
    delta_y = y_plus[i] - y_plus[i - 1]

    average_gradient = 0.5 * (g[i - 1] + g[i])

    U_plus[i] = U_plus[i - 1] + average_gradient * delta_y


# Normalized stress contributions
viscous_stress = g
reynolds_stress = (L * g)**2
total_stress = viscous_stress + reynolds_stress

# Largest absolute stress-balance error across the grid
max_stress_error = np.max(np.abs(total_stress - T))



# Display velocity, gradient, and stresses
print(
    f"{'y+':>8} {'U+':>12} {'g':>12} "
    f"{'Viscous':>12} {'Reynolds':>12} {'Total':>12}"
)

for i in range(n_points):
    print(
        f"{y_plus[i]:8.2f} "
        f"{U_plus[i]:12.6f} "
        f"{g[i]:12.6f} "
        f"{viscous_stress[i]:12.6f} "
        f"{reynolds_stress[i]:12.6f} "
        f"{total_stress[i]:12.6f}"
    )

print(f"\nMaximum stress error = {max_stress_error:.3e}")
print(f"Centreline velocity U+ = {U_plus[-1]:.6f}")

# Plot the mean velocity profile
plt.figure(figsize=(7, 5))

plt.plot(y_plus, U_plus, "o-", label="Prandtl model")

plt.xlabel(r"$y^+$")
plt.ylabel(r"$U^+$")
plt.title(f"Mean velocity: Re_tau = {re_tau:g}, N = {n_points}")

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the velocity figure .
plt.savefig(output_folder / "velocity_profile.png", dpi=300, bbox_inches="tight")

# Plot normalized shear-stress contributions
plt.figure(figsize=(7, 5))

plt.plot(
    y_plus, viscous_stress,
    "o-", label="Viscous stress"
)

plt.plot(
    y_plus, reynolds_stress,
    "s-", label="Reynolds shear stress"
)

plt.plot(
    y_plus, total_stress,
    "o-", label="Calculated total"
)

plt.plot(
    y_plus, T,
    "k--", label="Prescribed total"
)

plt.xlabel(r"$y^+$")
plt.ylabel(r"Shear stress / $\tau_w$")
plt.title(f"Stress balance: Re_tau = {re_tau:g}, N = {n_points}")

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the stress figure before displaying the plots.
plt.savefig(output_folder / "stress_balance.png", dpi=300, bbox_inches="tight")
print(f"Plots saved in: {output_folder}")

# Display both figures
plt.show()
