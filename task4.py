from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1. INPUTS
# Choose the nominal case: 180, 395, or 590.
re_tau = 590
n_points = 5121
kappa = 0.41

# Actual Reynolds numbers recorded in the DNS file headers
dns_re_values = {
    180: 178.12,
    395: 392.24,
    590: 587.19,
}

dns_re_tau = dns_re_values[re_tau]


# 2. CLASSICAL PRANDTL SOLVER
def solve_channel(re_tau, n_points, kappa=0.41):
    """Calculate velocity and stresses on one uniform grid."""

    # Grid, mixing length, and prescribed total stress
    y_plus = np.linspace(0.0, re_tau, n_points)
    L = kappa * y_plus
    T = 1.0 - y_plus / re_tau

    # Mean velocity gradient
    g = 2.0 * T / (
        1.0 + np.sqrt(1.0 + 4.0 * L**2 * T)
    )

    # Integrate from the wall, where U+ = 0
    U_plus = np.zeros(n_points)

    for i in range(1, n_points):
        delta_y = y_plus[i] - y_plus[i - 1]
        average_gradient = 0.5 * (g[i - 1] + g[i])

        U_plus[i] = (
            U_plus[i - 1] + average_gradient * delta_y
        )

    # Stress contributions normalized by wall stress
    viscous_stress = g
    reynolds_stress = (L * g)**2
    total_stress = viscous_stress + reynolds_stress

    max_stress_error = np.max(np.abs(total_stress - T))

    return {
        "y_plus": y_plus,
        "U_plus": U_plus,
        "viscous_stress": viscous_stress,
        "reynolds_stress": reynolds_stress,
        "total_stress": total_stress,
        "max_stress_error": max_stress_error,
    }


# 3. LOAD DNS MEAN VELOCITY
assignment_folder = Path(__file__).resolve().parents[2]
data_folder = assignment_folder / "worked_example" / "data"
output_folder = Path(__file__).resolve().parent

means_file = data_folder / f"chan{re_tau}.means"
means = np.loadtxt(means_file, comments="#")

# These quantities are already normalized
dns_y_over_H = means[:, 0]
dns_y_plus = means[:, 1]
dns_U_plus = means[:, 2]

print(f"DNS file: {means_file.name}")
print(f"Data points: {means.shape[0]}")
print(f"Columns: {means.shape[1]}")

print("\nFirst ten points:")
print(f"{'y/H':>12} {'y+':>12} {'U+':>12}")

for i in range(10):
    print(
        f"{dns_y_over_H[i]:12.6f} "
        f"{dns_y_plus[i]:12.6f} "
        f"{dns_U_plus[i]:12.6f}"
    )

print(f"\nCentreline y/H: {dns_y_over_H[-1]:.6f}")
print(f"Centreline y+: {dns_y_plus[-1]:.6f}")
print(f"Centreline U+: {dns_U_plus[-1]:.6f}")


# 4. CALCULATE THE MODEL
model = solve_channel(re_tau, n_points, kappa)

model_y_plus = model["y_plus"]
model_U_plus = model["U_plus"]

print(f"\nModel Re_tau: {re_tau:.2f}")
print(f"DNS Re_tau: {dns_re_tau:.2f}")
print(f"Model centreline U+: {model_U_plus[-1]:.6f}")
print(f"DNS centreline U+: {dns_U_plus[-1]:.6f}")


# 5. RESEARCH-PAPER PLOTTING STYLE
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# 6. VELOCITY COMPARISON
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.6, 3.4),
    constrained_layout=True
)

# Exclude y+ = 0 only from the logarithmic plot
model_positive = model_y_plus > 0
dns_positive = dns_y_plus > 0

# Linear coordinates
axes[0].plot(
    model_y_plus, model_U_plus,
    color="#286090", linewidth=1.4,
    label=rf"Prandtl, $Re_\tau={re_tau:g}$"
)

axes[0].plot(
    dns_y_plus, dns_U_plus,
    "o", color="black", markersize=3,
    markerfacecolor="none", markeredgewidth=0.7,
    label=rf"DNS, $Re_\tau={dns_re_tau:g}$"
)

# Semilogarithmic coordinates
axes[1].semilogx(
    model_y_plus[model_positive],
    model_U_plus[model_positive],
    color="#286090", linewidth=1.4
)

axes[1].semilogx(
    dns_y_plus[dns_positive],
    dns_U_plus[dns_positive],
    "o", color="black", markersize=3,
    markerfacecolor="none", markeredgewidth=0.7
)

axes[0].set_title("(a) Linear coordinates", loc="left")
axes[1].set_title("(b) Semilogarithmic coordinates", loc="left")

axes[0].set_xlim(0, re_tau)
axes[1].set_xlim(0.05, re_tau)

for ax in axes:
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_ylim(bottom=0)
    ax.tick_params(
        which="both", direction="in", top=True, right=True
    )

axes[0].legend(frameon=False, fontsize=9)

# Save velocity plots without overwriting other cases
for extension in ("png", "svg"):
    figure_path = output_folder / f"velocity_dns_{re_tau}.{extension}"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")


# 7. LOAD DNS REYNOLDS SHEAR STRESS
stress_file = data_folder / f"chan{re_tau}.reystress"
stress_data = np.loadtxt(stress_file, comments="#")

dns_stress_y_plus = stress_data[:, 1]

# Column 5 contains <u'v'> / u_tau^2: negate it.
dns_R_plus = -stress_data[:, 5]

model_R_plus = model["reynolds_stress"]

print(f"\nModel peak Reynolds stress+: {np.max(model_R_plus):.6f}")
print(f"DNS peak Reynolds stress+: {np.max(dns_R_plus):.6f}")


# 8. REYNOLDS SHEAR-STRESS COMPARISON
stress_fig, stress_axes = plt.subplots(
    1, 2,
    figsize=(7.6, 3.4),
    constrained_layout=True
)

for ax in stress_axes:
    ax.plot(
        model_y_plus, model_R_plus,
        color="#286090", linewidth=1.4,
        label=rf"Prandtl, $Re_\tau={re_tau:g}$"
    )

    ax.plot(
        dns_stress_y_plus, dns_R_plus,
        "o", color="black", markersize=3,
        markerfacecolor="none", markeredgewidth=0.7,
        label=rf"DNS, $Re_\tau={dns_re_tau:g}$"
    )

    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$-\overline{u'v'}/u_\tau^2$")
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.tick_params(
        which="both", direction="in", top=True, right=True
    )

stress_axes[0].set_title("(a) Reynolds shear stress", loc="left")
stress_axes[0].set_xlim(0, re_tau)
stress_axes[0].legend(frameon=False, fontsize=9)

stress_axes[1].set_title("(b) Near-wall detail", loc="left")
stress_axes[1].set_xlim(0, 30)

# Save stress plots without overwriting other cases
for extension in ("png", "svg"):
    figure_path = output_folder / f"stress_dns_{re_tau}.{extension}"
    stress_fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")

plt.show()