import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def solve_channel(re_tau, n_points, kappa =0.41):

    #calculate classical prandtl channel flow on one uniform grid

    #grid, mixing length and total stress

    y_plus = np.linspace(0.0, re_tau, n_points)

    L= kappa * y_plus
    T= 1.0 - y_plus / re_tau

    #mean velocity gradient

    g= 2.0 * T / (1.0 + np.sqrt (1.0 + 4.0  * L**2 * T))

    # integrate from the wall
    U_plus = np.zeros(n_points)

    for i in range(1, n_points):
        delta_y = y_plus[i] - y_plus[i-1]
        average_gradient = 0.5 * (g[i-1] + g[i])

        U_plus[i] = U_plus[i-1] + average_gradient * delta_y


    #Stress contributions

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



# Physical cases required by Task 3
reynolds_numbers = [180, 395, 590]
kappa = 0.41

# Starting resolution; higher-Re cases still need a refinement check
n_points = 5121

# Store each solution under its Reynolds number
solutions = {}

# Check the current grid against twice as many intervals
fine_n = 2 * n_points - 1

velocity_tolerance = 0.005
stress_tolerance = 0.001

print(f"\nGrid check: {n_points} versus {fine_n} points")

print(
    f"{'Re_tau':>8} "
    f"{'Max dU+':>14} "
    f"{'Max dR+':>14} "
    f"{'Pass?':>8}"
)

for re_tau in reynolds_numbers:
    # Current solution already calculated above
    coarse = solve_channel(re_tau, n_points, kappa)
    solutions[re_tau] = coarse

    # Same physical case on a finer grid
    fine = solve_channel(re_tau, fine_n, kappa)
    fine_y = fine["y_plus"]

    # Compare both profiles at the fine-grid positions
    coarse_U = np.interp(
        fine_y, coarse["y_plus"], coarse["U_plus"]
    )

    coarse_R = np.interp(
        fine_y, coarse["y_plus"], coarse["reynolds_stress"]
    )

    max_U_difference = np.max(
        np.abs(coarse_U - fine["U_plus"])
    )

    max_R_difference = np.max(
        np.abs(coarse_R - fine["reynolds_stress"])
    )

    passed = (
        max_U_difference < velocity_tolerance
        and max_R_difference < stress_tolerance
    )

    status = "YES" if passed else "NO"

    print(
        f"{re_tau:8d} "
        f"{max_U_difference:14.6e} "
        f"{max_R_difference:14.6e} "
        f"{status:>8}"
    )

# Print centreline velocity and peak Reynolds shear stress
print("\nCalculated profile summary:")

print(
    f"{'Re_tau':>8} "
    f"{'Centreline U+':>16} "
    f"{'Peak R+':>12} "
    f"{'Peak y+':>12} "
    f"{'Peak y/H':>12}"
)

for re_tau in reynolds_numbers:
    result = solutions[re_tau]

    # Index of the largest Reynolds shear-stress value
    peak_index = np.argmax(result["reynolds_stress"])

    peak_stress = result["reynolds_stress"][peak_index]
    peak_y_plus = result["y_plus"][peak_index]
    peak_y_over_H = peak_y_plus / re_tau

    Uc = result["U_plus"][-1]

    print(
        f"{re_tau:8d} "
        f"{Uc:16.5f} "
        f"{peak_stress:12.5f} "
        f"{peak_y_plus:12.5f} "
        f"{peak_y_over_H:12.5f}"
    )

#plotting 
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

# One color and line style per Reynolds number
colors = ["#286090", "#A65628", "#487B65"]
line_styles = ["-", "--", "-."]

fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8))

fig.subplots_adjust(
    left=0.11, right=0.97,
    bottom=0.10, top=0.82,
    hspace=0.42, wspace=0.35
)

for re_tau, color, style in zip(
    reynolds_numbers, colors, line_styles
):
    result = solutions[re_tau]

    y_plus = result["y_plus"]
    y_over_H = y_plus / re_tau

    label = rf"$Re_\tau = {re_tau}$"

    # (a) Mean velocity in wall units
    axes[0, 0].plot(
        y_plus, result["U_plus"],
        color=color, linestyle=style,
        linewidth=1.4, label=label
    )

    # (b) Reynolds shear stress
    axes[0, 1].plot(
        y_over_H, result["reynolds_stress"],
        color=color, linestyle=style,
        linewidth=1.4
    )

    # (c) Viscous shear stress
    axes[1, 0].plot(
        y_over_H, result["viscous_stress"],
        color=color, linestyle=style,
        linewidth=1.4
    )

    # (d) Calculated total shear stress
    axes[1, 1].plot(
        y_over_H, result["total_stress"],
        color=color, linestyle=style,
        linewidth=1.4
    )

# Exact total-stress distribution for comparison
axes[1, 1].plot(
    [0, 1], [1, 0],
    color="black", linestyle=":",
    linewidth=1.2, label=r"$1-y/H$"
)

# Titles and labels
axes[0, 0].set_title("(a) Mean velocity", loc="left")
axes[0, 0].set_xlabel(r"$y^+$")
axes[0, 0].set_ylabel(r"$U^+$")
axes[0, 0].set_xlim(0, max(reynolds_numbers))

axes[0, 1].set_title("(b) Reynolds shear stress", loc="left")
axes[0, 1].set_ylabel(r"$-\overline{u'v'}/u_\tau^2$")

axes[1, 0].set_title("(c) Viscous shear stress", loc="left")
axes[1, 0].set_ylabel(r"$\tau_\nu/\tau_w$")

axes[1, 1].set_title("(d) Total shear stress", loc="left")
axes[1, 1].set_ylabel(r"$\tau_{\mathrm{tot}}/\tau_w$")

# The three stress panels use the same normalized domain
for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
    ax.set_xlabel(r"$y/H$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

for ax in axes.flat:
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.tick_params(which="major", length=3.5, width=0.7)
    ax.tick_params(which="minor", length=2, width=0.5)
    ax.grid(False)

# Shared Reynolds-number legend
handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center", bbox_to_anchor=(0.54, 0.95),
    ncol=3, frameon=False, handlelength=3
)

# Identify the exact reference only in the total-stress panel
axes[1, 1].legend(frameon=False, loc="upper right")

fig.suptitle(
    rf"Classical Prandtl model: $\kappa={kappa:g}$, "
    rf"$N={n_points}$",
    fontsize=11, y=0.99
)

# Save beside task3.py
output_folder = Path(__file__).resolve().parent

for extension in ("png", "svg"):
    figure_path = output_folder / f"task3_profiles.{extension}"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")

plt.show()