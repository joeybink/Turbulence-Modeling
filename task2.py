from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 1. INPUTS: keep the physics fixed and change only the grid.
re_tau = 180.0
kappa = 0.41
grid_sizes = [11, 21, 41, 81, 161, 321, 641, 1281, 2561, 5121]
velocity_tolerance = 0.005
stress_tolerance = 0.001
near_wall_limit = 30.0
output_folder = Path(__file__).resolve().parent


# 2. SOLVER: the same equations and integration used in Task 1.
def solve_channel(re_tau, n_points, kappa=0.41):
    """Return velocity and normalized stresses on one uniform grid."""
    y_plus = np.linspace(0.0, re_tau, n_points)
    L = kappa * y_plus
    T = 1.0 - y_plus / re_tau
    g = 2.0 * T / (1.0 + np.sqrt(1.0 + 4.0 * L**2 * T))

    # Start at U+(0) = 0, then add each trapezoidal area.
    U_plus = np.zeros(n_points)
    for i in range(1, n_points):
        delta_y = y_plus[i] - y_plus[i - 1]
        average_gradient = 0.5 * (g[i - 1] + g[i])
        U_plus[i] = U_plus[i - 1] + average_gradient * delta_y

    reynolds_stress = (L * g)**2
    total_stress = g + reynolds_stress

    return {
        "y_plus": y_plus,
        "U_plus": U_plus,
        "reynolds_stress": reynolds_stress,
        "max_stress_error": np.max(np.abs(total_stress - T)),
    }


# 3. RUN EACH GRID: store profiles and compare centreline velocities.
solutions = {}
previous_Uc = None
print(f"{'N':>6} {'Spacing y+':>14} {'Centreline U+':>16} "
      f"{'Change (%)':>14} {'Stress error':>14}")

for n_points in grid_sizes:
    result = solve_channel(re_tau, n_points, kappa)
    solutions[n_points] = result
    spacing = re_tau / (n_points - 1)
    Uc = result["U_plus"][-1]
    stress_error = result["max_stress_error"]

    change_text = "-"
    if previous_Uc is not None:
        change = 100.0 * abs(Uc - previous_Uc) / abs(Uc)
        change_text = f"{change:.6f}"

    print(f"{n_points:6d} {spacing:14.6f} {Uc:16.8f} "
          f"{change_text:>14} {stress_error:14.3e}")
    previous_Uc = Uc


# 4. COMPARE PROFILES: interpolate coarse values onto the fine grid.
# These differences include interpolation error; they are not exact errors.
print("\nSuccessive-grid profile comparisons:")
print(f"{'Coarse N':>10} {'Fine N':>10} {'Max dU+':>14} "
      f"{'Near-wall dU+':>16} {'Max dR+':>14} {'Pass?':>8}")

for i in range(len(grid_sizes) - 1):
    coarse_n = grid_sizes[i]
    fine_n = grid_sizes[i + 1]
    coarse = solutions[coarse_n]
    fine = solutions[fine_n]
    fine_y = fine["y_plus"]

    coarse_U = np.interp(fine_y, coarse["y_plus"], coarse["U_plus"])
    coarse_R = np.interp(fine_y, coarse["y_plus"], coarse["reynolds_stress"])
    velocity_difference = np.abs(coarse_U - fine["U_plus"])
    stress_difference = np.abs(coarse_R - fine["reynolds_stress"])

    max_U = np.max(velocity_difference)
    max_R = np.max(stress_difference)
    near_wall_U = np.max(velocity_difference[fine_y <= near_wall_limit])
    passed = max_U < velocity_tolerance and max_R < stress_tolerance
    status = "YES" if passed else "NO"

    print(f"{coarse_n:10d} {fine_n:10d} {max_U:14.6e} "
          f"{near_wall_U:16.6e} {max_R:14.6e} {status:>8}")


# 5. PLOT: compact journal-style panels with consistent curve identities.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.7,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 9,
    "savefig.facecolor": "white",
})

plot_grids = [11, 41, 161, 641, 2561, 5121]
colors = ["#A65628", "#777777", "#487B65", "#78618D", "#286090", "#111111"]
line_styles = [":", "--", "-.", (0, (5, 2, 1, 2, 1, 2)), "-", (0, (6, 3))]

# A two-column figure width keeps labels legible when placed in a report.
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10,
                    top=0.79, hspace=0.40, wspace=0.30)

for n_points, color, style in zip(plot_grids, colors, line_styles):
    result = solutions[n_points]
    y = result["y_plus"]
    U = result["U_plus"]
    R = result["reynolds_stress"]
    marker = "o" if n_points == 11 else None

    for column in range(2):
        axes[0, column].plot(
            y, U, color=color, linestyle=style, linewidth=1.2,
            marker=marker, markersize=2.5, markerfacecolor="white",
            markeredgewidth=0.7, label=rf"$N = {n_points}$"
        )
        axes[1, column].plot(
            y, R, color=color, linestyle=style, linewidth=1.2,
            marker=marker, markersize=2.5, markerfacecolor="white",
            markeredgewidth=0.7
        )

axes[0, 0].set_title("(a) Mean velocity", loc="left", pad=7)
axes[0, 1].set_title("(b) Mean velocity: near wall", loc="left", pad=7)
axes[1, 0].set_title("(c) Reynolds shear stress", loc="left", pad=7)
axes[1, 1].set_title("(d) Reynolds shear stress: near wall", loc="left", pad=7)

for column in range(2):
    axes[0, column].set_ylabel(r"$U^+$")
    axes[1, column].set_ylabel(r"$-\overline{u'v'}/u_\tau^2$")
    limit = re_tau if column == 0 else min(near_wall_limit, re_tau)
    for row in range(2):
        axes[row, column].set_xlim(0, limit)
        axes[row, column].set_xlabel(r"$y^+$")

for ax in axes.flat:
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="major", length=3.5, width=0.7, pad=4)
    ax.tick_params(which="minor", length=2, width=0.5)

# One legend; put the detailed interpretation in the report caption.
fig.text(0.54, 0.96,
         rf"Prandtl mixing-length model: $Re_\tau = {re_tau:g}$, $\kappa = {kappa:g}$",
         ha="center", fontsize=11)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.935),
           ncol=3, frameon=False, handlelength=3.6, columnspacing=2.0,
           labelspacing=0.5)

# PNG for preview; SVG preserves sharp lines and text at any size.
for extension in ("png", "svg"):
    figure_path = output_folder / f"grid_profile_comparison.{extension}"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")

plt.show()
