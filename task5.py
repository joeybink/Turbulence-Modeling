from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def solve_channel(re_tau, n_points, kappa = 0.41, A_plus=None):

# Grid, Prandtl damping function, Prandtl mixing length, and prescribed total stress
    y_plus = np.linspace(0.0, re_tau, n_points)

    if A_plus is not None:
        f_d = 1 - np.exp(-y_plus / A_plus)
        l_m = kappa * y_plus * f_d
    else:
        l_m = kappa * y_plus
    
    T = 1.0 - y_plus / re_tau
    
# Mean velocity gradient
    g = 2.0 * T / (
        1.0 + np.sqrt(1.0 + 4.0 * l_m**2 * T)
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
    reynolds_stress = (l_m * g)**2

    return {
        "y_plus": y_plus,
        "U_plus": U_plus,
        "reynolds_stress": reynolds_stress,
    }
       
def run_task5():
    
    # Inputs
    reynolds = [180, 395, 590]
    n_points = 5121
    kappa = 0.41
    A_plus = 25 #(Damping coefficient). Source: Sumer, B.M. & Fuhrman, D.R. (2020) Turbulence in Coastal and Civil Engineering, World Scientific. https://doi.org/10.1142/10829

    dns_re_values = {
            180: 178.12,
            395: 392.24,
            590: 587.19,
    }


    output_folder = Path(__file__).resolve().parent
    data_folder = output_folder / "data"
    

    # RESEARCH-PAPER PLOTTING STYLE
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


    for re_tau in reynolds:
        model_old = solve_channel(re_tau, n_points)
        model_driest = solve_channel(re_tau, n_points, A_plus = A_plus)

        dns_re_tau = dns_re_values.get(re_tau, re_tau)

        means_file = data_folder / f"chan{re_tau}.means"
        stress_file = data_folder / f"chan{re_tau}.reystress"

        dns_means = np.loadtxt(means_file, comments="#")
        dns_stress = np.loadtxt(stress_file, comments='#')

        dns_y_plus = dns_means[:, 1]
        dns_U_plus = dns_means[:, 2]
        dns_stress_y_plus = dns_stress[:, 1]
        dns_R_plus = -dns_stress[:, 5]  

        # 4. PLOTTING COMPARISONS
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True)

        # Velocity Comparison
        axes[0].plot(
            model_old["y_plus"], model_old["U_plus"],
            "--", color="#A65628", linewidth=1.4,
            label=f"Prandtl (No damping)"
        )
        axes[0].plot(
            model_driest["y_plus"], model_driest["U_plus"],
            "-", color="#286090", linewidth=1.4,
            label=f"van Driest ($A^+={A_plus}$)"
        )
        axes[0].plot(
            dns_y_plus, dns_U_plus,
            "o", color="black", markersize=3,
            markerfacecolor="none", markeredgewidth=0.7,
            label=rf"DNS, $Re_\tau={dns_re_tau:g}$"
        )
        axes[0].set_title(f"(a) Mean velocity ($Re_\\tau={re_tau}$)", loc="left")
        axes[0].set_xlabel(r"$y^+$")
        axes[0].set_ylabel(r"$U^+$")
        axes[0].set_xlim(0, re_tau)
        axes[0].set_ylim(bottom=0)
        axes[0].legend(frameon=False, fontsize=9)

        # Reynolds Stress Comparison
        axes[1].plot(
            model_old["y_plus"], model_old["reynolds_stress"],
            "--", color="#A65628", linewidth=1.4
        )
        axes[1].plot(
            model_driest["y_plus"], model_driest["reynolds_stress"],
            "-", color="#286090", linewidth=1.4
        )
        axes[1].plot(
            dns_stress_y_plus, dns_R_plus,
            "o", color="black", markersize=3,
            markerfacecolor="none", markeredgewidth=0.7
        )
        axes[1].set_title(f"(b) Reynolds shear stress ($Re_\\tau={re_tau}$)", loc="left")
        axes[1].set_xlabel(r"$y^+$")
        axes[1].set_ylabel(r"$-\overline{u'v'}/u_\tau^2$")
        axes[1].set_xlim(0, 100)
        axes[1].set_ylim(bottom=0)

        # Zoomed in plot for behavior close to the wall
        axes[2].plot(model_old["y_plus"], model_old["reynolds_stress"], "--", color="#A65628", linewidth=1.4)
        axes[2].plot(model_driest["y_plus"], model_driest["reynolds_stress"], "-", color="#286090", linewidth=1.4)
        axes[2].plot(dns_stress_y_plus, dns_R_plus, "o", color="black", markersize=3, markerfacecolor="none", markeredgewidth=0.7)
        
        axes[2].set_title(f"(c) Reynolds shear stress close to the wall ($Re_\\tau={re_tau}$)", loc="left")
        axes[2].set_xlabel(r"$y^+$")
        axes[2].set_xlim(0, 50) 
        axes[2].set_ylim(bottom=0)

    
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)

        
        for extension in ("png", "svg"):
            figure_path = output_folder / f"task5_comparison_{re_tau}.{extension}"
            fig.savefig(figure_path, dpi=600, bbox_inches="tight")
            print(f"Figure saved to: {figure_path}")
        
    
        plt.close(fig)

if __name__ == "__main__":
    run_task5()