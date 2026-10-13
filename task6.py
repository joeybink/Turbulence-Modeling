from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from task5 import solve_channel

def calculate_laminar_profile(y_plus, re_tau):

    return y_plus - (y_plus**2) / (2.0 * re_tau)



def run_task6():
    # Inputs
    reynolds = [180, 395, 590]
    n_points = 5121
    A_plus = 25 
    output_folder = Path(__file__).resolve().parent

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

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

    for idx, re_tau in enumerate(reynolds):
        model_data = solve_channel(re_tau, n_points, A_plus=A_plus)
        
        y_plus = model_data["y_plus"]
        u_turb = model_data["U_plus"]
        u_lam = calculate_laminar_profile(y_plus, re_tau)

        ax = axes[idx]
        ax.plot(y_plus, u_turb, "-", color="#286090", linewidth=1.5, label="Turbulent (van Driest)")
        ax.plot(y_plus, u_lam, "--", color="#A65628", linewidth=1.5, label="Laminar Reference")
        
        ax.set_title(rf"$Re_\tau = {re_tau}$")
        ax.set_xlabel(r"$y^+$")
        
        if idx == 0:
            ax.set_ylabel(r"$U^+$")
            ax.legend(frameon=False)
            
        ax.set_xlim(0, re_tau)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()

    # Save outputs
    for extension in ("png", "svg"):
        figure_path = output_folder / f"task6_laminar_comparison.{extension}"
        fig.savefig(figure_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {figure_path}")
        
    plt.show()

if __name__ == "__main__":
    run_task6()