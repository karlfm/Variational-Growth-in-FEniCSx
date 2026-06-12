"""
Plotting for the variational growth solver (ODE_growth.py).

All matplotlib code lives here so that ODE_growth.py stays a pure solver.
The functions take plain arrays/dicts, so they can be driven either from an
in-memory run (ODE_growth.py imports and calls them) or standalone from the
exported JSON:

    python plot_growth.py                       # uses outputs/simple_growth_data.json
    python plot_growth.py path/to/data.json
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def plot_evolution(R, states, p_overload, out_path):
    """Evolution of stresses, stretches and growth across sampled snapshots.

    `states` is a list of dicts (earliest → latest), each with keys:
    sigma_rr, sigma_tt, alpha_r, alpha_theta, gamma_r, gamma_theta.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
    for j, st in enumerate(states):
        frac = (j + 1) / (len(states) + 1)
        col = plt.cm.Blues(0.25 + 0.55 * frac)
        if j == 0:
            lbl = "Initial (overloaded)"
        elif j == len(states) - 1:
            lbl = "Final (remodeled)"
        else:
            lbl = None

        axes[0, 0].plot(R, st["sigma_rr"], color=col, lw=1.2, alpha=0.85, label=lbl)
        axes[1, 0].plot(R, st["sigma_tt"], color=col, lw=1.2, alpha=0.85)
        axes[0, 1].plot(R, st["alpha_r"], color=col, lw=1.2, alpha=0.85)
        axes[1, 1].plot(R, st["alpha_theta"], color=col, lw=1.2, alpha=0.85)
        axes[0, 2].plot(R, st["gamma_r"], color=col, lw=1.2, alpha=0.85)
        axes[1, 2].plot(R, st["gamma_theta"], color=col, lw=1.2, alpha=0.85)

    axes[1, 0].axhline(0, color="black", ls="--", lw=0.8, alpha=0.7)
    axes[1, 0].legend(fontsize=7, loc="best")

    axes[0, 0].set_ylabel(r"$\sigma^{rr}$"); axes[0, 0].set_title("Radial stress")
    axes[1, 0].set_ylabel(r"$\sigma^{\theta\theta}$"); axes[1, 0].set_title("Hoop stress")
    axes[0, 1].set_ylabel(r"$\alpha_r$"); axes[0, 1].set_title("Radial stretch")
    axes[1, 1].set_ylabel(r"$\alpha_\theta$"); axes[1, 1].set_title("Hoop stretch")
    axes[0, 2].set_ylabel(r"$\gamma_r$"); axes[0, 2].set_title("Radial growth")
    axes[1, 2].set_ylabel(r"$\gamma_\theta$"); axes[1, 2].set_title("Hoop growth")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("$R$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=9,
               title="Time: lighter blue earlier, darker blue later",
               title_fontsize=9)

    fig.suptitle(rf"Growth: $p={p_overload}$",
                 fontsize=13)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual(R, sigma_rr, sigma_tt, out_path):
    """Residual radial and hoop stress in the unloaded (p=0) grown config."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    axes[0].plot(R, sigma_rr, lw=1.5, color="C0")
    axes[1].plot(R, sigma_tt, lw=1.5, color="C0")
    axes[0].set_ylabel(r"$\sigma^{rr}$"); axes[0].set_title("Residual radial stress")
    axes[1].set_ylabel(r"$\sigma^{\theta\theta}$"); axes[1].set_title("Residual hoop stress")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$R$")
    fig.suptitle("Residual stress at $p=0$ after growth", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_from_json(json_path, output_dir=OUTPUT_DIR):
    """Regenerate both figures from an exported simple_growth_data.json."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    R = data["R"]
    meta = data["metadata"]
    plot_evolution(
        R, data["states"], meta["p"],
        os.path.join(output_dir, "simple_growth.png"),
    )

    res = data.get("residual")
    if res is not None:
        plot_residual(
            R, res["sigma_rr"], res["sigma_tt"],
            os.path.join(output_dir, "simple_growth_residual_stress.png"),
        )
    else:
        print("  (no 'residual' block in JSON — skipping residual-stress plot)")


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(OUTPUT_DIR, "simple_growth_data.json")
    plot_from_json(json_path)
    print(f"Saved plots to {OUTPUT_DIR}/ from {json_path}")
