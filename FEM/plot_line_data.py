# Plot radial profiles from line_data.csv produced by pressurized_cylinder.py.
# 2x2 panels: column 1 = radial stress, hoop stress; column 2 = radial stretch, hoop stretch.
# Run: python plot_line_data.py [line_data.csv] [output.png]
import sys
import numpy as np
import matplotlib.pyplot as plt

csv_file = sys.argv[1] if len(sys.argv) > 1 else "line_data_no_growth.csv"
out_file = sys.argv[2] if len(sys.argv) > 2 else "line_data_no_growth.png"

data = np.genfromtxt(csv_file, delimiter=",", names=True)
r = data["r"]

fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True)

panels = [
    (ax[0, 0], data["sigma_rr"],       r"$\sigma_{rr}$",           "tab:blue"),
    (ax[1, 0], data["sigma_theta"],    r"$\sigma_{\theta\theta}$", "tab:red"),
    (ax[0, 1], data["lambda_r"],       r"$\lambda_r$",             "tab:green"),
    (ax[1, 1], data["lambda_theta"],   r"$\lambda_\theta$",        "tab:orange"),
]
for a, y, label, color in panels:
    a.plot(r, y, color=color, lw=2)
    a.set_ylabel(label)
    a.grid(alpha=0.3)
ax[0, 0].axhline(0.0, color="k", lw=0.8, ls=":")   # zero reference on stresses
ax[1, 0].axhline(0.0, color="k", lw=0.8, ls=":")
ax[0, 1].axhline(1.0, color="k", lw=0.8, ls=":")   # undeformed reference on stretches
ax[1, 1].axhline(1.0, color="k", lw=0.8, ls=":")
ax[1, 0].set_xlabel(r"$r$ (reference radial coordinate)")
ax[1, 1].set_xlabel(r"$r$ (reference radial coordinate)")
fig.suptitle("Pressurized cylinder: stress and stretch at mid-height")
fig.tight_layout()
fig.savefig(out_file, dpi=200)
print(f"Wrote {out_file}")