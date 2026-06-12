"""
Cross-validate the plane-strain FEM (growing_cylinder.py) against the analytic
ODE shooting solver (Volumetric_growth.py) for the Section 4.8 volumetric
growth law. Produces the paper's six-panel figure (Fig. 6 style) with the
growth history drawn light -> dark blue (old -> new), overlaying both solvers,
and prints a numerical agreement check on the converged state.

Project layout (this script sits at the project root):
    ODE/Volumetric_growth.py     analytic shooting solver
    FEM/growing_cylinder.py      plane-strain FEM (writes FEM/fem_history.npz)
    compare_growth.py            <- here

Recommended workflow:
    1) run the FEM once:    python FEM/growing_cylinder.py   (writes FEM/fem_history.npz)
    2) run the comparison:  python compare_growth.py
If FEM/fem_history.npz is missing, the script tries to import and run the FEM
live (needs dolfinx); failing that, it plots the ODE alone.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_dir(name):
    """Locate the ODE/ or FEM/ folder relative to this script (or CWD)."""
    for cand in (os.path.join(ROOT, name),
                 os.path.join(ROOT, "..", name),
                 os.path.join(os.getcwd(), name)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.join(ROOT, name)              # default location even if absent


ODE_DIR = _find_dir("ODE")                       # holds Volumetric_growth.py
FEM_DIR = _find_dir("FEM")                       # holds growing_cylinder.py + fem_history.npz
sys.path.insert(0, ODE_DIR)
sys.path.insert(0, FEM_DIR)

# ---- matched problem parameters (MUST equal those in growing_cylinder.py) ----
RI, RO = 1.0, 2.0
MU, P, KG, W = 1.0, 0.5, 1.0, 1.0
N_GRID = 16                      # ODE radial grid points
N_SNAP = 8                       # number of blue history curves to draw
ODE_DT, ODE_MAXSTEPS, ODE_TOL = 0.01, 3000, 1e-4
FEM_CACHE = os.path.join(FEM_DIR, "fem_history.npz")

FIELDS = ["sigma_rr", "sigma_theta", "mean_stress", "gamma", "alpha_theta", "ln_J"]


# =====================================================================
# 1. ODE solver -> list of per-step field dicts
# =====================================================================
def run_ode():
    from Volumetric_growth import VolumetricCylinderSolver
    R = np.linspace(RI, RO, N_GRID)
    n = len(R)
    init = VolumetricCylinderSolver(R, np.ones(n), np.ones(n), MU, P, kg=KG, w_v=W)
    final, hist = init.run(dt=ODE_DT, max_steps=ODE_MAXSTEPS, tol=ODE_TOL,
                           print_every=0, return_history=True)
    snaps = []
    for st in hist:
        _, r, p = st.solve()
        snaps.append({
            "R": R,
            "sigma_rr": st.cauchy_radial(r, p),
            "sigma_theta": st.cauchy_hoop(r, p),
            "mean_stress": st.mean_stress(r, p),
            "gamma": st.gr.copy(),
            "alpha_theta": r / (R * st.gt),
            "ln_J": st.log_J(),
        })
    return snaps


# =====================================================================
# 2. FEM history: cache file -> live import -> unavailable
# =====================================================================
def _snaps_from_arrays(d):
    R = d["R"]
    out = []
    for i in range(d["sigma_rr"].shape[0]):
        g = d["gamma"][i]
        srr, stt = d["sigma_rr"][i], d["sigma_theta"][i]
        out.append({
            "R": R,
            "sigma_rr": srr,
            "sigma_theta": stt,
            "mean_stress": 0.5 * (srr + stt),
            "gamma": g,
            "alpha_theta": d["alpha_theta"][i],
            "ln_J": 2.0 * np.log(g),          # plane strain: J = gamma^2
        })
    return out


def load_fem():
    if os.path.exists(FEM_CACHE):
        print(f"[FEM] loading cached history from {os.path.basename(FEM_CACHE)}")
        return _snaps_from_arrays(np.load(FEM_CACHE)), "cache"
    try:
        from growing_cylinder import run_fem
    except Exception as exc:
        print(f"[FEM] no cache and cannot import run_fem "
              f"({exc.__class__.__name__}: {exc}); plotting ODE only.")
        return None, "none"
    print("[FEM] running live (this needs dolfinx)...")
    _, snaps = run_fem(p_target=P, kg=KG, w=W, return_snapshots=True)
    return snaps, "live"


# =====================================================================
# 3. helpers
# =====================================================================
def subsample(snaps, k):
    idx = np.unique(np.round(np.linspace(0, len(snaps) - 1, k)).astype(int))
    return [snaps[i] for i in idx]


def agreement(ode_final, fem_final):
    """Max abs difference per field on a common reference-radius grid."""
    lo = max(ode_final["R"].min(), fem_final["R"].min())
    hi = min(ode_final["R"].max(), fem_final["R"].max())
    rg = np.linspace(lo, hi, 50)
    report = {}
    for f in FIELDS:
        a = np.interp(rg, ode_final["R"], ode_final[f])
        b = np.interp(rg, fem_final["R"], fem_final[f])
        denom = max(np.max(np.abs(a)), 1e-12)
        report[f] = (np.max(np.abs(a - b)), np.max(np.abs(a - b)) / denom)
    return rg, report


# =====================================================================
# 4. plotting
# =====================================================================
PANELS = [
    ("sigma_rr",    r"$\sigma^{rr}$",                 "Radial stress"),
    ("mean_stress", r"$\frac{1}{2}(\sigma^{rr}+\sigma^{\theta\theta})$", "Mean stress (driver)"),
    ("gamma",       r"$\gamma_r=\gamma_\theta$",      "Growth (isotropic)"),
    ("sigma_theta", r"$\sigma^{\theta\theta}$",       "Hoop stress"),
    ("alpha_theta", r"$\alpha_\theta$",               "Hoop stretch"),
    ("ln_J",        r"$\ln J$",                       r"$\ln J=\ln(\gamma_r\gamma_\theta)$"),
]


def plot_six_panel(ode_snaps, fem_snaps, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ode = subsample(ode_snaps, N_SNAP)
    fem = subsample(fem_snaps, N_SNAP) if fem_snaps is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    ax = axes.flat

    def blue(j, n):
        return plt.cm.Blues(0.25 + 0.55 * (j + 1) / (n + 1))

    for k, (key, ylab, title) in enumerate(PANELS):
        a = ax[k]
        for j, st in enumerate(ode):
            lw = 2.2 if j == len(ode) - 1 else 1.1
            a.plot(st["R"], st[key], color=blue(j, len(ode)), lw=lw, alpha=0.9, zorder=2)
        if fem is not None:
            for j, st in enumerate(fem):
                me = max(1, len(st["R"]) // 8)        # thin the markers
                a.plot(st["R"][::me], st[key][::me], color=blue(j, len(fem)),
                       ls="none", marker="o", ms=3.4, mfc="none", mew=1.0, zorder=3)
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10)
        a.grid(True, alpha=0.3)
    for a in axes[1]:
        a.set_xlabel("$R$")

    handles = [Line2D([], [], color=plt.cm.Blues(0.75), lw=2, label="ODE (lines)")]
    if fem is not None:
        handles.append(Line2D([], [], color=plt.cm.Blues(0.75), ls="none",
                              marker="o", ms=4, mfc="none", label="FEM (markers)"))
    handles += [Line2D([], [], color=plt.cm.Blues(0.4), lw=2, label="initial"),
                Line2D([], [], color=plt.cm.Blues(0.95), lw=2, label="final")]
    ax[0].legend(handles=handles, fontsize=8, loc="best")

    fig.suptitle(rf"Volumetric growth, plane strain:  $p={P}$,  $k_g={KG}$,  "
                 rf"$R_i={RI}$, $R_o={RO}$   (light$\to$dark = old$\to$new)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_agreement(ode_final, fem_final, rg, report, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axx, key, lab in [(axes[0], "sigma_rr", r"$\sigma^{rr}$"),
                          (axes[1], "sigma_theta", r"$\sigma^{\theta\theta}$")]:
        axx.plot(ode_final["R"], ode_final[key], "-", color="tab:blue", lw=2, label="ODE")
        axx.plot(fem_final["R"], fem_final[key], "o", color="tab:red", ms=3.5,
                 mfc="none", label="FEM")
        axx.set_xlabel("$R$"); axx.set_ylabel(lab)
        axx.set_title(rf"{lab}: max$|$FEM$-$ODE$|$ = {report[key][0]:.2e}", fontsize=10)
        axx.grid(True, alpha=0.3); axx.legend(fontsize=9)
    fig.suptitle("Converged-state agreement (FEM vs ODE)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_path}")


# =====================================================================
# 5. main
# =====================================================================
def main():
    print("=== ODE ===")
    ode_snaps = run_ode()
    print(f"  {len(ode_snaps)} ODE history states")

    print("=== FEM ===")
    fem_snaps, src = load_fem()
    if fem_snaps is not None:
        print(f"  {len(fem_snaps)} FEM history states (source: {src})")

    print("=== plotting ===")
    plot_six_panel(ode_snaps, fem_snaps, os.path.join(ROOT, "growth_comparison.png"))

    if fem_snaps is not None:
        ode_final, fem_final = ode_snaps[-1], fem_snaps[-1]
        rg, report = agreement(ode_final, fem_final)
        plot_agreement(ode_final, fem_final, rg, report,
                       os.path.join(ROOT, "growth_agreement.png"))
        print("\n=== converged-state agreement (max over R) ===")
        print(f"  {'field':<14}{'abs diff':>12}{'rel diff':>12}")
        worst = 0.0
        for f in FIELDS:
            ad, rd = report[f]
            worst = max(worst, rd)
            print(f"  {f:<14}{ad:>12.3e}{rd:>12.2%}")
        tol = 0.05
        print(f"\n  {'PASS' if worst < tol else 'FAIL'}: worst relative diff "
              f"= {worst:.2%} (tol {tol:.0%})")
    else:
        print("\n  (FEM unavailable -> ODE-only figure; run growing_cylinder.py "
              "to generate fem_history.npz, then re-run.)")


if __name__ == "__main__":
    main()