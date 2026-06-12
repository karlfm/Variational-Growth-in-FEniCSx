"""
Variational *volumetric* growth solver for a pressurized cylinder.

Implements the volumetric growth law of Section 4.8 of
"Boundary Conditions Select the Homeostatic Target in Variational Growth".

Isotropic growth is assumed,

    γ_r = γ_θ = J^{1/2},          J = det F^g = γ_r γ_θ,

together with the growth energy  W_g = (k_g / 2) (ln J)^2.  With the
logarithmic growth variable v = ln J and the dissipation potential
D = ½ w v̇², variation with respect to v gives the gradient flow (eq. 16)

    w v̇ = ½ (σ^rr + σ^θθ) − k_g ln J,

which is the gradient flow of the (coercive, dissipative) Lyapunov
functional V = −L + P + W_g.  The volumetric driver is the *mean* stress,
and the law equilibrates (eq. 17) at

    ln J = (σ^rr + σ^θθ) / (2 k_g),

with k_g setting the residual mean stress the tissue tolerates.  Unlike the
mean-subtracted hoop law (Section 4.7 / Mean_stress_difference_growth.py),
the transmural hoop-stress gradient is *not* removed: volumetric growth
regulates the overall stress level, not its distribution.

The kinematic / equilibrium / stress equations are identical to the
hoop-growth solver; only the growth driver and the γ_r = γ_θ constraint
differ.
"""

import numpy as np
import os
import json

try:
    from scipy.optimize import brentq
except ModuleNotFoundError:                     # scipy-free fallback (bisection)
    def brentq(f, a, b, xtol=1e-12, maxiter=200):
        fa, fb = f(a), f(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            raise ValueError("brentq fallback: f(a), f(b) must bracket a root")
        for _ in range(maxiter):
            m = 0.5 * (a + b)
            fm = f(m)
            if fm == 0.0 or 0.5 * (b - a) < xtol:
                return m
            if fa * fm < 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class VolumetricCylinderSolver:

    def __init__(self, R, gr, gt, mu, bc, kg=1.0, w_v=1.0):
        self.R  = np.asarray(R, dtype=float)        # material radius
        self.gr = np.asarray(gr, dtype=float)       # radial growth field (γ_r)
        self.gt = np.asarray(gt, dtype=float)       # hoop growth field (γ_θ); γ_r = γ_θ
        self.mu = float(mu)                         # shear modulus
        self.bc = float(bc)                         # inner radial stress σ^rr(Ri) = -bc
        self.kg = float(kg)                         # growth-energy stiffness (sets residual mean stress)
        self.w_v = float(w_v)                       # dissipation / inverse growth rate
        self.N  = len(R)

    # ── Kinematics: r(R) from incompressibility, r² = ri² + 2∫ γr γθ s ds ──
    def compute_r(self, ri):
        f = self.gr * self.gt * self.R
        I = np.zeros(self.N, dtype=float)
        for i in range(1, self.N):
            ds = self.R[i] - self.R[i - 1]
            I[i] = I[i - 1] + 0.5 * (f[i - 1] + f[i]) * ds
        return np.sqrt(ri**2 + 2.0 * I)

    # ── Pressure p(R) from balance of momentum (Div P = 0) for neo-Hookean ──
    def compute_p(self, r):
        s = self.R
        dgt_ds = np.gradient(self.gt, s)
        integrand = self.mu * (
            2.0 * s * self.gt**2 / r**2
            + 2.0 * s**2 * self.gt * dgt_ds / r**2
            - s**3 * self.gr * self.gt**3 / r**4
            - self.gr / (s * self.gt)
        )
        Ri = s[0]
        rRi = r[0]
        gfRi = self.gt[0]
        sigma_rr_Ri = -self.bc
        p_i = self.mu * (Ri**2 / rRi**2) * gfRi**2 - sigma_rr_Ri
        p = np.zeros(self.N, dtype=float)
        p[0] = p_i
        for i in range(1, self.N):
            ds = s[i] - s[i - 1]
            p[i] = p[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * ds
        return p

    # ── Cauchy stress ─────────────────────────────────────────────
    def cauchy_radial(self, r, p):
        # σ^rr = μ (R²/r²) γθ² − p ; radial elastic stretch α_r = (R/r) γθ.
        a_r2 = (self.R * self.gt / r) ** 2
        return self.mu * a_r2 - p

    def cauchy_hoop(self, r, p):
        # σ^θθ = μ (r²/R²) /γθ² − p
        a_t2 = (r / (self.R * self.gt)) ** 2
        return self.mu * a_t2 - p

    # ── BVP solver: outer Cauchy radial stress = 0 ────────────────
    def solve(self):
        def obj(ri):
            r = self.compute_r(ri)
            p = self.compute_p(r)
            return self.cauchy_radial(r, p)[-1]

        lo, hi = 0.3 * self.R[0], 8.0 * self.R[-1]
        pts = np.linspace(lo, hi, 100)
        vals = np.array([obj(x) for x in pts])

        # Prefer a sign change going from + to - (physical branch).
        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k + 1]):
                if vals[k] > 0 and vals[k + 1] < 0:
                    ri = brentq(obj, pts[k], pts[k + 1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p

        for k in range(len(vals) - 1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k + 1]):
                if vals[k] * vals[k + 1] < 0:
                    ri = brentq(obj, pts[k], pts[k + 1], xtol=1e-12)
                    r = self.compute_r(ri)
                    p = self.compute_p(r)
                    return ri, r, p

        raise RuntimeError("No root found")

    # ── Volumetric growth driver ──────────────────────────────────
    def mean_stress(self, r, p):
        """½(σ^rr + σ^θθ) — the volumetric (mean-stress) driver."""
        return 0.5 * (self.cauchy_radial(r, p) + self.cauchy_hoop(r, p))

    def log_J(self):
        """v = ln J = ln(γ_r γ_θ)."""
        return np.log(self.gr * self.gt)

    def growth_rate_v(self, r, p):
        """v̇ = [ ½(σ^rr + σ^θθ) − k_g ln J ] / w   (eq. 16)."""
        return (self.mean_stress(r, p) - self.kg * self.log_J()) / self.w_v

    # ── Time stepping (forward Euler on v = ln J, isotropic) ──────
    def run(self, dt, max_steps=10000, tol=1e-4, print_every=500,
            return_history=False):
        cur = self
        history = [cur] if return_history else None
        for i in range(1, max_steps + 1):
            _, r, p = cur.solve()
            vdot = cur.growth_rate_v(r, p)
            mr = np.max(np.abs(vdot))
            if mr < tol:
                print(f"  converged at step {i}, max|v̇|={mr:.2e}")
                if return_history:
                    return cur, history
                return cur
            # v_{n+1} = v_n + dt v̇ ; isotropic: γ_r = γ_θ = exp(v/2) = J^{1/2}
            v_new = cur.log_J() + dt * vdot
            g_new = np.exp(0.5 * v_new)
            cur = VolumetricCylinderSolver(
                self.R, g_new, g_new, self.mu, self.bc,
                kg=self.kg, w_v=self.w_v,
            )
            if return_history:
                history.append(cur)
            if print_every and i % print_every == 0:
                _, rr, pp = cur.solve()
                lnJ = cur.log_J()
                print(f"  step {i:>5d}  γ=[{cur.gr.min():.4f},{cur.gr.max():.4f}]"
                      f"  lnJ=[{lnJ.min():.4f},{lnJ.max():.4f}]"
                      f"  max|v̇|={mr:.2e}")
        print(f"  did not converge after {max_steps}, max|v̇|={mr:.2e}")
        if return_history:
            return cur, history
        return cur


# =====================================================================
#  Plotting (mirrors Figure 6 of the paper)
# =====================================================================
def plot_volumetric(R, states, p_overload, kg, out_path):
    """Six-panel evolution matching Figure 6:
        top:    σ^rr, mean-stress driver ½(σ^rr+σ^θθ), isotropic growth γ
        bottom: σ^θθ, hoop stretch α_θ, ln J
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("  (matplotlib not available — skipping plot)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
    for j, st in enumerate(states):
        frac = (j + 1) / (len(states) + 1)
        col = plt.cm.Blues(0.25 + 0.55 * frac)
        lw = 2.0 if j == len(states) - 1 else 1.2
        lbl = "Initial" if j == 0 else ("Final" if j == len(states) - 1 else None)

        axes[0, 0].plot(R, st["sigma_rr"], color=col, lw=lw, alpha=0.85, label=lbl)
        axes[0, 1].plot(R, st["mean_stress"], color=col, lw=lw, alpha=0.85)
        axes[0, 2].plot(R, st["gamma"], color=col, lw=lw, alpha=0.85)
        axes[1, 0].plot(R, st["sigma_tt"], color=col, lw=lw, alpha=0.85)
        axes[1, 1].plot(R, st["alpha_theta"], color=col, lw=lw, alpha=0.85)
        axes[1, 2].plot(R, st["log_J"], color=col, lw=lw, alpha=0.85)

    axes[0, 0].set_ylabel(r"$\sigma^{rr}$"); axes[0, 0].set_title("Radial stress")
    axes[0, 1].set_ylabel(r"$\frac{1}{2}\mathrm{tr}\,\sigma$"); axes[0, 1].set_title("Mean stress (driver)")
    axes[0, 2].set_ylabel(r"$\gamma_r=\gamma_\theta$"); axes[0, 2].set_title("Growth (isotropic)")
    axes[1, 0].set_ylabel(r"$\sigma^{\theta\theta}$"); axes[1, 0].set_title("Hoop stress")
    axes[1, 1].set_ylabel(r"$\alpha_\theta$"); axes[1, 1].set_title("Hoop stretch")
    axes[1, 2].set_ylabel(r"$\ln J$"); axes[1, 2].set_title(r"$\ln J=\ln(\gamma_r\gamma_\theta)$")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("$R$")
    axes[0, 0].legend(fontsize=8, loc="best")

    fig.suptitle(rf"Volumetric growth: $p={p_overload}$,  "
                 rf"$W_g=\frac{{k_g}}{{2}}(\ln J)^2$,  $k_g={kg}$", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =====================================================================
#  Demonstration: pressure overload + isotropic volumetric growth
# =====================================================================
if __name__ == "__main__":
    R  = np.linspace(1.0, 2.0, 16)
    N  = len(R)
    mu = 1.0
    p_overload = 0.5     # applied inner pressure (compressive σ^rr at inner)
    kg = 1.0             # growth-energy stiffness (Figure 6)
    w_v = 1.0

    print(f"=== Volumetric growth at p={p_overload}, k_g={kg} ===")
    init = VolumetricCylinderSolver(
        R, np.ones(N), np.ones(N), mu, p_overload, kg=kg, w_v=w_v,
    )

    _, r0, p0 = init.solve()
    m0 = init.mean_stress(r0, p0)
    print(f"  Initial mean stress ½tr(σ) range = [{m0.min():.4f}, {m0.max():.4f}]")

    final, history = init.run(
        dt=0.001, max_steps=20000, tol=5e-4, return_history=True
    )
    _, r_f, p_f = final.solve()

    # equilibrium check: ln J should approach (σ^rr+σ^θθ)/(2 k_g)  (eq. 17)
    lnJ_f = final.log_J()
    target = final.mean_stress(r_f, p_f) / kg          # = (σ^rr+σ^θθ)/(2 k_g)
    print(f"  Final ln J range          = [{lnJ_f.min():.4f}, {lnJ_f.max():.4f}]")
    print(f"  Eq.(17) target ½tr σ / k_g = [{target.min():.4f}, {target.max():.4f}]")
    print(f"  max|ln J − target|        = {np.max(np.abs(lnJ_f - target)):.2e}")

    # ── Sample 8 evenly-spaced snapshots ──
    n_steps = 8
    if len(history) >= 2:
        raw_idx = np.linspace(0, len(history) - 1, num=min(n_steps, len(history)))
        sampled_idx = np.unique(np.round(raw_idx).astype(int))
    else:
        sampled_idx = np.array([0], dtype=int)
    sampled_states = [history[i] for i in sampled_idx]

    sampled_data = []
    for j, state in enumerate(sampled_states):
        _, r_m, p_m = state.solve()
        sampled_data.append({
            "sample_number": int(j + 1),
            "history_index": int(sampled_idx[j]),
            "R": R.tolist(),
            "r": r_m.tolist(),
            "pressure_p": p_m.tolist(),
            "sigma_rr": state.cauchy_radial(r_m, p_m).tolist(),
            "sigma_tt": state.cauchy_hoop(r_m, p_m).tolist(),
            "mean_stress": state.mean_stress(r_m, p_m).tolist(),
            "alpha_r": (R * state.gt / r_m).tolist(),
            "alpha_theta": (r_m / (R * state.gt)).tolist(),
            "gamma": state.gr.tolist(),          # γ_r = γ_θ
            "log_J": state.log_J().tolist(),
            "J": (state.gr * state.gt).tolist(),
        })

    plot_volumetric(R, sampled_data, p_overload, kg,
                    os.path.join(OUTPUT_DIR, "volumetric_growth.png"))

    # ── Residual stress in the final configuration (unloaded) ─────
    residual_solver = VolumetricCylinderSolver(
        R, final.gr, final.gt, mu, 0.0, kg=kg, w_v=w_v,
    )
    ri_res, r_res, p_res = residual_solver.solve()
    srr_res = residual_solver.cauchy_radial(r_res, p_res)
    stt_res = residual_solver.cauchy_hoop(r_res, p_res)

    # ── Diagnostics ──────────────────────────────────────────────
    print(f"\n=== Diagnostics ===")
    print(f"  Reference Ri={R[0]:.4f}, Ro={R[-1]:.4f}")
    print(f"  Final r_i = {r_f[0]:.4f},  r_o = {r_f[-1]:.4f}")
    print(f"  γ range:   [{final.gr.min():.4f}, {final.gr.max():.4f}]")
    print(f"  J range:   [{(final.gr*final.gt).min():.4f}, {(final.gr*final.gt).max():.4f}]")
    print(f"  Residual σ^rr range: [{srr_res.min():.4f}, {srr_res.max():.4f}]")
    print(f"  Residual σ^θθ range: [{stt_res.min():.4f}, {stt_res.max():.4f}]")

    # ── Export JSON ──────────────────────────────────────────────
    export = {
        "metadata": {
            "model": "volumetric (Section 4.8)",
            "mu": float(mu),
            "p": float(p_overload),
            "kg": float(kg),
            "w_v": float(w_v),
            "dt": 0.001,
            "tol": 5e-4,
            "n_grid": int(N),
            "n_history_states": int(len(history)),
            "n_sampled_states": int(len(sampled_states)),
        },
        "R": R.tolist(),
        "r_final": r_f.tolist(),
        "sampled_indices": sampled_idx.tolist(),
        "states": sampled_data,
        "residual": {
            "r": r_res.tolist(),
            "sigma_rr": srr_res.tolist(),
            "sigma_tt": stt_res.tolist(),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "volumetric_growth_data.json"),
              "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print("\nSaved plot and data to outputs/.")
