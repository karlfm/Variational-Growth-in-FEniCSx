# Pressurized hollow cylinder, incompressible neo-Hookean material, PLANE STRAIN.
# Mixed (u, p) formulation, follower pressure (Neumann) on inner/outer walls,
# u_z = 0 enforced on the whole domain so lambda_z = 1 (plane strain). Isotropic
# *in-plane* volumetric growth Fg = diag(gamma, gamma, 1), Jg = gamma^2, driven by
# the in-plane mean Cauchy stress -- this matches the 2D plane-strain ODE shooting
# solver in ODE/Volumetric_growth.py so compare_cylinder.py can cross-validate them.
# Written for dolfinx 0.9.
#
# Run standalone:   python growing_cylinder.py [mesh.msh]
# Or import:        from growing_cylinder import run_fem
#                   _, snaps = run_fem(p_target=0.5, kg=1.0, w=1.0, return_snapshots=True)
# run_fem() writes fem_history.npz next to this file; compare_cylinder.py loads it.
import os
import sys
import numpy as np, ufl, basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh as dmesh, geometry
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

mu = 1.0                            # shear modulus
INNER, OUTER, BOTTOM, TOP = 1, 2, 3, 4
HERE = os.path.dirname(os.path.abspath(__file__))


def radius(p): return np.sqrt(p[0] ** 2 + p[1] ** 2)


def run_fem(p_target=0.5, kg=1.0, w=1.0, msh_file=None,
            n_steps=40, dt=0.2, n_line=60, write_outputs=True,
            return_snapshots=False, cache_path=None):
    """Solve the plane-strain growing cylinder and return its growth history.

    Parameters
    ----------
    p_target : float   final inner pressure (ramped up from 0); matches compare's P.
    kg, w    : float   growth-energy stiffness and dissipation coefficient.
    msh_file : str     gmsh mesh; defaults to cylinder.msh next to this file.
    n_steps, dt        forward-Euler growth steps and step size.
    n_line             number of radial sample points for the history snapshots.

    Returns
    -------
    (final_snapshot, snapshots) if return_snapshots else final_snapshot.
    Each snapshot is a dict with keys R, sigma_rr, sigma_theta, mean_stress,
    gamma, alpha_theta, ln_J -- the same schema compare_cylinder.py consumes.
    Also writes fem_history.npz (stacked arrays) for the comparison's cache path.
    """
    if msh_file is None:
        msh_file = os.path.join(HERE, "cylinder.msh")
    if cache_path is None:
        cache_path = os.path.join(HERE, "fem_history.npz")

    # ---- Read the mesh (axis assumed along z, base at z = 0) ----
    mesh, _, _ = io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, 0, gdim=3)

    # ---- Infer geometry from the mesh coordinates ----
    x = mesh.geometry.x
    rr = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    Ri, Ro = rr.min(), rr.max()
    z0, H = x[:, 2].min(), x[:, 2].max()
    print(f"Mesh: Ri = {Ri:.4f}, Ro = {Ro:.4f}, z in [{z0:.4f}, {H:.4f}]")

    # ---- Tag boundary facets geometrically ----
    fdim = mesh.topology.dim - 1
    tol = 1e-6 * Ro
    markers = [(INNER,  lambda p: np.isclose(radius(p), Ri, atol=1e-3 * Ro)),
               (OUTER,  lambda p: np.isclose(radius(p), Ro, atol=1e-3 * Ro)),
               (BOTTOM, lambda p: np.isclose(p[2], z0, atol=tol)),
               (TOP,    lambda p: np.isclose(p[2], H, atol=tol))]
    indices, values = [], []
    for tag, marker in markers:
        facets = dmesh.locate_entities_boundary(mesh, fdim, marker)
        indices.append(facets)
        values.append(np.full_like(facets, tag))
    indices = np.hstack(indices).astype(np.int32)
    values = np.hstack(values).astype(np.int32)
    order = np.argsort(indices)
    facet_tags = dmesh.meshtags(mesh, fdim, indices[order], values[order])

    # ---- Mixed Taylor-Hood space: P2 displacement, P1 pressure ----
    Vel = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(3,))
    Pel = basix.ufl.element("P", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, basix.ufl.mixed_element([Vel, Pel]))
    wsol = fem.Function(W)
    u, p = ufl.split(wsol)
    v, q = ufl.TestFunctions(W)

    # ---- In-plane isotropic volumetric growth Fg = diag(gamma, gamma, 1) ----
    # gamma is a scalar field; Jg = det(Fg) = gamma^2 (plane strain: no z growth).
    # Growth energy Wg = kg/2 (ln Jg)^2 is independent of (u, p), so it does NOT
    # enter the mechanical weak form (it only drives the growth law below).
    Vg = fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1))
    gamma = fem.Function(Vg)           # isotropic in-plane growth stretch field
    gamma.x.array[:] = 1.0             # start from the ungrown state

    # ---- Incompressible neo-Hookean energy on the *elastic* part Fe = F Fg^{-1} ----
    # The reference integrand is weighted by Jg, so equilibrium is taken on the
    # grown configuration. p enforces the elastic incompressibility constraint
    # Je = det(Fe) = 1.
    F = ufl.Identity(3) + ufl.grad(u)
    J = ufl.det(F)                     # total Jacobian (used by the follower-pressure term)
    Fg_inv = ufl.as_matrix([[1.0 / gamma, 0, 0],
                            [0, 1.0 / gamma, 0],
                            [0, 0, 1.0]])
    Fe = F * Fg_inv                    # elastic deformation (no growth in z)
    Je = ufl.det(Fe)                   # elastic Jacobian, constrained to 1
    Jg = gamma ** 2                    # grown (in-plane) volume form
    psi = (mu / 2 * (ufl.tr(Fe.T * Fe) - 3) - p * (Je - 1)) * Jg
    Pi = psi * ufl.dx

    # ---- Neumann boundary conditions ----
    p_in, p_out = fem.Constant(mesh, 0.0), fem.Constant(mesh, 0.0)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    # Very weak Robin (elastic spring) on the outer wall: traction = -k_robin * u.
    # Removes the in-plane rigid-body modes.
    k_robin = fem.Constant(mesh, 1e-4 * mu / Ro)
    Res = ufl.derivative(Pi, wsol) \
        + p_in * J * ufl.dot(ufl.inv(F).T * n, v) * ds(INNER) \
        + p_out * J * ufl.dot(ufl.inv(F).T * n, v) * ds(OUTER) \
        + k_robin * ufl.dot(u, v) * ds(OUTER)

    # ---- Dirichlet BC: u_z = 0 on the WHOLE domain -> plane strain (lambda_z = 1) ----
    Wz, _ = W.sub(0).sub(2).collapse()
    uz_zero = fem.Function(Wz)
    dofs_z = fem.locate_dofs_geometrical(
        (W.sub(0).sub(2), Wz), lambda p: np.full(p.shape[1], True))
    bc_z = fem.dirichletbc(uz_zero, dofs_z, W.sub(0).sub(2))

    # ---- Newton solver with load stepping ----
    problem = NonlinearProblem(Res, wsol, bcs=[bc_z])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    opts[f"{ksp.getOptionsPrefix()}ksp_type"] = "preonly"
    opts[f"{ksp.getOptionsPrefix()}pc_type"] = "lu"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.WARNING)

    # ---- Cauchy stress on the elastic part (Je = 1) and right Cauchy-Green ----
    sigma = mu * Fe * Fe.T - p * ufl.Identity(3)
    C = F.T * F
    # In-plane mean Cauchy stress = (sigma_xx + sigma_yy)/2; the in-plane trace is
    # invariant under in-plane rotation, so this equals (sigma_rr + sigma_tt)/2.
    mean_inplane = (sigma[0, 0] + sigma[1, 1]) / 2

    # ---- Radial-line sampling machinery (set up once, evaluated each step) ----
    # Line: y = 0, z = (z0 + H)/2, r from Ri to Ro. Along it the x-axis is radial
    # and the y-axis hoop. Reference config = mesh, so r_line are reference radii R.
    T_DG = fem.functionspace(mesh, basix.ufl.element("DG", mesh.basix_cell(), 1, shape=(3, 3)))
    C_h, sig_h = fem.Function(T_DG), fem.Function(T_DG)
    tipts = T_DG.element.interpolation_points()
    C_expr = fem.Expression(C, tipts)
    sig_expr = fem.Expression(sigma, tipts)
    m_expr = fem.Expression(mean_inplane, Vg.element.interpolation_points())
    m_h = fem.Function(Vg)

    eps = 1e-4 * (Ro - Ri)             # keep points strictly inside the domain
    r_line = np.linspace(Ri + eps, Ro - eps, n_line)
    pts = np.zeros((n_line, 3))
    pts[:, 0], pts[:, 2] = r_line, 0.5 * (z0 + H)
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    cells = np.array([geometry.compute_colliding_cells(mesh, cand, pts).links(i)[0]
                      for i in range(n_line)])

    def sample_line():
        """Evaluate the current solution along the radial line -> snapshot dict."""
        C_h.interpolate(C_expr)
        sig_h.interpolate(sig_expr)
        Cv = C_h.eval(pts, cells)       # (N, 9) row-major tensor components
        sv = sig_h.eval(pts, cells)
        gv = gamma.eval(pts, cells)[:, 0]
        srr, stt = sv[:, 0], sv[:, 4]
        lam_t = np.sqrt(Cv[:, 4])       # total hoop stretch
        return {
            "R": r_line,
            "sigma_rr": srr,
            "sigma_theta": stt,
            "mean_stress": 0.5 * (srr + stt),
            "gamma": gv,
            "alpha_theta": lam_t / gv,  # elastic hoop stretch (matches ODE r/(R*gt))
            "ln_J": 2.0 * np.log(gv),   # plane strain: J = gamma^2
        }

    # ---- Step 1: ramp the internal pressure up to p_target (no growth yet) ----
    for load in np.linspace(p_target / 3.0, p_target, 3):
        p_in.value = float(load)        # set p_out.value here for outer pressure
        its, _ = solver.solve(wsol)
        print(f"p_in = {load:.3f}: converged in {its} Newton iterations")

    snapshots = [sample_line()]         # ungrown, fully pressurized state

    # ---- Step 2: in-plane volumetric growth law (forward Euler) ----
    # v = ln Jg = 2 ln gamma; w v_dot = (in-plane mean stress) - kg v, so the
    # equilibrium is ln Jg = (sigma_rr + sigma_tt)/(2 kg) -- identical in form to
    # the ODE (Eqs. 16-17). We re-solve mechanical equilibrium after each increment.
    for step in range(n_steps):
        m_h.interpolate(m_expr)                       # in-plane mean stress of current sol
        v_old = 2.0 * np.log(gamma.x.array)           # v = ln Jg = 2 ln gamma
        v_new = v_old + (dt / w) * (m_h.x.array - kg * v_old)
        dmax = float(np.max(np.abs(v_new - v_old)))
        gamma.x.array[:] = np.exp(v_new / 2.0)        # gamma = exp(v/2) = J^{1/2}
        its, _ = solver.solve(wsol)                   # re-equilibrate on the grown metric
        snapshots.append(sample_line())
        print(f"growth step {step:2d}: max|dv| = {dmax:.2e}, "
              f"gamma in [{gamma.x.array.min():.4f}, {gamma.x.array.max():.4f}], "
              f"Newton its = {its}")
        if dmax < 1e-5:
            print("growth converged"); break

    # ---- Cache the history for compare_cylinder.py ----
    def stack(key):
        return np.array([s[key] for s in snapshots])
    np.savez(cache_path, R=r_line,
             sigma_rr=stack("sigma_rr"), sigma_theta=stack("sigma_theta"),
             gamma=stack("gamma"), alpha_theta=stack("alpha_theta"))
    print(f"Wrote {os.path.basename(cache_path)} "
          f"({len(snapshots)} history states).")

    if write_outputs:
        # ---- Output displacement (interpolated to P1) for ParaView ----
        V1 = fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1, shape=(3,)))
        u_out = fem.Function(V1)
        u_out.interpolate(wsol.sub(0).collapse())
        u_out.name = "displacement"
        with io.XDMFFile(mesh.comm, os.path.join(HERE, "cylinder_displacement.xdmf"), "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_out)

        # ---- Final radial line data (lambda + Cauchy stress) as CSV ----
        fin = snapshots[-1]
        lam_t = fin["alpha_theta"] * fin["gamma"]
        data = np.column_stack([r_line, fin["gamma"], lam_t,
                                fin["sigma_rr"], fin["sigma_theta"]])
        np.savetxt(os.path.join(HERE, "line_data.csv"), data, delimiter=",",
                   comments="", header="r,gamma,lambda_theta,sigma_rr,sigma_theta")
        print("Wrote line_data.csv:")
        print(f"  inner wall: lam_theta = {lam_t[0]:.4f}, "
              f"sigma_rr = {fin['sigma_rr'][0]:+.4f} (≈ -p_in)")
        print(f"  outer wall: lam_theta = {lam_t[-1]:.4f}, "
              f"sigma_rr = {fin['sigma_rr'][-1]:+.4f} (≈ 0)")

    final = snapshots[-1]
    return (final, snapshots) if return_snapshots else final


if __name__ == "__main__":
    msh = sys.argv[1] if len(sys.argv) > 1 else None
    run_fem(p_target=0.5, kg=1.0, w=1.0, msh_file=msh)
