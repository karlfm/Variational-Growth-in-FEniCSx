# Pressurized hollow cylinder, incompressible neo-Hookean material.
# Mixed (u, p) formulation, follower pressure (Neumann) on inner/outer walls,
# Dirichlet BCs on top/bottom. Written for dolfinx 0.9.
# Run: python pressurized_cylinder.py [mesh.msh]   (default: cylinder.msh)
import sys
import numpy as np, ufl, basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, log, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

mu = 1.0                            # shear modulus
INNER, OUTER, BOTTOM, TOP = 1, 2, 3, 4

# ---- Read the mesh (gmsh .msh file, axis assumed along z, base at z=0) ----
msh_file = sys.argv[1] if len(sys.argv) > 1 else "cylinder.msh"
mesh, _, _ = io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, 0, gdim=3)

# ---- Infer geometry from the mesh coordinates ----
x = mesh.geometry.x
r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
Ri, Ro = r.min(), r.max()
z0, H = x[:, 2].min(), x[:, 2].max()
print(f"Mesh: Ri = {Ri:.4f}, Ro = {Ro:.4f}, z in [{z0:.4f}, {H:.4f}]")

# ---- Tag boundary facets geometrically ----
def radius(p): return np.sqrt(p[0] ** 2 + p[1] ** 2)
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
w = fem.Function(W)
u, p = ufl.split(w)
v, q = ufl.TestFunctions(W)

# ---- Isotropic volumetric growth Fg = gamma * I (Sec. 4.8 of the paper) ----
# gamma is a scalar field; Jg = det(Fg) = gamma^3 is the grown volume form.
# Growth energy Wg = kg/2 (ln Jg)^2 is independent of (u, p), so it does NOT
# enter the mechanical weak form (it only drives the growth law below).
kg = 1.0                            # growth-energy stiffness (sets residual mean stress)
w_g = 1.0                           # growth dissipation coefficient
Vg = fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1))
gamma = fem.Function(Vg)            # isotropic growth stretch field
gamma.x.array[:] = 1.0             # start from the ungrown state

# ---- Incompressible neo-Hookean energy on the *elastic* part Fe = F Fg^{-1} ----
# The reference integrand is weighted by Jg, so equilibrium is taken on the
# (possibly non-Euclidean) grown configuration. p enforces the elastic
# incompressibility constraint Je = det(Fe) = 1.
F = ufl.Identity(3) + ufl.grad(u)
J = ufl.det(F)                      # total Jacobian (used by the follower-pressure term)
Fe = F / gamma                      # elastic deformation
Je = ufl.det(Fe)                    # elastic Jacobian, constrained to 1
Jg = gamma**3                       # grown volume form
psi = (mu / 2 * (ufl.tr(Fe.T * Fe) - 3) - p * (Je - 1)) * Jg
Pi = psi * ufl.dx

# ---- Neumann boundary conditions ----
p_in, p_out = fem.Constant(mesh, 0.0), fem.Constant(mesh, 0.0)
n = ufl.FacetNormal(mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
# Very weak Robin (elastic spring) on the outer wall: traction = -k_robin * u.
# Removes the in-plane rigid-body modes
k_robin = fem.Constant(mesh, 1e-4 * mu / Ro)
R = ufl.derivative(Pi, w) \
    + p_in * J * ufl.dot(ufl.inv(F).T * n, v) * ds(INNER) \
    + p_out * J * ufl.dot(ufl.inv(F).T * n, v) * ds(OUTER) \
    + k_robin * ufl.dot(u, v) * ds(OUTER)

# ---- Dirichlet BCs: u_z = 0 on both top and bottom (radial sliding free) ----
Wz, _ = W.sub(0).sub(2).collapse()
uz_zero = fem.Function(Wz)
dofs_bot = fem.locate_dofs_topological((W.sub(0).sub(2), Wz), fdim, facet_tags.find(BOTTOM))
bc_bot = fem.dirichletbc(uz_zero, dofs_bot, W.sub(0).sub(2))
dofs_top = fem.locate_dofs_topological((W.sub(0).sub(2), Wz), fdim, facet_tags.find(TOP))
bc_top = fem.dirichletbc(uz_zero, dofs_top, W.sub(0).sub(2))

# ---- Newton solver with load stepping ----
problem = NonlinearProblem(R, w, bcs=[bc_bot, bc_top])
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
ksp = solver.krylov_solver
opts = PETSc.Options()
opts[f"{ksp.getOptionsPrefix()}ksp_type"] = "preonly"
opts[f"{ksp.getOptionsPrefix()}pc_type"] = "lu"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)

# ---- Step 1: apply the full internal pressure (no growth yet) ----
for load in np.linspace(0.1, 0.30, 3):   # ramp inner pressure up to 0.3*mu
    p_in.value = load                     # set p_out.value here for outer pressure
    its, converged = solver.solve(w)
    print(f"p_in = {load:.3f}: converged in {its} Newton iterations")

# ---- Step 2: volumetric growth law from the variational principle ----
# Vary V = -L + P + Wg with respect to the isotropic growth Fg = gamma*I. Using
# the log-volumetric growth variable v = ln Jg and dissipation D = w_g/2 v_dot^2,
#     w_g v_dot = dL/dv - dWg/dv = (1/3) tr(sigma) - kg ln Jg,
# i.e. the driver is the mean Cauchy stress and the equilibrium is
#     ln Jg = tr(sigma) / (3 kg).
# This is the 3D analogue of Eqs. (16)-(17) (the 2D paper has 1/2 instead of 1/3,
# and the mean stress is the in-plane mean). We integrate it with forward Euler,
# re-solving mechanical equilibrium after each growth increment.
sigma = mu * Fe * Fe.T - p * ufl.Identity(3)        # Cauchy stress (Je = 1)
mean_stress = ufl.tr(sigma) / 3
m_expr = fem.Expression(mean_stress, Vg.element.interpolation_points())
m_h = fem.Function(Vg)

dt = 0.2
for step in range(40):
    m_h.interpolate(m_expr)                          # mean stress of current solution
    v_old = 3.0 * np.log(gamma.x.array)              # v = ln Jg = 3 ln gamma
    v_new = v_old + (dt / w_g) * (m_h.x.array - kg * v_old)
    dmax = float(np.max(np.abs(v_new - v_old)))
    gamma.x.array[:] = np.exp(v_new / 3.0)           # update isotropic growth
    its, converged = solver.solve(w)                 # re-equilibrate on the grown metric
    print(f"growth step {step:2d}: max|dv| = {dmax:.2e}, "
          f"gamma in [{gamma.x.array.min():.4f}, {gamma.x.array.max():.4f}], "
          f"Newton its = {its}")
    if dmax < 1e-5:
        print("growth converged"); break

# ---- Output displacement (interpolated to P1) for ParaView ----
V1 = fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1, shape=(3,)))
u_out = fem.Function(V1)
u_out.interpolate(w.sub(0).collapse())
u_out.name = "displacement"
with io.XDMFFile(mesh.comm, "cylinder_displacement.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u_out)

# ---- Sample stretch and Cauchy stress along a radial line at mid-height ----
# Line: y = 0, z = (z0 + H)/2, r from Ri to Ro. Along this line the x-axis is
# the radial direction and the y-axis the hoop direction.
from dolfinx import geometry
C = F.T * F                                  # total right Cauchy-Green tensor
sigma = mu * Fe * Fe.T - p * ufl.Identity(3) # Cauchy stress on the elastic part (Je = 1)
T_DG = fem.functionspace(mesh, basix.ufl.element("DG", mesh.basix_cell(), 1, shape=(3, 3)))
C_h, sig_h = fem.Function(T_DG), fem.Function(T_DG)
ipts = T_DG.element.interpolation_points()
C_h.interpolate(fem.Expression(C, ipts))
sig_h.interpolate(fem.Expression(sigma, ipts))

N = 60
eps = 1e-4 * (Ro - Ri)                       # keep points strictly inside the domain
r_line = np.linspace(Ri + eps, Ro - eps, N)
pts = np.zeros((N, 3))
pts[:, 0], pts[:, 2] = r_line, 0.5 * (z0 + H)
tree = geometry.bb_tree(mesh, mesh.topology.dim)
cand = geometry.compute_collisions_points(tree, pts)
cells = np.array([geometry.compute_colliding_cells(mesh, cand, pts).links(i)[0]
                  for i in range(N)])
Cv = C_h.eval(pts, cells)                    # (N, 9), row-major tensor components
sv = sig_h.eval(pts, cells)
gv = gamma.eval(pts, cells)[:, 0]            # isotropic growth stretch along the line
lam_r, lam_t, lam_z = np.sqrt(Cv[:, 0]), np.sqrt(Cv[:, 4]), np.sqrt(Cv[:, 8])
data = np.column_stack([r_line, lam_r, lam_t, lam_z, sv[:, 0], sv[:, 4], sv[:, 8], gv])
np.savetxt("line_data.csv", data, delimiter=",", comments="",
           header="r,lambda_r,lambda_theta,lambda_z,sigma_rr,sigma_theta,sigma_zz,gamma")
print("Wrote line_data.csv:")
print(f"  inner wall: lam_theta = {lam_t[0]:.4f}, sigma_rr = {sv[0, 0]:+.4f} (≈ -p_in)")
print(f"  outer wall: lam_theta = {lam_t[-1]:.4f}, sigma_rr = {sv[-1, 0]:+.4f} (≈ 0)")