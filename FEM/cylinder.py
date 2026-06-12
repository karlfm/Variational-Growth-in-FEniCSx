# Generate a hollow cylinder mesh for pressurized_cylinder.py.
# Writes cylinder.msh (gmsh format) with a physical volume group,
# which dolfinx's gmshio.read_from_msh requires.
# Run: python make_cylinder.py [output.msh]   (default: cylinder.msh)
import sys
import gmsh

Ri, Ro, H = 1.0, 1.5, 1.0   # inner radius, outer radius, height (axis along z)
h_mesh = 0.15               # max element size

out_file = sys.argv[1] if len(sys.argv) > 1 else "cylinder.msh"

gmsh.initialize()
gmsh.model.add("hollow_cylinder")
outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, Ro)
inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, Ri)
gmsh.model.occ.cut([(3, outer)], [(3, inner)])
gmsh.model.occ.synchronize()

# Physical volume group (needed by dolfinx when reading the file)
gmsh.model.addPhysicalGroup(3, [t for d, t in gmsh.model.getEntities(3)], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_mesh)
gmsh.model.mesh.generate(3)
gmsh.write(out_file)
gmsh.finalize()
print(f"Wrote {out_file} (Ri={Ri}, Ro={Ro}, H={H}, h={h_mesh})")