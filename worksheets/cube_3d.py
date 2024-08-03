import numpy as np
import gmsh
import sys
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_matrix, assemble_vector, create_vector, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities
from ufl import (FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)
from petsc4py.PETSc import ScalarType
from basix.ufl import element

def generate_3d_mesh():
    """
    Generates a 3D mesh using GMSH, including a box with a spherical cut-out.
    The mesh is saved to `.msh` and `.geo_unrolled` files.

    This function uses GMSH to define a geometric model, create a mesh with a specific resolution, and save it.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("DFG 3D")

    # Define geometric shapes
    channel = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    cylinder = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.3)

    # Perform Boolean operations
    fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)])

    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    assert (volumes == fluid[0])

    # Define physical groups
    fluid_marker = 0
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    surfaces = gmsh.model.occ.getEntities(dim=2)
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 2, 3, 4
    walls = []
    obstacles = []

    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0, 0.5, 0.5]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_marker)
            gmsh.model.setPhysicalName(surface[0], inlet_marker, "Inflow")
        elif np.allclose(com, [1, 0.5, 0.5]):
            gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_marker)
            gmsh.model.setPhysicalName(surface[0], outlet_marker, "Outflow")
        elif np.allclose(com, [0.5, 0.5, 0.5]):
            obstacles.append(surface[1])
        else:
            walls.append(surface[1])

    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
    gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")

    # Define mesh fields
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

    r = 0.5
    resolution = r / 10
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * 0.1)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 1)

    inlet_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
    inlet_thre = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write("project/cube3d_mesh/mesh_out.msh")
    gmsh.write("project/cube3d_mesh/mmesh_out.geo_unrolled")

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

def solve_navier_stokes_3d():
    """
    Solves the 3D Navier-Stokes equations using the finite element method.
    The solution is written to XDMF files at each time step.

    This function initializes the mesh and function spaces, applies boundary conditions,
    defines the variational problem, and solves it using PETSc.
    """
    # Load mesh
    file = io.XDMFFile(MPI.COMM_WORLD, "project/concorde_3D/Concorde_SD_envelop_1000_30.xdmf", "r")
    mesh = file.read_mesh()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    facet_tags = file.read_meshtags(mesh, "Facet tags")
    cell_tags = file.read_meshtags(mesh, "Cell tags")

    file_res = io.XDMFFile(mesh.comm, "project/concorde_3D/res_30.xdmf", "w")
    file_res.write_mesh(mesh)

    # Constants
    degree = 1
    epsilon = 1e-6
    t = 0
    T = 1
    num_steps = 100
    dt = T / num_steps

    # Define function spaces
    v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2, tdim)
    s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_cg2)
    Q = FunctionSpace(mesh, s_cg1)

    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define boundary conditions
    def bc_u_inflow_value(x):
        v0x = -11.6
        return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.ones(x.shape[1]) * v0x))

    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    u_R_inflow = Function(V)
    u_R_inflow.interpolate(bc_u_inflow_value)

    obstacle = facet_tags.find(1800)
    outer_walls = facet_tags.find(1700)
    inflow = facet_tags.find(1500)
    outer_flow = facet_tags.find(1600)

    obstacle_dofs = locate_dofs_topological(V, 2, obstacle)
    Obstacle_u = dirichletbc(u_noslip, obstacle_dofs, V)
    outer_walls_dofs = locate_dofs_topological(V, 2, outer_walls)
    Border_u = dirichletbc(u_noslip, outer_walls_dofs, V)
    inflow_dofs = locate_dofs_topological(V, 2, inflow)
    Inflow_u = dirichletbc(u_R_inflow, inflow_dofs)
    outer_flow_dofs = locate_dofs_topological(Q, 2, outer_flow)
    Outlet_p = dirichletbc(PETSc.ScalarType(0), outer_flow_dofs, Q)

    bcu = [Border_u, Obstacle_u, Inflow_u]
    bcp = [Outlet_p]

    # Define variational problems
    u_n = Function(V)
    u_n.name = "u_n"
    U = 0.5 * (u_n + u)
    n = FacetNormal(mesh)
    f = Constant(mesh, PETSc.ScalarType((0, 0, 0)))
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(1))
    rho = Constant(mesh, PETSc.ScalarType(1))

    def epsilon(u):
        return sym(nabla_grad(u))

    def sigma(u, p):
        return 2 * mu * epsilon(u) - p * Identity(len(u))

    # Variational problem for velocity
    p_n = Function(Q)
    p_n.name = "p_n"
    F1 = rho * dot((u - u_n) / k, v) * dx
    F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    F1 += inner(sigma(U, p_n), epsilon(v)) * dx
    F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
    F1 -= dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))

    A1 = assemble_matrix(a1, bcs=bcu)
    A1.assemble()
    b1 = create_vector(L1)

    # Variational problem for pressure
    u_ = Function(V)
    a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
    L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    # Variational problem for velocity correction
    p_ = Function(Q)
    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    # Solvers
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.HYPRE)
    pc1.setHYPREType("boomeramg")

    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.BCGS)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    # Time-stepping loop
    v0x = -11.6
    v0y = 0.6
    v0x_increment = (v0x * 10 - v0x) / num_steps
    v0y_increment = (v0y * 10 - v0y) / num_steps

    for i in range(num_steps):
        print("i =", i)
        t += dt
        v0x += v0x_increment
        v0y += v0y_increment

        def bc_u_inflow_value(x):
            return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.ones(x.shape[1]) * v0x))
        
        u_R_inflow = Function(V)
        u_R_inflow.interpolate(bc_u_inflow_value)
        Inflow_u = dirichletbc(u_R_inflow, inflow_dofs)
        bcu = [Border_u, Obstacle_u, Inflow_u]

        # Step 1: Velocity update
        with b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_.vector)
        u_.x.scatter_forward()

        # Step 2: Pressure correction
        with b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, p_.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction
        with b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)

        # Update variable with solution
        p_n.x.array[:] = p_.x.array[:]
        P1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
        u_n = Function(FunctionSpace(mesh, P1))
        u_n.interpolate(u_)

        # Write solutions to file
        file_res.write_function(u_n, t)
        file_res.write_function(p_n, t)

if __name__ == "__main__":
    generate_3d_mesh()
    solve_navier_stokes_3d()