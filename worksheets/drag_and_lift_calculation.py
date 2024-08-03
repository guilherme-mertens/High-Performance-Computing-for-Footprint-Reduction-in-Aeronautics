import warnings
import numpy as np
import gmsh
import sys
from math import pi
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags_from_entities
from dolfinx.cpp.mesh import cell_entity_type
from dolfinx.io import distribute_entity_data
from dolfinx.graph import adjacencylist
from dolfinx.mesh import create_mesh
from dolfinx.cpp.mesh import to_type
from dolfinx.cpp.io import perm_gmsh
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

import math
import ufl
from dolfinx import fem, io
from petsc4py import PETSc
import matplotlib.pyplot as plt
from ufl import ds, dx, inner, TrialFunctions, TestFunctions, FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement, div, dot, sym
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological, functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import locate_entities, meshtags
from basix.ufl import element

warnings.filterwarnings("ignore")


def initialize_gmsh():
    """
    Initialize the GMSH environment.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("DFG 3D")


def create_geometry():
    """
    Create the geometry of the problem using GMSH.
    """
    channel = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    cylinder = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.3)
    fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    assert (volumes == fluid[0])
    return volumes, fluid


def add_physical_groups(volumes):
    """
    Add physical groups to the GMSH model.
    """
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
            inlet = surface[1]
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

    return obstacles, inlet


def create_mesh(obstacles, inlet):
    """
    Create the mesh using GMSH.
    """
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


def finalize_gmsh():
    """
    Finalize the GMSH environment.
    """
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def main_generate_3d_mesh():
    """
    Main function to generate the 3D mesh.
    """
    initialize_gmsh()
    volumes, fluid = create_geometry()
    obstacles, inlet = add_physical_groups(volumes)
    create_mesh(obstacles, inlet)
    finalize_gmsh()


def load_mesh_and_tags(filename):
    """
    Load mesh and tags from an XDMF file.

    Parameters:
    -----------
    filename : str
        The path to the XDMF file.

    Returns:
    --------
    mesh : dolfinx.mesh.Mesh
        The loaded mesh.
    facet_tags : dolfinx.mesh.meshtags
        The facet tags.
    cell_tags : dolfinx.mesh.meshtags
        The cell tags.
    """
    file = XDMFFile(MPI.COMM_WORLD, filename, "r")
    mesh = file.read_mesh()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_tags = file.read_meshtags(mesh, "Facet tags")
    cell_tags = file.read_meshtags(mesh, "Cell tags")
    return mesh, facet_tags, cell_tags


def save_results(file, mesh, u_n, p_n):
    """
    Save the results to an XDMF file.

    Parameters:
    -----------
    file : dolfinx.io.XDMFFile
        The XDMF file to save the results.
    mesh : dolfinx.mesh.Mesh
        The mesh.
    u_n : dolfinx.fem.Function
        The velocity function.
    p_n : dolfinx.fem.Function
        The pressure function.
    """
    file.write_mesh(mesh)
    file.write_function(u_n)
    file.write_function(p_n)


def define_function_spaces(mesh):
    """
    Define function spaces for velocity and pressure.

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh.

    Returns:
    --------
    V : dolfinx.fem.FunctionSpace
        Function space for velocity.
    Q : dolfinx.fem.FunctionSpace
        Function space for pressure.
    """
    v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2, mesh.topology.dim)
    s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_cg2)
    Q = FunctionSpace(mesh, s_cg1)
    return V, Q


def bc_u_inflow_value(x, v0x):
    """
    Define the inflow boundary condition value for velocity.

    Parameters:
    -----------
    x : numpy.ndarray
        The array of coordinates.
    v0x : float
        The velocity component in the x direction.

    Returns:
    --------
    numpy.ndarray
        The velocity boundary condition values.
    """
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.ones(x.shape[1]) * v0x))


def define_boundary_conditions(V, Q, mesh, facet_tags, v0x):
    """
    Define boundary conditions for the problem.

    Parameters:
    -----------
    V : dolfinx.fem.FunctionSpace
        Function space for velocity.
    Q : dolfinx.fem.FunctionSpace
        Function space for pressure.
    mesh : dolfinx.mesh.Mesh
        The mesh.
    facet_tags : dolfinx.mesh.meshtags
        The facet tags.
    v0x : float
        The velocity component in the x direction.

    Returns:
    --------
    bcu : list
        List of velocity boundary conditions.
    bcp : list
        List of pressure boundary conditions.
    """
    u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    u_R_inflow = Function(V)
    u_R_inflow.interpolate(lambda x: bc_u_inflow_value(x, v0x))

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
    return bcu, bcp


def define_variational_problem(V, Q, mesh, dt):
    """
    Define the variational problem for the Navier-Stokes equations.

    Parameters:
    -----------
    V : dolfinx.fem.FunctionSpace
        Function space for velocity.
    Q : dolfinx.fem.FunctionSpace
        Function space for pressure.
    mesh : dolfinx.mesh.Mesh
        The mesh.
    dt : float
        The time step size.

    Returns:
    --------
    tuple
        The variational forms and assembled matrices.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    u_n = Function(V)
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

    p_n = Function(Q)

    F1 = rho * dot((u - u_n) / k, v) * dx
    F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    F1 += inner(sigma(U, p_n), epsilon(v)) * dx
    F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
    F1 -= dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))

    A1 = assemble_matrix(a1)
    A1.assemble()
    b1 = create_vector(L1)

    u_ = Function(V)
    a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
    L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
    A2 = assemble_matrix(a2)
    A2.assemble()
    b2 = create_vector(L2)

    p_ = Function(Q)
    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    return a1, L1, A1, b1, u_, a2, L2, A2, b2, p_, a3, L3, A3, b3, u_n, p_n


def setup_solvers(A1, A2, A3):
    """
    Set up the solvers for the Navier-Stokes equations.

    Parameters:
    -----------
    A1 : PETSc.Mat
        Assembled matrix for the first variational problem.
    A2 : PETSc.Mat
        Assembled matrix for the second variational problem.
    A3 : PETSc.Mat
        Assembled matrix for the third variational problem.

    Returns:
    --------
    tuple
        The solvers for the variational problems.
    """
    solver1 = PETSc.KSP().create(A1.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.HYPRE)
    pc1.setHYPREType("boomeramg")

    solver2 = PETSc.KSP().create(A2.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.BCGS)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    solver3 = PETSc.KSP().create(A3.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    return solver1, solver2, solver3


def navier_stokes_solver(mesh, facet_tags, cell_tags, num_steps, dt, v0x, v0y):
    """
    Solve the Navier-Stokes equations for a given mesh and boundary conditions.

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh.
    facet_tags : dolfinx.mesh.meshtags
        The facet tags.
    cell_tags : dolfinx.mesh.meshtags
        The cell tags.
    num_steps : int
        Number of time steps.
    dt : float
        Time step size.
    v0x : float
        Initial velocity component in the x direction.
    v0y : float
        Initial velocity component in the y direction.
    """
    V, Q = define_function_spaces(mesh)
    bcu, bcp = define_boundary_conditions(V, Q, mesh, facet_tags, v0x)
    a1, L1, A1, b1, u_, a2, L2, A2, b2, p_, a3, L3, A3, b3, u_n, p_n = define_variational_problem(V, Q, mesh, dt)
    solver1, solver2, solver3 = setup_solvers(A1, A2, A3)

    file_res = XDMFFile(mesh.comm, "project/concorde_3D/res_30.xdmf", "w")
    save_results(file_res, mesh, u_n, p_n)

    v0x_increment = (v0x * 10 - v0x) / num_steps
    v0y_increment = (v0y * 10 - v0y) / num_steps
    t = 0

    for i in range(num_steps):
        print(f"Step {i + 1}/{num_steps}")
        t += dt
        v0x += v0x_increment
        v0y += v0y_increment

        u_R_inflow = Function(V)
        u_R_inflow.interpolate(lambda x: bc_u_inflow_value(x, v0x))
        inflow_dofs = locate_dofs_topological(V, 2, facet_tags.find(1500))
        Inflow_u = dirichletbc(u_R_inflow, inflow_dofs)
        bcu[-1] = Inflow_u

        # Step 1: Tentative velocity step
        with b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_.vector)
        u_.x.scatter_forward()

        # Step 2: Pressure correction step
        with b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, p_.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)
        u_n.x.array[:] = u_.x.array[:]
        p_n.x.array[:] = p_.x.array[:]

        save_results(file_res, mesh, u_n, p_n, t)

    file_res.close()


if __name__ == "__main__":
    mesh, facet_tags, cell_tags = main_generate_3d_mesh()
    navier_stokes_solver(mesh, facet_tags, cell_tags, num_steps=100, dt=0.01, v0x=-11.6, v0y=0.6)





