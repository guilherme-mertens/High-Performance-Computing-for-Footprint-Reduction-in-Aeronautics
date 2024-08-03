from mpi4py import MPI
from dolfinx import io, fem, mesh, plot
from basix.ufl import element, mixed_element
from ufl import TrialFunctions, TestFunctions, inner, grad, dx, SpatialCoordinate
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
import numpy as np

def read_mesh_and_tags(filename):
    """
    Read mesh and facet tags from an XDMF file.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        tuple: The mesh and facet tags.
    """
    file = io.XDMFFile(MPI.COMM_WORLD, filename, "r")
    msh = file.read_mesh()
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facet_tags = file.read_meshtags(msh, "Facet tags")
    return msh, facet_tags

def create_function_space(msh):
    """
    Create a mixed function space for the Poisson problem.

    Parameters:
        msh (dolfinx.mesh.Mesh): The mesh.

    Returns:
        dolfinx.fem.FunctionSpace: The created mixed function space.
    """
    V_el = element("Lagrange", msh.basix_cell(), 1)
    Q_el = element("Lagrange", msh.basix_cell(), 1)
    VQ_el = mixed_element([V_el, Q_el])
    W = fem.functionspace(msh, VQ_el)
    return W

def define_boundary_conditions(W, facet_tags, u_D=ScalarType(0.)):
    """
    Define Dirichlet boundary conditions for the mixed Poisson problem.

    Parameters:
        W (dolfinx.fem.FunctionSpace): The function space.
        facet_tags (dolfinx.mesh.MeshTags): The facet tags.
        u_D (dolfinx.fem.Constant, optional): Dirichlet boundary condition value. Defaults to 0.

    Returns:
        list: The list of boundary conditions.
    """
    boundary_conditions = []
    for marker_id in [1, 2, 3, 4]:
        dofs_0 = fem.locate_dofs_topological(W.sub(0), 1, facet_tags.find(marker_id))
        dofs_1 = fem.locate_dofs_topological(W.sub(1), 1, facet_tags.find(marker_id))
        bc_0 = fem.dirichletbc(u_D if marker_id != 1 else ScalarType(1.), dofs_0, W.sub(0))
        bc_1 = fem.dirichletbc(u_D if marker_id != 1 else ScalarType(1.), dofs_1, W.sub(1))
        boundary_conditions.extend([bc_0, bc_1])
    return boundary_conditions

def solve_mixed_poisson(msh, W, boundary_conditions):
    """
    Solve the mixed Poisson problem.

    Parameters:
        msh (dolfinx.mesh.Mesh): The mesh.
        W (dolfinx.fem.FunctionSpace): The function space.
        boundary_conditions (list): The list of boundary conditions.

    Returns:
        tuple: The solutions uh and ph.
    """
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    x = SpatialCoordinate(msh)
    u_e1 = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f1 = 2 * ufl.pi ** 2 * u_e1
    f2 = 1

    a = (inner(grad(u), grad(v)) + inner(grad(p), grad(q))) * dx
    L = (inner(f1, v) + inner(f2, q)) * dx

    problem = LinearProblem(a, L, bcs=boundary_conditions)
    wh = problem.solve()
    (uh, ph) = wh.split()
    return uh, ph

def save_results(msh, uh, ph, filename):
    """
    Save the results to an XDMF file.

    Parameters:
        msh (dolfinx.mesh.Mesh): The mesh.
        uh (dolfinx.fem.Function): The solution uh.
        ph (dolfinx.fem.Function): The solution ph.
        filename (str): The path to the output XDMF file.
    """
    with io.XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)
        file.write_function(ph)

def visualize_solution(W, uh):
    """
    Visualize the solution using PyVista.

    Parameters:
        W (dolfinx.fem.FunctionSpace): The function space.
        uh (dolfinx.fem.Function): The solution uh.
    """
    try:
        import pyvista
        cells, types, x = plot.vtk_mesh(W.sub(1).collapse()[0])
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = np.split(uh.x.array.real, 2)[0]
        grid.set_active_scalars("u")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        if pyvista.OFF_SCREEN:
            pyvista.start_xvfb(wait=0.1)
            plotter.screenshot("uh_poisson.png")
        else:
            plotter.show()
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution")
        print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")

def main():
    """
    Main function to solve the mixed Poisson problem and save the results.
    """
    msh, facet_tags = read_mesh_and_tags("worksheet1/mesh_out.xdmf")
    W = create_function_space(msh)
    boundary_conditions = define_boundary_conditions(W, facet_tags)
    uh, ph = solve_mixed_poisson(msh, W, boundary_conditions)
    save_results(msh, uh, ph, "worksheet1/out_ex_1_2/result.xdmf")
    visualize_solution(W, uh)

if __name__ == "__main__":
    main()