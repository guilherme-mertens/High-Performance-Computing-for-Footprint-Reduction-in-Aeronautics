from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
import numpy as np
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import TrialFunction, TestFunction
from ufl import inner, dot, grad, dx, ds
import matplotlib.pyplot as plt
import time

def create_mesh(N):
    """
    Create a rectangular mesh [0,1]x[0,1] with N subdivisions.

    Parameters:
        N (int): Number of subdivisions along each axis.

    Returns:
        dolfinx.mesh.Mesh: The created mesh.
    """
    return mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(N, N),
        cell_type=mesh.CellType.triangle,
    )

def poisson_solver(N, degree):
    """
    Solve the Poisson problem on a unit square with Dirichlet boundary conditions.

    Parameters:
        N (int): Number of subdivisions for the mesh.
        degree (int): Degree of the Lagrange finite elements.

    Returns:
        tuple: Solution and exact solution.
    """
    msh = create_mesh(N)
    V = fem.functionspace(msh, ("Lagrange", degree))

    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

    u = TrialFunction(V)
    v = TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    u_e = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 2 * ufl.pi**2 * u_e

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh, u_e

def error_l2(eh):
    """
    Compute the L2 error norm.

    Parameters:
        eh (dolfinx.fem.Function): Error function.

    Returns:
        float: L2 error norm.
    """
    comm = eh.function_space.mesh.comm
    error = fem.form((eh)**2 * dx)
    E = np.sqrt(comm.allreduce(fem.assemble_scalar(error), MPI.SUM))
    return E

def error_h1(eh):
    """
    Compute the H1 error norm.

    Parameters:
        eh (dolfinx.fem.Function): Error function.

    Returns:
        float: H1 error norm.
    """
    comm = eh.function_space.mesh.comm
    error_H1 = fem.form(dot(grad(eh), grad(eh)) * dx + (eh)**2 * dx)
    E_H1 = np.sqrt(comm.allreduce(fem.assemble_scalar(error_H1), op=MPI.SUM))
    return E_H1

def main():
    """
    Main function to solve the Poisson problem and plot errors.
    """
    Ns = np.arange(2, 100, 10)
    Es_l2 = np.zeros(len(Ns), dtype=default_scalar_type)
    Es_h1 = np.zeros(len(Ns), dtype=default_scalar_type)
    hs = np.zeros(len(Ns), dtype=np.float64)

    for i, N in enumerate(Ns):
        uh, u_e = poisson_solver(N, degree=1)
        Es_l2[i] = error_l2((uh - u_e))
        Es_h1[i] = error_h1((uh - u_e))
        hs[i] = 1. / Ns[i]

    plt.figure()
    plt.plot(np.log(hs), np.log(Es_l2))
    plt.plot(np.log(hs), np.log(Es_h1))
    plt.legend(["L2 norm", "H1 norm"])
    plt.title("Degree 1")
    plt.show()

if __name__ == "__main__":
    main()
