import numpy as np
import math
import ufl
from dolfinx import fem, io
from petsc4py import PETSc
import matplotlib.pyplot as plt
from ufl import ds, dx, inner, TrialFunctions, TestFunctions
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from typing import Callable
from dolfinx.io import VTXWriter
from basix.ufl import element
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological, functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import locate_entities, meshtags
from ufl import (FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)


class NavierStokes:
    """Class to solve the Navier-Stokes equations."""

    def __init__(self) -> None:
        self.epsilon = 1e-6

    def solve(self, mesh, facet_tags, T: float, num_steps: int,
              inflow_speed: Callable = lambda x: np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))):
        """
        Solves the Navier-Stokes equations.

        Parameters:
            mesh: The computational mesh.
            facet_tags: Facet tags for boundary conditions.
            T: Final time.
            num_steps: Number of time steps.
            inflow_speed: Function for inflow boundary condition.

        Returns:
            velocity, pressure, time: Solution arrays for velocity, pressure, and time.
        """
        t = 0
        dt = T / num_steps
        tdim = mesh.topology.dim
        fdim = mesh.topology.dim - 1

        V, Q, u, v, p, q = self._create_function_spaces(mesh)
        bcu, bcp = self._create_boundary_conditions(mesh, facet_tags, V, Q, inflow_speed)

        u_n, p_n, U, n, f, k, mu, rho = self._initialize_solution_functions(mesh, V, Q, dt)

        a1, L1, A1, b1 = self._create_tentative_velocity_problem(V, Q, u, v, U, n, f, k, mu, rho, u_n, p_n, bcu)
        a2, L2, A2, b2 = self._create_pressure_correction_problem(Q, p, q, p_n, rho, k, u_)
        a3, L3, A3, b3 = self._create_velocity_correction_problem(V, u, v, k, p_, p_n, u_)

        solver1, solver2, solver3 = self._initialize_solvers(A1, A2, A3, mesh)

        time, velocity, pressure = [], [], []

        for i in range(num_steps):
            print(f"Step {i + 1}/{num_steps}")
            t += dt
            u_, p_ = self._time_step(a1, L1, b1, a2, L2, b2, a3, L3, b3, solver1, solver2, solver3, bcu, bcp, u_, p_, u_n, p_n)
            time.append(t)
            velocity.append(u_n)
            pressure.append(p_n)

        return velocity, pressure, time

    def save(self, mesh, u, p, t, path: str):
        """
        Saves the solutions to a file.

        Parameters:
            mesh: The computational mesh.
            u: Velocity solutions.
            p: Pressure solutions.
            t: Time steps.
            path: File path for saving.
        """
        file_res = io.XDMFFile(mesh.comm, path, "w")
        file_res.write_mesh(mesh)
        for velocity, pressure, time in zip(u, p, t):
            velocity.name = "velocity"
            pressure.name = "pressure"
            file_res.write_function(velocity, time)
            file_res.write_function(pressure, time)

    def _create_function_spaces(self, mesh):
        tdim = mesh.topology.dim
        v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2, tdim)
        s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, v_cg2)
        Q = FunctionSpace(mesh, s_cg1)
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)
        return V, Q, u, v, p, q

    def _create_boundary_conditions(self, mesh, facet_tags, V, Q, inflow_speed):
        obstacle = facet_tags.find(4)
        outer_walls = facet_tags.find(3)
        inflow = facet_tags.find(1)

        u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        u_inflow_values = Function(V)
        u_inflow_values.interpolate(inflow_speed)

        obstacle_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, obstacle)
        Obstacle_u = dirichletbc(u_noslip, obstacle_dofs, V)

        outer_walls_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, outer_walls)
        Border_u = dirichletbc(u_noslip, outer_walls_dofs, V)

        inflow_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, inflow)
        Inflow_u = dirichletbc(u_inflow_values, inflow_dofs)

        outer_walls_dofs = locate_dofs_topological(Q, mesh.topology.dim - 1, outer_walls)
        Outlet_p = dirichletbc(PETSc.ScalarType(0), outer_walls_dofs, Q)

        bcu = [Border_u, Obstacle_u, Inflow_u]
        bcp = [Outlet_p]
        return bcu, bcp

    def _initialize_solution_functions(self, mesh, V, Q, dt):
        u_n = Function(V)
        u_n.name = "u_n"
        p_n = Function(Q)
        p_n.name = "p_n"
        U = 0.5 * (u_n + u)
        n = FacetNormal(mesh)
        f = Constant(mesh, PETSc.ScalarType([0] * mesh.geometry.dim))
        k = Constant(mesh, PETSc.ScalarType(dt))
        mu = Constant(mesh, PETSc.ScalarType(1))
        rho = Constant(mesh, PETSc.ScalarType(1))
        return u_n, p_n, U, n, f, k, mu, rho

    def _create_tentative_velocity_problem(self, V, Q, u, v, U, n, f, k, mu, rho, u_n, p_n, bcu):
        def epsilon(u):
            return sym(nabla_grad(u))

        def sigma(u, p):
            return 2 * mu * epsilon(u) - p * Identity(len(u))

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
        return a1, L1, A1, b1

    def _create_pressure_correction_problem(self, Q, p, q, p_n, rho, k, u_):
        a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)
        return a2, L2, A2, b2

    def _create_velocity_correction_problem(self, V, u, v, k, p_, p_n, u_):
        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)
        return a3, L3, A3, b3

    def _initialize_solvers(self, A1, A2, A3, mesh):
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.MINRES)
        solver1.getPC().setType(PETSc.PC.Type.HYPRE)

        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        solver2.getPC().setType(PETSc.PC.Type.HYPRE)

        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.MINRES)
        solver3.getPC().setType(PETSc.PC.Type.HYPRE)

        return solver1, solver2, solver3

    def _time_step(self, a1, L1, b1, a2, L2, b2, a3, L3, b3, solver1, solver2, solver3, bcu, bcp, u_, p_, u_n, p_n):
        with b1.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], bcs=[bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_.vector)
        u_.x.scatter_forward()

        with b2.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], bcs=[bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, p_.vector)
        p_.x.scatter_forward()

        with b3.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b3, L3)
        apply_lifting(b3, [a3], bcs=[bcu])
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b3, bcu)
        solver3.solve(b3, u_.vector)
        u_.x.scatter_forward()

        u_n.x.array[:] = u_.x.array
        p_n.x.array[:] = p_.x.array

        return u_, p_

def main():
    from dolfinx import mesh, fem
    from mpi4py import MPI

    # Create mesh and function spaces
    mesh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [10, 10], mesh.CellType.triangle)
    facet_tags = mesh.create_meshtags(mesh, 1, locate_entities_boundary(mesh, 1, lambda x: np.full(x.shape[1], True)))
    
    ns_solver = NavierStokes()
    velocity, pressure, time = ns_solver.solve(mesh, facet_tags, T=1.0, num_steps=10)
    ns_solver.save(mesh, velocity, pressure, time, "results.xdmf")


if __name__ == "__main__":
    main()
