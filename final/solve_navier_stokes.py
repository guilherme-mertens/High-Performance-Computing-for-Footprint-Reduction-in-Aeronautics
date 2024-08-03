import numpy as np
import math
import ufl
 
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io
from petsc4py import PETSc

import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind
from ufl import ds, dx, inner, TrialFunctions, TestFunctions

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from typing import TYPE_CHECKING
import pyvista
from dolfinx.io import VTXWriter
from basix.ufl import element
from dolfinx.fem import Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological, functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.mesh import locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, FiniteElement, Identity, TestFunction, TrialFunction, VectorElement,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)


from typing import Callable


class NavierStokes:

    def __init__(self) -> None:
        self.epsilon = 1e-6

    def solve(self, mesh, facet_tags, T, num_steps,
               inflow_speed: Callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))),
    ):
        t = 0
        dt = T / num_steps
        tdim = mesh.topology.dim
        fdim = mesh.topology.dim - 1
        v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2, tdim)
        s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, v_cg2)
        Q = FunctionSpace(mesh, s_cg1)
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)


        obstacle = facet_tags.find( 4 )
        outer_walls = facet_tags.find( 3 )
        inflow = facet_tags.find( 1 )


        u_noslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        u_inflow_values = Function(V)
        u_inflow_values.interpolate(inflow_speed)



        obstacle_dofs = locate_dofs_topological(V, fdim, obstacle)
        Obstacle_u = dirichletbc(u_noslip, obstacle_dofs, V)

        outer_walls_dofs = locate_dofs_topological(V, fdim, outer_walls)
        Border_u = dirichletbc(u_noslip, outer_walls_dofs, V)


        inflow_dofs = locate_dofs_topological(V, fdim, inflow)
        Inflow_u = dirichletbc(u_inflow_values, inflow_dofs)

        outer_walls_dofs = locate_dofs_topological(Q, fdim, outer_walls)
        Outlet_p = dirichletbc(PETSc.ScalarType(0), outer_walls_dofs, Q)





        bcu = [Border_u, Obstacle_u, Inflow_u]# [Obstacle_u, Border_u, Inflow_u]
        bcp = [Outlet_p]  


        u_n = Function(V)
        u_n.name = "u_n"
        U = 0.5 * (u_n + u)
        n = FacetNormal(mesh)
        f = Constant(mesh, PETSc.ScalarType([0]*tdim))
        k = Constant(mesh, PETSc.ScalarType(dt))
        mu = Constant(mesh, PETSc.ScalarType(1))
        rho = Constant(mesh, PETSc.ScalarType(1))
        # Define strain-rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor


        def sigma(u, p):
            return 2 * mu * epsilon(u) - p * Identity(len(u))


        # Define the variational problem for the first step
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


        # Define variational problem for step 2
        u_ = Function(V)
        a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)

        # Define variational problem for step 3
        p_ = Function(Q)
        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)


        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        time = []
        velocity = []
        pressure = []

        for i in range(num_steps):
            # Update current time step
            print(" i =",i)
            t += dt

            # Step 1: Tentative veolcity step
            with b1.localForm() as loc_1:
                loc_1.set(0)
            assemble_vector(b1, L1)
            apply_lifting(b1, [a1], [bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, bcu)
            solver1.solve(b1, u_.vector)
            u_.x.scatter_forward()

            # Step 2: Pressure corrrection step
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
            u_.x.scatter_forward()
            # Update variable with solution form this time step
            # u_n.x.array[:] = u_.x.array[:]
            p_n.x.array[:] = p_.x.array[:]

            P1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
            u_n = Function(functionspace(mesh, P1))
            u_n.interpolate(u_)
            
            # Write solutions to file
            time.append(t)
            velocity.append(u_n)
            pressure.append(p_n)

        return velocity, pressure, time
        


    

    def save(self, mesh, u, p, t, path: str):
        file_res = io.XDMFFile(mesh.comm, path, "w")
        file_res.write_mesh(mesh)
        for velocity, pressure, time in zip(u, p, t):
            velocity.name = "velocity"
            pressure.name = "pressure"
            
            file_res.write_function(velocity, time)
            file_res.write_function(pressure, time)


    def save_as_vtk(self,):
        pass







if __name__=="__main__":
    from generate_3d_airplane import Airplane3D
    a = Airplane3D("final/Concorde_SD_envelop_1000.geo")
    mesh, _, facets = a.get_mesh(-30)
    navier_stokes = NavierStokes()
    def bc_u_inflow_value(x):
        H = 100  # escolha o valor correspondente ao tamanho do seu retângulo
        amplitude = 0.1
        frequency = 10
        turbulent_velocity = -1/24 * (x[1,:] - H) * (x[1,:] + H) + amplitude * np.sin(2 * np.pi * frequency * x[1] / H)
        
        return np.stack((turbulent_velocity, np.zeros(x.shape[1]), np.zeros(x.shape[1])))

    navier_stokes.solve(mesh, facets, 0.1, 10, bc_u_inflow_value)



