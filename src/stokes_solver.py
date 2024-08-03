from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from dolfinx import cpp as _cpp
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx.fem import Constant, Function, dirichletbc, functionspace
from dolfinx.mesh import locate_entities_boundary
from ufl import div, dx, grad, inner, Identity, ds, FacetNormal, dot
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem
from typing import Callable

class Stokes:
    def __init__(self) -> None:
        pass

    def solve(self, mesh, facet_tags, inflow_speed: Callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))):
        comm = MPI.COMM_WORLD

        def mpi_print(s):
            print(f"Rank {comm.rank}: {s}")

        mpi_print(f"Ghost cells (global numbering): {mesh.topology.index_map(mesh.topology.dim).ghosts}")

        P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        P1 = element("Lagrange", mesh.basix_cell(), 1)
        TH = mixed_element([P2, P1])
        W = functionspace(mesh, TH)

        vector_space, _ = W.sub(0).collapse()
        u_zero = Function(vector_space)
        u_R_inflow = Function(vector_space)
        u_R_inflow.interpolate(inflow_speed)

        fdim = mesh.topology.dim - 1
        bc1 = self._create_bc(W, vector_space, facet_tags, u_R_inflow, 1500, fdim)
        bc3 = self._create_bc(W, vector_space, facet_tags, u_zero, 1700, fdim)
        bc4 = self._create_bc(W, vector_space, facet_tags, u_zero, 1800, fdim)
        bcs = [bc1, bc3, bc4]

        a, L = self._define_variational_form(W, mesh)

        w = Function(W)
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        w = problem.solve()
        u, p = w.split()

        J, drag, lift = self._compute_quantities(mesh, facet_tags, u, p)

        return J, drag, lift, u, p

    def _create_bc(self, W, vector_space, facet_tags, func, tag, fdim):
        dofs = fem.locate_dofs_topological((W.sub(0), vector_space), fdim, facet_tags.find(tag))
        return dirichletbc(func, dofs, W.sub(0))

    def _define_variational_form(self, W, mesh):
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        epsilon = 1e-6
        a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u) + epsilon * p * q) * dx
        f = Constant(mesh, [PETSc.ScalarType(0)] * mesh.topology.dim)
        L = inner(f, v) * dx
        return a, L

    def _compute_quantities(self, mesh, facet_tags, u, p):
        J = fem.form(0.5 * inner(grad(u), grad(u)) * dx)
        J = fem.assemble_scalar(J)

        df, dp = self._define_forces(mesh)

        sigma = -p * Identity(len(u)) + grad(u) + grad(u).T
        n = -FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=1800)

        Fd_equation = fem.form(dot(dot(sigma, n), df) * ds)
        Fl_equation = fem.form(dot(dot(sigma, n), dp) * ds)

        Fd = fem.assemble_scalar(Fd_equation)
        Fl = fem.assemble_scalar(Fl_equation)

        return J, Fd, Fl

    def _define_forces(self, mesh):
        if mesh.topology.dim == 2:
            return fem.Constant(mesh, ScalarType((1., 0.))), fem.Constant(mesh, ScalarType((0., 1.)))
        elif mesh.topology.dim == 3:
            return fem.Constant(mesh, ScalarType((1., 0., 0.))), fem.Constant(mesh, ScalarType((0., 1., 0.)))

    def save(self, mesh, u, p, path: str):
        with XDMFFile(MPI.COMM_WORLD, f"{path}/velocity.xdmf", "w") as ufile_xdmf:
            u.x.scatter_forward()
            P1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
            u1 = Function(functionspace(mesh, P1))
            u1.interpolate(u)
            ufile_xdmf.write_mesh(mesh)
            ufile_xdmf.write_function(u1)

        with XDMFFile(MPI.COMM_WORLD, f"{path}/pressure.xdmf", "w") as pfile_xdmf:
            p.x.scatter_forward()
            pfile_xdmf.write_mesh(mesh)
            pfile_xdmf.write_function(p)
