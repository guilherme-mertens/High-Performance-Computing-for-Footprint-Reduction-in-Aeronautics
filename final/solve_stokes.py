from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from dolfinx import cpp as _cpp

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)


from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, Identity, ds, FacetNormal, dot, nabla_grad
from dolfinx import io
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem


from typing import Callable


class Stokes:

    def __init__(self) -> None:
        pass

    def solve(self, mesh, facet_tags,
               inflow_speed: Callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    ):

        comm = MPI.COMM_WORLD
        def mpi_print(s):
            print(f"Rank {comm.rank}: {s}")
        tdim = mesh.topology.dim
        mpi_print(f"Ghost cells (global numbering): {mesh.topology.index_map(tdim).ghosts}")
        P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        P1 = element("Lagrange", mesh.basix_cell(), 1)



        # Create the Taylot-Hood function space
        TH = mixed_element([P2, P1])
        W = functionspace(mesh, TH)


        # No slip boundary condition
        vector_space, _ = W.sub(0).collapse()
        u_zero = Function(vector_space)

        
        vector_space, _ = W.sub(0).collapse()
        u_R_inflow = Function(vector_space)
        u_R_inflow.interpolate(inflow_speed)
        


        fdim = mesh.topology.dim - 1

        

        dofs4 = fem.locate_dofs_topological((W.sub(0), vector_space), fdim, facet_tags.find( 1800 )) # Obstacle
        bc4 = dirichletbc(u_zero, dofs4, W.sub(0))


        dofs3 = fem.locate_dofs_topological((W.sub(0), vector_space), fdim, facet_tags.find( 1700 ))  # Walls
        bc3 = dirichletbc(u_zero, dofs3,  W.sub(0))


        dofs1 = fem.locate_dofs_topological((W.sub(0), vector_space), fdim, facet_tags.find( 1500 ))  # Inflow
        bc1 = dirichletbc(u_R_inflow, dofs1, W.sub(0))
        bcs = [bc4, bc3, bc1]




        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        epsilon = 1e-6
        a = (inner(nabla_grad(u), nabla_grad(v)) - p * div(v) + q * div(u) + epsilon*p*q)*dx 
        f = Constant(mesh, [PETSc.ScalarType(0)]*tdim)
        L = inner(f, v)*dx

        # Create the function to contain the solution
        w = Function(W)

        # Solve the system
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        w = problem.solve()

        (u, p) = w.split()




        # Compute objective
        J = fem.form(.5 * inner(ufl.nabla_grad(u), ufl.nabla_grad(u))*dx)
        J = fem.assemble_scalar(J)


        if tdim == 2:
            df =  fem.Constant(mesh, ScalarType((1.,0.))) #Constant((1.0, 0.0))
            dp =  fem.Constant(mesh, ScalarType((0.,1.))) #Constant((0.0, 1.0))
        elif tdim == 3:
            # REMEMBER TO VERIFY THIS
            df =  fem.Constant(mesh, ScalarType((0.,0., 1.))) #Constant((1.0, 0.0))
            dp =  fem.Constant(mesh, ScalarType((0.,1., 0.))) #Constant((0.0, 1.0))



        sigma = -p * Identity(len(u)) + grad(u) + grad(u).T



        n = -FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data = facet_tags, subdomain_id = 1800)


        Fd_equation = fem.form(dot(dot(sigma, n), df) * ds)


        Fl_equation = fem.form(dot(dot(sigma, n), dp) * ds)



        return J, fem.assemble_scalar(Fd_equation), fem.assemble_scalar(Fl_equation), u, p
    

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







if __name__=="__main__":
    """
    from generate_2d_airplane import Airplane2D
    airplane = Airplane2D("final/airplane_2d/normal_concordia_2d_coordinates.csv", attack_angle=0, box_size=25, mesh_size=5, points_scale=100)
    airplane.generate_mesh()
    mesh, _, facets = airplane.get_mesh()
    stokes = Stokes()
    inflow_speed = lambda x: (np.stack((-1/24*(x[1,:]-25)*(x[1,:]), np.zeros(x.shape[1]))))
    stokes.solve(mesh, facets, inflow_speed)"""
    """
    from dolfinx import io
    from mpi4py import MPI

    mesh , cell_tags , facet_tags = io.gmshio.read_from_msh("final/cube_3d/mesh_out.msh", MPI.COMM_WORLD , 0 , gdim =3)
    stokes = Stokes()
    inflow_speed = lambda x: (np.stack((-1/24*(x[1,:]-25)*(x[1,:]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
    j, drag, lift, u, p = stokes.solve(mesh, facet_tags, inflow_speed)
    print(drag)
    print(lift)"""
    pass



