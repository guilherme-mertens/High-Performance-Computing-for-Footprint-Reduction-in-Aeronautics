# from generate_2d_airplane import Airplane2D
from generate_3d_airplane import Airplane3D
from solve_navier_stokes import NavierStokes
from solve_stokes import Stokes
import pandas as pd
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from typing import Union

from mpi4py import MPI
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
import dolfinx as dfx
from dolfinx import io
from mpi4py import MPI

class ONERA:

    def __init__(self) -> None:
        self.epsilon = 1e-5
        self.stokes = Stokes()
        self.navier_stokes = NavierStokes()
        

        # self.airplane_2d_concorde = Airplane2D("final/airplane_2d/normal_concordia_2d_coordinates.csv")
        # self.airplane_2d_concorde_broken_nose = Airplane2D("final/airplane_2d/a.csv")

        self.airplane_3d_concorde = Airplane3D("/usr/users/st76o/st76o_1/project/final/Concorde_SD_envelop_1000.geo")


    def generate_2d_concorde(self, attack_angle=0):
        return self.airplane_2d_concorde.get_mesh(attack_angle)

    def generate_2d_concorde_broken_nose(self, attack_angle=0):
        return self.airplane_2d_concorde_broken_nose.get_mesh(attack_angle=0)

    def generate_3d_concorde(self, attack_angle=0):
        return self.airplane_3d_concorde.get_mesh(attack_angle)

    def generate_3d_concorde_broken_nose(self, attack_angle=0):
        pass


    # def plot_mesh(self, airplane_class: Union[Airplane3D, Airplane2D], attack_angle=0):
    #     pass


    def solve_stokes(self, mesh, facets, inflow=None):
        if inflow:
            return self.stokes.solve(mesh, facets, inflow)
        else:
            return self.stokes.solve(mesh, facets)

    def solve_navier_stokes(self, mesh, facets):
        return self.navier_stokes.solve(mesh, facets)
    
    def solve_and_save_stokes(self, mesh, facets):
        return self.stokes.solve(mesh, facets)

    def solve_and_save_navier_stokes(self, mesh, facets):
        return self.navier_stokes.solve(mesh, facets)

    def solve_and_save_stokes_in_range_angle(self, generare_airplane_function: callable, save: bool = False, path: str = None,
                                             angles: list = list(np.arange(-50, 50, 20)),
                                             inflow: callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
                                             ):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        j_values_ls = []
        fd_values_ls = []
        fl_values_ls = []
        angle_ls=[]
        for angle in angles:


            print(angle)
            mesh, _, facets = generare_airplane_function(angle)
            # file = io.XDMFFile(MPI.COMM_WORLD, "/usr/users/st76o/st76o_1/project/final/mesh.xdmf", "r")
            # msh = file.read_mesh()
            j, fd, fl, u, p = self.solve_stokes(mesh, facets, inflow)
            
            j_val = np.array(j, dtype='float64')
            fd_val = np.array(fd, dtype='float64')
            fl_val = np.array(fl, dtype='float64')

            # Initialize empty numpy arrays for the sums; these will be used to store the result of the reduction.
            # Note: For MPI_Reduce, only the root process (typically rank 0) will have the meaningful summed values.
            # For MPI_Allreduce, every process will get the final summed values.
            global_j = np.empty_like(j_val)
            global_fd = np.empty_like(fd_val)
            global_fl = np.empty_like(fl_val)

            # Perform the reduction to sum up the values. Replace 'MPI_SUM' with the appropriate MPI operation if needed.
            comm.Reduce([j_val, MPI.DOUBLE], [global_j, MPI.DOUBLE], op=MPI.SUM, root=0)
            comm.Reduce([fd_val, MPI.DOUBLE], [global_fd, MPI.DOUBLE], op=MPI.SUM, root=0)
            comm.Reduce([fl_val, MPI.DOUBLE], [global_fl, MPI.DOUBLE], op=MPI.SUM, root=0)

            if rank == 0:
                # Append the results and the angle to the lists on rank 0
                j_values_ls.append(global_j.item())  # Convert numpy array to scalar
                fd_values_ls.append(global_fd.item())  # Convert numpy array to scalar
                fl_values_ls.append(global_fl.item())  # Convert numpy array to scalar
                angle_ls.append(angle)


            # Only rank 0 executes this block
        dict_to_pandas = {'angle': angle_ls, 'j': j_values_ls, 'fl': fl_values_ls, 'fd': fd_values_ls}

        df = pd.DataFrame(dict_to_pandas)
        print(df)
        if save:
            # Ensure the saving path is not None and valid
            if path is not None:
                df.to_csv(path, index=False)
            else:
                print("No path provided for saving.")

        return df


    def plot_stokes_results(self, path):
        df = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

        plt.subplot(2, 2, 1)
        plt.plot(df["angle"], df["fd"])
        plt.title("Fd")

        plt.subplot(2, 2, 2)
        plt.plot(df["angle"], df["fl"])
        plt.title("Fl")

        plt.subplot(2, 2, 3)
        plt.plot(df["angle"], df["fl"] / df["fd"])
        plt.title("Fl / Fd")

        plt.subplot(2, 2, 4)
        plt.plot(df["angle"], df["j"])
        plt.title("J")

        plt.show()


    


if __name__=="__main__":
    onera = ONERA()
    inflow_speed=500
    inflow = lambda x: (np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), -inflow_speed*np.ones(x.shape[1]))))
    result = onera.solve_and_save_stokes_in_range_angle(onera.generate_3d_concorde, save=True,
                                                         path=f"/usr/users/st76o/st76o_1/project/final/sla_3d_v{inflow_speed}.csv", inflow=inflow)
    # onera.plot_stokes_results(f"/usr/users/st76o/st76o_1/project/final/sla_3d1_v{inflow_speed}.csv")