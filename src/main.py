from generate_3d_airplane import Airplane3D
from solve_navier_stokes import NavierStokes
from solve_stokes import Stokes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable

from mpi4py import MPI
from dolfinx.io import XDMFFile

class ONERA:
    """
    Class to manage the generation of aircraft meshes and solve Stokes and Navier-Stokes problems.
    """

    def __init__(self) -> None:
        """
        Initializes the ONERA class, setting up Stokes and Navier-Stokes solvers
        and loading the 3D Concorde airplane model.
        """
        self.epsilon = 1e-5
        self.stokes = Stokes()
        self.navier_stokes = NavierStokes()
        self.airplane_3d_concorde = Airplane3D("/usr/users/st76o/st76o_1/project/final/Concorde_SD_envelop_1000.geo")

    def generate_3d_concorde(self, attack_angle=0):
        """
        Generates the 3D mesh for the Concorde aircraft at a given angle of attack.

        :param attack_angle: Angle of attack for mesh generation.
        :return: 3D mesh of the Concorde aircraft.
        """
        return self.airplane_3d_concorde.get_mesh(attack_angle)

    def solve_stokes(self, mesh, facets, inflow=None):
        """
        Solves the Stokes problem for the given mesh and facets.

        :param mesh: The mesh to solve the problem on.
        :param facets: Facets of the mesh.
        :param inflow: Optional function to define the inflow velocity.
        :return: Results of the Stokes problem solution.
        """
        if inflow:
            return self.stokes.solve(mesh, facets, inflow)
        else:
            return self.stokes.solve(mesh, facets)

    def solve_navier_stokes(self, mesh, facets):
        """
        Solves the Navier-Stokes problem for the given mesh and facets.

        :param mesh: The mesh to solve the problem on.
        :param facets: Facets of the mesh.
        :return: Results of the Navier-Stokes problem solution.
        """
        return self.navier_stokes.solve(mesh, facets)

    def solve_and_save_stokes(self, mesh, facets):
        """
        Solves and saves the results of the Stokes problem for the given mesh and facets.

        :param mesh: The mesh to solve the problem on.
        :param facets: Facets of the mesh.
        :return: Results of the Stokes problem solution.
        """
        return self.stokes.solve(mesh, facets)

    def solve_and_save_navier_stokes(self, mesh, facets):
        """
        Solves and saves the results of the Navier-Stokes problem for the given mesh and facets.

        :param mesh: The mesh to solve the problem on.
        :param facets: Facets of the mesh.
        :return: Results of the Navier-Stokes problem solution.
        """
        return self.navier_stokes.solve(mesh, facets)

    def solve_and_save_stokes_in_range_angle(self, generare_airplane_function: Callable, save: bool = False, path: str = None,
                                             angles: list = list(np.arange(-50, 50, 20)),
                                             inflow: Callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))):
        """
        Solves and saves the results of the Stokes problem over a range of angles of attack.

        :param generare_airplane_function: Function to generate the aircraft mesh.
        :param save: Whether to save the results.
        :param path: Path to save the results.
        :param angles: List of angles of attack to solve for.
        :param inflow: Function to define the inflow velocity.
        :return: DataFrame with results for each angle.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        j_values_ls = []
        fd_values_ls = []
        fl_values_ls = []
        angle_ls = []

        for angle in angles:
            print(angle)
            mesh, _, facets = generare_airplane_function(angle)
            j, fd, fl, u, p = self.solve_stokes(mesh, facets, inflow)

            j_val = np.array(j, dtype='float64')
            fd_val = np.array(fd, dtype='float64')
            fl_val = np.array(fl, dtype='float64')

            global_j = np.empty_like(j_val)
            global_fd = np.empty_like(fd_val)
            global_fl = np.empty_like(fl_val)

            comm.Reduce([j_val, MPI.DOUBLE], [global_j, MPI.DOUBLE], op=MPI.SUM, root=0)
            comm.Reduce([fd_val, MPI.DOUBLE], [global_fd, MPI.DOUBLE], op=MPI.SUM, root=0)
            comm.Reduce([fl_val, MPI.DOUBLE], [global_fl, MPI.DOUBLE], op=MPI.SUM, root=0)

            if rank == 0:
                j_values_ls.append(global_j.item())
                fd_values_ls.append(global_fd.item())
                fl_values_ls.append(global_fl.item())
                angle_ls.append(angle)

        if rank == 0:
            dict_to_pandas = {'angle': angle_ls, 'j': j_values_ls, 'fl': fl_values_ls, 'fd': fd_values_ls}
            df = pd.DataFrame(dict_to_pandas)
            print(df)
            if save:
                if path is not None:
                    df.to_csv(path, index=False)
                else:
                    print("No path provided for saving.")
            return df

    def plot_stokes_results(self, path):
        """
        Plots the results of the Stokes problem from a CSV file.

        :param path: Path to the CSV file with the results.
        """
        df = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

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


if __name__ == "__main__":
    onera = ONERA()
    inflow_speed = 500
    inflow = lambda x: (np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), -inflow_speed * np.ones(x.shape[1]))))
    result = onera.solve_and_save_stokes_in_range_angle(onera.generate_3d_concorde, save=True,
                                                        path=f"/usr/users/st76o/st76o_1/project/final/sla_3d_v{inflow_speed}.csv", inflow=inflow)
    # onera.plot_stokes_results(f"/usr/users/st76o/st76o_1/project/final/sla_3d1_v{inflow_speed}.csv")
