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
    Classe para gerenciar a geração de malhas de aeronaves e resolver problemas de Stokes e Navier-Stokes.
    """

    def __init__(self) -> None:
        """
        Inicializa a classe ONERA, configurando as soluções de Stokes e Navier-Stokes
        e carregando o modelo de avião 3D da Concorde.
        """
        self.epsilon = 1e-5
        self.stokes = Stokes()
        self.navier_stokes = NavierStokes()
        self.airplane_3d_concorde = Airplane3D("/usr/users/st76o/st76o_1/project/final/Concorde_SD_envelop_1000.geo")

    def generate_3d_concorde(self, attack_angle=0):
        """
        Gera a malha 3D para a aeronave Concorde em um determinado ângulo de ataque.

        :param attack_angle: Ângulo de ataque para a geração da malha.
        :return: Malha 3D da aeronave Concorde.
        """
        return self.airplane_3d_concorde.get_mesh(attack_angle)

    def solve_stokes(self, mesh, facets, inflow=None):
        """
        Resolve o problema de Stokes para a malha e os facetas fornecidas.

        :param mesh: A malha para resolver o problema.
        :param facets: Facetas da malha.
        :param inflow: Função opcional para definir a velocidade de entrada.
        :return: Resultados da solução do problema de Stokes.
        """
        if inflow:
            return self.stokes.solve(mesh, facets, inflow)
        else:
            return self.stokes.solve(mesh, facets)

    def solve_navier_stokes(self, mesh, facets):
        """
        Resolve o problema de Navier-Stokes para a malha e os facetas fornecidas.

        :param mesh: A malha para resolver o problema.
        :param facets: Facetas da malha.
        :return: Resultados da solução do problema de Navier-Stokes.
        """
        return self.navier_stokes.solve(mesh, facets)

    def solve_and_save_stokes(self, mesh, facets):
        """
        Resolve e salva os resultados do problema de Stokes para a malha e facetas fornecidas.

        :param mesh: A malha para resolver o problema.
        :param facets: Facetas da malha.
        :return: Resultados da solução do problema de Stokes.
        """
        return self.stokes.solve(mesh, facets)

    def solve_and_save_navier_stokes(self, mesh, facets):
        """
        Resolve e salva os resultados do problema de Navier-Stokes para a malha e facetas fornecidas.

        :param mesh: A malha para resolver o problema.
        :param facets: Facetas da malha.
        :return: Resultados da solução do problema de Navier-Stokes.
        """
        return self.navier_stokes.solve(mesh, facets)

    def solve_and_save_stokes_in_range_angle(self, generare_airplane_function: Callable, save: bool = False, path: str = None,
                                             angles: list = list(np.arange(-50, 50, 20)),
                                             inflow: Callable = lambda x: (np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))):
        """
        Resolve e salva os resultados do problema de Stokes em uma faixa de ângulos de ataque.

        :param generare_airplane_function: Função para gerar a malha da aeronave.
        :param save: Se deve salvar os resultados.
        :param path: Caminho para salvar os resultados.
        :param angles: Lista de ângulos de ataque para resolver.
        :param inflow: Função para definir a velocidade de entrada.
        :return: DataFrame com os resultados para cada ângulo.
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
        Plota os resultados do problema de Stokes a partir de um arquivo CSV.

        :param path: Caminho do arquivo CSV com os resultados.
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
