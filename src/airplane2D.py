import pandas as pd
import gmsh
import sys
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import math
from dolfinx import io

class Airplane2D:
    """
    Classe para gerar a malha 2D de um avião a partir de um arquivo de silhueta.
    
    Atributos:
        airplane_points (list): Pontos da silhueta do avião.
        x_mean (float): Média das coordenadas x dos pontos da silhueta.
        y_mean (float): Média das coordenadas y dos pontos da silhueta.
        size (float): Tamanho da caixa de simulação.
        lc (float): Tamanho da malha.
    """
    def __init__(self, silhouette_file, box_size=1, mesh_size=0.1, points_scale=3000, x_bias=0, y_bias=0) -> None:
        self.airplane_points, self.x_mean, self.y_mean = self.read_points(silhouette_file, points_scale, x_bias, y_bias)
        self.size = box_size
        self.lc = mesh_size

    def get_mesh(self, attack_angle):
        """
        Gera a malha com um determinado ângulo de ataque.

        Args:
            attack_angle (float): Ângulo de ataque.

        Returns:
            mesh (Mesh): Malha gerada.
            cell_tags (MeshTags): Tags das células da malha.
            facet_tags (MeshTags): Tags das facetas da malha.
        """
        self.generate_mesh(attack_angle)
        mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        return mesh, cell_tags, facet_tags

    def read_points(self, file, points_scale, x_bias, y_bias):
        """
        Lê os pontos da silhueta do arquivo.

        Args:
            file (str): Caminho para o arquivo CSV contendo os pontos da silhueta.
            points_scale (float): Escala dos pontos.
            x_bias (float): Deslocamento em x.
            y_bias (float): Deslocamento em y.

        Returns:
            tuple: (pontos da silhueta, média x, média y)
        """
        points = pd.read_csv(file)
        points = points[["X", "Y"]]
        points = points / points_scale
        points["Y"] += y_bias
        points["X"] += x_bias + 0.3
        x_mean = points["X"].mean()
        y_mean = points["Y"].mean()
        x = points["X"].to_list()
        y = points["Y"].to_list()
        return zip(x, y), x_mean, y_mean

    def save_mesh(self, path):
        """
        Salva a malha em um arquivo.

        Args:
            path (str): Caminho para salvar a malha.
        """
        gmsh.write(path)

    def save_as_vtk(self):
        """Salva a malha em formato VTK."""
        pass

    def plot_mesh(self):
        """Plota a malha gerada."""
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
        gmsh.finalize()

    def rotate(self, points, angle, origin=(0, 0)):
        """
        Rotaciona pontos em torno de uma origem.

        Args:
            points (list): Lista de pontos a serem rotacionados.
            angle (float): Ângulo de rotação em graus.
            origin (tuple): Ponto de origem para a rotação.

        Returns:
            list: Pontos rotacionados.
        """
        angle_rad = math.radians(angle)
        ref_x, ref_y = origin
        rotated_points = []
        for point in points:
            x, y = point
            translated_x = x - ref_x
            translated_y = y - ref_y
            rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
            rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
            rotated_x += ref_x
            rotated_y += ref_y
            rotated_points.append([rotated_x, rotated_y])
        return rotated_points

    def generate_mesh(self, attack_angle):
        """Gera a malha do avião com um determinado ângulo de ataque."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mesh")

        p1 = gmsh.model.geo.addPoint(-self.size, -self.size, 0, self.lc)
        p2 = gmsh.model.geo.addPoint(self.size, -self.size, 0, self.lc)
        p3 = gmsh.model.geo.addPoint(self.size, self.size, 0, self.lc)
        p4 = gmsh.model.geo.addPoint(-self.size, self.size, 0, self.lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        self.airplane_points = self.rotate(self.airplane_points, attack_angle, origin=(self.x_mean, self.y_mean))
        airfoil_points = []
        for xi, yi in self.airplane_points:
            pid = gmsh.model.geo.addPoint(xi, yi, 0, self.lc / 10)
            airfoil_points.append(pid)

        airfoil_lines = []
        for i in range(len(airfoil_points) - 1):
            lid = gmsh.model.geo.addLine(airfoil_points[i], airfoil_points[i + 1])
            airfoil_lines.append(lid)
        airfoil_lines.append(gmsh.model.geo.addLine(airfoil_points[-1], airfoil_points[0]))

        outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        wing_loop = gmsh.model.geo.addCurveLoop(airfoil_lines)
        plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop, wing_loop])

        gmsh.model.geo.synchronize()

        fluid = gmsh.model.addPhysicalGroup(2, [plane_surface], 5)
        gmsh.model.setPhysicalName(1, fluid, "Fluid")

        inflow = gmsh.model.addPhysicalGroup(1, [l4], 1)
        gmsh.model.setPhysicalName(1, inflow, "Inflow")

        outflow = gmsh.model.addPhysicalGroup(1, [l2], 2)
        gmsh.model.setPhysicalName(1, outflow, "Outflow")

        walls = gmsh.model.addPhysicalGroup(1, [l3, l1], 3)
        gmsh.model.setPhysicalName(1, walls, "Walls")

        obstacle = gmsh.model.addPhysicalGroup(1, airfoil_lines, 4)
        gmsh.model.setPhysicalName(1, obstacle, "Obstacle")

        gmsh.model.mesh.generate(2)

if __name__ == "__main__":
    airplane = Airplane2D("final/airplane_2d/normal_concordia_2d_coordinates.csv")
    airplane.generate_mesh(attack_angle=-50)
    airplane.plot_mesh()
