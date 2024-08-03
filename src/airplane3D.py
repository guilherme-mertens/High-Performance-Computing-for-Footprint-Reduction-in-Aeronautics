import os
from dolfinx import io
from mpi4py import MPI
from math import radians
import gmsh
import sys

class Airplane3D:
    """
    Class to generate the 3D mesh of an aircraft from a GEO file.

    Attributes:
        base_file (str): Path to the base GEO file.
        new_angle_file (str): Path to the GEO file with the new angle of attack.
        mesh_file (str): Path to the generated mesh file.
    """
    def __init__(self, base_file) -> None:
        self.base_file = base_file
        self.new_angle_file = '.'.join([self.base_file.split(".")[0] + "newangle", "geo"])
        self.mesh_file = '.'.join([self.base_file.split(".")[0], "msh"])

    def change_attack_angle(self, file, attack_angle):
        """
        Changes the angle of attack in the GEO file.

        Args:
            file (str): Path to the GEO file.
            attack_angle (float): Angle of attack in degrees.
        """
        with open(file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if 'angle' in line:
                line = line.replace('angle', str(radians(attack_angle)))
            new_lines.append(line)

        with open(self.new_angle_file, 'w') as f:
            f.writelines(new_lines)

    def geo_to_msh(self, file):
        """
        Converts a GEO file to an MSH file using Gmsh.

        Args:
            file (str): Path to the GEO file.
        """
        os.system(f'gmsh -3 -format msh2 {file} -o {self.mesh_file}')

    def save_mesh_file(self, attack_angle):
        """
        Saves the mesh with a given angle of attack.

        Args:
            attack_angle (float): Angle of attack.
        """
        self.change_attack_angle(self.base_file, attack_angle)
        self.geo_to_msh(self.new_angle_file)

    def get_mesh(self, attack_angle):
        """
        Obtains the mesh with a given angle of attack.

        Args:
            attack_angle (float): Angle of attack.

        Returns:
            mesh (Mesh): Generated mesh.
        """
        self.save_mesh_file(attack_angle)
        with io.XDMFFile(MPI.COMM_WORLD, self.mesh_file, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            cell_tags = xdmf.read_meshtags(mesh, name="Grid")
            facet_tags = xdmf.read_meshtags(mesh, name="Grid")
        return mesh, cell_tags, facet_tags
    
    def plot_mesh(self, attack_angle):
        self.change_attack_angle(self.base_file, attack_angle)
        self.geo_to_msh(self.new_angle_file)

        gmsh.initialize()


        gmsh.open(self.mesh_file)


        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

        self.delete_auxiliar_files()
        gmsh.finalize()

if __name__ == "__main__":
    airplane_3d = Airplane3D("final/airplane_3d/airplane3d.geo")
    airplane_3d.save_mesh_file(attack_angle=30)