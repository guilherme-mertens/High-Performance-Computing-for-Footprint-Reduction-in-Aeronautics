
import os
from dolfinx import io
from mpi4py import MPI
# import gmsh
from math import radians
import sys

class Airplane3D:

    def __init__(self, base_file) -> None:
        self.base_file = base_file
        self.new_angle_file = '.'.join([self.base_file.split(".")[0]+"newangle", "geo"])
        self.mesh_file = '.'.join([self.base_file.split(".")[0], "msh"])


    

    def change_attack_angle(self, file, attack_angle):
        with open(file, 'r') as f:
            linhas = f.readlines()

        # Localiza e substitui o valor "k"
        novas_linhas = []
        for linha in linhas:
            if 'angle' in linha:
                linha = linha.replace('angle', str(radians(attack_angle)))
            novas_linhas.append(linha)

        # Escreve as linhas modificadas para o novo arquivo
        with open(self.new_angle_file, 'w') as f:
            f.writelines(novas_linhas)
            

    def geo_to_msh(self, file):
        os.system(f'gmsh -3 -format msh2 {file} -o {self.mesh_file}')



    def save_mesh_file(self, attack_angle):
        self.change_attack_angle(self.base_file, attack_angle)
        self.geo_to_msh(self.new_angle_file)
        self.delete_auxiliar_files(remove_mesh=False)

    def save_as_vtk(self,):
        pass

    # def plot_mesh(self, attack_angle):
    #     self.change_attack_angle(self.base_file, attack_angle)
    #     self.geo_to_msh(self.new_angle_file)

    #     gmsh.initialize()


    #     gmsh.open(self.mesh_file)


    #     if '-nopopup' not in sys.argv:
    #         gmsh.fltk.run()

    #     self.delete_auxiliar_files()
    #     gmsh.finalize()



    def delete_auxiliar_files(self, remove_angle=False, remove_mesh=False):
        if os.path.exists(self.new_angle_file):
            if remove_angle:
                os.remove(self.new_angle_file)

        if os.path.exists(self.mesh_file):
            if remove_mesh:
                os.remove(self.mesh_file)



    def get_mesh(self, attack_angle):
        self.change_attack_angle(self.base_file, attack_angle)

        self.geo_to_msh(self.new_angle_file)
        

        mesh , cell_tags , facet_tags = io.gmshio.read_from_msh(self.mesh_file, MPI.COMM_WORLD , 0 , gdim =3)
        # with io.XDMFFile( MPI.COMM_WORLD , "/usr/users/st76o/st76o_1/project/final/mesh.xdmf", "w") as xdmf :   
        #     xdmf.write_mesh ( mesh )
        #     mesh.topology.create_connectivity(2, 3)
        #     xdmf.write_meshtags ( facet_tags , mesh.geometry )
        #     xdmf.write_meshtags ( cell_tags , mesh.geometry )
        self.delete_auxiliar_files()

        return mesh , cell_tags , facet_tags




if __name__=="__main__":

    a = Airplane3D("final/Concorde_SD_envelop_1000.geo")
    a.save_mesh_file(-30)

    """
    //+
Rotate {{1,0,0}, {0, 0, 0}, k} { Volume{1}; };
    """