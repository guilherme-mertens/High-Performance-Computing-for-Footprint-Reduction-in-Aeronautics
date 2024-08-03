import unittest
import os
from generate_3d_airplane import Airplane3D

class TestAirplane3D(unittest.TestCase):
    def test_change_attack_angle(self):
        airplane = Airplane3D("test_airplane.geo")
        airplane.change_attack_angle("test_airplane.geo", 30)
        self.assertTrue(os.path.exists(airplane.new_angle_file))

    def test_geo_to_msh(self):
        airplane = Airplane3D("test_airplane.geo")
        airplane.geo_to_msh("test_airplane.geo")
        self.assertTrue(os.path.exists(airplane.mesh_file))

if __name__ == "__main__":
    unittest.main()
