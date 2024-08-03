import unittest
from generate_2d_airplane import Airplane2D

class TestAirplane2D(unittest.TestCase):
    def test_read_points(self):
        airplane = Airplane2D("test_silhouette.csv")
        points, x_mean, y_mean = airplane.read_points("test_silhouette.csv", 3000, 0, 0)
        self.assertIsInstance(points, zip)
        self.assertIsInstance(x_mean, float)
        self.assertIsInstance(y_mean, float)

    def test_rotate(self):
        airplane = Airplane2D("test_silhouette.csv")
        points = [(1, 0), (0, 1)]
        rotated_points = airplane.rotate(points, 90)
        expected_points = [(0, 1), (-1, 0)]
        self.assertAlmostEqual(rotated_points[0][0], expected_points[0][0])
        self.assertAlmostEqual(rotated_points[0][1], expected_points[0][1])
        self.assertAlmostEqual(rotated_points[1][0], expected_points[1][0])
        self.assertAlmostEqual(rotated_points[1][1], expected_points[1][1])

if __name__ == "__main__":
    unittest.main()
