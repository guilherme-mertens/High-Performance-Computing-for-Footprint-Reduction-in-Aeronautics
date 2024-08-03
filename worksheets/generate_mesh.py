import gmsh
import sys
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

def generate_mesh(mesh_file):
    """
    Generate a 2D mesh with a rectangular domain and a circular obstacle.

    Parameters:
    -----------
    mesh_file : str
        Path to save the generated mesh in MSH format.
    """
    gmsh.initialize()
    gmsh.model.add("mesh")

    lc = 0.1  # characteristic length for mesh size

    # Add points for the rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(2, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
    r = 0.125
    pc1 = gmsh.model.geo.addPoint(0.5, 0.5 - r, 0, lc / 10)
    pc2 = gmsh.model.geo.addPoint(0.5, 0.5 + r, 0, lc / 10)
    pc = gmsh.model.geo.addPoint(0.5, 0.5, 0, lc / 100)

    # Add lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a circle
    circle1 = gmsh.model.geo.addCircleArc(pc1, pc, pc2)
    circle2 = gmsh.model.geo.addCircleArc(pc2, pc, pc1)

    # Create curve loops and plane surfaces
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    circle_loop = gmsh.model.geo.addCurveLoop([circle1, circle2])
    plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop, circle_loop])

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Define physical groups
    fluid = gmsh.model.addPhysicalGroup(2, [plane_surface])
    gmsh.model.setPhysicalName(2, fluid, "Fluid")

    inflow = gmsh.model.addPhysicalGroup(1, [l4])
    gmsh.model.setPhysicalName(1, inflow, "Inflow")

    outflow = gmsh.model.addPhysicalGroup(1, [l2])
    gmsh.model.setPhysicalName(1, outflow, "Outflow")

    walls = gmsh.model.addPhysicalGroup(1, [l1, l3])
    gmsh.model.setPhysicalName(1, walls, "Walls")

    obstacle = gmsh.model.addPhysicalGroup(1, [circle1, circle2])
    gmsh.model.setPhysicalName(1, obstacle, "Obstacle")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh
    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.write(mesh_file)

    # Finalize gmsh
    gmsh.finalize()

    return mesh, cell_tags, facet_tags

def main():
    """
    Main function to generate and save the mesh.
    """
    mesh_file = "worksheet2/mesh_out.msh"
    generate_mesh(mesh_file)
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

if __name__ == "__main__":
    main()
