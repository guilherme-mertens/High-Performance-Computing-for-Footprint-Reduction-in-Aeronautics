import gmsh
import sys
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI

def generate_mesh():
    """
    Generate a mesh with marked boundaries using gmsh.

    Returns:
        tuple: The generated mesh, cell tags, and facet tags.
    """
    gmsh.initialize()

    gmsh.model.add("mesh")

    # Define the geometry
    lc = 0.1  # characteristic length for mesh size

    # Add points for the rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(2, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)

    pc1 = gmsh.model.geo.addPoint(0.5, 0.25, 0, lc/10)
    pc2 = gmsh.model.geo.addPoint(0.5, 0.75, 0, lc/10)
    pc = gmsh.model.geo.addPoint(0.5, 0.5, 0, lc/100)

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

    # Synchronize necessary before meshing
    gmsh.model.geo.synchronize()

    # Define Physical Groups for boundaries
    fluid = gmsh.model.addPhysicalGroup(2, [plane_surface], 0)
    gmsh.model.setPhysicalName(1, fluid, "Fluid")

    inflow = gmsh.model.addPhysicalGroup(1, [l4], 1)
    gmsh.model.setPhysicalName(1, inflow, "Inflow")

    outflow = gmsh.model.addPhysicalGroup(1, [l2], 2)
    gmsh.model.setPhysicalName(1, outflow, "Outflow")

    walls = gmsh.model.addPhysicalGroup(1, [l1, l3], 3)
    gmsh.model.setPhysicalName(1, walls, "Walls")

    circle = gmsh.model.addPhysicalGroup(1, [circle1, circle2], 4)
    gmsh.model.setPhysicalName(1, circle, "Circle")

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)

    gmsh.finalize()

    return mesh, cell_tags, facet_tags

if __name__ == "__main__":
    mesh, cell_tags, facet_tags = generate_mesh()
