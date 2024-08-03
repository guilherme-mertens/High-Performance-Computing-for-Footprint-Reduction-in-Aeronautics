import gmsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from compute_stokes import solve_stokes

def rotate(x, y, theta):
    """
    Rotate points around the origin by theta degrees.

    Parameters:
    -----------
    x : np.ndarray
        X-coordinates of points.
    y : np.ndarray
        Y-coordinates of points.
    theta : float
        Angle in degrees.

    Returns:
    --------
    xr : np.ndarray
        Rotated X-coordinates.
    yr : np.ndarray
        Rotated Y-coordinates.
    """
    theta_rad = np.radians(theta)
    xr = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    yr = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    return xr, yr

def naca4_full(number, chord=0.1, n=50, angle_of_attack=0):
    """
    Generate NACA 4-digit airfoil coordinates.

    Parameters:
    -----------
    number : str
        NACA 4-digit number (e.g., '2412').
    chord : float
        Chord length of the airfoil.
    n : int
        Number of points to generate along the chord.
    angle_of_attack : float
        Angle of attack in degrees.

    Returns:
    --------
    x_full : np.ndarray
        X-coordinates of the airfoil.
    y_full : np.ndarray
        Y-coordinates of the airfoil.
    """
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0
    
    x = np.linspace(0, 1, n)
    x = (0.5 * (1 - np.cos(np.pi * x))) * chord
    
    yt = 5 * t * chord * (0.2969 * np.sqrt(x/chord) - 0.1260 * (x/chord) - 0.3516 * (x/chord)**2 + 0.2843 * (x/chord)**3 - 0.1015 * (x/chord)**4)
    
    yc = np.where(x < p * chord, m * (x / np.power(p, 2)) * (2 * p - (x / chord)), m * ((chord - x) / np.power(1-p, 2)) * (1 + (x / chord) - 2 * p))
    dyc_dx = np.where(x < p * chord, 2*m / np.power(p, 2) * (p - x / chord), 2*m / np.power(1-p, 2) * (p - x / chord))
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Rotate coordinates for angle of attack
    xu, yu = rotate(xu, yu, angle_of_attack)
    xl, yl = rotate(xl, yl, angle_of_attack)
    
    # Combine upper and lower surfaces into one loop without duplicating the trailing edge
    x_full = np.concatenate([xu[:-1], xl[::-1]])[:-1]
    y_full = np.concatenate([yu[:-1], yl[::-1]])[:-1]
    
    return x_full, y_full

def create_NACA_mesh(theta, camber):
    """
    Create a mesh around a NACA wing profile in a rectangular domain.

    Parameters:
    -----------
    theta : float
        Angle of the centerline of the NACA wing with respect to the flow direction.
    camber : str
        NACA 4-digit number (e.g., '2412').

    Returns:
    --------
    mesh : dolfinx.mesh.Mesh
        Generated mesh.
    facet_tags : dolfinx.mesh.MeshTags
        Facet tags for boundary conditions.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("mesh")

    # Define the geometry
    lc = 0.1  # characteristic length for mesh size

    # Add points for the rectangle
    p1 = gmsh.model.geo.addPoint(-15, -6, 0, lc)
    p2 = gmsh.model.geo.addPoint(15, -6, 0, lc)
    p3 = gmsh.model.geo.addPoint(15, 6, 0, lc)
    p4 = gmsh.model.geo.addPoint(-15, 6, 0, lc)

    # Add lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    center_x = 0
    center_y = 0
    x, y = naca4_full(camber, chord=1.0, angle_of_attack=theta)
    x, y = x + center_x, y + center_y

    airfoil_points = []
    for xi, yi in zip(x, y):
        pid = gmsh.model.geo.addPoint(xi, yi, 0, lc/10)
        airfoil_points.append(pid)

    # Add lines for the airfoil
    airfoil_lines = []
    for i in range(len(airfoil_points) - 1):
        lid = gmsh.model.geo.addLine(airfoil_points[i], airfoil_points[i + 1])
        airfoil_lines.append(lid)
    # Close the airfoil loop by connecting the last point back to the first
    airfoil_lines.append(gmsh.model.geo.addLine(airfoil_points[-1], airfoil_points[0]))

    # Create curve loops and plane surfaces
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    wing_loop = gmsh.model.geo.addCurveLoop(airfoil_lines)
    plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop, wing_loop])

    # Synchronize necessary before meshing
    gmsh.model.geo.synchronize()

    # Define Physical Groups for boundaries
    fluid = gmsh.model.addPhysicalGroup(2, [plane_surface], 0)  # Fluid
    gmsh.model.setPhysicalName(2, fluid, "Fluid")

    inflow = gmsh.model.addPhysicalGroup(1, [l4], 1)  # Inflow
    gmsh.model.setPhysicalName(1, inflow, "Inflow")

    outflow = gmsh.model.addPhysicalGroup(1, [l2], 2)  # Outflow
    gmsh.model.setPhysicalName(1, outflow, "Outflow")

    walls = gmsh.model.addPhysicalGroup(1, [l3, l1], 3)  # Walls
    gmsh.model.setPhysicalName(1, walls, "Walls")

    obstacle = gmsh.model.addPhysicalGroup(1, airfoil_lines, 4)  # Obstacle
    gmsh.model.setPhysicalName(1, obstacle, "Obstacle")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    gmsh.finalize()

    return mesh, facet_tags

def run_simulations(cambers, angles):
    """
    Run simulations for different camber values and angles.

    Parameters:
    -----------
    cambers : list of str
        List of NACA 4-digit camber values.
    angles : list of float
        List of angles in degrees.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the results of the simulations.
    """
    camber_values = []
    angle_values = []
    j_values = []
    fd_values = []
    fl_values = []

    for camber in cambers:
        for angle in angles:
            mesh, facets = create_NACA_mesh(angle, camber)
            j, fd, fl, u, p = solve_stokes(mesh, facets)
            camber_values.append(camber)
            angle_values.append(angle)
            j_values.append(j)
            fd_values.append(fd)
            fl_values.append(fl)
            print(f"Camber: {camber}, Angle: {angle}")

    dict_to_pandas = {'camber': camber_values, 'angle': angle_values, 'j': j_values, 'fl': fl_values, 'fd': fd_values}
    df = pd.DataFrame(dict_to_pandas)
    return df

def plot_results(file_path):
    """
    Plot the results of the simulation.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the results.
    """
    df = pd.read_csv(file_path, index_col=0)
    grouped = df.groupby('camber')
    names = grouped.groups.keys()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    for name, group in grouped:
        plt.plot(group["angle"], group["fd"], label=name)
    plt.title("Fd")
    plt.legend()

    plt.subplot(2, 2, 2)
    for name, group in grouped:
        plt.plot(group["angle"], group["fl"], label=name)
    plt.title("Fl")
    plt.legend()

    plt.subplot(2, 2, 3)
    for name, group in grouped:
        plt.plot(group["angle"], group["fl"] / group["fd"], label=name)
    plt.title("Fl / Fd")
    plt.legend()

    plt.subplot(2, 2, 4)
    for name, group in grouped:
        plt.plot(group["angle"], group["j"], label=name)
    plt.title("J")
    plt.legend()

    fig.suptitle("All Cambers")
    plt.show()

def main():
    """
    Main function to run simulations, save results, and plot them.
    """
    cambers = ["2412", "6112", "9912", "0012", "8812"]
    angles = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    results_df = run_simulations(cambers, angles)
    results_df.to_csv("results_ex3_2.csv")

    plot_results("results_ex3_2.csv")

if __name__ == "__main__":
    main()
