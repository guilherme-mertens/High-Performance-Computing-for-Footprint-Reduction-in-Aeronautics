from dolfinx import io
from mpi4py import MPI

def read_mesh_from_msh(mesh_file, comm, rank, gdim):
    """
    Read a mesh from an MSH file and return the mesh, cell tags, and facet tags.

    Parameters:
    -----------
    mesh_file : str
        Path to the MSH file.
    comm : MPI.Comm
        MPI communicator.
    rank : int
        Rank of the process.
    gdim : int
        Geometric dimension of the mesh.

    Returns:
    --------
    msh : dolfinx.mesh.Mesh
        Loaded mesh.
    cell_tags : dolfinx.mesh.MeshTags
        Mesh tags for cells.
    facet_tags : dolfinx.mesh.MeshTags
        Mesh tags for facets.
    """
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(mesh_file, comm, rank, gdim)
    return msh, cell_tags, facet_tags

def write_mesh_to_xdmf(mesh, cell_tags, facet_tags, xdmf_file):
    """
    Write the mesh, cell tags, and facet tags to an XDMF file.

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        Mesh to be written.
    cell_tags : dolfinx.mesh.MeshTags
        Mesh tags for cells.
    facet_tags : dolfinx.mesh.MeshTags
        Mesh tags for facets.
    xdmf_file : str
        Path to the XDMF file.
    """
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

def main():
    """
    Main function to read a mesh from an MSH file and write it to an XDMF file.
    """
    mesh_file = "worksheet2/mesh_out.msh"
    xdmf_file = "worksheet2/mesh_out.xdmf"
    
    msh, cell_tags, facet_tags = read_mesh_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    write_mesh_to_xdmf(msh, cell_tags, facet_tags, xdmf_file)

if __name__ == "__main__":
    main()
