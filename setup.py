from setuptools import setup, find_packages

setup(
    name='airplane_mesh_generator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'gmsh',
        'dolfinx',
        'mpi4py',
    ],
    entry_points={
        'console_scripts': [
            'generate_2d_airplane=generate_2d_airplane:main',
            'generate_3d_airplane=generate_3d_airplane:main',
        ],
    },
)