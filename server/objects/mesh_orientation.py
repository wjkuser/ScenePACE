"""Coordinate conversion used by SAGE's local asset library."""
import numpy as np
import trimesh

def apply_entry_rotation(mesh_dict, x_degrees: float=90.0, z_degrees: float=180.0):
    """Rotate render and physics meshes into the pipeline's upright frame."""
    x_transform = trimesh.transformations.rotation_matrix(np.radians(x_degrees), [1, 0, 0])
    z_transform = trimesh.transformations.rotation_matrix(np.radians(z_degrees), [0, 0, 1])
    for key in ('mesh', 'mesh_watertight'):
        mesh = mesh_dict.get(key)
        if mesh is not None:
            mesh.apply_transform(x_transform)
            mesh.apply_transform(z_transform)
    return mesh_dict
