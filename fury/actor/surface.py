"""Surface actors for FURY."""

import logging
import os

import numpy as np

from fury.geometry import buffer_to_geometry, create_mesh
from fury.io import load_image_texture
from fury.lib import MeshBasicMaterial
from fury.material import _create_mesh_material, validate_opacity
from fury.utils import generate_planar_uvs


def surface(
    vertices,
    faces,
    *,
    material="phong",
    colors=None,
    texture=None,
    texture_axis="xy",
    texture_coords=None,
    opacity=1.0,
):
    """Create a surface mesh actor from vertices and faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        The vertex positions of the surface mesh.
    faces : ndarray, shape (M, 3)
        The indices of the vertices that form each triangular face.
    material : str, optional
        The material type for the surface mesh. Options are 'phong' and 'basic'. This
        option only works with colors is passed.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    texture : str, optional
        Path to the texture image file.
    texture_axis : str, optional
        The axis to generate UV coordinates for the texture. Options are 'xy', 'yz',
        and 'xz'. This option only works with texture is passed.
    texture_coords : ndarray, shape (N, 2), optional
        Predefined UV coordinates for the texture mapping. If not provided, they will
        be generated based on the `texture_axis`.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    Mesh
        A mesh actor containing the generated surface with the specified properties.
    """
    geo = None
    mat = None

    opacity = validate_opacity(opacity)

    if colors is not None:
        if texture is not None:
            logging.warning("Texture will be ignored when colors are provided.")

        if isinstance(colors, np.ndarray) and colors.shape[0] == vertices.shape[0]:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
                colors=colors,
            )
            mat = _create_mesh_material(
                material=material, mode="vertex", opacity=opacity
            )
        elif isinstance(colors, (tuple, list)) and len(colors) == 3:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
            )
            mat = _create_mesh_material(color=colors, opacity=opacity)
        else:
            raise ValueError(
                "Colors must be either an ndarray with shape (N, 3) or (N, 4), "
                "or a tuple/list of length 3 for RGB colors."
            )
    elif texture is not None:
        if not os.path.exists(texture):
            raise FileNotFoundError(f"Texture file '{texture}' not found.")

        logging.warning(
            "texture option currently only supports planar projection,"
            " the plane can be provided by texture_axis parameter."
        )

        tex = load_image_texture(texture)
        if texture_coords is None:
            texture_coords = generate_planar_uvs(vertices, axis=texture_axis)
        elif (
            texture_coords.shape[0] != vertices.shape[0] or texture_coords.shape[1] != 2
        ):
            raise ValueError(
                "texture_coords must be an ndarray with shape (N, 2) "
                "where N is the number of vertices."
            )
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"),
            indices=faces.astype("int32"),
            texcoords=texture_coords.astype("float32"),
        )
        mat = MeshBasicMaterial(map=tex, opacity=opacity)
    else:
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"), indices=faces.astype("int32")
        )
        mat = _create_mesh_material(material=material, opacity=opacity)

    obj = create_mesh(geo, mat)
    return obj
