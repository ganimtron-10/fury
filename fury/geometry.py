"""Geometry utilities for FURY."""

from PIL import Image as PILImage
import numpy as np

from fury.lib import (
    Geometry,
    Image,
    ImageBasicMaterial,
    Line,
    Mesh,
    MeshBasicMaterial,
    MeshPhongMaterial,
    Points,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    Text,
    TextMaterial,
    Texture,
)


def buffer_to_geometry(positions, **kwargs):
    """Convert a buffer to a geometry object.

    Parameters
    ----------
    positions : array_like
        The positions buffer.
    **kwargs : dict
        A dict of attributes to define on the geometry object. Keys can be
        "colors", "normals", "texcoords", "indices", etc.

    Returns
    -------
    Geometry
        The geometry object.

    Raises
    ------
    ValueError
        If positions array is empty or None.
    """
    if positions is None:
        raise ValueError("positions array cannot be empty or None.")

    geo = Geometry(positions=positions, **kwargs)
    return geo


def create_mesh(geometry, material):
    """Create a mesh object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be either MeshPhongMaterial or MeshBasicMaterial.

    Returns
    -------
    Mesh
        The mesh object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of MeshPhongMaterial or MeshBasicMaterial.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(material, (MeshPhongMaterial, MeshBasicMaterial)):
        raise TypeError(
            "material must be an instance of MeshPhongMaterial or MeshBasicMaterial."
        )

    mesh = Mesh(geometry=geometry, material=material)
    return mesh


def create_line(geometry, material):
    """
    Create a line object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object.

    Returns
    -------
    Line
        The line object.
    """
    line = Line(geometry=geometry, material=material)
    return line


def line_buffer_separator(line_vertices, color=None):
    """
    Create a line buffer with separators between segments.

    Parameters
    ----------
    line_vertices : list of array_like
        The line vertices as a list of segments (each segment is an array of points).
    color : array_like, optional
        The color of the line segments.

    Returns
    -------
    positions : array_like
        The positions buffer with NaN separators.
    colors : array_like, optional
        The colors buffer with NaN separators (if color is provided).
    """

    total_vertices = sum(len(segment) for segment in line_vertices)
    total_size = total_vertices + len(line_vertices) - 1

    positions_result = np.empty((total_size, 3), dtype=np.float32)

    if color is None:
        color = np.asarray((1, 1, 1, 1), dtype=np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)

    if (len(color) == 3 or len(color) == 4) and color.ndim == 1:
        color = np.tile(color, (len(line_vertices), 1))
        color_mode = "line"
    elif len(color) == len(line_vertices) and color.ndim == 2:
        color_mode = "line"
    elif len(color) == len(line_vertices) and color.ndim > 2:
        color_mode = "vertex"
    elif len(color) == total_vertices:
        color_mode = "vertex_flattened"
    else:
        raise ValueError(
            "Color array size doesn't match either vertex count or segment count"
        )

    colors_result = np.empty((total_size, color.shape[-1]), dtype=np.float32)

    idx = 0
    color_idx = 0

    for i, segment in enumerate(line_vertices):
        segment_length = len(segment)

        positions_result[idx : idx + segment_length] = segment

        if color_mode == "vertex":
            colors_result[idx : idx + segment_length] = color[i]
            color_idx += segment_length

        elif color_mode == "line":
            colors_result[idx : idx + segment_length] = np.tile(
                color[i], (segment_length, 1)
            )
        elif color_mode == "vertex_flattened":
            colors_result[idx : idx + segment_length] = color[
                color_idx : color_idx + segment_length
            ]
            color_idx += segment_length

        idx += segment_length

        if i < len(line_vertices) - 1:
            positions_result[idx] = np.nan
            colors_result[idx] = np.nan
            idx += 1

    return positions_result, colors_result


def create_point(geometry, material):
    """Create a point object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be either PointsMaterial, PointsGaussianBlobMaterial,
        or PointsMarkerMaterial.

    Returns
    -------
    Points
        The point object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of PointsMaterial, PointsGaussianBlobMaterial, or PointsMarkerMaterial.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(
        material, (PointsMaterial, PointsGaussianBlobMaterial, PointsMarkerMaterial)
    ):
        raise TypeError(
            "material must be an instance of PointsMaterial, "
            "PointsGaussianBlobMaterial or PointsMarkerMaterial."
        )

    point = Points(geometry=geometry, material=material)
    return point


def create_text(text, material, **kwargs):
    """Create a text object.

    Parameters
    ----------
    text : str
        The text content.
    material : TextMaterial
        The material object.
    **kwargs : dict
        Additional properties like font_size, anchor, etc.

    Returns
    -------
    Text
        The text object.

    Raises
    ------
    TypeError
        If text is not a string or material is not an instance of TextMaterial.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(material, TextMaterial):
        raise TypeError("material must be an instance of TextMaterial.")

    text = Text(text=text, material=material, **kwargs)
    return text


def create_image(image_input, material, **kwargs):
    """Create an image object.

    Parameters
    ----------
    image_input : str or np.ndarray, optional
        The image content.
    material : Material
        The material object.
    **kwargs : dict, optional
        Additional properties like position, visible, etc.

    Returns
    -------
    Image
        The image object.
    """
    if isinstance(image_input, str):
        image = np.flipud(np.array(PILImage.open(image_input)).astype(np.float32))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim not in (2, 3):
            raise ValueError("image_input must be a 2D or 3D NumPy array.")
        if image_input.ndim == 3 and image_input.shape[2] not in (1, 3, 4):
            raise ValueError("image_input must have 1, 3, or 4 channels.")
        image = image_input
    else:
        raise TypeError("image_input must be a file path (str) or a NumPy array.")

    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    if image.max() > 1.0 or image.min() < 0.0:
        if image.max() == image.min():
            raise ValueError("Cannot normalize an image with constant pixel values.")
        image = (image - image.min()) / (image.max() - image.min())

    if not isinstance(material, ImageBasicMaterial):
        raise TypeError("material must be an instance of ImageBasicMaterial.")

    image = Image(
        Geometry(grid=Texture(image.astype(np.float32), dim=2)), material=material
    )
    return image


def rotate_vector(v, axis, angle):
    """Rotate a vector `v` around an axis `axis` by an angle `angle`.

    Parameters
    ----------
    v : array_like
        The vector to be rotated.
    axis : array_like
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    returns
    -------
    array_like
        The rotated vector.
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (
        v * cos_theta
        + np.cross(axis, v) * sin_theta
        + axis * np.dot(axis, v) * (1 - cos_theta)
    )


def prune_colinear(arr, colinear_threshold=0.9999):
    """Prune colinear points from the array.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        The input array of points.
    colinear_threshold : float, optional
        The threshold for colinearity. Points are considered colinear if the
        cosine of the angle between them is greater than or equal to this value.

    Returns
    -------
    ndarray, shape (3,)
        The pruned array with colinear points removed.
    """
    keep = [arr[0]]
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - keep[-1]
        v2 = arr[i + 1] - arr[i]
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        if (
            np.linalg.norm(v1 / np.linalg.norm(v1) - v2 / np.linalg.norm(v2))
            >= colinear_threshold
        ):
            keep.append(arr[i])
    keep.append(arr[-1])
    return np.stack(keep)


def axes_for_dir(d, prev_x=None):
    """Compute the axes for a given direction vector.

    Parameters
    ----------
    d : ndarray, shape (3,)
        The direction vector.
    prev_x : ndarray, shape (3,), optional
        The previous x-axis vector.

    Returns
    -------
    x : ndarray, shape (3,)
        The x-axis vector.
    y : ndarray, shape (3,)
        The y-axis vector.
    """
    d /= np.linalg.norm(d)
    if prev_x is None:
        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(d, up)) > 0.9:
            up = np.array([0, 1, 0], dtype=np.float32)
        x = np.cross(up, d)
    else:
        x = prev_x - d * np.dot(prev_x, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return x, y
