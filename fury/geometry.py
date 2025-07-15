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

    line_vertices = np.asarray(line_vertices, dtype=np.float32)
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
    elif len(color) == len(line_vertices) and color.ndim == line_vertices.ndim:
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


def generate_ring(
    inner_radius, outer_radius, radial_segments, circumferential_segments
):
    """
    Generate geometry data for ring.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the ring (radius of the hole).
    outer_radius : float
        The outer radius of the ring.
    radial_segments : int
        Number of segments along the radial direction.
    circumferential_segments : int
        Number of segments around the circumference.

    Returns
    -------
    positions : ndarray
        Array of vertex positions (x, y, 0).
    normals : ndarray
        Array of vertex normals (0, 0, 1).
    texcoords : ndarray
        Array of texture coordinates (u, v).
    indices : ndarray
        Array of triangle indices.
    """
    inner_radius = max(0, float(inner_radius))
    outer_radius = max(inner_radius, float(outer_radius))

    if radial_segments < 1:
        raise ValueError("radial_segments must be greater than or equal to 1")
    if circumferential_segments < 3:
        raise ValueError("circumferential_segments must be greater than or equal to 3")
    if not (0 <= inner_radius < outer_radius):
        raise ValueError(
            "inner_radius must be greater than equal to 0 and less than outer_radius"
        )

    # Number of vertices along radial and circumferential directions
    nr = radial_segments + 1
    nc = circumferential_segments

    # Generate radial and angular coordinates
    radii = np.linspace(inner_radius, outer_radius, nr, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, nc, endpoint=False, dtype=np.float32)

    # Create a grid of radii and angles
    rr, aa = np.meshgrid(radii, angles)
    rr, aa = rr.flatten(), aa.flatten()

    # Convert to Cartesian coordinates (x, y, z=0)
    x = rr * np.cos(aa)
    y = rr * np.sin(aa)
    positions = np.column_stack([x, y, np.zeros_like(x)])

    # Texture coordinates: map radial distance to [0, 1], angle to [0, 1]
    texcoords = np.zeros((nc * nr, 2), dtype=np.float32)
    texcoords[:, 0] = (rr - inner_radius) / (
        outer_radius - inner_radius
    )  # u: radial distance
    texcoords[:, 1] = aa / (2 * np.pi)  # v: angular coordinate
    # Flip v to match typical texture coordinate orientation
    texcoords[:, 1] = 1 - texcoords[:, 1]

    # Normals: all point along +z (0, 0, 1)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (nc * nr, 1))

    # Generate indices for triangles
    indices = []
    for i in range(nc):
        for j in range(radial_segments):
            # Vertex indices for a quad (counter-clockwise)
            v0 = i * nr + j
            v1 = i * nr + (j + 1)
            v2 = ((i + 1) % nc) * nr + j
            v3 = ((i + 1) % nc) * nr + (j + 1)

            # Two triangles per quad
            indices.append([v0, v1, v3])  # First triangle
            indices.append([v0, v3, v2])  # Second triangle

    indices = np.array(indices, dtype=np.uint32)

    return positions, normals, texcoords, indices


def ring_geometry(
    inner_radius=0, outer_radius=10, radial_segments=1, circumferential_segments=16
):
    """
    Generate a Ring geometry.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the ring (radius of the hole).
    outer_radius : float
        The outer radius of the ring.
    radial_segments : int
        Number of segments along the radial direction.
    circumferential_segments : int
        Number of segments around the circumference.

    Returns
    -------
    Geometry
        A geometry object representing the requested ring.
    """
    positions, normals, texcoords, indices = generate_ring(
        inner_radius, outer_radius, radial_segments, circumferential_segments
    )

    return Geometry(
        indices=indices,
        positions=positions,
        normals=normals,
        texcoords=texcoords,
    )
