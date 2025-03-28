import numpy as np
from scipy.ndimage import map_coordinates

from fury.colormap import line_colors
from fury.decorators import warn_on_args_to_kwargs
from fury.lib import (
    VTK_DOUBLE,
    VTK_FLOAT,
    VTK_INT,
    VTK_UNSIGNED_CHAR,
    Actor,
    AlgorithmOutput,
    CellArray,
    Glyph3D,
    ImageData,
    Matrix3x3,
    Matrix4x4,
    Points,
    PolyData,
    PolyDataMapper,
    PolyDataNormals,
    Transform,
    TransformPolyDataFilter,
    numpy_support,
)


def remove_observer_from_actor(actor, id):
    """Remove the observer with the given id from the actor.

    Parameters
    ----------
    actor : vtkActor
    id : int
        id of the observer to remove

    """
    if not hasattr(actor, "GetMapper"):
        raise ValueError("Invalid actor")

    mapper = actor.GetMapper()
    if not hasattr(mapper, "RemoveObserver"):
        raise ValueError("Invalid mapper")
    mapper.RemoveObserver(id)


def set_input(vtk_object, inp):
    """Set Generic input function which takes into account VTK 5 or 6.

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -----
    This can be used in the following way::
        from fury.utils import set_input
        poly_mapper = set_input(PolyDataMapper(), poly_data)

    """
    if isinstance(inp, (PolyData, ImageData)):
        vtk_object.SetInputData(inp)
    elif isinstance(inp, AlgorithmOutput):
        vtk_object.SetInputConnection(inp)
    vtk_object.Update()
    return vtk_object


def numpy_to_vtk_points(points):
    """Convert Numpy points array to a vtk points array.

    Parameters
    ----------
    points : ndarray

    Returns
    -------
    vtk_points : vtkPoints()

    """
    vtk_points = Points()
    vtk_points.SetData(numpy_support.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points


def numpy_to_vtk_colors(colors):
    """Convert Numpy color array to a vtk color array.

    Parameters
    ----------
    colors: ndarray

    Returns
    -------
    vtk_colors : vtkDataArray

    Notes
    -----
    If colors are not already in UNSIGNED_CHAR you may need to multiply by 255.

    Examples
    --------
    >>> import numpy as np
    >>> from fury.utils import numpy_to_vtk_colors
    >>> rgb_array = np.random.rand(100, 3)
    >>> vtk_colors = numpy_to_vtk_colors(255 * rgb_array)

    """
    vtk_colors = numpy_support.numpy_to_vtk(
        np.asarray(colors), deep=True, array_type=VTK_UNSIGNED_CHAR
    )
    return vtk_colors


@warn_on_args_to_kwargs()
def numpy_to_vtk_cells(data, *, is_coords=True):
    """Convert numpy array to a vtk cell array.

    Parameters
    ----------
    data : ndarray
        points coordinate or connectivity array (e.g triangles).
    is_coords : ndarray
        Select the type of array. default: True.

    Returns
    -------
    vtk_cell : vtkCellArray
        connectivity + offset information

    """
    if isinstance(data, (list, np.ndarray)):
        offsets_dtype = np.int64
    else:
        offsets_dtype = np.dtype(data._offsets.dtype)
        if offsets_dtype.kind == "u":
            offsets_dtype = np.dtype(offsets_dtype.name[1:])
    data = np.array(data, dtype=object)
    nb_cells = len(data)

    # Get lines_array in vtk input format
    connectivity = data.flatten() if not is_coords else []
    offset = [
        0,
    ]
    current_position = 0

    cell_array = CellArray()

    for i in range(nb_cells):
        current_len = len(data[i])
        offset.append(offset[-1] + current_len)

        if is_coords:
            end_position = current_position + current_len
            connectivity += list(range(current_position, end_position))
            current_position = end_position

    connectivity = np.array(connectivity, offsets_dtype)
    offset = np.array(offset, dtype=offsets_dtype)

    vtk_array_type = numpy_support.get_vtk_array_type(offsets_dtype)
    cell_array.SetData(
        numpy_support.numpy_to_vtk(offset, deep=True, array_type=vtk_array_type),
        numpy_support.numpy_to_vtk(connectivity, deep=True, array_type=vtk_array_type),
    )

    cell_array.SetNumberOfCells(nb_cells)
    return cell_array


@warn_on_args_to_kwargs()
def numpy_to_vtk_image_data(
    array, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), deep=True
):
    """Convert numpy array to a vtk image data.

    Parameters
    ----------
    array : ndarray
        pixel coordinate and colors values.
    spacing : (float, float, float) (optional)
        sets the size of voxel (unit of space in each direction x,y,z)
    origin : (float, float, float) (optional)
        sets the origin at the given point
    deep : bool (optional)
        decides the type of copy(ie. deep or shallow)

    Returns
    -------
    vtk_image : vtkImageData

    """
    if array.ndim not in [2, 3]:
        raise IOError("only 2D (L, RGB, RGBA) or 3D image available")

    vtk_image = ImageData()
    depth = 1 if array.ndim == 2 else array.shape[2]

    vtk_image.SetDimensions(array.shape[1], array.shape[0], depth)
    vtk_image.SetExtent(0, array.shape[1] - 1, 0, array.shape[0] - 1, 0, 0)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    temp_arr = np.flipud(array)
    temp_arr = temp_arr.reshape(array.shape[1] * array.shape[0], depth)
    temp_arr = np.ascontiguousarray(temp_arr, dtype=array.dtype)
    vtk_array_type = numpy_support.get_vtk_array_type(array.dtype)
    uchar_array = numpy_support.numpy_to_vtk(
        temp_arr, deep=deep, array_type=vtk_array_type
    )
    vtk_image.GetPointData().SetScalars(uchar_array)
    return vtk_image


def map_coordinates_3d_4d(input_array, indices):
    """Evaluate input_array at the given indices using trilinear interpolation.

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

    """
    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i], indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


@warn_on_args_to_kwargs()
def lines_to_vtk_polydata(lines, *, colors=None):
    """Create a vtkPolyData with lines and colors.

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), list of arrays, tuple (3,), array (K,)
        If None or False, a standard orientation colormap is used for every
        line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, 3) is given, where K is the number of points of all
        lines then every point is colored with a different RGB color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    Returns
    -------
    poly_data : vtkPolyData
    color_is_scalar : bool, true if the color array is a single scalar
        Scalar array could be used with a colormap lut
        None if no color was used

    """
    # Get the 3d points_array
    if lines.__class__.__name__ == "ArraySequence":
        points_array = lines._data
    else:
        points_array = np.vstack(lines)

    if points_array.size == 0:
        raise ValueError("Empty lines/streamlines data.")

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_cell_array = numpy_to_vtk_cells(lines)

    # Create the poly_data
    poly_data = PolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_cell_array)

    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    nb_points = len(points_array)
    nb_lines = len(lines)
    lines_range = range(nb_lines)
    points_per_line = [len(lines[i]) for i in lines_range]
    points_per_line = np.array(points_per_line, np.intp)

    color_is_scalar = False
    if points_array.size:
        if colors is None or colors is False:
            # set automatic rgb colors
            cols_arr = line_colors(lines)
            colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
            vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
        else:
            cols_arr = np.asarray(colors)
            if cols_arr.dtype == object:  # colors is a list of colors
                vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
            else:
                if len(cols_arr) == nb_points:
                    if cols_arr.ndim == 1:  # values for every point
                        vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                        color_is_scalar = True
                    elif cols_arr.ndim == 2:  # map color to each point
                        vtk_colors = numpy_to_vtk_colors(255 * cols_arr)

                elif cols_arr.ndim == 1:
                    if len(cols_arr) == nb_lines:  # values for every streamline
                        cols_arrx = []
                        for i, value in enumerate(colors):
                            cols_arrx += lines[i].shape[0] * [value]
                        cols_arrx = np.array(cols_arrx)
                        vtk_colors = numpy_support.numpy_to_vtk(cols_arrx, deep=True)
                        color_is_scalar = True
                    else:  # the same colors for all points
                        vtk_colors = numpy_to_vtk_colors(
                            np.tile(255 * cols_arr, (nb_points, 1))
                        )

                elif cols_arr.ndim == 2:  # map color to each line
                    colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                    vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
                else:  # colormap
                    #  get colors for each vertex
                    cols_arr = map_coordinates_3d_4d(cols_arr, points_array)
                    vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                    color_is_scalar = True

        vtk_colors.SetName("colors")
        poly_data.GetPointData().SetScalars(vtk_colors)

    return poly_data, color_is_scalar


def get_polydata_lines(line_polydata):
    """Convert vtk polydata to a list of lines ndarrays.

    Parameters
    ----------
    line_polydata : vtkPolyData

    Returns
    -------
    lines : list
        List of N curves represented as 2D ndarrays

    """
    lines_vertices = numpy_support.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = numpy_support.vtk_to_numpy(line_polydata.GetLines().GetData())

    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]

        next_idx = current_idx + line_len + 1
        line_range = lines_idx[current_idx + 1 : next_idx]

        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines


def get_polydata_triangles(polydata):
    """Get triangles (ndarrays Nx3 int) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        triangles

    """
    vtk_polys = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    # test if its really triangles
    if not (vtk_polys[::4] == 3).all():
        raise AssertionError("Shape error: this is not triangles")
    return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T


def get_polydata_vertices(polydata):
    """Get vertices (ndarrays Nx3 int) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        points, represented as 2D ndarrays

    """
    return numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())


def get_polydata_tcoord(polydata):
    """Get texture coordinates (ndarrays Nx2 float) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 2)
        Tcoords, represented as 2D ndarrays. None if there are no texture
        in the vtk polydata.

    """
    vtk_tcoord = polydata.GetPointData().GetTCoords()
    if vtk_tcoord is None:
        return None

    return numpy_support.vtk_to_numpy(vtk_tcoord)


def get_polydata_normals(polydata):
    """Get vertices normal (ndarrays Nx3 int) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        Normals, represented as 2D ndarrays (Nx3). None if there are no normals
        in the vtk polydata.

    """
    vtk_normals = polydata.GetPointData().GetNormals()
    if vtk_normals is None:
        return None

    return numpy_support.vtk_to_numpy(vtk_normals)


def get_polydata_tangents(polydata):
    """Get vertices tangent (ndarrays Nx3 int) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        Tangents, represented as 2D ndarrays (Nx3). None if there are no
        tangents in the vtk polydata.

    """
    vtk_tangents = polydata.GetPointData().GetTangents()
    if vtk_tangents is None:
        return None

    return numpy_support.vtk_to_numpy(vtk_tangents)


def get_polydata_colors(polydata):
    """Get points color (ndarrays Nx3 int) from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        Colors. None if no normals in the vtk polydata.

    """
    vtk_colors = polydata.GetPointData().GetScalars()
    if vtk_colors is None:
        return None

    return numpy_support.vtk_to_numpy(vtk_colors)


@warn_on_args_to_kwargs()
def get_polydata_field(polydata, field_name, *, as_vtk=False):
    """Get a field from a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData
    field_name : str
    as_vtk : optional
        By default, ndarray is returned.

    Returns
    -------
    output : ndarray or vtkDataArray
        Field data. The return type depends on the value of the as_vtk
        parameter. None if the field is not found.

    """
    vtk_field_data = polydata.GetFieldData().GetArray(field_name)
    if vtk_field_data is None:
        return None
    if as_vtk:
        return vtk_field_data
    return numpy_support.vtk_to_numpy(vtk_field_data)


@warn_on_args_to_kwargs()
def add_polydata_numeric_field(polydata, field_name, field_data, *, array_type=VTK_INT):
    """Add a field to a vtk polydata.

    Parameters
    ----------
    polydata : vtkPolyData
    field_name : str
    field_data : bool, int, float, double, numeric array or ndarray
    array_type : vtkArrayType

    """
    vtk_field_data = numpy_support.numpy_to_vtk(
        field_data, deep=True, array_type=array_type
    )
    vtk_field_data.SetName(field_name)
    polydata.GetFieldData().AddArray(vtk_field_data)
    return polydata


def set_polydata_primitives_count(polydata, primitives_count):
    """Add primitives count to polydata.

    Parameters
    ----------
    polydata: vtkPolyData
    primitives_count : int

    """
    add_polydata_numeric_field(
        polydata, "prim_count", primitives_count, array_type=VTK_INT
    )


def get_polydata_primitives_count(polydata):
    """Get primitives count from actor's polydata.

    Parameters
    ----------
    polydata: vtkPolyData

    Returns
    -------
    primitives count : int

    """
    return get_polydata_field(polydata, "prim_count")[0]


def primitives_count_to_actor(actor, primitives_count):
    """Add primitives count to actor's polydata.

    Parameters
    ----------
    actor: :class: `UI` or `vtkProp3D` actor
    primitives_count : int

    """
    polydata = actor.GetMapper().GetInput()
    set_polydata_primitives_count(polydata, primitives_count)


def primitives_count_from_actor(actor):
    """Get primitives count from actor's polydata.

    Parameters
    ----------
    actor: :class: `UI` or `vtkProp3D` actor

    Returns
    -------
    primitives count : int

    """
    polydata = actor.GetMapper().GetInput()
    return get_polydata_primitives_count(polydata)


def set_polydata_triangles(polydata, triangles):
    """Set polydata triangles with a numpy array (ndarrays Nx3 int).

    Parameters
    ----------
    polydata : vtkPolyData
    triangles : array (N, 3)
        triangles, represented as 2D ndarrays (Nx3)

    """
    vtk_cells = CellArray()
    vtk_cells = numpy_to_vtk_cells(triangles, is_coords=False)
    polydata.SetPolys(vtk_cells)
    return polydata


def set_polydata_vertices(polydata, vertices):
    """Set polydata vertices with a numpy array (ndarrays Nx3 int).

    Parameters
    ----------
    polydata : vtkPolyData
    vertices : vertices, represented as 2D ndarrays (Nx3)

    """
    vtk_points = Points()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
    polydata.SetPoints(vtk_points)
    return polydata


def set_polydata_normals(polydata, normals):
    """Set polydata normals with a numpy array (ndarrays Nx3 int).

    Parameters
    ----------
    polydata : vtkPolyData
    normals : normals, represented as 2D ndarrays (Nx3) (one per vertex)

    """
    vtk_normals = numpy_support.numpy_to_vtk(normals, deep=True)
    # VTK does not require a specific name for the normals array, however, for
    # readability purposes, we set it to "Normals"
    vtk_normals.SetName("Normals")
    polydata.GetPointData().SetNormals(vtk_normals)
    return polydata


def set_polydata_tangents(polydata, tangents):
    """Set polydata tangents with a numpy array (ndarrays Nx3 int).

    Parameters
    ----------
    polydata : vtkPolyData
    tangents : tangents, represented as 2D ndarrays (Nx3) (one per vertex)

    """
    vtk_tangents = numpy_support.numpy_to_vtk(tangents, deep=True, array_type=VTK_FLOAT)
    # VTK does not require a specific name for the tangents array, however, for
    # readability purposes, we set it to "Tangents"
    vtk_tangents.SetName("Tangents")
    polydata.GetPointData().SetTangents(vtk_tangents)
    return polydata


@warn_on_args_to_kwargs()
def set_polydata_colors(polydata, colors, *, array_name="colors"):
    """Set polydata colors with a numpy array (ndarrays Nx3 int).

    Parameters
    ----------
    polydata : vtkPolyData
    colors : colors, represented as 2D ndarrays (Nx3)
        colors are uint8 [0,255] RGB for each points

    """
    vtk_colors = numpy_support.numpy_to_vtk(
        colors, deep=True, array_type=VTK_UNSIGNED_CHAR
    )
    nb_components = colors.shape[1]
    vtk_colors.SetNumberOfComponents(nb_components)
    vtk_colors.SetName(array_name)
    polydata.GetPointData().SetScalars(vtk_colors)
    return polydata


def set_polydata_tcoords(polydata, tcoords):
    """
    Set polydata texture coordinates with a numpy array (ndarrays Nx2 float).

    Parameters
    ----------
    polydata : vtkPolyData
    tcoords : texture coordinates, represented as 2D ndarrays (Nx2)
        (one per vertex range (0, 1))

    """
    vtk_tcoords = numpy_support.numpy_to_vtk(tcoords, deep=True, array_type=VTK_FLOAT)
    polydata.GetPointData().SetTCoords(vtk_tcoords)
    return polydata


def update_polydata_normals(polydata):
    """Generate and update polydata normals.

    Parameters
    ----------
    polydata : vtkPolyData

    """
    normals_gen = set_input(PolyDataNormals(), polydata)
    normals_gen.ComputePointNormalsOn()
    normals_gen.ComputeCellNormalsOn()
    normals_gen.SplittingOff()
    # normals_gen.FlipNormalsOn()
    # normals_gen.ConsistencyOn()
    # normals_gen.AutoOrientNormalsOn()
    normals_gen.Update()

    vtk_normals = normals_gen.GetOutput().GetPointData().GetNormals()
    polydata.GetPointData().SetNormals(vtk_normals)


def get_polymapper_from_polydata(polydata):
    """Get vtkPolyDataMapper from a vtkPolyData.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    poly_mapper : vtkPolyDataMapper

    """
    poly_mapper = set_input(PolyDataMapper(), polydata)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.InterpolateScalarsBeforeMappingOn()
    poly_mapper.Update()
    poly_mapper.StaticOn()
    return poly_mapper


def get_actor_from_polymapper(poly_mapper):
    """Get actor from a vtkPolyDataMapper.

    Parameters
    ----------
    poly_mapper : vtkPolyDataMapper

    Returns
    -------
    actor : actor

    """
    actor = Actor()
    actor.SetMapper(poly_mapper)
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()

    return actor


def get_actor_from_polydata(polydata):
    """Get actor from a vtkPolyData.

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    actor : actor

    """
    poly_mapper = get_polymapper_from_polydata(polydata)
    return get_actor_from_polymapper(poly_mapper)


@warn_on_args_to_kwargs()
def get_actor_from_primitive(
    vertices,
    triangles,
    *,
    colors=None,
    normals=None,
    backface_culling=True,
    prim_count=1,
):
    """Get actor from a vtkPolyData.

    Parameters
    ----------
    vertices : (Mx3) ndarray
        XYZ coordinates of the object
    triangles: (Nx3) ndarray
        Indices into vertices; forms triangular faces.
    colors: (Nx3) or (Nx4) ndarray
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
        N is equal to the number of vertices.
    normals: (Nx3) ndarray
        normals, represented as 2D ndarrays (Nx3) (one per vertex)
    backface_culling: bool
        culling of polygons based on orientation of normal with respect to
        camera. If backface culling is True, polygons facing away from camera
        are not drawn. Default: True
    prim_count: int, optional
        primitives count to be associated with the actor

    Returns
    -------
    actor : actor

    """
    # Create a Polydata
    pd = PolyData()
    set_polydata_vertices(pd, vertices)
    set_polydata_triangles(pd, triangles)
    set_polydata_primitives_count(pd, prim_count)
    if isinstance(colors, np.ndarray):
        if len(colors) != len(vertices):
            msg = "Vertices and Colors should have the same size."
            msg += " Please, update your color array or use the function "
            msg += "``fury.primitive.repeat_primitives`` to normalize your "
            msg += "color array before calling this function. e.g."
            raise ValueError(msg)

        set_polydata_colors(pd, colors, array_name="colors")
    if isinstance(normals, np.ndarray):
        set_polydata_normals(pd, normals)

    current_actor = get_actor_from_polydata(pd)
    current_actor.GetProperty().SetBackfaceCulling(backface_culling)
    return current_actor


@warn_on_args_to_kwargs()
def repeat_sources(
    centers,
    colors,
    *,
    active_scalars=1.0,
    directions=None,
    source=None,
    vertices=None,
    faces=None,
    orientation=None,
):
    """Transform a vtksource to glyph."""
    if source is None and faces is None:
        raise IOError("A source or faces should be defined")

    if np.array(colors).ndim == 1:
        colors = np.tile(colors, (len(centers), 1))

    pts = numpy_to_vtk_points(np.ascontiguousarray(centers))
    cols = numpy_to_vtk_colors(255 * np.ascontiguousarray(colors))
    cols.SetName("colors")
    if isinstance(active_scalars, (float, int)):
        active_scalars = np.tile(active_scalars, (len(centers), 1))
    if isinstance(active_scalars, np.ndarray):
        ascalars = numpy_support.numpy_to_vtk(
            np.asarray(active_scalars), deep=True, array_type=VTK_DOUBLE
        )
        ascalars.SetName("active_scalars")

    if directions is not None:
        directions_fa = numpy_support.numpy_to_vtk(
            np.asarray(directions), deep=True, array_type=VTK_DOUBLE
        )
        directions_fa.SetName("directions")

    polydata_centers = PolyData()
    polydata_geom = PolyData()

    if faces is not None:
        set_polydata_vertices(polydata_geom, vertices)
        set_polydata_triangles(polydata_geom, faces)

    polydata_centers.SetPoints(pts)
    polydata_centers.GetPointData().AddArray(cols)
    set_polydata_primitives_count(polydata_centers, len(centers))

    if directions is not None:
        polydata_centers.GetPointData().AddArray(directions_fa)
        polydata_centers.GetPointData().SetActiveVectors("directions")
    if isinstance(active_scalars, np.ndarray):
        polydata_centers.GetPointData().AddArray(ascalars)
        polydata_centers.GetPointData().SetActiveScalars("active_scalars")

    glyph = Glyph3D()
    if faces is None:
        if orientation is not None:
            transform = Transform()
            transform.SetMatrix(numpy_to_vtk_matrix(orientation))
            rtrans = TransformPolyDataFilter()
            rtrans.SetInputConnection(source.GetOutputPort())
            rtrans.SetTransform(transform)
            source = rtrans
        glyph.SetSourceConnection(source.GetOutputPort())
    else:
        glyph.SetSourceData(polydata_geom)
    glyph.SetInputData(polydata_centers)
    glyph.SetOrient(True)
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetVectorModeToUseVector()
    glyph.Update()

    mapper = PolyDataMapper()
    mapper.SetInputData(glyph.GetOutput())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")

    actor = Actor()
    actor.SetMapper(mapper)
    return actor


def apply_affine_to_actor(act, affine):
    """Apply affine matrix `affine` to the actor `act`.

    Parameters
    ----------
    act: Actor

    affine: (4, 4) array-like
        Homogeneous affine, for 3D object.

    Returns
    -------
    transformed_act: Actor

    """
    act.SetUserMatrix(numpy_to_vtk_matrix(affine))
    return act


def apply_affine(aff, pts):
    """Apply affine matrix `aff` to points `pts`.

    Returns result of application of `aff` to the *right* of `pts`.  The
    coordinate dimension of `pts` should be the last.
    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3 - maybe it will just be N by 3. The return value is the
    transformed points, in this case::
    res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
    transformed_pts = res.T
    This routine is more general than 3D, in that `aff` can have any shape
    (N,N), and `pts` can have any shape, as long as the last dimension is for
    the coordinates, and is therefore length N-1.

    Parameters
    ----------
    aff : (N, N) array-like

        Homogeneous affine, for 3D points, will be 4 by 4. Contrary to first
        appearance, the affine will be applied on the left of `pts`.
    pts : (..., N-1) array-like
        Points, where the last dimension contains the coordinates of each
        point.  For 3D, the last dimension will be length 3.

    Returns
    -------
    transformed_pts : (..., N-1) array
        transformed points

    Notes
    -----
    Copied from nibabel to remove dependency.

    Examples
    --------
    >>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    >>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    >>> # Just to show that in the simple 3D case, it is equivalent to:
    >>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    >>> # But `pts` can be a more complicated shape:
    >>> pts = pts.reshape((2,2,3))
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[[14, 14, 24],
            [16, 17, 28]],
    <BLANKLINE>
           [[20, 23, 36],
            [24, 29, 44]]]...)

    """
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1, :-1]
    trans = aff[:-1, -1]
    res = np.dot(pts, rzs.T) + trans[None, :]
    return res.reshape(shape)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def vtk_matrix_to_numpy(matrix):
    """Convert VTK matrix to numpy array."""
    if matrix is None:
        return None

    size = (4, 4)
    if isinstance(matrix, Matrix3x3):
        size = (3, 3)

    mat = np.zeros(size)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] = matrix.GetElement(i, j)

    return mat


def numpy_to_vtk_matrix(array):
    """Convert a numpy array to a VTK matrix."""
    if array is None:
        return None

    if array.shape == (4, 4):
        matrix = Matrix4x4()
    elif array.shape == (3, 3):
        matrix = Matrix3x3()
    else:
        raise ValueError("Invalid matrix shape: {0}".format(array.shape))

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])

    return matrix


def get_bounding_box_sizes(actor):
    """Get the bounding box sizes of an actor."""
    X1, X2, Y1, Y2, Z1, Z2 = actor.GetBounds()
    return (X2 - X1, Y2 - Y1, Z2 - Z1)


@warn_on_args_to_kwargs()
def get_grid_cells_position(shapes, *, aspect_ratio=16 / 9.0, dim=None):
    """Construct a XY-grid based on the cells content shape.

    This function generates the coordinates of every grid cell. The width and
    height of every cell correspond to the largest width and the largest height
    respectively. The grid dimensions will automatically be adjusted to respect
    the given aspect ratio unless they are explicitly specified.

    The grid follows a row-major order with the top left corner being at
    coordinates (0,0,0) and the bottom right corner being at coordinates
    (nb_cols*cell_width, -nb_rows*cell_height, 0). Note that the X increases
    while the Y decreases.

    Parameters
    ----------
    shapes : list of tuple of int
        The shape (width, height) of every cell content.
    aspect_ratio : float (optional)
        Aspect ratio of the grid (width/height). Default: 16:9.
    dim : tuple of int (optional)
        Dimension (nb_rows, nb_cols) of the grid, if provided.

    Returns
    -------
    ndarray
        3D coordinates of every grid cell.

    """
    cell_shape = np.r_[np.max(shapes, axis=0), 0]
    cell_aspect_ratio = cell_shape[0] / cell_shape[1]

    count = len(shapes)
    if dim is None:
        # Compute the number of rows and columns.
        n_cols = np.ceil(np.sqrt(count * aspect_ratio / cell_aspect_ratio))
        n_rows = np.ceil(count / n_cols)
    else:
        n_rows, n_cols = dim

    if n_cols * n_rows < count:
        msg = "Size is too small, it cannot contain at least {} elements."
        raise ValueError(msg.format(count))

    # Use indexing="xy" so the cells are in row-major (C-order). Also,
    # the Y coordinates are negative so the cells are order from top to bottom.
    X, Y, Z = np.meshgrid(np.arange(n_cols), -np.arange(n_rows), [0], indexing="xy")
    return cell_shape * np.array([X.flatten(), Y.flatten(), Z.flatten()]).T


def shallow_copy(vtk_object):
    """Create a shallow copy of a given `vtkObject` object."""
    copy = vtk_object.NewInstance()
    copy.ShallowCopy(vtk_object)
    return copy


@warn_on_args_to_kwargs()
def rotate(actor, *, rotation=(90, 1, 0, 0)):
    """Rotate actor around axis by angle.

    Parameters
    ----------
    actor : actor or other prop
    rotation : tuple
        Rotate with angle w around axis x, y, z. Needs to be provided
        in the form (w, x, y, z).

    """
    prop3D = actor
    center = np.array(prop3D.GetCenter())

    oldMatrix = prop3D.GetMatrix()
    orig = np.array(prop3D.GetOrigin())

    newTransform = Transform()
    newTransform.PostMultiply()
    if prop3D.GetUserMatrix() is not None:
        newTransform.SetMatrix(prop3D.GetUserMatrix())
    else:
        newTransform.SetMatrix(oldMatrix)

    newTransform.Translate(*(-center))
    newTransform.RotateWXYZ(*rotation)
    newTransform.Translate(*center)

    # now try to get the composite of translate, rotate, and scale
    newTransform.Translate(*(-orig))
    newTransform.PreMultiply()
    newTransform.Translate(*orig)

    if prop3D.GetUserMatrix() is not None:
        newTransform.GetMatrix(prop3D.GetUserMatrix())
    else:
        prop3D.SetPosition(newTransform.GetPosition())
        prop3D.SetScale(newTransform.GetScale())
        prop3D.SetOrientation(newTransform.GetOrientation())


def rgb_to_vtk(data):
    """RGB or RGBA images to VTK arrays.

    Parameters
    ----------
    data : ndarray
        Shape can be (X, Y, 3) or (X, Y, 4)

    Returns
    -------
    vtkImageData

    """
    grid = ImageData()
    grid.SetDimensions(data.shape[1], data.shape[0], 1)
    nd = data.shape[-1]
    vtkarr = numpy_support.numpy_to_vtk(
        np.flip(data.swapaxes(0, 1), axis=1).reshape((-1, nd), order="F")
    )
    vtkarr.SetName("Image")
    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars("Image")
    grid.GetPointData().Update()
    return grid


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(N, 3).

    Parameters
    ----------
    array : ndarray
        Shape (N, 3)

    Returns
    -------
    norm_array

    """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def normals_from_v_f(vertices, faces):
    """Calculate normals from vertices and faces.

    Parameters
    ----------
    verices : ndarray
    faces : ndarray

    Returns
    -------
    normals : ndarray
        Shape same as vertices

    """
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


def tangents_from_direction_of_anisotropy(normals, direction):
    """Calculate tangents from normals and a 3D vector representing the
       direction of anisotropy.

    Parameters
    ----------
    normals : normals, represented as 2D ndarrays (Nx3) (one per vertex)
    direction : tuple (3,) or array (3,)

    Returns
    -------
    output : array (N, 3)
        Tangents, represented as 2D ndarrays (Nx3).

    """
    tangents = np.cross(normals, direction)
    binormals = normalize_v3(np.cross(normals, tangents))
    return normalize_v3(np.cross(normals, binormals))


def triangle_order(vertices, faces):
    """Determine the winding order of a given set of vertices and a triangle.

    Parameters
    ----------
    vertices : ndarray
        array of vertices making up a shape
    faces : ndarray
        array of triangles

    Returns
    -------
    order : int
        If the order is counter clockwise (CCW), returns True.
        Otherwise, returns False.

    """
    v1 = vertices[faces[0]]
    v2 = vertices[faces[1]]
    v3 = vertices[faces[2]]

    # https://stackoverflow.com/questions/40454789/computing-face-normals-and-winding
    m_orient = np.ones((4, 4))
    m_orient[0, :3] = v1
    m_orient[1, :3] = v2
    m_orient[2, :3] = v3
    m_orient[3, :3] = 0

    val = np.linalg.det(m_orient)

    return bool(val > 0)


def change_vertices_order(triangle):
    """Change the vertices order of a given triangle.

    Parameters
    ----------
    triangle : ndarray, shape(1, 3)
        array of 3 vertices making up a triangle

    Returns
    -------
    new_triangle : ndarray, shape(1, 3)
        new array of vertices making up a triangle in the opposite winding
        order of the given parameter

    """
    return np.array([triangle[2], triangle[1], triangle[0]])


@warn_on_args_to_kwargs()
def fix_winding_order(vertices, triangles, *, clockwise=False):
    """Return corrected triangles.

    Given an ordering of the triangle's three vertices, a triangle can appear
    to have a clockwise winding or counter-clockwise winding.
    Clockwise means that the three vertices, in order, rotate clockwise around
    the triangle's center.

    Parameters
    ----------
    vertices : ndarray
        array of vertices corresponding to a shape
    triangles : ndarray
        array of triangles corresponding to a shape
    clockwise : bool
        triangle order type: clockwise (default) or counter-clockwise.

    Returns
    -------
    corrected_triangles : ndarray
        The corrected order of the vert parameter

    """
    corrected_triangles = triangles.copy()
    desired_order = clockwise
    for nb, face in enumerate(triangles):
        current_order = triangle_order(vertices, face)
        if desired_order != current_order:
            corrected_triangles[nb] = change_vertices_order(face)
    return corrected_triangles


@warn_on_args_to_kwargs()
def vertices_from_actor(actor, *, as_vtk=False):
    """Access to vertices from actor.

    Parameters
    ----------
    actor : actor
    as_vtk: bool, optional
        by default, ndarray is returned.

    Returns
    -------
    vertices : ndarray

    """
    vtk_array = actor.GetMapper().GetInput().GetPoints().GetData()
    if as_vtk:
        return vtk_array

    return numpy_support.vtk_to_numpy(vtk_array)


@warn_on_args_to_kwargs()
def colors_from_actor(actor, *, array_name="colors", as_vtk=False):
    """Access colors from actor which uses polydata.

    Parameters
    ----------
    actor : actor
    array_name: str
    as_vtk: bool, optional
        by default, numpy array is returned.

    Returns
    -------
    output : array (N, 3)
        Colors

    """
    return array_from_actor(actor, array_name=array_name, as_vtk=as_vtk)


def normals_from_actor(act):
    """Access normals from actor which uses polydata.

    Parameters
    ----------
    act : actor

    Returns
    -------
    output : array (N, 3)
        Normals

    """
    polydata = act.GetMapper().GetInput()
    return get_polydata_normals(polydata)


def tangents_from_actor(act):
    """Access tangents from actor which uses polydata.

    Parameters
    ----------
    act : actor

    Returns
    -------
    output : array (N, 3)
        Tangents

    """
    polydata = act.GetMapper().GetInput()
    return get_polydata_tangents(polydata)


@warn_on_args_to_kwargs()
def array_from_actor(actor, array_name, *, as_vtk=False):
    """Access array from actor which uses polydata.

    Parameters
    ----------
    actor : actor
    array_name: str
    as_vtk_type: bool, optional
        by default, ndarray is returned.

    Returns
    -------
    output : array (N, 3)

    """
    vtk_array = actor.GetMapper().GetInput().GetPointData().GetArray(array_name)
    if vtk_array is None:
        return None
    if as_vtk:
        return vtk_array

    return numpy_support.vtk_to_numpy(vtk_array)


def normals_to_actor(act, normals):
    """Set normals to actor which uses polydata.

    Parameters
    ----------
    act : actor
    normals : normals, represented as 2D ndarrays (Nx3) (one per vertex)

    Returns
    -------
    actor

    """
    polydata = act.GetMapper().GetInput()
    set_polydata_normals(polydata, normals)
    return act


def tangents_to_actor(act, tangents):
    """Set tangents to actor which uses polydata.

    Parameters
    ----------
    act : actor
    tangents : tangents, represented as 2D ndarrays (Nx3) (one per vertex)

    """
    polydata = act.GetMapper().GetInput()
    set_polydata_tangents(polydata, tangents)
    return act


def compute_bounds(actor):
    """Compute Bounds of actor.

    Parameters
    ----------
    actor : actor

    """
    actor.GetMapper().GetInput().ComputeBounds()


@warn_on_args_to_kwargs()
def update_actor(actor, *, all_arrays=True):
    """Update actor.

    Parameters
    ----------
    actor : actor
    all_arrays: bool, optional
        if False, only vertices are updated
        if True, all arrays associated to the actor are updated
        Default: True

    """
    pd = actor.GetMapper().GetInput()
    pd.GetPoints().GetData().Modified()
    if all_arrays:
        nb_array = pd.GetPointData().GetNumberOfArrays()
        for i in range(nb_array):
            pd.GetPointData().GetArray(i).Modified()


def get_bounds(actor):
    """Return Bounds of actor.

    Parameters
    ----------
    actor : actor

    Returns
    -------
    vertices : ndarray

    """
    return actor.GetMapper().GetInput().GetBounds()


def represent_actor_as_wireframe(actor):
    """Returns the actor wireframe.

    Parameters
    ----------
    actor : actor

    Returns
    -------
    actor : actor

    """
    return actor.GetProperty().SetRepresentationToWireframe()


def update_surface_actor_colors(actor, colors):
    """Update colors of a surface actor.

    Parameters
    ----------
    actor : surface actor
    colors : ndarray of shape (N, 3) having colors. The colors should be in the
        range [0, 1].

    """
    actor.GetMapper().GetInput().GetPointData().SetScalars(
        numpy_to_vtk_colors(255 * colors)
    )


@warn_on_args_to_kwargs()
def color_check(pts_len, *, colors=None):
    """Returns a VTK scalar array containing colors information for each one of
    the points according to the policy defined by the parameter colors.

    Parameters
    ----------
    pts_len : int
        length of points ndarray
    colors : None or tuple (3D or 4D) or array/ndarray (N, 3 or 4)
        If None a predefined color is used for each point.
        If a tuple of color is used. Then all points will have the same color.
        If an array (N, 3 or 4) is given, where N is equal to the number of
        points. Then every point is colored with a different RGB(A) color.

    Returns
    -------
    color_array : vtkDataArray
        vtk scalar array with name 'colors'.
    global_opacity : float
        returns 1 if the colors array doesn't contain opacity otherwise -1.
        If colors array has 4 dimensions, it checks values of the fourth
        dimension. If the value is the same, then assign it to global_opacity.

    """
    global_opacity = 1
    if colors is None:
        # Automatic RGB colors
        colors = np.asarray((1, 1, 1))
        color_array = numpy_to_vtk_colors(np.tile(255 * colors, (pts_len, 1)))
    elif type(colors) is tuple:
        global_opacity = 1 if len(colors) == 3 else colors[3]
        colors = np.asarray(colors)
        color_array = numpy_to_vtk_colors(np.tile(255 * colors, (pts_len, 1)))
    elif isinstance(colors, np.ndarray):
        colors = np.asarray(colors)
        if colors.shape[1] == 4:
            opacities = np.unique(colors[:, 3])
            global_opacity = opacities[0] if len(opacities) == 1 else -1
        color_array = numpy_to_vtk_colors(255 * colors)
    color_array.SetName("colors")

    return color_array, global_opacity


def is_ui(actor):
    """Method to check if the passed actor is `UI` or `vtkProp3D`

    Parameters
    ----------
    actor: :class: `UI` or `vtkProp3D`
        actor that is to be checked

    """
    return all(hasattr(actor, attr) for attr in ["add_to_scene", "_setup"])


@warn_on_args_to_kwargs()
def set_actor_origin(actor, *, center=None):
    """Change the origin of an actor to a custom position.

    Parameters
    ----------
    actor: Actor
        The actor object to change origin for.
    center: ndarray, optional, default: None
        The new center position. If `None`, the origin will be set to the mean
        of the actor's vertices.

    """
    vertices = vertices_from_actor(actor)
    if center is None:
        center = np.mean(vertices)
    vertices[:] -= center
    update_actor(actor)


def minmax_norm(data, axis=1):
    """Returns the min-max normalization of data along an axis.

    Parameters
    ----------
    data: ndarray
        2D array
    axis: int, optional
        axis for the function to be applied on

    Returns
    -------
    output : ndarray

    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim == 1:
        data = np.array([data])
    elif data.ndim > 2:
        raise ValueError("the dimension of the array dimension must be 2.")

    minimum = data.min(axis=axis)
    maximum = data.max(axis=axis)
    if np.array_equal(minimum, maximum):
        return data
    if axis == 0:
        return (data - minimum) / (maximum - minimum)
    if axis == 1:
        return (data - minimum[:, None]) / (maximum - minimum)[:, None]
