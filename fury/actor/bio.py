"""Medical actors for FURY."""

import numpy as np

from fury.actor import (
    apply_affine_to_actor,
    apply_affine_to_group,
    data_slicer,
    show_slices,
    vector_field_slicer,
)
from fury.utils import get_transformed_cube_bounds


def volume_slicer(
    data,
    *,
    affine=None,
    value_range=None,
    opacity=1.0,
    interpolation="linear",
    visibility=(True, True, True),
    initial_slices=None,
):
    """Visualize a 3D volume data as a slice.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z) or (X, Y, Z, 3)
        The 3D volume data to be sliced.
    affine : ndarray, shape (4, 4), optional
        The affine transformation matrix to apply to the data.
    value_range : tuple, optional
        The minimum and maximum values for the color mapping.
        If None, the range is determined from the data.
    opacity : float, optional
        The opacity of the slice. Takes values from 0 (fully transparent) to 1 (opaque).
    interpolation : str, optional
        The interpolation method for the slice. Options are 'linear' and 'nearest'.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.
    initial_slices : tuple, optional
        A tuple of three initial slice positions in the x, y, and z dimensions,
        respectively. If None, the slices are initialized to the middle of the volume.

    Returns
    -------
    Group
        An actor containing the generated slice with the specified properties.
    """

    obj = data_slicer(
        data,
        value_range=value_range,
        opacity=opacity,
        interpolation=interpolation,
        visibility=visibility,
        initial_slices=initial_slices,
    )

    if affine is not None:
        apply_affine_to_group(obj, affine)
        bounds = obj.get_bounding_box()
        show_slices(obj, np.asarray(bounds).mean(axis=0))

    return obj


def peaks_slicer(
    peak_dirs,
    *,
    affine=None,
    peak_values=1.0,
    actor_type="thin_line",
    cross_section=None,
    colors=None,
    opacity=1.0,
    thickness=1.0,
    visibility=(True, True, True),
):
    """Visualize peaks as lines in 3D space.

    Parameters
    ----------
    peak_dirs : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The directions of the peaks.
    affine : ndarray, shape (4, 4), optional
        The affine transformation matrix to apply to the peak directions.
    peak_values : float or ndarray, optional
        The values associated with each peak direction. If a single float is provided,
        it is applied uniformly to all peaks.
    actor_type : str, optional
        The type of actor to create for the peaks. Options are 'thin_line' and
        'line'.
    cross_section : float, optional
        The cross-section size for the peaks. If None, it defaults to a small value.
    colors : ndarray, shape (N, 3) or None, optional
        The colors for each peak direction. If None, a default color is used.
    opacity : float, optional
        The opacity of the peaks. Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the peaks if `actor_type` is 'thick_line'.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the peaks in the x,
        y, and z dimensions, respectively.

    Returns
    -------
    VectorField
        An actor containing the generated peaks with the specified properties.
    """
    obj = vector_field_slicer(
        peak_dirs,
        scales=peak_values,
        actor_type=actor_type,
        cross_section=cross_section,
        colors=colors,
        opacity=opacity,
        thickness=thickness,
        visibility=visibility,
    )

    if affine is not None:
        apply_affine_to_actor(obj, affine)
        bounds = get_transformed_cube_bounds(
            affine, np.zeros((3,)), np.asarray(obj.field_shape)
        )
        obj.get_bounding_box = lambda: bounds
        obj.cross_section = np.asarray(bounds).mean(axis=0)

    return obj
