import numpy as np
import pytest

from fury.actor import data_slicer
from fury.lib import AffineTransform, Group, Mesh, RecursiveTransform, WorldObject
from fury.utils import (
    apply_affine_to_actor,
    apply_affine_to_group,
    create_sh_basis_matrix,
    generate_planar_uvs,
    get_lmax,
    get_n_coeffs,
    get_slices,
    get_transformed_cube_bounds,
    set_group_opacity,
    set_group_visibility,
    show_slices,
)


@pytest.fixture
def actor():
    return Mesh()


@pytest.fixture
def group_slicer():
    data = np.random.rand(10, 20, 30)
    return data_slicer(data)


# Test cases for set_group_visibility
def test_set_group_visibility_type_error():
    with pytest.raises(TypeError):
        set_group_visibility("not a group", True)


def test_set_group_visibility_single_bool():
    group = Group()
    group.visible = False
    set_group_visibility(group, True)
    assert group.visible is True


def test_set_group_visibility_list(group_slicer):
    visibility = [True, False, True]
    set_group_visibility(group_slicer, visibility)
    for actor, vis in zip(group_slicer.children, visibility, strict=False):
        assert actor.visible == vis


def test_set_group_visibility_tuple(group_slicer):
    visibility = (False, True, False)
    set_group_visibility(group_slicer, visibility)
    for actor, vis in zip(group_slicer.children, visibility, strict=False):
        assert actor.visible == vis


# Test cases for set_opacity
def test_set_opacity_type_error():
    with pytest.raises(TypeError):
        set_group_opacity("not a group", 0.5)


def test_set_opacity_valid(group_slicer):
    set_group_opacity(group_slicer, 0.7)
    for child in group_slicer.children:
        assert round(child.material.opacity, 2) == 0.7


# Test cases for get_slices
def test_get_slices_type_error():
    with pytest.raises(TypeError):
        get_slices("not a group")


def test_get_slices_value_error(actor):
    group = Group()
    group.add(actor, Mesh())
    with pytest.raises(ValueError):
        get_slices(group)


def test_get_slices_attribute_error(actor):
    group = Group()
    group.add(actor, Mesh(), Mesh())
    with pytest.raises(AttributeError):
        get_slices(group)


def test_get_slices_valid(group_slicer):
    for i, child in enumerate(group_slicer.children):
        child.material.plane = (0, 0, 0, i * 10)
    result = get_slices(group_slicer)
    expected = np.array([0, 10, 20])
    assert np.array_equal(result, expected)


# Test cases for show_slices
def test_show_slices_type_error():
    with pytest.raises(TypeError):
        show_slices("not a group", (1, 2, 3))


def test_show_slices_valid(group_slicer):
    for child in group_slicer.children:
        child.material.plane = (1, 2, 3, 0)
    position = (10, 20, 30)
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        expected_plane = (1, 2, 3, position[i])
        np.testing.assert_equal(child.material.plane, expected_plane)


def test_show_slices_with_list(group_slicer):
    position = [5, 6, 7]
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        assert child.material.plane[-1] == position[i]


def test_apply_affine_to_actor_valid_input():
    """Test apply_affine_to_actor with valid inputs."""
    # Create a mock actor
    actor = WorldObject()
    original_local = actor.local
    original_world = actor.world

    # Create test affine matrix
    affine = np.eye(4)
    affine[:3, :3] = 2 * np.eye(3)  # Scale by 2

    apply_affine_to_actor(actor, affine)

    # Check that the transforms were updated
    assert actor.local != original_local
    assert actor.world != original_world
    assert isinstance(actor.local, AffineTransform)
    assert isinstance(actor.world, RecursiveTransform)
    assert np.allclose(actor.local.matrix, affine)


def test_apply_affine_to_actor_invalid_actor():
    """Test apply_affine_to_actor with invalid actor type."""
    with pytest.raises(TypeError, match="actor must be an instance of WorldObject"):
        apply_affine_to_actor("not_an_actor", np.eye(4))


def test_apply_affine_to_actor_invalid_affine():
    """Test apply_affine_to_actor with invalid affine matrix."""
    actor = WorldObject()

    with pytest.raises(ValueError):
        # Wrong shape
        apply_affine_to_actor(actor, np.eye(3))

    with pytest.raises(ValueError):
        # Not a numpy array
        apply_affine_to_actor(
            actor, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )


def test_apply_affine_to_group_valid_input(group_slicer):
    """Test apply_affine_to_group with valid inputs."""

    # Store original transforms
    original_transforms = [(c.local, c.world) for c in group_slicer.children]

    # Create test affine matrix
    affine = np.eye(4)
    affine[0, 3] = 10  # Translate in x by 10

    apply_affine_to_group(group_slicer, affine)

    # Check that all children were updated
    for child, (orig_local, orig_world) in zip(
        group_slicer.children, original_transforms, strict=False
    ):
        assert child.local != orig_local
        assert child.world != orig_world
        assert isinstance(child.local, AffineTransform)
        assert isinstance(child.world, RecursiveTransform)
        assert np.allclose(child.local.matrix, affine)


def test_apply_affine_to_group_empty_group():
    """Test apply_affine_to_group with an empty group."""
    group = Group()

    # This should not raise any exceptions
    apply_affine_to_group(group, np.eye(4))


def test_apply_affine_to_group_invalid_group():
    """Test apply_affine_to_group with invalid group type."""
    with pytest.raises(TypeError, match="group must be an instance of Group"):
        apply_affine_to_group("not_a_group", np.eye(4))


def test_affine_transform_properties():
    """Test that the affine transform properties are correctly set."""
    actor = WorldObject()
    affine = np.eye(4)
    affine[1, 1] = 3.0  # Scale y by 3

    apply_affine_to_actor(actor, affine)

    # Check transform properties
    assert actor.local.state_basis == "matrix"
    assert actor.local.is_camera_space == int(True)
    assert np.allclose(actor.local.matrix, affine)

    # Check recursive transform wraps the affine transform
    assert isinstance(actor.world, RecursiveTransform)
    assert actor.world.own == actor.local


def test_generate_planar_uvs_basic_projections():
    """Test generate_planar_uvs with all three projection axes using simple geometry"""
    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # XY projection
    xy_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="xy"), xy_expected)

    # XZ projection
    xz_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="xz"), xz_expected)

    # YZ projection
    yz_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="yz"), yz_expected)


def test_generate_planar_uvs_edge_cases():
    """Test generate_planar_uvs with various edge cases"""
    # All vertices same position
    with pytest.raises(ValueError):
        same_verts = np.array([[1.0, 1.0, 1.0]] * 3)
        generate_planar_uvs(same_verts)

    # Flat plane (zero range in one dimension)
    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the XY plane."
    ):
        flat_xy = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
        generate_planar_uvs(flat_xy, axis="xy")

    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the XZ plane."
    ):
        flat_xz = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
        generate_planar_uvs(flat_xz, axis="xz")

    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the YZ plane."
    ):
        flat_yz = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 2.0]])
        generate_planar_uvs(flat_yz, axis="yz")


def test_generate_planar_uvs_input_validation():
    """Test generate_planar_uvs input validation and error cases"""
    # Invalid axis
    with pytest.raises(ValueError, match="axis must be one of 'xy', 'xz', or 'yz'."):
        generate_planar_uvs(np.array([[1, 2, 3]]), axis="invalid")

    # Wrong array dimensions
    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([1, 2, 3]))  # 1D array

    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([[1, 2]]))  # Wrong shape

    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([[1, 2, 3]]))  # Single vertex

    # Empty array
    with pytest.raises(ValueError):
        generate_planar_uvs(np.empty((0, 3)))


def test_generate_planar_uvs_numerical_stability():
    """Test generate_planar_uvs with numerical edge cases"""
    # Very small range
    small_range = np.array([[1.0, 2.0, 3.0], [1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10]])
    result = generate_planar_uvs(small_range, axis="xy")
    assert not np.any(np.isnan(result))
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Very large coordinates
    large_coords = np.array([[1e20, 2e20, 3e20], [2e20, 3e20, 4e20]])
    result = generate_planar_uvs(large_coords, axis="yz")
    assert not np.any(np.isnan(result))
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Mixed positive and negative coordinates
    mixed_coords = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    result = generate_planar_uvs(mixed_coords, axis="xz")
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_get_lmax_standard_basis():
    """Test the standard basis type (default)."""
    assert get_lmax(3) == 1
    assert get_lmax(16) == 3
    assert get_lmax(24, basis_type="standard") == 4


def test_get_lmax_descoteaux07_basis():
    """Test the descoteaux07 basis type."""
    assert get_lmax(6, basis_type="descoteaux07") == 2
    assert get_lmax(28, basis_type="descoteaux07") == 6


def test_get_lmax_invalid_inputs():
    """Test invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        get_lmax(0)  # n_coeffs < 1
    with pytest.raises(ValueError):
        get_lmax(1.5)  # non-integer n_coeffs
    with pytest.raises(ValueError):
        get_lmax(10, basis_type="invalid")  # invalid basis_type


def test_get_n_coeffs_standard_basis():
    """Test the standard basis type (default)."""
    assert get_n_coeffs(1) == 4
    assert get_n_coeffs(3) == 16
    assert get_n_coeffs(4, basis_type="standard") == 25


def test_get_n_coeffs_descoteaux07_basis():
    """Test the descoteaux07 basis type."""
    assert get_n_coeffs(2, basis_type="descoteaux07") == 6
    assert get_n_coeffs(6, basis_type="descoteaux07") == 28


def test_get_n_coeffs_invalid_inputs():
    """Test invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        get_n_coeffs(-1)  # l_max < 0
    with pytest.raises(ValueError):
        get_n_coeffs(1.5)  # non-integer l_max
    with pytest.raises(ValueError):
        get_n_coeffs(2, basis_type="invalid")  # invalid basis_type


def test_get_n_coeffs_edge_cases():
    """Test edge cases for get_n_coeffs."""
    # l_max = 0 should give 1 coefficient
    assert get_n_coeffs(0) == 1
    assert get_n_coeffs(0, basis_type="descoteaux07") == 1

    # Test some higher values
    assert get_n_coeffs(5) == 36  # (5+1)^2 = 36
    assert get_n_coeffs(10) == 121  # (10+1)^2 = 121


def test_lmax_n_coeffs_inverse_relationship_standard():
    """Test that get_lmax and get_n_coeffs are inverse for standard basis."""
    test_lmax_values = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15]

    for l_max in test_lmax_values:
        # Test: get_lmax(get_n_coeffs(l_max)) == l_max
        n_coeffs = get_n_coeffs(l_max, basis_type="standard")
        recovered_lmax = get_lmax(n_coeffs, basis_type="standard")
        assert recovered_lmax == l_max, (
            f"Failed for l_max={l_max}: got {recovered_lmax}"
        )


def test_lmax_n_coeffs_inverse_relationship_descoteaux07():
    """Test that get_lmax and get_n_coeffs are inverse for descoteaux07 basis."""
    test_lmax_values = [0, 1, 2, 3, 4, 5, 6, 8, 10]

    for l_max in test_lmax_values:
        # Test: get_lmax(get_n_coeffs(l_max)) == l_max
        n_coeffs = get_n_coeffs(l_max, basis_type="descoteaux07")
        recovered_lmax = get_lmax(n_coeffs, basis_type="descoteaux07")
        assert recovered_lmax == l_max, (
            f"Failed for l_max={l_max}: got {recovered_lmax}"
        )


def test_n_coeffs_lmax_inverse_relationship_standard():
    """Test that get_n_coeffs and get_lmax are inverse for standard basis."""
    # Test with known valid n_coeffs values for standard basis: (l+1)^2
    test_n_coeffs_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]

    for n_coeffs in test_n_coeffs_values:
        # Test: get_n_coeffs(get_lmax(n_coeffs)) == n_coeffs
        l_max = get_lmax(n_coeffs, basis_type="standard")
        recovered_n_coeffs = get_n_coeffs(l_max, basis_type="standard")
        assert recovered_n_coeffs == n_coeffs, (
            f"Failed for n_coeffs={n_coeffs}: got {recovered_n_coeffs}"
        )


def test_n_coeffs_lmax_inverse_relationship_descoteaux07():
    """Test that get_n_coeffs and get_lmax are inverse for descoteaux07 basis."""
    # Test with known valid n_coeffs values for descoteaux07 basis
    test_n_coeffs_values = [1, 6, 15, 28, 45, 66, 91, 120, 153]

    for n_coeffs in test_n_coeffs_values:
        # Test: get_n_coeffs(get_lmax(n_coeffs)) == n_coeffs
        l_max = get_lmax(n_coeffs, basis_type="descoteaux07")
        recovered_n_coeffs = get_n_coeffs(l_max, basis_type="descoteaux07")
        assert recovered_n_coeffs == n_coeffs, (
            f"Failed for n_coeffs={n_coeffs}: got {recovered_n_coeffs}"
        )


def test_both_functions_consistency():
    """Test consistency between both functions with various input combinations."""
    # Test that both functions handle basis_type parameter consistently
    l_max = 4

    # Standard basis
    n_coeffs_std = get_n_coeffs(l_max, basis_type="standard")
    recovered_lmax_std = get_lmax(n_coeffs_std, basis_type="standard")
    assert recovered_lmax_std == l_max

    # Descoteaux07 basis
    n_coeffs_desc = get_n_coeffs(l_max, basis_type="descoteaux07")
    recovered_lmax_desc = get_lmax(n_coeffs_desc, basis_type="descoteaux07")
    assert recovered_lmax_desc == l_max

    # Verify that the two basis types give different results
    assert n_coeffs_std != n_coeffs_desc


def test_create_sh_basis_matrix_input_validation():
    """Test invalid inputs raise ValueError."""
    # Invalid vertices (not a 2D array with shape (N, 3))
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([1, 2, 3]), 1)  # 1D array
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[1, 2]]), 1)  # Shape (N, 2)

    # Invalid l_max (non-integer or negative)
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[0, 0, 1]]), -1)
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[0, 0, 1]]), 1.5)


def test_create_sh_basis_matrix_l_max_zero():
    """Test l_max=0 (only the constant SH term)."""
    vertices = np.array([[0, 0, 1], [1, 0, 0]])
    B = create_sh_basis_matrix(vertices, l_max=0)
    assert B.shape == (2, 1)  # (N_vertices, 1 coefficient)
    assert np.allclose(B, 1 / (2 * np.sqrt(np.pi)))  # Y_0^0 = 1/(2√π)


def test_create_sh_basis_matrix_basic_output_shape():
    """Verify output shape matches (N, (l_max+1)^2)."""
    vertices = np.random.randn(10, 3)  # 10 random vertices
    for l_max in [1, 2, 3]:
        B = create_sh_basis_matrix(vertices, l_max)
        assert B.shape == (10, (l_max + 1) ** 2)


def test_create_sh_basis_matrix_known_values():
    """Test against known SH values at specific points."""
    # North pole (theta=0, phi=undefined)
    vertices = np.array([[0, 0, 1]])
    B = create_sh_basis_matrix(vertices, l_max=1)

    # Expected values for l_max=1:
    # Y_0^0 = 1/(2√π)
    # Y_1^{-1} = 0 (due to sin(phi) term, but phi is undefined at pole)
    # Y_1^0 = √(3/4π)*cos(0) = √(3/4π)
    # Y_1^1 = 0 (due to cos(phi) term)
    expected = np.array([[1 / (2 * np.sqrt(np.pi)), 0, np.sqrt(3 / (4 * np.pi)), 0]])
    assert np.allclose(B, expected, atol=1e-6)


def test_get_transformed_cube_bounds_valid_input():
    """Test function with valid inputs returns correct bounds"""
    affine_matrix = np.eye(4)
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_invalid_vertex_dimensions():
    """Test function raises ValueError for non-3D vertices"""
    affine_matrix = np.eye(4)

    with pytest.raises(ValueError, match="must be 3D coordinates"):
        get_transformed_cube_bounds(
            affine_matrix, np.array([1, 2]), np.array([4, 5, 6])
        )

    with pytest.raises(ValueError, match="must be 3D coordinates"):
        get_transformed_cube_bounds(
            affine_matrix, np.array([1, 2, 3]), np.array([4, 5])
        )


def test_get_transformed_cube_bounds_invalid_matrix_shape():
    """Test function raises ValueError for non-4x4 matrix"""
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    with pytest.raises(ValueError, match="must be a 4x4 numpy array"):
        get_transformed_cube_bounds(np.eye(3), vertex1, vertex2)

    with pytest.raises(ValueError, match="must be a 4x4 numpy array"):
        get_transformed_cube_bounds("not_a_matrix", vertex1, vertex2)


def test_get_transformed_cube_bounds_translation():
    """Test function correctly handles translation"""
    affine_matrix = np.array(
        [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
    )
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([11, 22, 33]), np.array([14, 25, 36])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_scaling():
    """Test function correctly handles scaling"""
    affine_matrix = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]])
    vertex1 = np.array([1, 1, 1])
    vertex2 = np.array([2, 2, 2])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([2, 3, 4]), np.array([4, 6, 8])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_degenerate_case():
    """Test function handles single-point cube correctly"""
    affine_matrix = np.eye(4)
    vertex1 = np.array([5, 5, 5])
    vertex2 = np.array([5, 5, 5])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([5, 5, 5]), np.array([5, 5, 5])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])
