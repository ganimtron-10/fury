import numpy as np
import pytest

from fury.utils import (
    create_sh_basis_matrix,
    generate_planar_uvs,
    get_lmax,
    get_n_coeffs,
    get_transformed_cube_bounds,
)


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
