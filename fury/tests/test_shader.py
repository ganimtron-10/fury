import numpy as np

from fury.actor import SphGlyph, VectorField
from fury.actor.curved import Streamlines
from fury.actor.planar import LineProjection
from fury.geometry import line_buffer_separator
from fury.lib import load_wgsl
from fury.primitive import prim_sphere
from fury.shader import (
    LineProjectionComputeShader,
    SphGlyphComputeShader,
    StreamlinesShader,
    VectorFieldArrowShader,
    VectorFieldComputeShader,
    VectorFieldShader,
    VectorFieldThinShader,
)


def test_VectorFieldComputeShader_initialization():
    """Test VectorFieldComputeShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["workgroup_size"] == 64
    assert shader.type == "compute"


def test_VectorFieldComputeShader_get_render_info():
    """Test VectorFieldComputeShader.get_render_info()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup


def test_VectorFieldComputeShader_get_pipeline_info():
    """Test VectorFieldComputeShader.get_pipeline_info()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_VectorFieldComputeShader_get_code():
    """Test VectorFieldComputeShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_compute.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldThinShader_initialization():
    """Test VectorFieldThinShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldThinShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape


def test_VectorFieldThinShader_get_code():
    """Test VectorFieldThinShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldThinShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_thin_render.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldShader_initialization():
    """Test VectorFieldShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["line_type"] == "segment"


def test_VectorFieldShader_get_code():
    """Test VectorFieldShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_render.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldArrowShader_initialization():
    """Test VectorFieldArrowShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldArrowShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["line_type"] == "arrow"


def test_VectorFieldArrowShader_inheritance():
    """Test VectorFieldArrowShader inheritance."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldArrowShader(wobject)

    assert isinstance(shader, VectorFieldShader)
    assert hasattr(shader, "get_code")
    assert shader["line_type"] == "arrow"


def test_shaders_with_multiple_vectors_per_voxel():
    """Test shaders with multiple vectors per voxel."""
    field = np.random.rand(5, 5, 5, 10, 3)
    vectors_per_voxel = 10
    wobject = VectorField(field)

    # Test compute shader
    compute_shader = VectorFieldComputeShader(wobject)
    assert compute_shader["num_vectors"] == vectors_per_voxel

    # Test thin shader
    thin_shader = VectorFieldThinShader(wobject)
    assert thin_shader["num_vectors"] == vectors_per_voxel

    # Test regular shader
    reg_shader = VectorFieldShader(wobject)
    assert reg_shader["num_vectors"] == vectors_per_voxel

    # Test arrow shader
    arrow_shader = VectorFieldArrowShader(wobject)
    assert arrow_shader["num_vectors"] == vectors_per_voxel


def test_SphGlyphComputeShader_initialization():
    """Test SphGlyphComputeShader initialization."""
    coefficients = np.random.rand(5, 5, 5, 15)
    n_coeffs = coefficients.shape[-1]  # Exclude the last dimension (vector components)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    assert shader["n_coeffs"] == n_coeffs
    assert shader["data_shape"] == (5, 5, 5)
    assert shader["workgroup_size"] == (64, 1, 1)
    assert shader.type == "compute"


def test_SphGlyphComputeShader_get_render_info():
    """Test SphGlyphComputeShader.get_render_info()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup


def test_SphGlyphComputeShader_get_pipeline_info():
    """Test SphGlyphComputeShader.get_pipeline_info()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_SphGlyphComputeShader_get_code():
    """Test SphGlyphComputeShader.get_code()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)
    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("sph_glyph_compute.wgsl", package_name="fury.wgsl") == code


def test_streamline_shader_get_code():
    """Test StreamlineShader.get_code()."""
    # Create sample lines data for Streamline constructor
    lines = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]
    lines_positions, lines_colors = line_buffer_separator(lines, color=(1, 0, 0))
    print(lines_positions)
    wobject = Streamlines(lines_positions, colors=lines_colors)
    shader = StreamlinesShader(wobject)
    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("streamline_render.wgsl", package_name="fury.wgsl") == code


def test_LineProjectionComputeShader_initialization():
    """Test LineProjectionComputeShader initialization."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[3, 3, 3], [4, 4, 4]]),
    ]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    assert shader["num_lines"] == wobject.num_lines
    assert shader["num_lines"] == 2
    assert shader["workgroup_size"] == 64
    assert shader.type == "compute"


def test_LineProjectionComputeShader_get_render_info():
    """Test LineProjectionComputeShader.get_render_info()."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[2, 2, 2], [3, 3, 3]]),
        np.array([[4, 4, 4], [5, 5, 5]]),
    ]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup

    # Test specific calculation: ceil(num_lines / workgroup_size)
    expected_workgroups = int(np.ceil(wobject.num_lines / shader["workgroup_size"]))
    assert render_info["indices"][0] == expected_workgroups
    assert render_info["indices"] == (expected_workgroups, 1, 1)


def test_LineProjectionComputeShader_get_pipeline_info():
    """Test LineProjectionComputeShader.get_pipeline_info()."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_LineProjectionComputeShader_get_code():
    """Test LineProjectionComputeShader.get_code()."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert len(code) > 0
    assert load_wgsl("line_projection_compute.wgsl", package_name="fury.wgsl") == code


def test_LineProjectionComputeShader_with_different_line_configurations():
    """Test LineProjectionComputeShader with different line configurations."""
    # Test with single line
    single_line = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject_single = LineProjection(single_line)
    shader_single = LineProjectionComputeShader(wobject_single)

    assert shader_single["num_lines"] == 1
    render_info_single = shader_single.get_render_info(wobject_single, {})
    assert render_info_single["indices"][0] == 1  # Should have 1 workgroup for 1 line

    # Test with many lines
    many_lines = [np.array([[i, i, i], [i + 1, i + 1, i + 1]]) for i in range(100)]
    wobject_many = LineProjection(many_lines)
    shader_many = LineProjectionComputeShader(wobject_many)

    assert shader_many["num_lines"] == 100
    render_info_many = shader_many.get_render_info(wobject_many, {})
    expected_workgroups = int(np.ceil(100 / 64))  # 100 lines, 64 workgroup size
    assert render_info_many["indices"][0] == expected_workgroups

    # Test with empty lines (single point lines)
    empty_lines = [np.array([[0, 0, 0]])]
    wobject_empty = LineProjection(empty_lines)
    shader_empty = LineProjectionComputeShader(wobject_empty)

    assert shader_empty["num_lines"] == 1


def test_LineProjectionComputeShader_with_custom_parameters():
    """Test LineProjectionComputeShader with LineProjection custom parameters."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[3, 3, 3], [4, 4, 4]]),
    ]

    # Test with custom plane
    wobject = LineProjection(
        lines,
        plane=(1, 0, 0, -1),
        colors=[(1, 0, 0), (0, 1, 0)],
        thickness=2.0,
        outline_thickness=0.5,
        opacity=0.8,
    )
    shader = LineProjectionComputeShader(wobject)

    assert shader["num_lines"] == 2
    assert wobject.plane[0] == 1


def test_LineProjectionComputeShader_inheritance():
    """Test LineProjectionComputeShader inheritance and base class functionality."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    # Should inherit from BaseShader
    assert hasattr(shader, "type")
    assert hasattr(shader, "get_pipeline_info")
    assert hasattr(shader, "get_render_info")
    assert hasattr(shader, "get_bindings")
    assert hasattr(shader, "get_code")

    # Should have compute shader type
    assert shader.type == "compute"
