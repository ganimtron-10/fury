"""Tests connecting physics bodies to FURY actors."""

import numpy as np
import numpy.testing as npt
import pytest

from fury import actor
from fury.actor import Group, actor_from_primitive
from fury.motion import Animation
from fury.physics import Box, Sphere, World
from fury.physics._math import quat_to_matrix
from fury.physics.bindings import direction_quaternions
from fury.primitive import prim_box


def _centers(n=3):
    return np.array([[i * 3.0, 1.0, -2.0] for i in range(n)])


def _world():
    return World(gravity=(0, 0, 0))


def test_box_actor_becomes_one_body_per_box():
    world = _world()
    boxes = actor.box(_centers(), scales=np.array([[2, 4, 6]] * 3))
    bodies = world.add(boxes)
    npt.assert_equal(len(bodies), 3)
    npt.assert_allclose(bodies.positions, _centers())
    npt.assert_allclose(world._size[:3], [[1, 2, 3]] * 3)


def test_impostor_sphere_uses_actor_radii():
    world = _world()
    balls = actor.sphere(_centers(), radii=np.array([0.5, 1.0, 1.5]))
    world.add(balls)
    npt.assert_allclose(world._size[:, 0], [0.5, 1.0, 1.5])


def test_mesh_sphere_uses_actor_radii():
    world = _world()
    balls = actor.sphere(_centers(), radii=0.7, impostor=False)
    world.add(balls)
    npt.assert_allclose(world._size[:, 0], 0.7)


def test_mesh_sphere_radius_matches_its_vertices():
    balls = actor.sphere(np.zeros((1, 3)), radii=0.7, impostor=False)
    radius = np.linalg.norm(balls.geometry.positions.data, axis=1).max()
    npt.assert_allclose(radius, 0.7, atol=1e-5)


def test_box_scale_is_full_edge_length():
    boxes = actor.box(np.zeros((1, 3)), scales=(2, 4, 6))
    span = np.ptp(boxes.geometry.positions.data, axis=0)
    npt.assert_allclose(span, [2, 4, 6])


def test_disk_and_square_inference():
    world = _world()
    world.add(actor.disk(_centers(), radii=np.array([1.0, 2.0, 3.0])))
    npt.assert_allclose(world._size[:3, 0], [1, 2, 3])
    world = _world()
    world.add(actor.square(_centers(2), scales=(2, 4, 1)))
    npt.assert_allclose(world._size[:2, :2], [[1, 2]] * 2)
    assert world._size[0, 2] > 0


def test_merged_mesh_vertices_follow_bodies():
    world = _world()
    boxes = actor.box(_centers(2), scales=2.0)
    bodies = world.add(boxes)
    per_box = boxes.geometry.positions.data.shape[0] // 2
    before = boxes.geometry.positions.data.copy()
    bodies.positions = bodies.positions + np.array([[1.0, 0, 0], [0, 5.0, 0]])
    world.sync()
    moved = boxes.geometry.positions.data
    npt.assert_allclose(moved[:per_box], before[:per_box] + [1, 0, 0], atol=1e-5)
    npt.assert_allclose(moved[per_box:], before[per_box:] + [0, 5, 0], atol=1e-5)


def test_merged_mesh_vertices_rotate_about_each_center():
    world = _world()
    boxes = actor.box(_centers(2), scales=2.0)
    bodies = world.add(boxes)
    half = np.sqrt(0.5)
    bodies.rotations = (0, half, 0, half)
    world.sync()
    per_box = boxes.geometry.positions.data.shape[0] // 2
    verts = boxes.geometry.positions.data[:per_box].astype(float) - _centers(2)[0]
    npt.assert_allclose(np.abs(verts).max(axis=0), [1, 1, 1], atol=1e-5)
    original = actor.box(np.zeros((1, 3)), scales=2.0).geometry.positions.data
    expected = original @ quat_to_matrix(np.array([[0, half, 0, half]]))[0].T
    npt.assert_allclose(verts, expected, atol=1e-5)


def test_per_primitive_colors_survive():
    world = _world()
    colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    boxes = actor.box(_centers(2), colors=colors)
    before = boxes.geometry.colors.data.copy()
    bodies = world.add(boxes)
    bodies.positions = bodies.positions + 1.0
    world.sync()
    npt.assert_array_equal(boxes.geometry.colors.data, before)


def test_billboard_centers_are_written():
    world = _world()
    balls = actor.sphere(_centers(2), radii=0.5)
    bodies = world.add(balls)
    bodies.positions = [[1, 2, 3], [4, 5, 6]]
    world.sync()
    data = balls.geometry.positions.data
    npt.assert_allclose(data[:6], [[1, 2, 3]] * 6)
    npt.assert_allclose(data[6:12], [[4, 5, 6]] * 6)
    npt.assert_allclose(balls.billboard_centers[1], [4, 5, 6])


def test_smooth_mesh_normals_rotate():
    world = _world()
    balls = actor.sphere(_centers(2), radii=0.5, impostor=False, smooth=True)
    bodies = world.add(balls)
    start = balls.geometry.normals.data.copy()
    half = np.sqrt(0.5)
    bodies.rotations = (0, 0, half, half)
    world.sync()
    rotated = start @ quat_to_matrix(np.array([[0, 0, half, half]]))[0].T
    npt.assert_allclose(balls.geometry.normals.data, rotated, atol=1e-5)


def test_flat_shaded_box_gets_no_normals_buffer():
    world = _world()
    boxes = actor.box(_centers(2))
    world.add(boxes)
    assert getattr(boxes.geometry, "normals", None) is None


def test_single_box_turns_about_its_own_center():
    world = _world()
    box = actor.box(np.array([[5.0, 0, 0]]), scales=2.0)
    bodies = world.add(box)
    npt.assert_allclose(bodies.positions[0], [5, 0, 0])
    npt.assert_allclose(box.local.position, [5, 0, 0])
    npt.assert_allclose(np.abs(box.geometry.positions.data).max(axis=0), 1, atol=1e-5)
    half = np.sqrt(0.5)
    bodies.rotations = (0, 0, half, half)
    bodies.positions = (5, 3, 0)
    world.sync()
    npt.assert_allclose(box.local.position, [5, 3, 0])
    npt.assert_allclose(box.local.rotation, [0, 0, half, half], atol=1e-6)


def test_directions_are_undone_for_the_rest_shape():
    world = _world()
    direction = np.array([[0.0, 1.0, 0.0]])
    boxes = actor.box(np.vstack([_centers(2)]), directions=(0, 1, 0), scales=(2, 4, 6))
    start = boxes.geometry.positions.data.copy()
    bodies = world.add(boxes)
    world.sync()
    npt.assert_allclose(boxes.geometry.positions.data, start, atol=1e-5)
    rotation = quat_to_matrix(bodies.rotations.copy())[0]
    npt.assert_allclose(rotation @ [1, 0, 0], direction[0], atol=1e-6)


def test_direction_quaternions_match_repeat_primitive():
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2.0], [1, 1, 1], [0, 0, 0.0]])
    matrices = quat_to_matrix(direction_quaternions(directions))
    for rotation, direction in zip(matrices[:4], directions[:4], strict=True):
        npt.assert_allclose(
            rotation @ [1, 0, 0], direction / np.linalg.norm(direction), atol=1e-9
        )
    npt.assert_allclose(matrices[4], np.eye(3))


def test_exact_negative_x_direction_is_stable():
    world = _world()
    boxes = actor.box(_centers(2), directions=(-1, 0, 0), scales=(2, 4, 6))
    start = boxes.geometry.positions.data.copy()
    world.add(boxes)
    world.sync()
    npt.assert_allclose(boxes.geometry.positions.data, start, atol=1e-5)


def test_actor_translation_is_baked_into_merged_mesh():
    world = _world()
    boxes = actor.box(_centers(2), scales=2.0)
    boxes.local.position = (10, 0, 0)
    start = boxes.geometry.positions.data.copy() + [10, 0, 0]
    bodies = world.add(boxes)
    npt.assert_allclose(bodies.positions, _centers(2) + [10, 0, 0])
    npt.assert_allclose(boxes.local.position, [0, 0, 0])
    world.sync()
    npt.assert_allclose(boxes.geometry.positions.data, start, atol=1e-5)


def test_actor_scale_is_applied_to_sizes():
    world = _world()
    box = actor.box(np.zeros((1, 3)), scales=2.0)
    box.local.scale = (2, 3, 4)
    world.add(box)
    npt.assert_allclose(world._size[0], [2, 3, 4])


def test_nonuniform_scale_rejected_for_spheres():
    ball = actor.sphere(np.zeros((1, 3)), radii=1.0, impostor=False)
    ball.local.scale = (1, 2, 3)
    with pytest.raises(ValueError, match="uniform"):
        _world().add(ball)


def test_billboard_translation_is_baked():
    world = _world()
    balls = actor.sphere(_centers(2), radii=0.5)
    balls.local.position = (0, 5, 0)
    bodies = world.add(balls)
    npt.assert_allclose(bodies.positions, _centers(2) + [0, 5, 0])
    npt.assert_allclose(balls.geometry.positions.data[0], _centers(2)[0] + [0, 5, 0])


def test_billboard_scale_rejected():
    balls = actor.sphere(_centers(2), radii=0.5)
    balls.local.scale = (2, 2, 2)
    with pytest.raises(ValueError, match="scale"):
        _world().add(balls)


def test_merged_mesh_under_transformed_parent_rejected():
    boxes = actor.box(_centers(2))
    parent = Group()
    parent.add(boxes)
    parent.local.position = (1, 0, 0)
    before = boxes.geometry.positions.data.copy()
    with pytest.raises(ValueError, match="parent"):
        _world().add(boxes)
    npt.assert_array_equal(boxes.geometry.positions.data, before)


def test_unknown_actor_needs_a_shape():
    cone = actor.cone(np.zeros((1, 3)))
    with pytest.raises(TypeError, match="shape="):
        _world().add(cone)


def test_unknown_actor_works_with_explicit_shape():
    world = _world()
    cone = actor.cone(np.array([[1.0, 2.0, 3.0]]))
    bodies = world.add(cone, shape=Box(0.5))
    npt.assert_allclose(bodies.positions[0], [1, 2, 3])
    bodies.positions = (0, 0, 0)
    world.sync()
    npt.assert_allclose(cone.local.position, [0, 0, 0])


def test_plain_group_with_explicit_shape_moves_by_transform():
    world = _world()
    group = Group()
    group.local.position = (1, 1, 1)
    bodies = world.add(group, shape=Sphere(1.0))
    npt.assert_allclose(bodies.positions[0], [1, 1, 1])
    bodies.positions = (2, 2, 2)
    world.sync()
    npt.assert_allclose(group.local.position, [2, 2, 2])


def test_plain_group_without_shape_raises():
    with pytest.raises(TypeError, match="shape="):
        _world().add(Group())


def test_body_options_pass_through():
    world = _world()
    bodies = world.add(actor.box(_centers(2)), mass=5.0, restitution=0.9, friction=0.1)
    npt.assert_allclose(bodies.masses, 5.0)
    npt.assert_allclose(world._restitution[:2], 0.9)


def test_static_actor():
    world = World()
    floor = world.add(actor.box(np.zeros((1, 3)), scales=(10, 1, 10)), kind="static")
    world.step(5)
    npt.assert_array_equal(floor.positions[0], [0, 0, 0])


def test_remove_drops_the_binding():
    world = _world()
    bodies = world.add(actor.box(_centers(2)))
    world.remove(bodies)
    assert world.bindings == ()
    world.sync()


def test_container_from_actor():
    world = World()
    tank = actor.box(np.array([[1.0, 2.0, 3.0]]), scales=(8, 6, 4), wireframe=True)
    box = world.add_container(actor=tank)
    npt.assert_allclose(box.half_extents, [4, 3, 2])
    npt.assert_allclose(box.positions[0], [1, 2, 3])
    box.velocities = [[0, 1.0, 0]]
    world.step(30)
    world.sync()
    npt.assert_allclose(tank.local.position, [1, 2.5, 3], atol=1e-6)


def test_container_needs_exactly_one_box():
    with pytest.raises(TypeError):
        World().add_container(actor=actor.box(_centers(2)))
    with pytest.raises(TypeError):
        World().add_container(actor=actor.sphere(np.zeros((1, 3)), impostor=False))


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"half_extents": 1.0, "follow_actor": True}],
)
def test_container_argument_errors(kwargs):
    with pytest.raises(ValueError):
        World().add_container(**kwargs)


def test_container_rejects_actor_and_size_together():
    tank = actor.box(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="either"):
        World().add_container(actor=tank, half_extents=1.0)


def test_follow_actor_requires_kinematic():
    with pytest.raises(ValueError, match="kinematic"):
        _world().add(actor.box(np.zeros((1, 3))), follow_actor=True)


def test_follow_actor_gets_velocity_from_pose_changes():
    world = _world()
    box = actor.box(np.zeros((1, 3)))
    body = world.add(box, kind="kinematic", follow_actor=True)
    box.local.position = (0.5, 0, 0)
    world.step(1)
    npt.assert_allclose(body.positions[0], [0.5, 0, 0], atol=1e-9)
    npt.assert_allclose(body.velocities[0], [30.0, 0, 0])
    world.step(1)
    npt.assert_allclose(body.velocities[0], 0.0)


def test_follow_actor_tracks_rotation():
    world = _world()
    box = actor.box(np.zeros((1, 3)))
    body = world.add(box, kind="kinematic", follow_actor=True)
    angle = 0.1
    box.local.rotation = (0, 0, np.sin(angle / 2), np.cos(angle / 2))
    world.step(1)
    npt.assert_allclose(body.angular_velocities[0], [0, 0, 6.0], atol=1e-6)


def test_follow_actor_treats_big_jumps_as_teleports():
    world = _world()
    box = actor.box(np.zeros((1, 3)))
    body = world.add(box, kind="kinematic", follow_actor=True)
    box.local.position = (500, 0, 0)
    world.step(1)
    npt.assert_allclose(body.positions[0], [500, 0, 0])
    npt.assert_allclose(body.velocities[0], 0)


def test_follow_requires_single_primitive():
    with pytest.raises(TypeError):
        _world().add(actor.box(_centers(2)), kind="kinematic", follow_actor=True)


def test_animation_drives_a_container_that_pushes_a_ball():
    world = World(gravity=(0, 0, 0))
    tank = actor.box(np.zeros((1, 3)), scales=4.0, wireframe=True)
    world.add_container(actor=tank, follow_actor=True, restitution=0.0)
    ball = world.add_bodies([[1.6, 0, 0]], Sphere(0.3), restitution=0.0, friction=0.0)

    animation = Animation()
    animation.add_actor(tank)
    animation.set_position(0.0, np.array([0.0, 0, 0]))
    animation.set_position(1.0, np.array([-2.0, 0, 0]))
    animation.set_rotation(0.0, np.array([0.0, 0, 0]))
    for frame in range(30):
        animation.update_animation(time=frame / 60)
        world.step(1)
    assert ball.velocities[0, 0] < -1.0


def _translated_parent(child):
    parent = Group()
    parent.add(child)
    parent.local.position = (10, 0, 0)
    return parent


def test_single_box_under_translated_parent_renders_at_its_body():
    world = _world()
    box = actor.box(np.array([[1.0, 0, 0]]))
    _translated_parent(box)
    body = world.add(box)
    npt.assert_allclose(body.positions[0], [11, 0, 0])
    npt.assert_allclose(box.world.position, [11, 0, 0], atol=1e-6)
    body.positions = [[12.0, 3.0, 0]]
    world.sync()
    npt.assert_allclose(box.world.position, [12, 3, 0], atol=1e-6)


def test_explicit_shape_under_translated_parent_renders_at_its_body():
    world = _world()
    group = Group()
    group.local.position = (-5, 0, 0)
    _translated_parent(group)
    body = world.add(group, shape=Sphere(0.5))
    npt.assert_allclose(body.positions[0], [5, 0, 0])
    world.sync()
    npt.assert_allclose(group.world.position, [5, 0, 0], atol=1e-6)


def test_follow_actor_under_translated_parent_uses_world_pose():
    world = _world()
    paddle = actor.box(np.zeros((1, 3)))
    _translated_parent(paddle)
    body = world.add(paddle, kind="kinematic", follow_actor=True)
    npt.assert_allclose(body.positions[0], [10, 0, 0])
    paddle.local.position = (0.5, 0, 0)
    world.step(1)
    npt.assert_allclose(body.positions[0], [10.5, 0, 0], atol=1e-9)
    npt.assert_allclose(paddle.world.position, [10.5, 0, 0], atol=1e-6)


def test_single_actor_can_be_added_to_a_second_world():
    box = actor.box(np.array([[1.0, 2.0, 3.0]]))
    _world().add(box)
    body = _world().add(box)
    npt.assert_allclose(body.positions[0], [1, 2, 3])
    npt.assert_allclose(box.world.position, [1, 2, 3], atol=1e-6)


def test_merged_actor_can_be_added_to_a_second_world_after_moving():
    first = _world()
    boxes = actor.box(_centers(2), directions=(0, 1, 0))
    bodies = first.add(boxes)
    bodies.positions = bodies.positions + [0, 10, 0]
    bodies.rotations = [0, 0, np.sin(0.3), np.cos(0.3)]
    first.sync()
    drawn = boxes.geometry.positions.data.copy()
    second = _world()
    again = second.add(boxes)
    npt.assert_allclose(again.positions, _centers(2) + [0, 10, 0])
    npt.assert_allclose(np.abs(again.rotations), np.abs(bodies.rotations), atol=1e-12)
    second.sync()
    npt.assert_allclose(boxes.geometry.positions.data, drawn, atol=1e-5)


@pytest.mark.parametrize("count", [1, 2])
def test_nonuniform_scale_follows_turned_primitives(count):
    boxes = actor.box(_centers(count), directions=(0, 1, 0))
    boxes.local.scale = (4, 1, 1)
    world = _world()
    world.add(boxes)
    extents = world._extents(quat_to_matrix(world._rotations))
    npt.assert_allclose(extents, [[2.0, 0.5, 0.5]] * count, atol=1e-9)


def test_nonuniform_scale_rejected_for_slanted_primitives():
    boxes = actor.box(np.zeros((1, 3)), directions=(1, 1, 0))
    boxes.local.scale = (4, 1, 1)
    with pytest.raises(ValueError, match="lined up"):
        _world().add(boxes)


@pytest.mark.parametrize("scale", [(0, 1, 1), (-1, 1, 1)])
def test_degenerate_or_mirrored_scale_rejected(scale):
    box = actor.box(np.zeros((1, 3)))
    box.local.scale = scale
    with pytest.raises(ValueError, match="scale"):
        _world().add(box)


def _actor_state(target):
    return (
        target.geometry.positions.data.copy(),
        np.asarray(target.local.matrix).copy(),
        np.asarray(target.primitive_info.centers).copy(),
    )


@pytest.mark.parametrize(
    "build, options",
    [
        (lambda: actor.box(np.array([[1.0, 2.0, 3.0]])), {"mass": -1.0}),
        (lambda: actor.box(_centers(2)), {"kind": "bogus"}),
    ],
)
def test_failed_add_leaves_the_actor_unchanged(build, options):
    target = build()
    target.local.position = (5, 0, 0)
    before = _actor_state(target)
    with pytest.raises(ValueError):
        _world().add(target, **options)
    for old, new in zip(before, _actor_state(target), strict=True):
        npt.assert_array_equal(new, old)
    assert getattr(target.geometry, "normals", None) is None


def test_failed_container_add_leaves_the_actor_unchanged():
    ball = actor.sphere(np.array([[1.0, 2.0, 3.0]]), radii=1.0, impostor=False)
    before = _actor_state(ball)
    with pytest.raises(TypeError):
        _world().add_container(actor=ball)
    for old, new in zip(before, _actor_state(ball), strict=True):
        npt.assert_array_equal(new, old)


@pytest.mark.parametrize("kind", ["box", "sphere", "disk", "square"])
def test_scaled_actor_keeps_its_size_in_a_second_world(kind):
    if kind == "box":
        target = actor.box(_centers(2))
        expected = [[1.0, 1.0, 1.0]] * 2
    elif kind == "sphere":
        target = actor.sphere(_centers(2), radii=1.0, impostor=False)
        expected = [[2.0, 2.0, 2.0]] * 2
    elif kind == "disk":
        target = actor.disk(_centers(2), radii=1.0)
        expected = [[2.0, 2.0, 2.0]] * 2
    else:
        target = actor.square(_centers(2))
        expected = [[1.0, 1.0, 0.05]] * 2
    target.local.scale = (2, 2, 2)
    first = _world()
    first.add(target)
    second = _world()
    second.add(target)
    npt.assert_allclose(second._size, first._size)
    npt.assert_allclose(second._size, expected)


def test_actor_under_unevenly_scaled_parent_rejected():
    box = actor.box(np.zeros((1, 3)))
    parent = Group()
    parent.add(box)
    parent.local.scale = (1, 2, 1)
    with pytest.raises(ValueError, match="non-uniform"):
        _world().add(box)


def test_unrepeated_primitive_is_placed_by_its_transform():
    vertices, faces = prim_box()
    single = actor_from_primitive(
        vertices, faces, np.array([[5.0, 0, 0]]), repeat_primitive=False
    )
    body = _world().add(single, shape=Box(0.5))
    npt.assert_allclose(body.positions[0], [5, 0, 0])
    npt.assert_allclose(single.world.position, [5, 0, 0], atol=1e-6)


@pytest.mark.parametrize("count", [1, 2])
def test_square_thickness_ignores_z_scale(count):
    target = actor.square(_centers(count))
    target.local.scale = (1, 1, 3)
    first = _world()
    first.add(target)
    second = _world()
    second.add(target)
    npt.assert_allclose(first._size, [[0.5, 0.5, 0.025]] * count)
    npt.assert_allclose(second._size, first._size)


def test_actor_under_sheared_parent_rejected():
    box = actor.box(np.zeros((1, 3)))
    parent = Group()
    parent.add(box)
    grandparent = Group()
    grandparent.add(parent)
    grandparent.local.scale = (2, 1, 1)
    parent.local.rotation = (0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8))
    with pytest.raises(ValueError, match="non-uniform"):
        _world().add(box)


def test_plain_billboards_need_a_shape():
    sprites = actor.billboard(_centers(2))
    with pytest.raises(TypeError, match="shape="):
        _world().add(sprites)
    bodies = _world().add(sprites, shape=Sphere(0.5))
    npt.assert_equal(len(bodies), 2)


def test_container_actor_rejects_center_and_squares():
    tank = actor.box(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="center"):
        _world().add_container(actor=tank, center=(5, 5, 5))
    with pytest.raises(TypeError, match="single box"):
        _world().add_container(actor=actor.square(np.zeros((1, 3)), scales=4))
