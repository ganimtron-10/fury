"""Tests for box shapes and box contacts."""

import numpy as np
import numpy.testing as npt
import pytest

from fury.physics import Box, Sphere, World
from fury.physics._math import quat_to_matrix
from fury.physics.narrowphase import box_box, box_sphere
from fury.physics.shapes import BOX, SPHERE, bounding_radius, inertia, volume


def _yaw(angle):
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])


def _roll(angle):
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])


def _floor(world, y=-1.0):
    return world.add_bodies([[0, y, 0]], Box((50, 1, 50)), kind="static", friction=0.6)


def _pair(q0=(0, 0, 0, 1), q1=(0, 0, 0, 1)):
    return quat_to_matrix(np.array([q0, q1], dtype=float))


def _ids():
    return np.array([0]), np.array([1])


def test_box_mass_properties():
    kind = np.array([BOX])
    size = np.array([[1.0, 2.0, 3.0]])
    npt.assert_allclose(volume(kind, size), [48.0])
    npt.assert_allclose(bounding_radius(kind, size), [np.sqrt(14)])
    npt.assert_allclose(inertia(kind, size, np.array([3.0])), [[13.0, 10.0, 5.0]])


def test_sphere_properties_are_unchanged_next_to_boxes():
    kind = np.array([SPHERE, BOX])
    size = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    npt.assert_allclose(volume(kind, size), [4 / 3 * np.pi * 8, 8.0])
    npt.assert_allclose(inertia(kind, size, np.ones(2))[0], [1.6, 1.6, 1.6])


@pytest.mark.parametrize("half", [0.0, -1.0, (1, 1), np.nan])
def test_invalid_half_extents(half):
    with pytest.raises(ValueError):
        World().add_bodies([[0, 0, 0]], Box(half))


def test_stacked_boxes_make_four_contacts():
    x = np.array([[0, 0, 0], [0, 0.95, 0]], float)
    size = np.full((2, 3), 0.5)
    c = box_box(x, _pair(), size, *_ids(), 0.01)
    npt.assert_equal(c.a.size, 4)
    npt.assert_allclose(c.separation, -0.05)
    npt.assert_allclose(np.abs(c.normals), [[0, 1, 0]] * 4)


@pytest.mark.parametrize("angle", [np.pi / 4, 0.7853, np.pi / 4 + 0.1, 0.0])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_turned_box_on_floor_makes_four_contacts(angle, order):
    x = np.array([[0, -1.0, 0], [0, -0.001, 0]], float)
    size = np.array([[25.0, 0.5, 25.0], [0.5, 0.5, 0.5]])
    rotation = _pair((0, 0, 0, 1), _yaw(angle))
    a, b = np.array([order[0]]), np.array([order[1]])
    c = box_box(x, rotation, size, a, b, 0.01)
    npt.assert_equal(c.a.size, 4)
    corners = np.round(c.points[:, [0, 2]], 6)
    npt.assert_equal(len(np.unique(corners, axis=0)), 4)


def test_separated_boxes_make_no_contacts():
    x = np.array([[0, 0, 0], [3.0, 0, 0]], float)
    size = np.full((2, 3), 0.5)
    assert box_box(x, _pair(), size, *_ids(), 0.01).a.size == 0


def test_contact_normal_points_from_a_to_b():
    x = np.array([[0, 0, 0], [0.9, 0, 0]], float)
    size = np.full((2, 3), 0.5)
    c = box_box(x, _pair(), size, *_ids(), 0.01)
    for a, b, n in zip(c.a, c.b, c.normals, strict=True):
        assert np.dot(n, x[b] - x[a]) > 0


def test_small_box_on_large_box_contacts_at_its_corners():
    x = np.array([[0, 0, 0], [0, 0.69, 0]], float)
    size = np.array([[2.0, 0.5, 2.0], [0.2, 0.2, 0.2]])
    c = box_box(x, _pair(), size, *_ids(), 0.01)
    npt.assert_equal(c.a.size, 4)
    npt.assert_allclose(np.abs(c.points[:, [0, 2]]), 0.2, atol=1e-6)


def test_crossed_planks_contact_at_the_overlap_corners():
    x = np.array([[0, 0, 0], [0, 0.39, 0]], float)
    size = np.array([[2.0, 0.2, 0.3], [0.3, 0.2, 2.0]])
    c = box_box(x, _pair(), size, *_ids(), 0.01)
    npt.assert_equal(c.a.size, 4)
    npt.assert_allclose(np.abs(c.points[:, [0, 2]]), 0.3, atol=1e-6)


def test_edge_on_edge_makes_a_single_contact():
    root2 = np.sqrt(2.0)
    q0 = np.array([0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    q1 = np.array([np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)])
    x = np.array([[0, 0, 0], [0, root2 - 0.02, 0]], float)
    size = np.full((2, 3), 0.5)
    c = box_box(x, _pair(q0, q1), size, *_ids(), 0.01)
    npt.assert_equal(c.a.size, 1)
    assert c.separation[0] < 0
    npt.assert_allclose(np.abs(c.normals[0]), [0, 1, 0], atol=1e-6)


def test_parallel_edges_produce_finite_contacts():
    x = np.array([[0, 0, 0], [0.99, 0.0, 0]], float)
    size = np.full((2, 3), 0.5)
    c = box_box(x, _pair(_yaw(0.0), _yaw(0.0)), size, *_ids(), 0.01)
    assert np.isfinite(c.points).all() and np.isfinite(c.normals).all()


def test_box_sphere_outside_face_edge_and_inside():
    size = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    r = _pair()
    box, ball = _ids()

    x = np.array([[0, 0, 0], [0, 1.4, 0]], float)
    c = box_sphere(x, r, size, box, ball, 0.01)
    npt.assert_allclose(c.separation, [-0.1])
    npt.assert_allclose(c.normals, [[0, 1, 0]])

    x = np.array([[0, 0, 0], [1.3, 1.3, 0]], float)
    c = box_sphere(x, r, size, box, ball, 0.01)
    npt.assert_allclose(c.normals, [[np.sqrt(0.5), np.sqrt(0.5), 0]])

    x = np.array([[0, 0, 0], [0.8, 0, 0]], float)
    c = box_sphere(x, r, size, box, ball, 0.01)
    npt.assert_allclose(c.normals, [[1, 0, 0]])
    npt.assert_allclose(c.separation, [-0.7])

    x = np.array([[0, 0, 0], [3.0, 0, 0]], float)
    assert box_sphere(x, r, size, box, ball, 0.01).a.size == 0


def test_box_rests_on_static_floor():
    world = World()
    _floor(world)
    box = world.add_bodies([[0, 0.6, 0]], Box(0.5), restitution=0.0)
    world.step(240)
    npt.assert_allclose(box.positions[0, 1], 0.5, atol=0.02)
    npt.assert_allclose(box.velocities[0], 0, atol=0.02)


def test_stack_of_boxes_stays_up():
    world = World()
    _floor(world)
    n = 6
    x = np.array([[0, 0.5 + i * 1.001, 0] for i in range(n)])
    boxes = world.add_bodies(x, Box(0.5), restitution=0.0, friction=0.6)
    world.step(480)
    npt.assert_allclose(boxes.positions[:, [0, 2]], 0, atol=0.03)
    assert np.diff(boxes.positions[:, 1]).min() > 0.97
    assert boxes.positions[-1, 1] > n - 0.5 - 0.3


def test_tilted_box_settles_flat():
    world = World()
    _floor(world)
    box = world.add_bodies([[0, 2.0, 0]], Box(0.5), restitution=0.0)
    box.rotations = _roll(0.6)
    world.step(600)
    npt.assert_allclose(box.positions[0, 1], 0.5, atol=0.03)
    rotation = quat_to_matrix(box.rotations.copy())[0]
    assert np.abs(rotation[1]).max() > 0.99


def test_ball_rolls_off_a_ramp():
    world = World()
    _floor(world)
    ramp = world.add_bodies([[0, 2, 0]], Box((4, 0.2, 2)), kind="static")
    ramp.rotations = _roll(-0.4)
    ball = world.add_bodies([[-2.0, 3.5, 0]], Sphere(0.3))
    world.step(180)
    assert ball.positions[0, 0] > -1.5


def test_friction_holds_a_box_on_a_shallow_ramp_but_not_when_slippery():
    def slide(friction):
        world = World(gravity=(-2.0, -9.81, 0))
        world.add_bodies(
            [[0, -1, 0]], Box((50, 1, 50)), kind="static", friction=friction
        )
        box = world.add_bodies(
            [[0, 0.5, 0]], Box(0.5), friction=friction, restitution=0
        )
        world.step(120)
        return box.positions[0, 0]

    npt.assert_allclose(slide(1.0), 0.0, atol=0.02)
    assert slide(0.01) < -1.0


def test_impact_off_centre_makes_box_spin():
    world = World(gravity=(0, 0, 0))
    box = world.add_bodies([[0, 0, 0]], Box(0.5))
    ball = world.add_bodies([[-3, 0.4, 0]], Sphere(0.2), mass=5.0, velocities=(6, 0, 0))
    world.step(60)
    assert np.linalg.norm(box.angular_velocities[0]) > 0.1
    assert ball.velocities[0, 0] < 6.0


def test_mixed_scene_stays_finite():
    rng = np.random.default_rng(2)
    world = World()
    _floor(world)
    x = rng.uniform((-3, 1, -3), (3, 7, 3), (120, 3))
    half = rng.uniform(0.2, 0.4, (60, 3))
    world.add_bodies(x[:60], Box(half), friction=0.4)
    world.add_bodies(x[60:], Sphere(rng.uniform(0.2, 0.4, 60)), restitution=0.4)
    world.step(180)
    assert np.isfinite(world._positions).all() and np.isfinite(world._velocities).all()
    npt.assert_allclose(np.linalg.norm(world._rotations, axis=1), 1.0, atol=1e-6)
    assert world._positions[:, 1].max() < 20
