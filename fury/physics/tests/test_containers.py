"""Tests for containers and scripted kinematic motion."""

import logging

import numpy as np
import numpy.testing as npt
import pytest

from fury.physics import Box, Container, KinematicBodySet, Sphere, World
from fury.physics._math import quat_to_matrix
from fury.physics.narrowphase import container_contacts


def _fill(world, count=40, half=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-half + 0.5, half - 0.5, (count, 3))
    x[:, 1] = rng.uniform(-half + 0.5, 0.0, count)
    a = world.add_bodies(x[: count // 2], Sphere(0.25), restitution=0.3)
    b = world.add_bodies(x[count // 2 :], Box(0.22), restitution=0.3)
    return a, b


def _inside(world, container, bodies, tolerance=0.3):
    rotation = quat_to_matrix(container.rotations.copy())[0]
    local = (bodies.positions - container.positions[0]) @ rotation
    return (np.abs(local) <= container.half_extents + tolerance).all()


def test_add_container_returns_kinematic_handle():
    world = World()
    box = world.add_container(half_extents=(4, 3, 2), center=(1, 2, 3))
    assert isinstance(box, Container)
    assert isinstance(box, KinematicBodySet)
    npt.assert_array_equal(box.half_extents, [4, 3, 2])
    npt.assert_array_equal(box.positions[0], [1, 2, 3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"half_extents": 0},
        {"half_extents": (1, -1, 1)},
        {"half_extents": 1, "center": (0, 0)},
        {"half_extents": 1, "friction": -1},
        {"half_extents": 1, "restitution": 2},
    ],
)
def test_invalid_container_arguments(kwargs):
    with pytest.raises(ValueError):
        World().add_container(**kwargs)


def test_ball_rests_on_container_floor():
    world = World()
    world.add_container(half_extents=2.0)
    ball = world.add_bodies([[0, 1.0, 0]], Sphere(0.5), restitution=0.0)
    world.step(240)
    npt.assert_allclose(ball.positions[0, 1], -1.5, atol=0.02)


def test_box_rests_on_container_floor_with_four_contacts():
    world = World()
    world.add_container(half_extents=2.0)
    box = world.add_bodies([[0, -1.4, 0]], Box(0.5), restitution=0.0)
    world.step(120)
    npt.assert_allclose(box.positions[0, 1], -1.5, atol=0.02)
    npt.assert_equal(world.contacts.a.size, 4)


def test_ball_bounces_off_every_wall():
    for axis in range(3):
        for sign in (-1.0, 1.0):
            world = World(gravity=(0, 0, 0))
            world.add_container(half_extents=2.0)
            velocity = np.zeros(3)
            velocity[axis] = sign * 4.0
            ball = world.add_bodies(
                [[0, 0, 0]],
                Sphere(0.5),
                velocities=velocity,
                restitution=1.0,
                friction=0.0,
            )
            world.step(60)
            npt.assert_allclose(ball.velocities[0, axis], -velocity[axis], atol=1e-6)


def test_fast_body_cannot_tunnel_through_wall():
    world = World(gravity=(0, 0, 0))
    world.add_container(half_extents=2.0)
    ball = world.add_bodies(
        [[0, 0, 0]], Sphere(0.1), velocities=(2000.0, 0, 0), restitution=0.0
    )
    world.step(5)
    assert abs(ball.positions[0, 0]) <= 2.0 + 40.0
    world.step(120)
    assert abs(ball.positions[0, 0]) <= 2.0


def test_open_top_has_no_ceiling():
    world = World(gravity=(0, 9.81, 0))
    box = world.add_container(half_extents=1.0, open_top=True)
    assert not box.walls[1]
    ball = world.add_bodies([[0, 0.5, 0]], Sphere(0.3))
    world.step(60)
    assert ball.positions[0, 1] > 1.0


def test_contact_normals_point_inward():
    world = World(gravity=(0, 0, 0), contact_margin=0.05)
    box = world.add_container(half_extents=1.0)
    world.add_bodies([[0.72, 0, 0]], Sphere(0.3))
    rotation = quat_to_matrix(world._rotations)
    contacts = container_contacts(world, rotation, box)
    npt.assert_equal(contacts.a[0], box.start)
    npt.assert_allclose(contacts.normals, [[-1, 0, 0]])


def test_shake_never_lets_bodies_escape_and_adds_energy():
    world = World()
    box = world.add_container(half_extents=3.0)
    balls, boxes = _fill(world)
    world.step(60)
    settled = (balls.masses * (balls.velocities**2).sum(1)).sum()
    box.shake(amplitude=0.5, frequency=3.0, axis=(1, 0.3, 0))
    energies = []
    for _ in range(120):
        world.step(1)
        energies.append((balls.masses * (balls.velocities**2).sum(1)).sum())
        assert _inside(world, box, balls) and _inside(world, box, boxes)
    assert max(energies) > 5 * settled


def test_shake_returns_to_start_after_whole_cycles():
    world = World()
    box = world.add_container(half_extents=2.0)
    box.shake(amplitude=0.5, frequency=2.0, duration=1.1)
    world.step(92)
    assert not box.is_moving
    npt.assert_allclose(box.positions[0], 0.0, atol=0.02)
    npt.assert_allclose(box.velocities[0], 0.0)


def test_shake_peak_displacement():
    world = World()
    box = world.add_container(half_extents=2.0)
    box.shake(amplitude=0.5, frequency=1.0, duration=1.0)
    peak = 0.0
    for _ in range(60):
        world.step(1)
        peak = max(peak, abs(box.positions[0, 0]))
    npt.assert_allclose(peak, 0.5, atol=0.01)


def test_shake_warns_when_too_violent(caplog):
    world = World()
    box = world.add_container(half_extents=2.0)
    world.add_bodies([[0, 0, 0]], Sphere(0.1))
    with caplog.at_level(logging.WARNING):
        box.shake(amplitude=2.0, frequency=5.0)
    assert "overlap visibly" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        box.shake(amplitude=0.01, frequency=1.0)
    assert caplog.text == ""


def test_moving_wall_transfers_velocity():
    world = World(gravity=(0, 0, 0))
    box = world.add_container(half_extents=2.0, restitution=0.0)
    ball = world.add_bodies([[1.6, 0, 0]], Sphere(0.3), restitution=0.0, friction=0.0)
    box.velocities = [[-3.0, 0, 0]]
    world.step(30)
    npt.assert_allclose(ball.velocities[0, 0], -3.0, atol=0.05)


def test_tilted_container_keeps_bodies_inside():
    world = World()
    box = world.add_container(half_extents=2.5)
    balls, boxes = _fill(world, count=30, half=2.5, seed=4)
    world.step(30)
    box.tilt(axis=(0, 0, 1), angle=np.pi / 2, duration=1.0)
    world.step(180)
    assert not box.is_moving
    assert _inside(world, box, balls) and _inside(world, box, boxes)
    rotation = quat_to_matrix(box.rotations.copy())[0]
    npt.assert_allclose(rotation[:, 1], [-1, 0, 0], atol=0.02)
    assert balls.positions[:, 1].mean() < -1.0


def test_rotating_container_drags_contents_around():
    world = World(gravity=(0, 0, 0))
    box = world.add_container(half_extents=1.0, friction=1.0)
    ball = world.add_bodies([[0.6, 0, 0.6]], Sphere(0.3), friction=1.0, restitution=0.0)
    box.spin(axis=(0, 1, 0), speed=4.0)
    world.step(120)
    assert np.linalg.norm(ball.velocities[0]) > 0.3


def test_rock_returns_to_start():
    world = World()
    box = world.add_container(half_extents=2.0)
    box.rock(axis=(0, 0, 1), angle=0.4, frequency=1.0, duration=1.0)
    extreme = 0.0
    for _ in range(61):
        world.step(1)
        extreme = max(extreme, 2 * abs(np.arcsin(box.rotations[0, 2])))
    npt.assert_allclose(extreme, 0.4, atol=0.02)
    npt.assert_allclose(box.rotations[0], [0, 0, 0, 1], atol=0.02)


def test_spin_with_pivot_stays_on_circle():
    world = World(gravity=(0, 0, 0))
    arm = world.add_bodies([[2.0, 0, 0]], Box(0.2), kind="kinematic")
    arm.spin(axis=(0, 1, 0), speed=3.0, pivot=(0, 0, 0))
    world.step(300)
    npt.assert_allclose(np.linalg.norm(arm.positions[0]), 2.0, atol=1e-6)


def test_spin_without_pivot_stays_in_place():
    world = World()
    arm = world.add_bodies([[2.0, 0, 0]], Box(0.2), kind="kinematic")
    arm.spin(axis=(0, 1, 0), speed=3.0)
    world.step(30)
    npt.assert_array_equal(arm.positions[0], [2.0, 0, 0])
    npt.assert_allclose(arm.angular_velocities[0], [0, 3, 0])


def test_timed_spin_stops():
    world = World()
    arm = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    arm.spin(speed=1.0, duration=0.5)
    world.step(40)
    assert not arm.is_moving
    npt.assert_array_equal(arm.angular_velocities[0], [0, 0, 0])


def test_move_by_and_move_to_reach_their_targets():
    world = World()
    body = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    body.move_by((3, 0, 0), duration=1.0)
    world.step(70)
    npt.assert_allclose(body.positions[0], [3, 0, 0], atol=0.01)
    body.move_to((3, 2, 0), duration=0.5)
    world.step(40)
    npt.assert_allclose(body.positions[0], [3, 2, 0], atol=0.01)


def test_move_preserves_group_offsets():
    world = World()
    bodies = world.add_bodies([[0, 0, 0], [1, 0, 0]], Box(0.2), kind="kinematic")
    bodies.move_to((0, 2, 0), duration=0.5)
    world.step(40)
    npt.assert_allclose(bodies.positions, [[0, 2, 0], [1, 2, 0]], atol=0.01)


def test_tilt_reaches_its_angle():
    world = World()
    body = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    body.tilt(axis=(0, 0, 1), angle=1.0, duration=1.0)
    world.step(70)
    npt.assert_allclose(2 * np.arcsin(body.rotations[0, 2]), 1.0, atol=0.01)


def test_follow_path_tracks_the_function():
    world = World()
    body = world.add_bodies([[1, 0, 0]], Box(0.2), kind="kinematic")
    body.follow_path(lambda t: (0.0, 2.0 * t, 0.0))
    world.step(30)
    npt.assert_allclose(body.positions[0], [1, 1.0, 0], atol=0.01)
    npt.assert_allclose(body.velocities[0], [0, 2.0, 0], atol=1e-6)


def test_motions_add_together():
    world = World()
    body = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    body.follow_path(lambda t: (t, 0.0, 0.0))
    body.follow_path(lambda t: (0.0, t, 0.0))
    world.step(60)
    npt.assert_allclose(body.positions[0], [1, 1, 0], atol=0.01)


def test_stop_cancels_motion():
    world = World()
    box = world.add_container(half_extents=2.0)
    box.shake(amplitude=0.2, frequency=1.0).spin(speed=1.0)
    world.step(5)
    box.stop()
    assert not box.is_moving
    held = box.positions[0].copy()
    world.step(5)
    npt.assert_array_equal(box.positions[0], held)


def test_remove_container_stops_collisions():
    world = World()
    box = world.add_container(half_extents=1.0)
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.3))
    world.remove(box)
    world.step(60)
    assert ball.positions[0, 1] < -1.0
    assert world.containers == ()


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.shake(amplitude=0),
        lambda b: b.shake(frequency=-1),
        lambda b: b.shake(axis=(0, 0, 0)),
        lambda b: b.shake(axis=(1, 0)),
        lambda b: b.shake(duration=0),
        lambda b: b.move_by((1, 0)),
        lambda b: b.move_by((1, 0, 0), duration=0),
        lambda b: b.tilt(duration=-1),
        lambda b: b.rock(frequency=0),
        lambda b: b.spin(pivot=(0, 0)),
        lambda b: b.spin(pivot=(0, np.nan, 0)),
        lambda b: b.spin(speed=np.nan),
        lambda b: b.tilt(angle=np.inf),
        lambda b: b.follow_path(lambda t: (t, 0)),
    ],
)
def test_invalid_motion_arguments(call):
    world = World()
    box = world.add_container(half_extents=2.0)
    with pytest.raises(ValueError):
        call(box)


def test_follow_path_requires_a_callable():
    world = World()
    box = world.add_container(half_extents=2.0)
    with pytest.raises(TypeError):
        box.follow_path((1, 2, 3))


def test_kinematic_paddle_sweeps_balls_aside():
    world = World()
    world.add_bodies([[0, -1, 0]], Box((20, 1, 20)), kind="static")
    paddle = world.add_bodies([[-3, 0.5, 0]], Box((0.2, 0.5, 2)), kind="kinematic")
    balls = world.add_bodies(
        [[-1.0, 0.31, -0.5], [-1.0, 0.31, 0.5]], Sphere(0.3), restitution=0.0
    )
    paddle.move_by((5, 0, 0), duration=1.0)
    world.step(90)
    assert (balls.positions[:, 0] > 1.0).all()


def test_bodies_outside_a_container_are_not_affected():
    world = World()
    world.add_bodies([[0, -1.5, 0]], Box((30, 0.5, 30)), kind="static")
    world.add_container(half_extents=1.0, center=(0, 0, 0))
    ball = world.add_bodies([[6.0, -0.7, 0]], Sphere(0.3))
    world.step(90)
    npt.assert_allclose(ball.positions[0, [0, 2]], [6.0, 0.0], atol=1e-6)


def test_two_containers_do_not_pull_on_each_others_bodies():
    world = World(gravity=(0, 0, 0))
    world.add_container(half_extents=1.0, center=(-5, 0, 0))
    world.add_container(half_extents=1.0, center=(5, 0, 0))
    balls = world.add_bodies([[-5.0, 0, 0], [5.0, 0, 0]], Sphere(0.3))
    world.step(120)
    npt.assert_allclose(balls.positions, [[-5.0, 0, 0], [5.0, 0, 0]], atol=1e-9)


def test_body_thrown_out_of_an_open_top_is_free():
    world = World()
    world.add_container(half_extents=1.0, open_top=True)
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.2), velocities=(3.0, 9.0, 0))
    world.step(90)
    assert ball.positions[0, 0] > 3.0


@pytest.mark.parametrize(
    "dt, duration", [(1 / 60, 1.0), (1 / 60, 0.25), (0.05, 0.11), (1 / 30, 0.1)]
)
def test_timed_motions_land_exactly_on_target(dt, duration):
    world = World(dt=dt, gravity=(0, 0, 0))
    mover = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    turner = world.add_bodies([[5, 0, 0]], Box(0.2), kind="kinematic")
    mover.move_by((10, 0, 0), duration=duration)
    turner.tilt(axis=(0, 0, 1), angle=np.pi / 2, duration=duration)
    world.step(int(np.ceil(duration / dt)) + 5)
    npt.assert_allclose(mover.positions[0], [10, 0, 0], atol=1e-9)
    npt.assert_allclose(2 * np.arcsin(turner.rotations[0, 2]), np.pi / 2, atol=1e-9)
    npt.assert_array_equal(mover.velocities, 0)


@pytest.mark.parametrize("frequency", [0.7, 1.3, 2.0])
def test_timed_shake_ends_where_it_began(frequency):
    world = World(gravity=(0, 0, 0))
    body = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    body.shake(amplitude=1.0, frequency=frequency, duration=0.5)
    world.step(150)
    npt.assert_allclose(body.positions[0], 0, atol=1e-9)


def test_long_spin_about_a_pivot_keeps_orientation_with_orbit():
    world = World(gravity=(0, 0, 0))
    body = world.add_bodies([[3, 0, 0]], Box(0.2), kind="kinematic")
    body.spin(axis=(0, 1, 0), speed=5.0, pivot=(0, 0, 0))
    world.step(6000)
    x, _, z = body.positions[0]
    orbit = np.arctan2(-z, x)
    turn = 2 * np.arctan2(body.rotations[0, 1], body.rotations[0, 3])
    npt.assert_allclose(np.angle(np.exp(1j * (orbit - turn))), 0, atol=1e-6)


@pytest.mark.parametrize("speed", [8.0, 15.0])
def test_fast_spinning_container_keeps_its_bodies(speed):
    world = World(dt=1 / 240)
    box = world.add_container(half_extents=1.0)
    grid = np.linspace(-0.7, 0.7, 3)
    centers = np.stack(np.meshgrid(grid, grid, grid), -1).reshape(-1, 3)
    balls = world.add_bodies(centers, Sphere(0.1))
    box.spin(axis=(0, 0, 1), speed=speed, duration=2.0)
    world.step(960)
    local = balls.positions @ quat_to_matrix(box.rotations)[0]
    assert np.all(np.abs(local) <= 1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.spin(speed=30.0),
        lambda b: b.rock(angle=1.0, frequency=5.0),
        lambda b: b.tilt(angle=3.0, duration=0.1),
        lambda b: b.move_by((5, 0, 0), duration=0.1),
        lambda b: b.follow_path(lambda t: (3 * np.sin(40 * t), 0, 0)),
        lambda b: b.shake(amplitude=0.12, frequency=3.0).move_by((0, 1.8, 0)),
    ],
)
def test_every_fast_motion_warns(call, caplog):
    world = World()
    box = world.add_container(half_extents=1.0)
    world.add_bodies([[0, 0, 0]], Sphere(0.1))
    with caplog.at_level(logging.WARNING):
        call(box)
        world.step(60)
    assert caplog.text.count("Slow the motion or reduce dt") == 1


def test_moving_a_body_out_by_hand_lets_it_go():
    world = World(gravity=(0, 0, 0))
    world.add_container(half_extents=1.0)
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.2))
    world.step(1)
    ball.positions = [[5.0, 0, 0]]
    world.step(30)
    npt.assert_allclose(ball.positions, [[5.0, 0, 0]])


def test_removing_a_container_lets_its_bodies_go():
    world = World(gravity=(0, 0, 0))
    box = world.add_container(half_extents=1.0)
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.2), velocities=(3.0, 0, 0))
    world.step(1)
    world.remove(box)
    npt.assert_array_equal(world._holder, -1)
    world.step(60)
    assert ball.positions[0, 0] > 2.5


def test_setting_one_body_lets_only_that_body_go():
    world = World(gravity=(0, 0, 0))
    world.add_container(half_extents=1.0)
    balls = world.add_bodies([[0.5, 0, 0], [-0.5, 0, 0]], Sphere(0.1))
    world.step(1)
    balls[1].velocity = (0.1, 0, 0)
    npt.assert_array_equal(world._holder[1:], [1, -1])


def test_moving_a_container_by_hand_keeps_only_bodies_still_inside():
    world = World(gravity=(0, 0, 0))
    box = world.add_container(half_extents=1.0)
    balls = world.add_bodies([[0.8, 0, 0], [-0.8, 0, 0]], Sphere(0.1))
    world.step(1)
    box.positions = [[1.5, 0, 0]]
    world.step(30)
    npt.assert_allclose(balls.positions[1], [-0.8, 0, 0])
    npt.assert_array_equal(world._holder[1:], [1, -1])


def test_a_slow_motion_after_a_fast_one_does_not_warn_again(caplog):
    world = World()
    box = world.add_container(half_extents=1.0)
    world.add_bodies([[0, 0, 0]], Sphere(0.1))
    with caplog.at_level(logging.WARNING):
        box.shake(amplitude=0.5, frequency=10.0)
        box.move_by((0.1, 0, 0), duration=5.0)
        world.step(30)
    assert caplog.text.count("Slow the motion or reduce dt") == 1


def test_path_that_turns_non_finite_raises():
    world = World()
    body = world.add_bodies([[0, 0, 0]], Box(0.2), kind="kinematic")
    body.follow_path(lambda t: (t if t < 0.05 else np.nan, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        world.step(10)
