"""Tests for the physics world, spheres and solver."""

import logging

import numpy as np
import numpy.testing as npt
import pytest

from fury.physics import Sphere, World
from fury.physics.broadphase import find_pairs
from fury.physics.narrowphase import sphere_sphere
from fury.physics.shapes import Shape
from fury.physics.solver import color_contacts


def _no_gravity(**kwargs):
    return World(gravity=(0, 0, 0), **kwargs)


def test_empty_world_steps():
    world = World()
    world.step(3)
    npt.assert_equal(world.step_count, 3)
    npt.assert_almost_equal(world.time, 3 / 60)


def test_free_fall_matches_integrated_gravity():
    world = World(dt=1 / 60, substeps=4)
    ball = world.add_bodies([[0.0, 100.0, 0.0]], Sphere(0.5), angular_damping=0.0)
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 1], -9.81, rtol=1e-9)
    npt.assert_allclose(ball.positions[0, 1], 100 - 0.5 * 9.81, atol=0.05)


def test_elastic_collision_swaps_velocities_and_conserves_momentum():
    world = _no_gravity()
    balls = world.add_bodies(
        [[-2.0, 0, 0], [2.0, 0, 0]], Sphere(0.5), restitution=1.0, friction=0.0
    )
    balls.velocities = [[3.0, 0, 0], [-1.0, 0, 0]]
    before = (balls.masses * balls.velocities[:, 0]).sum()
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [-1.0, 3.0], atol=1e-6)
    npt.assert_allclose((balls.masses * balls.velocities[:, 0]).sum(), before)


def test_inelastic_collision_moves_together():
    world = _no_gravity()
    balls = world.add_bodies(
        [[-1.0, 0, 0], [1.0, 0, 0]], Sphere(0.5), restitution=0.0, friction=0.0
    )
    balls.velocities = [[2.0, 0, 0], [0.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [1.0, 1.0], atol=1e-3)


def test_unequal_mass_elastic_collision():
    world = _no_gravity()
    world.restitution_threshold = 0.0
    heavy = world.add_bodies([[-2.0, 0, 0]], Sphere(0.5), mass=3.0, restitution=1.0)
    light = world.add_bodies([[2.0, 0, 0]], Sphere(0.5), mass=1.0, restitution=1.0)
    heavy.velocities = [[2.0, 0, 0]]
    world.step(240)
    npt.assert_allclose(heavy.velocities[0, 0], 1.0, atol=1e-6)
    npt.assert_allclose(light.velocities[0, 0], 3.0, atol=1e-6)


def test_static_sphere_never_moves_and_stops_a_ball():
    world = World(gravity=(0, -9.81, 0))
    anchor = world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static")
    ball = world.add_bodies([[0, 2.0, 0]], Sphere(0.5), restitution=0.0)
    world.step(240)
    npt.assert_array_equal(anchor.positions[0], [0, 0, 0])
    npt.assert_allclose(ball.positions[0, 1], 1.5, atol=0.03)
    npt.assert_allclose(ball.velocities[0], [0, 0, 0], atol=0.05)


def test_kinematic_body_follows_its_velocity_and_pushes():
    world = _no_gravity()
    pusher = world.add_bodies([[-2.0, 0, 0]], Sphere(0.5), kind="kinematic")
    ball = world.add_bodies([[0.0, 0, 0]], Sphere(0.5), restitution=0.0, friction=0.0)
    pusher.velocities = [[1.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(pusher.positions[0, 0], 0.0, atol=1e-9)
    assert ball.velocities[0, 0] > 0.5
    npt.assert_allclose(pusher.velocities[0], [1.0, 0, 0])


def test_gravity_does_not_affect_kinematic_body():
    world = World()
    body = world.add_bodies([[0, 5.0, 0]], Sphere(1.0), kind="kinematic")
    world.step(30)
    npt.assert_array_equal(body.positions[0], [0, 5.0, 0])


def test_no_energy_gain_in_closed_collisions():
    rng = np.random.default_rng(1)
    world = _no_gravity()
    grid = np.stack(np.meshgrid(*[np.arange(4.0)] * 3), -1).reshape(-1, 3) * 1.4
    balls = world.add_bodies(grid, Sphere(0.5), restitution=0.5, friction=0.2)
    balls.velocities = rng.uniform(-3, 3, grid.shape)

    def energy():
        return (balls.masses * (balls.velocities**2).sum(1)).sum()

    start = energy()
    world.step(240)
    assert energy() <= start * 1.01


def test_coincident_spheres_do_not_produce_nans():
    world = _no_gravity()
    balls = world.add_bodies([[0, 0, 0], [0, 0, 0]], Sphere(0.5))
    world.step(10)
    assert np.isfinite(balls.positions).all()
    assert np.linalg.norm(balls.positions[1] - balls.positions[0]) > 0


def test_impulse_force_and_torque():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(1.0), mass=2.0, angular_damping=0.0)
    ball.apply_impulse((4.0, 0, 0))
    npt.assert_allclose(ball.velocities[0], [2.0, 0, 0])

    ball.velocities = (0, 0, 0)
    ball.apply_force((0, 6.0, 0))
    world.step(1)
    npt.assert_allclose(ball.velocities[0, 1], 3.0 * world.dt)
    world.step(1)
    npt.assert_allclose(ball.velocities[0, 1], 3.0 * world.dt)

    ball.apply_torque((0, 0, 1.0))
    world.step(1)
    assert ball.angular_velocities[0, 2] > 0


def test_off_centre_impulse_spins_the_body():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(1.0), mass=1.0)
    ball.apply_impulse((0, 1.0, 0), at=(1.0, 0, 0))
    assert ball.angular_velocities[0, 2] > 0


def test_state_snapshots_are_read_only_and_setters_work():
    world = World()
    balls = world.add_bodies(np.zeros((2, 3)), Sphere(0.5))
    with pytest.raises(ValueError):
        balls.positions[0, 0] = 1.0
    balls.positions = [[1, 2, 3], [4, 5, 6]]
    npt.assert_array_equal(balls[1].position, [4, 5, 6])
    balls[-1].velocity = (1, 0, 0)
    npt.assert_array_equal(balls.velocities[1], [1, 0, 0])
    balls.rotations = (0, 0, 2.0, 0)
    npt.assert_allclose(balls.rotations[0], [0, 0, 1, 0])
    with pytest.raises(ValueError):
        balls.rotations = (0, 0, 0, 0)


def test_body_set_container_protocol():
    world = World()
    balls = world.add_bodies(np.zeros((3, 3)), Sphere(0.5))
    assert len(balls) == 3
    assert len(list(balls)) == 3
    with pytest.raises(IndexError):
        balls[3]


def test_removing_a_set_keeps_other_sets_valid():
    world = _no_gravity()
    first = world.add_bodies([[0, 0, 0]], Sphere(0.5))
    second = world.add_bodies([[5, 0, 0]], Sphere(0.5))
    third = world.add_bodies([[9, 0, 0]], Sphere(0.5))
    world.remove(second)
    npt.assert_equal(world.count, 2)
    npt.assert_array_equal(first.positions[0], [0, 0, 0])
    npt.assert_array_equal(third.positions[0], [9, 0, 0])
    world.step(1)
    with pytest.raises(ValueError):
        world.remove(second)


def test_adding_bodies_during_a_run():
    world = _no_gravity()
    world.add_bodies([[0, 0, 0]], Sphere(0.5))
    world.step(2)
    late = world.add_bodies([[3, 0, 0]], Sphere(0.5), velocities=(-5.0, 0, 0))
    world.step(60)
    assert late.velocities[0, 0] > -5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density": 0.0},
        {"density": -1.0},
        {"mass": 0.0},
        {"mass": np.inf},
        {"kind": "floppy"},
        {"velocities": [[1.0, 0, 0], [2.0, 0, 0]]},
        {"velocities": (np.inf, 0, 0)},
        {"gravity_scale": np.nan},
    ],
)
def test_invalid_body_arguments(kwargs):
    with pytest.raises(ValueError):
        World().add_bodies([[0, 0, 0]], Sphere(0.5), **kwargs)


@pytest.mark.parametrize(
    "positions, shape",
    [
        (np.zeros((0, 3)), Sphere(0.5)),
        (np.zeros(3), Sphere(0.5)),
        (np.zeros((2, 3)), Sphere(0.0)),
        (np.zeros((2, 3)), Sphere([1.0, 2.0, 3.0])),
        (np.array([[np.nan, 0.0, 0.0]]), Sphere(0.5)),
    ],
)
def test_invalid_positions_and_shapes(positions, shape):
    with pytest.raises(ValueError):
        World().add_bodies(positions, shape)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"dt": np.nan},
        {"substeps": 0},
        {"max_colors": -1},
        {"gravity": (0, 1)},
        {"gravity": (0, np.nan, 0)},
        {"contact_hertz": np.nan},
    ],
)
def test_invalid_world_arguments(kwargs):
    with pytest.raises(ValueError):
        World(**kwargs)


def test_broadphase_finds_only_nearby_pairs():
    x = np.array([[0, 0, 0], [0.9, 0, 0], [5, 0, 0]], float)
    r = np.full(3, 0.5)
    i, j = find_pairs(x, np.full((3, 3), 0.5), r, np.ones(3, bool), 0.01)
    npt.assert_array_equal(np.c_[i, j], [[0, 1]])


def test_broadphase_handles_one_huge_body():
    x = np.array(
        [[0, -50, 0], [0, 0.4, 0], [3, 0.4, 0], [3, 0.4, 0.5], [8, 9, 9]], float
    )
    r = np.array([50, 0.5, 0.5, 0.5, 0.5])
    i, j = find_pairs(x, r[:, None] * np.ones((5, 3)), r, np.ones(5, bool), 0.01)
    found = {(int(a), int(b)) for a, b in zip(i, j, strict=True)}
    assert (0, 1) in found and (0, 2) in found and (2, 3) in found
    assert all(1 != a or b != 2 for a, b in found)


def test_broadphase_with_few_candidates():
    x = np.zeros((2, 3))
    r = np.full(2, 0.5)
    i, j = find_pairs(x, np.full((2, 3), 0.5), r, np.array([True, False]), 0.01)
    npt.assert_equal((i.size, j.size), (0, 0))


def test_sphere_contact_is_at_midpoint_of_gap():
    x = np.array([[0, 0, 0], [0.9, 0, 0]], float)
    size = np.full((2, 3), 0.5)
    c = sphere_sphere(x, size, np.array([0]), np.array([1]), 0.01)
    npt.assert_allclose(c.separation, [-0.1])
    npt.assert_allclose(c.normals, [[1, 0, 0]])
    npt.assert_allclose(c.points, [[0.45, 0, 0]])


def test_coloring_never_repeats_a_dynamic_body():
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 30, (2, 200))
    b[a == b] = (b[a == b] + 1) % 30
    dynamic = np.ones(31, bool)
    batches, leftover = color_contacts(a, b, dynamic, max_colors=40)
    assert leftover.size == 0
    assert sum(index.size for index in batches) == 200
    for index in batches:
        used = np.concatenate([a[index], b[index]])
        assert np.unique(used).size == used.size


def test_coloring_lets_static_bodies_repeat():
    a = np.array([5, 5, 5])
    b = np.array([0, 1, 2])
    dynamic = np.array([True, True, True, True, True, False])
    batches, _ = color_contacts(a, b, dynamic, max_colors=8)
    assert len(batches) == 1


def test_coloring_with_zero_colors_is_all_leftover():
    batches, leftover = color_contacts(
        np.array([0]), np.array([1]), np.ones(2, bool), 0
    )
    assert batches == []
    npt.assert_array_equal(leftover, [0])


def test_jacobi_only_world_is_still_conservative():
    world = _no_gravity(max_colors=0)
    balls = world.add_bodies(
        [[-2.0, 0, 0], [2.0, 0, 0]], Sphere(0.5), restitution=1.0, friction=0.0
    )
    balls.velocities = [[3.0, 0, 0], [-1.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [-1.0, 3.0], atol=1e-6)


def test_results_are_repeatable():
    def run():
        world = World()
        rng = np.random.default_rng(9)
        x = rng.uniform(-1.5, 1.5, (30, 3))
        world.add_bodies(x, Sphere(0.3))
        world.step(30)
        return world.sets[0].positions

    npt.assert_array_equal(run(), run())


def test_held_snapshot_does_not_change_after_a_step():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.5), velocities=(1.0, 0, 0))
    before = ball.positions
    world.step(10)
    npt.assert_array_equal(before[0], [0, 0, 0])
    assert ball.positions[0, 0] > 0


def test_static_body_rejects_velocity():
    world = _no_gravity()
    with pytest.raises(ValueError, match="Static"):
        world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static", velocities=(1, 0, 0))
    wall = world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static")
    for assign in (
        lambda: setattr(wall, "velocities", (1, 0, 0)),
        lambda: setattr(wall, "angular_velocities", (1, 0, 0)),
        lambda: setattr(wall[0], "velocity", (1, 0, 0)),
        lambda: setattr(wall[0], "angular_velocity", (1, 0, 0)),
    ):
        with pytest.raises(ValueError, match="Static"):
            assign()
    world.step(10)
    npt.assert_array_equal(wall.positions[0], [0, 0, 0])


def test_removed_bodies_refuse_every_access():
    world = _no_gravity()
    gone = world.add_bodies([[5, 0, 0]], Sphere(0.5))
    kept = world.add_bodies([[9, 0, 0]], Sphere(0.5))
    body = gone[0]
    world.remove(gone)
    for use in (
        lambda: gone.positions,
        lambda: setattr(gone, "positions", (1, 1, 1)),
        lambda: gone.apply_impulse((1, 0, 0)),
        lambda: body.position,
    ):
        with pytest.raises(ValueError, match="removed"):
            use()
    npt.assert_array_equal(kept.positions[0], [9, 0, 0])
    npt.assert_array_equal(kept.velocities[0], [0, 0, 0])


def test_set_bookkeeping_is_read_only():
    world = World()
    balls = world.add_bodies(np.zeros((2, 3)), Sphere(0.5))
    assert world.sets == (balls,)
    assert balls.world is world
    for name in ("world", "start", "count"):
        with pytest.raises(AttributeError):
            setattr(balls, name, 1)


def _slide_on_sphere(heading):
    world = World(gravity=(0, -9.81, 0))
    world.add_bodies([[0, -100.0, 0]], Sphere(100.0), kind="static", friction=0.5)
    direction = np.array([np.cos(heading), 0.0, np.sin(heading)])
    ball = world.add_bodies(
        [[0, 0.5, 0]],
        Sphere(0.5),
        velocities=4.0 * direction,
        friction=0.5,
        restitution=0.0,
    )
    world.step(6)
    velocity = ball.velocities[0]
    deceleration = (4.0 - np.linalg.norm(velocity[[0, 2]])) / (6 * world.dt)
    return deceleration, np.arctan2(velocity[2], velocity[0])


@pytest.mark.parametrize("heading", [0.0, np.pi / 6, np.pi / 4, np.pi / 3, 2.0])
def test_friction_does_not_depend_on_sliding_direction(heading):
    deceleration, final_heading = _slide_on_sphere(heading)
    npt.assert_allclose(deceleration, 0.5 * 9.81, rtol=0.03)
    npt.assert_allclose(final_heading, heading, atol=1e-3)


def test_zero_friction_keeps_a_ball_sliding():
    world = World(gravity=(0, -9.81, 0))
    world.add_bodies([[0, -100.0, 0]], Sphere(100.0), kind="static", friction=0.0)
    ball = world.add_bodies(
        [[0, 0.5, 0]], Sphere(0.5), velocities=(4.0, 0, 0), friction=0.0, restitution=0
    )
    world.step(6)
    npt.assert_allclose(ball.velocities[0, 0], 4.0, atol=0.02)


def test_friction_coefficients_combine_as_geometric_mean():
    world = World(gravity=(0, -9.81, 0))
    world.add_bodies([[0, -100.0, 0]], Sphere(100.0), kind="static", friction=1.0)
    ball = world.add_bodies(
        [[0, 0.5, 0]], Sphere(0.5), velocities=(4.0, 0, 0), friction=0.25, restitution=0
    )
    world.step(6)
    deceleration = (4.0 - ball.velocities[0, 0]) / (6 * world.dt)
    npt.assert_allclose(deceleration, 0.5 * 9.81, rtol=0.03)


def test_restitution_uses_the_larger_coefficient():
    world = _no_gravity()
    balls = world.add_bodies([[-2.0, 0, 0]], Sphere(0.5), restitution=1.0, friction=0)
    other = world.add_bodies([[2.0, 0, 0]], Sphere(0.5), restitution=0.0, friction=0)
    balls.velocities = (2.0, 0, 0)
    other.velocities = (-2.0, 0, 0)
    world.step(120)
    npt.assert_allclose(balls.velocities[0, 0], -2.0, atol=1e-6)
    npt.assert_allclose(other.velocities[0, 0], 2.0, atol=1e-6)


@pytest.mark.parametrize("max_colors", [8, 0])
def test_three_body_impact_keeps_its_energy(max_colors):
    world = _no_gravity(max_colors=max_colors)
    balls = world.add_bodies(
        [[-2.0, 0, 0], [0, 0, 0], [2.0, 0, 0]],
        Sphere(0.5),
        restitution=1.0,
        friction=0.0,
    )
    balls.velocities = [[3.0, 0, 0], [0, 0, 0], [-3.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [-3.0, 0.0, 3.0], atol=1e-4)


def test_restitution_threshold_stops_slow_bounces():
    def rebound(threshold):
        world = _no_gravity()
        world.restitution_threshold = threshold
        balls = world.add_bodies(
            [[-1.0, 0, 0], [1.0, 0, 0]], Sphere(0.5), restitution=1.0, friction=0.0
        )
        balls.velocities = [[0.5, 0, 0], [-0.5, 0, 0]]
        world.step(240)
        return balls.velocities[0, 0]

    npt.assert_allclose(rebound(0.0), -0.5, atol=1e-6)
    npt.assert_allclose(rebound(5.0), 0.0, atol=1e-3)


@pytest.mark.parametrize("mass", [1e13, 1e18])
def test_very_heavy_bodies_still_collide(mass):
    world = _no_gravity()
    balls = world.add_bodies(
        [[-1.0, 0, 0], [1.0, 0, 0]], Sphere(0.5), mass=mass, restitution=1.0, friction=0
    )
    balls.velocities = [[1.0, 0, 0], [-1.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [-1.0, 1.0], atol=1e-6)


def test_linear_damping_slows_motion():
    world = _no_gravity()
    ball = world.add_bodies(
        [[0, 0, 0]], Sphere(0.5), velocities=(1.0, 0, 0), linear_damping=1.0
    )
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 0], np.exp(-1.0), atol=0.01)


def test_angular_damping_slows_spin():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.5), angular_damping=1.0)
    ball.angular_velocities = (0, 0, 1.0)
    world.step(60)
    npt.assert_allclose(ball.angular_velocities[0, 2], np.exp(-1.0), atol=0.01)


def test_gravity_scale_multiplies_gravity():
    world = World()
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.5), gravity_scale=0.5)
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 1], -0.5 * 9.81)


def test_spin_changes_orientation_at_the_expected_rate():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.5), angular_damping=0.0)
    ball.angular_velocities = (0, 0, np.pi / 2)
    world.step(60)
    half = np.sqrt(0.5)
    npt.assert_allclose(np.abs(ball.rotations[0]), [0, 0, half, half], atol=1e-2)


def test_sphere_inertia_sets_spin_from_an_impulse():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(1.0), mass=5.0)
    ball.apply_impulse((0, 2.0, 0), at=(1.0, 0, 0))
    npt.assert_allclose(ball.angular_velocities[0], [0, 0, 1.0])


def test_force_applied_off_centre_spins_the_body():
    world = _no_gravity()
    ball = world.add_bodies([[0, 0, 0]], Sphere(1.0), mass=5.0, angular_damping=0.0)
    ball.apply_force((0, 2.0, 0), at=(1.0, 0, 0))
    world.step(1)
    npt.assert_allclose(ball.angular_velocities[0, 2], world.dt)


def test_selections_affect_only_the_selected_bodies():
    world = _no_gravity()
    balls = world.add_bodies(
        [[0, 0, 0], [5, 0, 0], [9, 0, 0]], Sphere(0.5), mass=1.0, angular_damping=0.0
    )
    balls.apply_force((6.0, 0, 0), where=[1])
    balls.apply_torque((0, 0, 1.0), where=np.array([False, False, True]))
    world.step(1)
    npt.assert_allclose(balls.velocities[:, 0], [0, 6.0 * world.dt, 0])
    assert balls.angular_velocities[2, 2] > 0
    assert balls.angular_velocities[0, 2] == 0


def test_empty_selection_does_nothing():
    world = _no_gravity()
    balls = world.add_bodies(np.zeros((1, 3)), Sphere(0.5))
    balls.apply_impulse((1, 0, 0), where=[])
    npt.assert_array_equal(balls.velocities[0], 0)


def test_body_handle_forces_and_rotation():
    world = _no_gravity()
    balls = world.add_bodies(
        [[0, 0, 0], [5, 0, 0]], Sphere(0.5), mass=1.0, angular_damping=0.0
    )
    balls[1].apply_force((6.0, 0, 0))
    balls[1].apply_torque((0, 0, 1.0))
    world.step(1)
    npt.assert_allclose(balls.velocities[1, 0], 6.0 * world.dt)
    npt.assert_array_equal(balls.velocities[0], 0)
    assert balls.angular_velocities[1, 2] > 0
    balls[0].rotation = (0, 0, 0, 2.0)
    npt.assert_allclose(balls[0].rotation, [0, 0, 0, 1])
    with pytest.raises(ValueError):
        balls[0].rotation = (0, 0, 0, 0)


def test_max_push_speed_limits_depenetration():
    def gap(limit):
        world = _no_gravity(max_push_speed=limit)
        balls = world.add_bodies([[0, 0, 0], [0.5, 0, 0]], Sphere(0.5), restitution=0)
        world.step(1)
        return np.ptp(balls.positions[:, 0])

    assert gap(0.1) < 0.51 < gap(3.0)


def test_contact_hertz_changes_recovery():
    def gap(hertz):
        world = _no_gravity(contact_hertz=hertz)
        balls = world.add_bodies([[0, 0, 0], [0.9, 0, 0]], Sphere(0.5), restitution=0)
        world.step(2)
        return np.ptp(balls.positions[:, 0])

    assert gap(30.0) > gap(5.0)


def test_warm_starting_carries_impulses_between_steps():
    world = World()
    world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static")
    ball = world.add_bodies([[0, 1.5, 0]], Sphere(0.5), restitution=0.0)
    world.step(120)
    assert world._solver._keys.size == 1
    impulse = world._solver._impulses[0, 0]
    expected = (ball.masses[0] * 9.81) * world.dt / world.substeps
    npt.assert_allclose(impulse, expected, rtol=0.05)
    started = world._solver._warm_start_impulses(world._solver._keys)
    npt.assert_allclose(started[0], impulse)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_changing_bodies_forgets_remembered_impulses(change):
    world = World()
    world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static")
    world.add_bodies([[0, 1.5, 0]], Sphere(0.5), restitution=0.0)
    extra = world.add_bodies([[9, 9, 9]], Sphere(0.5))
    world.step(60)
    assert world._solver._keys.size
    if change == "add":
        world.add_bodies([[-9, 9, 9]], Sphere(0.5))
    else:
        world.remove(extra)
    assert world._solver._keys.size == 0


def test_removing_bodies_clears_reported_contacts():
    world = _no_gravity()
    balls = world.add_bodies([[0, 0, 0], [0.9, 0, 0]], Sphere(0.5))
    world.step(1)
    assert world.contacts.a.size == 1
    world.remove(balls)
    assert world.contacts.a.size == 0


def test_automatic_margin_follows_body_size():
    world = World()
    world.add_bodies([[0, 0, 0]], Sphere(2.0))
    npt.assert_allclose(world.margin, 0.2)
    fixed = World(contact_margin=0.5)
    fixed.add_bodies([[0, 0, 0]], Sphere(2.0))
    npt.assert_allclose(fixed.margin, 0.5)


def test_two_immovable_bodies_produce_no_contacts():
    world = World()
    world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static")
    world.add_bodies([[0.5, 0, 0]], Sphere(1.0), kind="kinematic")
    world.step(1)
    assert world.contacts.a.size == 0


def test_mass_splitting_shares_a_jacobi_batch_evenly():
    world = _no_gravity(max_colors=0)
    balls = world.add_bodies(
        [[-0.95, 0, 0], [0, 0, 0], [0.95, 0, 0]], Sphere(0.5), restitution=0.0
    )
    world.step(60)
    assert np.isfinite(balls.positions).all()
    npt.assert_allclose(balls.positions[1], 0.0, atol=1e-9)
    assert np.diff(balls.positions[:, 0]).min() > 0.97


def test_ignored_mass_arguments_warn(caplog):
    world = World()
    with caplog.at_level(logging.WARNING):
        world.add_bodies([[0, 0, 0]], Sphere(1.0), kind="static", mass=-5.0)
    assert "mass ignored" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        world.add_bodies(
            [[2, 0, 0]],
            Sphere(1.0),
            kind="kinematic",
            gravity_scale=0.5,
            linear_damping=3.0,
            angular_damping=1.0,
        )
    for name in ("gravity_scale", "linear_damping", "angular_damping"):
        assert name in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        world.add_bodies([[5, 0, 0]], Sphere(1.0), mass=2.0, density=9.0)
    assert "density is ignored" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        world.add_bodies([[9, 0, 0]], Sphere(1.0), density=2.0)
    assert caplog.text == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"friction": -0.1},
        {"friction": np.nan},
        {"restitution": 1.5},
        {"restitution": -0.5},
        {"restitution": np.nan},
        {"linear_damping": -1.0},
        {"angular_damping": -1.0},
        {"friction": [0.1, 0.2]},
    ],
)
def test_invalid_surface_properties(kwargs):
    with pytest.raises(ValueError):
        World().add_bodies([[0, 0, 0]], Sphere(0.5), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_hertz": 0},
        {"contact_damping": -1},
        {"max_push_speed": 0},
        {"contact_margin": -0.1},
        {"restitution_threshold": -1.0},
    ],
)
def test_invalid_contact_settings(kwargs):
    with pytest.raises(ValueError):
        World(**kwargs)


@pytest.mark.parametrize("mass", [1.0, 1e-9, 1e-12])
def test_three_body_impact_does_not_depend_on_mass(mass):
    world = _no_gravity()
    balls = world.add_bodies(
        [[-2.0, 0, 0], [0, 0, 0], [2.0, 0, 0]],
        Sphere(0.5),
        mass=mass,
        restitution=1.0,
        friction=0.0,
    )
    balls.velocities = [[3.0, 0, 0], [0, 0, 0], [-3.0, 0, 0]]
    world.step(120)
    npt.assert_allclose(balls.velocities[:, 0], [-3.0, 0.0, 3.0], atol=1e-4)


def test_body_handle_reads_and_writes_its_state():
    world = _no_gravity()
    balls = world.add_bodies(np.zeros((2, 3)), Sphere(0.5), mass=2.0)
    body = balls[1]
    assert body.bodies is balls and body.index == 1
    body.position = (1, 2, 3)
    body.velocity = (4, 0, 0)
    body.angular_velocity = (0, 5, 0)
    npt.assert_array_equal(balls.positions[1], [1, 2, 3])
    npt.assert_array_equal(body.velocity, [4, 0, 0])
    npt.assert_array_equal(body.angular_velocity, [0, 5, 0])
    body.apply_impulse((2.0, 0, 0))
    npt.assert_array_equal(body.velocity, [5, 0, 0])
    npt.assert_array_equal(balls.velocities[0], 0)


def test_body_handle_attributes_are_read_only():
    body = World().add_bodies(np.zeros((1, 3)), Sphere(0.5))[0]
    for name in ("bodies", "index"):
        with pytest.raises(AttributeError):
            setattr(body, name, 5)


def test_body_getters_and_masses_are_read_only():
    balls = World().add_bodies(np.zeros((1, 3)), Sphere(0.5))
    for values in (balls.masses, balls[0].position, balls[0].velocity):
        with pytest.raises(ValueError):
            values[0] = 1.0


@pytest.mark.parametrize("kind", ["static", "kinematic"])
def test_forces_on_immovable_bodies_are_rejected(kind):
    world = _no_gravity()
    bodies = world.add_bodies([[0, 0, 0]], Sphere(0.5), kind=kind)
    for apply in (
        lambda: bodies.apply_impulse((1, 0, 0)),
        lambda: bodies.apply_force((1, 0, 0)),
        lambda: bodies.apply_torque((1, 0, 0)),
        lambda: bodies[0].apply_impulse((1, 0, 0)),
    ):
        with pytest.raises(ValueError, match="dynamic"):
            apply()


def test_broadphase_matches_brute_force_with_mixed_sizes():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.uniform(-20, 20, (n, 3))
    r = rng.choice([0.2, 0.3, 1.0, 5.0, 12.0], n)
    extents = r[:, None] * np.ones((n, 3))
    i, j = find_pairs(x, extents, r, np.ones(n, bool), 0.01)
    found = {(int(a), int(b)) for a, b in zip(i, j, strict=True)}
    expected = {
        (a, b)
        for a in range(n)
        for b in range(a + 1, n)
        if np.linalg.norm(x[a] - x[b]) <= r[a] + r[b] + 0.01
        and (np.abs(x[a] - x[b]) - extents[a] - extents[b] <= 0.01).all()
    }
    assert expected <= found
    assert len(found) == len(set(found)) == i.size


def test_broadphase_keeps_coincident_bodies_of_different_sizes():
    x = np.zeros((2, 3))
    r = np.array([0.1, 10.0])
    i, j = find_pairs(x, r[:, None] * np.ones((2, 3)), r, np.ones(2, bool), 0.01)
    npt.assert_array_equal(np.c_[i, j], [[0, 1]])


def test_base_shape_has_no_sizes():
    with pytest.raises(NotImplementedError):
        Shape().sizes(1)


@pytest.mark.parametrize(
    "name, value",
    [
        ("positions", [[np.nan, 0, 0]]),
        ("velocities", [[0, np.inf, 0]]),
        ("angular_velocities", [[0, 0, np.nan]]),
        ("positions", np.zeros((3, 3))),
    ],
)
def test_invalid_state_assignment(name, value):
    world = World()
    balls = world.add_bodies([[0, 0, 0], [2, 0, 0]], Sphere(0.5))
    before = balls.positions
    with pytest.raises(ValueError):
        setattr(balls, name, value)
    npt.assert_array_equal(balls.positions, before)


@pytest.mark.parametrize(
    "name, value",
    [
        ("position", [np.nan, 0, 0]),
        ("rotation", [np.nan, 0, 0, 1]),
        ("rotation", [0, 0, 0, 0]),
        ("velocity", [0, np.inf, 0]),
        ("angular_velocity", [0, 0, np.nan]),
        ("position", [1.0, 2.0]),
    ],
)
def test_invalid_single_body_assignment(name, value):
    world = World()
    balls = world.add_bodies([[0, 0, 0], [2, 0, 0]], Sphere(0.5))
    before = balls.positions, balls.rotations
    with pytest.raises(ValueError):
        setattr(balls[0], name, value)
    npt.assert_array_equal(balls.positions, before[0])
    npt.assert_array_equal(balls.rotations, before[1])
    world.step(1)
    assert np.all(np.isfinite(balls.positions))


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("apply_impulse", {"impulse": (np.nan, 0, 0)}),
        ("apply_impulse", {"impulse": (1, 0, 0), "at": (np.inf, 0, 0)}),
        ("apply_force", {"force": (0, np.inf, 0)}),
        ("apply_torque", {"torque": (0, 0, np.nan)}),
    ],
)
def test_non_finite_pushes_are_rejected(method, kwargs):
    world = World(gravity=(0, 0, 0))
    balls = world.add_bodies([[0, 0, 0]], Sphere(0.5))
    with pytest.raises(ValueError):
        getattr(balls, method)(**kwargs)
    world.step(1)
    npt.assert_array_equal(balls.velocities, 0)


def test_shape_must_be_a_shape():
    with pytest.raises(TypeError):
        World().add_bodies([[0, 0, 0]], 0.5)
