"""Tests for force fields and explosions."""

import numpy as np
import numpy.testing as npt
import pytest

from fury import physics
from fury.physics import Box, Sphere, World


def _world():
    return World(gravity=(0, 0, 0))


def _ball(world, position=(0, 0, 0), mass=1.0, **kwargs):
    return world.add_bodies([position], Sphere(0.1), mass=mass, **kwargs)


def test_uniform_force_scales_with_mass():
    world = _world()
    light, heavy = _ball(world, (0, 0, 0), 1.0), _ball(world, (5, 0, 0), 4.0)
    world.add_force(physics.uniform((4.0, 0, 0)))
    world.step(60)
    npt.assert_allclose(light.velocities[0, 0], 4.0)
    npt.assert_allclose(heavy.velocities[0, 0], 1.0)


def test_drag_slows_a_moving_body():
    world = _world()
    ball = _ball(world, velocities=(10.0, 0, 0))
    world.add_force(physics.drag(0.5))
    world.step(120)
    assert 1.0 < ball.velocities[0, 0] < 10.0


def test_wind_pushes_towards_its_velocity_and_settles():
    world = _world()
    ball = _ball(world)
    world.add_force(physics.wind((3.0, 0, 0), coefficient=5.0))
    world.step(300)
    npt.assert_allclose(ball.velocities[0, 0], 3.0, atol=0.02)


def test_attractor_pulls_light_and_heavy_bodies_equally():
    world = _world()
    light = _ball(world, (5, 0, 0), 1.0)
    heavy = _ball(world, (-5, 0, 0), 10.0)
    world.add_force(physics.attractor((0, 0, 0), strength=20.0))
    world.step(30)
    assert light.velocities[0, 0] < 0
    npt.assert_allclose(light.velocities[0, 0], -heavy.velocities[0, 0], atol=1e-6)


def test_negative_attractor_repels():
    world = _world()
    ball = _ball(world, (2, 0, 0))
    world.add_force(physics.attractor((0, 0, 0), strength=-20.0))
    world.step(10)
    assert ball.velocities[0, 0] > 0


def test_attractor_at_its_own_center_is_finite():
    world = _world()
    ball = _ball(world, (0, 0, 0))
    world.add_force(physics.attractor((0, 0, 0)))
    world.step(5)
    assert np.isfinite(ball.positions).all()


def test_attractor_leaves_kinematic_and_static_alone():
    world = _world()
    fixed = world.add_bodies([[3, 0, 0]], Sphere(0.1), kind="static")
    moving = world.add_bodies([[-3, 0, 0]], Sphere(0.1), kind="kinematic")
    world.add_force(physics.attractor((0, 0, 0)))
    world.step(10)
    npt.assert_array_equal(fixed.positions[0], [3, 0, 0])
    npt.assert_array_equal(moving.velocities[0], 0)


def test_vortex_swirls_around_its_axis():
    world = _world()
    ball = _ball(world, (2, 0, 0))
    world.add_force(physics.vortex(strength=5.0))
    world.step(5)
    velocity = ball.velocities[0]
    assert abs(velocity[2]) > 10 * abs(velocity[0])
    npt.assert_allclose(velocity[1], 0)


def test_vortex_pull_goes_inward():
    world = _world()
    ball = _ball(world, (2, 0, 0))
    world.add_force(physics.vortex(strength=0.0, pull=3.0))
    world.step(5)
    assert ball.velocities[0, 0] < 0


def test_mutual_gravity_conserves_momentum():
    rng = np.random.default_rng(3)
    world = _world()
    x = rng.uniform(-5, 5, (20, 3))
    bodies = world.add_bodies(x, Sphere(0.05), mass=rng.uniform(1, 5, 20))
    world.add_force(physics.mutual_gravity(strength=1.0))
    world.step(60)
    momentum = (bodies.masses[:, None] * bodies.velocities).sum(0)
    npt.assert_allclose(momentum, 0, atol=1e-9)
    assert np.linalg.norm(bodies.velocities, axis=1).max() > 0


def test_two_body_orbit_stays_bound():
    world = World(gravity=(0, 0, 0), dt=1 / 240)
    star = world.add_bodies([[0, 0, 0]], Sphere(0.05), mass=1000.0)
    planet = world.add_bodies([[10.0, 0, 0]], Sphere(0.05), mass=1.0)
    planet.velocities = (0, 0, np.sqrt(1000.0 / 10.0))
    world.add_force(physics.mutual_gravity(strength=1.0, softening=1e-3))
    for _ in range(600):
        world.step(1)
        distance = np.linalg.norm(planet.positions[0] - star.positions[0])
        assert 9.0 < distance < 11.0


def test_mutual_gravity_with_one_body_is_zero():
    world = _world()
    ball = _ball(world)
    world.add_force(physics.mutual_gravity())
    world.step(1)
    npt.assert_array_equal(ball.velocities[0], 0)


def test_custom_field_and_removal():
    world = _world()
    ball = _ball(world)

    def up(w):
        result = np.zeros((w.count, 3))
        result[:, 1] = 2.0
        return result

    world.add_force(up)
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 1], 2.0)
    world.remove_force(up)
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 1], 2.0)


def test_field_with_wrong_shape_is_reported():
    world = _world()
    _ball(world)
    world.add_force(lambda w: np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        world.step(1)


def test_add_force_requires_a_callable():
    with pytest.raises(TypeError):
        World().add_force((0, 1, 0))


@pytest.mark.parametrize(
    "make",
    [
        lambda: physics.uniform((1, 2)),
        lambda: physics.drag(-1),
        lambda: physics.wind((1, 0)),
        lambda: physics.wind((1, 0, 0), coefficient=-1),
        lambda: physics.attractor((0, 0)),
        lambda: physics.attractor((0, 0, 0), softening=0),
        lambda: physics.vortex(axis=(0, 0, 0)),
        lambda: physics.mutual_gravity(softening=0),
        lambda: physics.uniform((np.nan, 0, 0)),
        lambda: physics.drag(np.nan),
        lambda: physics.wind((1, 0, 0), coefficient=np.inf),
        lambda: physics.attractor((0, 0, 0), softening=np.nan),
        lambda: physics.attractor((0, 0, 0), strength=np.inf),
        lambda: physics.vortex(strength=np.nan),
        lambda: physics.vortex(pull=np.inf),
        lambda: physics.mutual_gravity(strength=np.nan),
    ],
)
def test_invalid_field_arguments(make):
    with pytest.raises(ValueError):
        make()


def test_explode_throws_bodies_outward_with_falloff():
    world = _world()
    near = _ball(world, (1, 0, 0))
    far = _ball(world, (3, 0, 0))
    outside = _ball(world, (9, 0, 0))
    hit = world.explode((0, 0, 0), strength=10.0, radius=4.0)
    npt.assert_equal(hit, 2)
    npt.assert_allclose(near.velocities[0], [7.5, 0, 0])
    npt.assert_allclose(far.velocities[0], [2.5, 0, 0])
    npt.assert_array_equal(outside.velocities[0], 0)


def test_explode_gives_mass_independent_speed():
    world = _world()
    light = _ball(world, (1, 0, 0), 1.0)
    heavy = _ball(world, (-1, 0, 0), 50.0)
    world.explode((0, 0, 0), strength=8.0, radius=2.0)
    npt.assert_allclose(light.velocities[0, 0], -heavy.velocities[0, 0])


def test_explode_at_a_body_center():
    world = _world()
    ball = _ball(world)
    npt.assert_equal(world.explode((0, 0, 0), strength=5.0, radius=1.0), 1)
    npt.assert_allclose(ball.velocities[0], [0, 5.0, 0])


def test_explode_skips_static_and_kinematic():
    world = _world()
    world.add_bodies([[1, 0, 0]], Box(0.5), kind="static")
    world.add_bodies([[-1, 0, 0]], Box(0.5), kind="kinematic")
    npt.assert_equal(world.explode((0, 0, 0)), 0)


@pytest.mark.parametrize(
    "center, kwargs",
    [
        ((0, 0), {}),
        ((0, 0, 0), {"radius": 0}),
        ((0, 0, 0), {"radius": np.inf}),
        ((0, 0, 0), {"radius": np.nan}),
        ((0, 0, 0), {"strength": np.nan}),
        ((np.nan, 0, 0), {}),
    ],
)
def test_explode_invalid_arguments(center, kwargs):
    world = World()
    ball = world.add_bodies([[1.0, 0, 0]], Sphere(0.5))
    with pytest.raises(ValueError):
        world.explode(center, **kwargs)
    npt.assert_array_equal(ball.velocities, 0)


@pytest.mark.parametrize(
    "field, settle",
    [(lambda: physics.drag(1.0), 0.0), (lambda: physics.wind((3.0, 0, 0)), 3.0)],
)
def test_drag_and_wind_stay_stable_on_light_bodies(field, settle):
    world = World()
    ball = world.add_bodies([[0, 5, 0]], Sphere(0.1), velocities=(0, 0, 2.0))
    world.gravity = np.zeros(3)
    world.add_force(field())
    speeds = []
    for _ in range(120):
        world.step(1)
        speeds.append(np.linalg.norm(ball.velocities[0] - [settle, 0, 0]))
    assert np.all(np.diff(speeds) <= 1e-12)
    npt.assert_allclose(ball.velocities[0], [settle, 0, 0], atol=1e-6)


def test_drag_matches_exponential_decay():
    world = _world()
    ball = _ball(world, velocities=(10.0, 0, 0))
    world.add_force(physics.drag(0.5))
    world.step(60)
    npt.assert_allclose(ball.velocities[0, 0], 10.0 * np.exp(-0.5), rtol=1e-9)


def test_a_field_can_remove_itself_without_skipping_others():
    world = _world()
    ball = _ball(world)

    def once(world):
        world.remove_force(once)
        return np.zeros((world.count, 3))

    world.add_force(once)
    world.add_force(physics.uniform((1.0, 0, 0)))
    world.step(1)
    assert ball.velocities[0, 0] > 0


def test_removing_an_unknown_field_raises():
    with pytest.raises(ValueError, match="not added"):
        World().remove_force(physics.drag(1.0))


def test_drag_and_wind_together_stay_stable_on_light_bodies():
    world = World(gravity=(0, 0, 0))
    ball = world.add_bodies([[0, 0, 0]], Sphere(0.05), mass=1e-4, velocities=(1, 0, 0))
    world.add_force(physics.drag(10.0))
    world.add_force(physics.wind((0, 0, 0), coefficient=10.0))
    world.add_force(physics.wind((2.0, 0, 0), coefficient=20.0))
    speeds = []
    for _ in range(30):
        world.step(1)
        speeds.append(abs(ball.velocities[0, 0] - 1.0))
    assert np.all(np.isfinite(speeds))
    npt.assert_allclose(ball.velocities[0, 0], 1.0, atol=1e-9)


def test_non_finite_field_output_raises():
    world = _world()
    _ball(world)
    world.add_force(lambda world: np.full((world.count, 3), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        world.step(1)


def test_fields_follow_bodies_after_an_earlier_set_is_removed():
    world = _world()
    first = _ball(world, (0, 0, 0))
    second = _ball(world, (5, 0, 0))
    world.add_force(physics.uniform((1.0, 0, 0)))
    world.remove(first)
    world.step(1)
    assert second.velocities[0, 0] > 0
