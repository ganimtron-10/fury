"""Tests for the GPU sphere world."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree

from fury import physics, window
from fury.physics import GPUWorld


def _drop(count=400, radius=0.1, seed=0, **options):
    options.setdefault("substeps", 4)
    world = GPUWorld(**options)
    rng = np.random.default_rng(seed)
    centers = rng.uniform((-0.8, 0.1, -0.8), (0.8, 1.9, 0.8), (count, 3))
    spheres = world.add_spheres(centers, radii=radius)
    world.set_container(center=(0, 1.0, 0), size=(2.0, 2.0, 2.0), open_top=True)
    return world, spheres


def test_free_fall_matches_gravity():
    world = GPUWorld(gravity=(0, -10.0, 0), substeps=1)
    world.add_spheres([[0, 0, 0]], radii=0.1)
    world.step(30)
    npt.assert_allclose(world.velocities[0], (0, -5.0, 0), atol=1e-4)
    expected = -10.0 * world.dt**2 * 30 * 31 / 2
    npt.assert_allclose(world.positions[0, 1], expected, rtol=1e-4)


def test_spheres_settle_inside_the_container_without_overlapping():
    world, _ = _drop()
    world.step(240)
    positions = world.positions
    assert np.all(np.abs(positions[:, [0, 2]]) <= 0.9 + 1e-3)
    assert positions[:, 1].min() >= 0.1 - 1e-3
    distances, _ = cKDTree(positions).query(positions, k=2)
    assert distances[:, 1].min() > 0.2 * 0.85
    assert np.linalg.norm(world.velocities, axis=1).mean() < 0.2
    assert world.overflow == 0


def test_static_spheres_do_not_move_and_block_dynamic_ones():
    world = GPUWorld(substeps=4)
    world.add_spheres([[0, 0, 0]], radii=0.5, static=True)
    world.add_spheres([[0, 1.5, 0]], radii=0.2)
    world.step(120)
    positions = world.positions
    npt.assert_allclose(positions[0], (0, 0, 0))
    npt.assert_allclose(positions[1, 1], 0.7, atol=0.02)


def test_heavier_sphere_is_pushed_less():
    world = GPUWorld(gravity=(0, 0, 0), substeps=1, iterations=1)
    world.add_spheres([[0, 0, 0]], radii=0.5, masses=1.0)
    world.add_spheres([[0.8, 0, 0]], radii=0.5, masses=3.0)
    world.step(1)
    light, heavy = world.positions[:, 0]
    assert -light > 3 * (heavy - 0.8) * 0.9


def test_ground_plane_holds_spheres():
    world = GPUWorld(substeps=4)
    world.add_spheres([[0, 2, 0], [3, 5, 0]], radii=0.25)
    world.set_ground(offset=-1.0)
    world.step(240)
    npt.assert_allclose(world.positions[:, 1], -0.75, atol=0.01)


def test_open_top_lets_spheres_out_and_closed_top_keeps_them():
    for open_top, escaped in ((True, True), (False, False)):
        world = GPUWorld(gravity=(0, 0, 0), substeps=2)
        world.add_spheres([[0, 0, 0]], radii=0.1, velocities=(0, 5, 0))
        world.set_container(size=(1, 1, 1), open_top=open_top)
        world.step(30)
        assert (world.positions[0, 1] > 0.5) == escaped


def test_rotated_container_is_respected():
    world = GPUWorld(gravity=(0, 0, 0), substeps=2)
    world.add_spheres([[0, 0, 0]], radii=0.1, velocities=(5, 5, 0))
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    world.set_container(size=(2, 2, 2), rotation=rotation, open_top=False)
    world.step(60)
    local = rotation.T @ world.positions[0]
    assert np.all(np.abs(local) <= 0.9 + 1e-3)
    assert local[0] > 0.85


def test_shaking_container_moves_and_stirs_the_spheres():
    world, _ = _drop(count=200)
    world.step(120)
    container = world.container
    container.shake(amplitude=0.2, frequency=2.0, duration=0.5)
    assert container.is_moving
    world.step(10)
    assert np.linalg.norm(world.velocities[:, 0], ord=np.inf) > 0.5
    world.step(40)
    assert not container.is_moving
    npt.assert_allclose(container.center, (0, 1.0, 0), atol=1e-9)


def test_tilt_turns_the_container():
    world, _ = _drop(count=50)
    world.container.tilt(axis=(0, 0, 1), angle=0.5, duration=0.5)
    world.step(40)
    expected = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    npt.assert_allclose(world.container.rotation[:2, :2], expected, atol=1e-6)


@pytest.mark.parametrize(
    "field, check",
    [
        (physics.uniform((2.0, 0, 0)), lambda v: v[0, 0] > 0.9),
        (
            physics.wind((3.0, 0, 0), coefficient=5.0),
            lambda v: abs(v[0, 0] - 3 * (1 - np.exp(-2.5))) < 1e-3,
        ),
        (physics.attractor((5, 0, 0), strength=20.0), lambda v: v[0, 0] > 0),
        (physics.vortex(strength=5.0), lambda v: abs(v[0, 2]) > 0.1),
    ],
)
def test_force_fields_act_like_the_cpu_ones(field, check):
    world = GPUWorld(gravity=(0, 0, 0), substeps=2)
    world.add_spheres([[1, 0, 0]], radii=0.1, masses=2.0)
    world.add_force(field)
    world.step(60)
    assert check(world.velocities)


def test_uniform_force_matches_cpu_world():
    gpu = GPUWorld(gravity=(0, 0, 0), substeps=1)
    gpu.add_spheres([[0, 0, 0]], radii=0.1, masses=2.0)
    gpu.add_force(physics.uniform((4.0, 0, 0)))
    cpu = physics.World(gravity=(0, 0, 0))
    ball = cpu.add_bodies([[0, 0, 0]], physics.Sphere(0.1), mass=2.0)
    cpu.add_force(physics.uniform((4.0, 0, 0)))
    gpu.step(60)
    cpu.step(60)
    npt.assert_allclose(gpu.velocities[0], ball.velocities[0, :3], atol=1e-4)


def test_removed_force_stops_acting():
    world = GPUWorld(gravity=(0, 0, 0), substeps=1)
    world.add_spheres([[0, 0, 0]], radii=0.1)
    field = world.add_force(physics.uniform((1.0, 0, 0)))
    world.step(10)
    world.remove_force(field)
    speed = world.velocities[0, 0]
    world.step(10)
    npt.assert_allclose(world.velocities[0, 0], speed)


def test_unsupported_fields_are_rejected():
    world = GPUWorld()
    with pytest.raises(TypeError):
        world.add_force(physics.mutual_gravity())
    with pytest.raises(TypeError):
        world.add_force(lambda w: None)


def test_explode_throws_spheres_once_with_falloff():
    world = GPUWorld(gravity=(0, 0, 0), substeps=1)
    world.add_spheres([[1, 0, 0], [3, 0, 0], [9, 0, 0]], radii=0.1)
    world.explode((0, 0, 0), strength=8.0, radius=4.0)
    world.step(1)
    first = world.velocities
    npt.assert_allclose(first[:, 0], (6.0, 2.0, 0.0), atol=1e-4)
    world.step(1)
    npt.assert_allclose(world.velocities, first, atol=1e-5)


def test_max_speed_limits_blasts():
    world = GPUWorld(gravity=(0, 0, 0), substeps=1, max_speed=3.0)
    world.add_spheres([[0.5, 0, 0]], radii=0.1)
    world.explode((0, 0, 0), strength=100.0, radius=1.0)
    world.step(1)
    npt.assert_allclose(np.linalg.norm(world.velocities[0]), 3.0, rtol=1e-4)


def test_advance_runs_fixed_steps():
    world = GPUWorld(dt=0.01, max_steps_per_frame=5)
    world.add_spheres([[0, 0, 0]])
    assert world.advance(elapsed=0.035) == 3
    assert world.advance(elapsed=0.006) == 1
    assert world.advance() == 1
    npt.assert_allclose(world.time, 0.05)
    world.pause()
    assert world.advance(elapsed=1.0) == 0


def test_on_step_runs_before_every_step():
    world = GPUWorld()
    world.add_spheres([[0, 0, 0]])
    times = []
    callback = world.on_step(lambda w: times.append(w.time))
    world.step(3)
    world.remove_on_step(callback)
    world.step(1)
    npt.assert_allclose(times, [0, world.dt, 2 * world.dt])


def test_overflow_counts_dropped_grid_insertions():
    world = GPUWorld(gravity=(0, 0, 0), bucket_size=2, substeps=1)
    world.add_spheres(np.zeros((5, 3)), radii=0.1)
    world.step(1)
    assert world.overflow >= 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"substeps": 0},
        {"friction": 1.5},
        {"restitution": -0.1},
        {"over_relaxation": 2.5},
        {"dt": 0},
        {"bucket_size": 1.5},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GPUWorld(**kwargs)


def test_invalid_spheres():
    world = GPUWorld()
    with pytest.raises(ValueError):
        world.add_spheres(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        world.add_spheres(np.zeros((2, 3)), radii=(0.1, -0.1))
    with pytest.raises(ValueError):
        world.add_spheres(np.zeros((2, 3)), radii=(0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        world.add_spheres([[np.nan, 0, 0]])
    with pytest.raises(RuntimeError):
        world.step(1)
    world.add_spheres([[0, 0, 0]])
    world.step(1)
    with pytest.raises(RuntimeError):
        world.add_spheres([[1, 0, 0]])


def test_spheres_render_from_the_gpu_buffer():
    world = GPUWorld(gravity=(0, 0, 0))
    red = world.add_spheres([[-1, 0, 0]], radii=0.5, colors=(1, 0, 0))
    blue = world.add_spheres([[1, 0, 0]], radii=0.5, colors=(0, 0, 1))
    scene = window.Scene()
    scene.add(red, blue)
    show_manager = window.ShowManager(
        scene=scene, size=(200, 200), window_type="offscreen", pixel_ratio=1
    )
    show_manager.add_physics(world)
    show_manager.render()
    show_manager.window.draw()
    image = show_manager.snapshot()
    left = image[60:140, 10:100].astype(int)
    right = image[60:140, 100:190].astype(int)
    assert np.sum((left[..., 0] > 150) & (left[..., 2] < 60)) > 500
    assert np.sum((right[..., 2] > 150) & (right[..., 0] < 60)) > 500
    assert np.sum(left[..., 2] > left[..., 0] + 50) == 0

    world.add_force(physics.uniform((0, 30.0, 0)))
    world.step(20)
    show_manager.render()
    show_manager.window.draw()
    moved = show_manager.snapshot()
    assert moved[100, 20:180, :3].max() < image[100, 20:180, :3].max()


def test_gpu_world_records_a_gif(tmp_path):
    world, spheres = _drop(count=100)
    scene = window.Scene()
    scene.add(spheres)
    show_manager = window.ShowManager(
        scene=scene, size=(160, 160), window_type="offscreen", pixel_ratio=1
    )
    show_manager.add_physics(world)
    frames = show_manager.record_physics(tmp_path / "drop.gif", duration=0.2, fps=10)
    assert frames == 2
    assert (tmp_path / "drop.gif").stat().st_size > 0
    npt.assert_allclose(world.time, 0.2, atol=1e-9)


def test_open_top_releases_spheres_above_the_rim_sideways():
    world = GPUWorld(gravity=(0, 0, 0), substeps=2)
    world.add_spheres([[0, 0.8, 0]], radii=0.1, velocities=(5, 5, 0))
    world.set_container(size=(2, 2, 2), open_top=True)
    world.step(30)
    assert world.positions[0, 0] > 2.0


def test_ground_friction_slows_sliding_at_mu_g():
    world = GPUWorld(substeps=8, friction=0.3)
    world.add_spheres([[0, 0.1, 0]], radii=0.1, velocities=(2, 0, 0))
    world.set_ground()
    world.step(30)
    expected = 2.0 - 0.3 * 9.81 * 0.5
    npt.assert_allclose(world.velocities[0, 0], expected, atol=0.05)
    assert abs(world.velocities[0, 1]) < 1e-3


def test_resting_sphere_stays_still():
    world = GPUWorld(substeps=4)
    world.add_spheres([[0, 0.1, 0]], radii=0.1)
    world.set_ground()
    world.step(120)
    npt.assert_allclose(world.velocities[0], 0, atol=1e-3)
    npt.assert_allclose(world.positions[0, 1], 0.1, atol=1e-3)


def test_restitution_bounces_back():
    world = GPUWorld(substeps=8, restitution=0.8)
    world.add_spheres([[0, 1.0, 0]], radii=0.1)
    world.set_ground()
    heights = []
    for _ in range(120):
        world.step(1)
        heights.append(world.positions[0, 1])
    first_bounce = max(heights[30:])
    assert 0.4 < first_bounce < 0.75


def test_large_static_sphere_is_an_obstacle_outside_the_grid():
    world = GPUWorld(substeps=4)
    world.add_spheres([[0, 0, 0]], radii=1.0, static=True)
    rng = np.random.default_rng(3)
    centers = rng.uniform((-0.3, 1.3, -0.3), (0.3, 2.0, 0.3), (150, 3))
    world.add_spheres(centers, radii=0.05)
    world.step(90)
    distances = np.linalg.norm(world.positions[1:], axis=1)
    assert distances.min() > 1.05 * 0.95
    assert world.overflow == 0


def test_coincident_spheres_separate():
    world = GPUWorld(gravity=(0, 0, 0), substeps=2)
    world.add_spheres([[0, 0, 0], [0, 0, 0]], radii=0.1)
    world.step(10)
    gap = np.linalg.norm(world.positions[0] - world.positions[1])
    assert gap > 0.19


def test_container_move_spin_and_rock():
    world, _ = _drop(count=20)
    world.container.move_by((1.0, 0, 0), duration=0.5)
    world.step(40)
    npt.assert_allclose(world.container.center, (1.0, 1.0, 0), atol=1e-6)
    world.container.spin(axis=(0, 1, 0), speed=np.pi, duration=1.0)
    world.step(70)
    npt.assert_allclose(world.container.rotation[0, 0], -1.0, atol=1e-6)
    world.container.rock(angle=0.2, frequency=1.0, duration=1.0)
    world.step(70)
    assert not world.container.is_moving
    npt.assert_allclose(world.container.rotation[0, 0], -1.0, atol=1e-6)


def test_container_needs_a_rotation_matrix():
    world = GPUWorld()
    with pytest.raises(ValueError):
        world.set_container(rotation=np.diag((2.0, 1.0, 1.0)))


def test_warns_when_moving_radii_vary_a_lot(caplog):
    world = GPUWorld()
    world.add_spheres(np.arange(30).reshape(10, 3), radii=0.1)
    world.add_spheres([[50, 0, 0]], radii=1.0)
    world.step(1)
    assert "median radius" in caplog.text


def test_spheres_outside_an_open_box_are_left_alone():
    world = GPUWorld(substeps=4)
    world.add_spheres(
        [[3, 2, 0], [0.9, 1.5, 0]], radii=0.1, velocities=[(0, 0, 0), (4, 0, 0)]
    )
    world.set_container(size=(2, 2, 2), open_top=True)
    world.set_ground(offset=-1.0)
    world.step(90)
    positions = world.positions
    npt.assert_allclose(positions[0, [0, 2]], (3, 0), atol=1e-4)
    assert positions[1, 0] > 1.0


def test_buffer_sizing_options_are_fixed_once_started():
    world = GPUWorld()
    world.add_spheres([[0, 0, 0]])
    world.substeps = 3
    world.bucket_size = 16
    world.step(1)
    with pytest.raises(RuntimeError):
        world.substeps = 4
    with pytest.raises(RuntimeError):
        world.bucket_size = 32
    world.iterations = 3
    world.step(1)
    assert world.substeps == 3 and world.bucket_size == 16


def test_explosion_inside_a_pile_stays_finite_and_capped():
    world, _ = _drop(count=300)
    world.step(120)
    world.explode((0, 0.2, 0), strength=30.0, radius=2.0)
    world.step(5)
    speeds = np.linalg.norm(world.velocities, axis=1)
    assert np.all(np.isfinite(world.positions))
    assert speeds.max() <= world.max_speed * (1 + 1e-5)
    assert speeds.mean() > 1.0


def test_pile_moves_with_a_sliding_container():
    world, _ = _drop(count=150)
    world.step(150)
    before = world.positions.mean(0)
    world.container.move_by((1.0, 0, 0), duration=3.0)
    world.step(200)
    positions = world.positions
    assert positions[:, 0].mean() - before[0] > 0.7
    assert np.all(np.abs(positions[:, [0, 2]] - (1.0, 0.0)) <= 0.9 + 1e-3)


def test_fast_shake_keeps_small_spheres_inside():
    world = GPUWorld(substeps=6)
    rng = np.random.default_rng(5)
    world.add_spheres(rng.uniform(-0.45, 0.45, (1500, 3)), radii=0.01)
    box = world.set_container(size=(1, 1, 1))
    box.shake(amplitude=0.5, frequency=2.0, duration=1.0)
    for _ in range(4):
        world.step(15)
        local = world.positions - box.center
        assert np.all(np.abs(local) <= 0.5 + 1e-3)


def test_spheres_outside_a_closed_box_are_left_alone():
    world = GPUWorld(gravity=(0, 0, 0))
    world.add_spheres([[1.2, 0, 0], [0, 1.3, 0]], radii=0.1)
    world.set_container(size=(2, 2, 2))
    world.step(10)
    npt.assert_allclose(world.positions, [[1.2, 0, 0], [0, 1.3, 0]], atol=1e-6)


def test_static_spheres_ignore_given_velocities():
    world = GPUWorld()
    world.add_spheres([[0, 0, 0]], radii=0.5, static=True, velocities=(3, 0, 0))
    world.step(5)
    npt.assert_allclose(world.velocities[0], 0)
    npt.assert_allclose(world.positions[0], 0)


@pytest.mark.parametrize(
    "name, value",
    [
        ("iterations", 0),
        ("friction", 5.0),
        ("restitution", 2.0),
        ("over_relaxation", 0.5),
        ("max_speed", -1.0),
    ],
)
def test_setting_invalid_options_is_rejected(name, value):
    world = GPUWorld()
    with pytest.raises(ValueError):
        setattr(world, name, value)


def test_friction_between_pressed_spheres_stops_their_slip():
    for friction, stops in ((1.0, True), (0.0, False)):
        world = GPUWorld(gravity=(-9.81, 0, 0), substeps=4, friction=friction)
        world.add_spheres([[0, 0, 0]], radii=0.1, static=True)
        world.add_spheres([[0.2, 0, 0]], radii=0.1, velocities=(0, 0, 1.0))
        world.step(12)
        slip = abs(world.velocities[1, 2])
        assert (slip < 0.05) == stops


@pytest.mark.parametrize("seed", [0, 1])
def test_friction_between_spheres_holds_a_heap_without_creeping(seed):
    heights = {}
    for friction in (0.0, 1.0):
        world = GPUWorld(substeps=4, friction=friction)
        rng = np.random.default_rng(seed)
        centers = rng.uniform((-0.3, 0.05, -0.3), (0.3, 2.0, 0.3), (600, 3))
        world.add_spheres(centers, radii=0.05)
        world.set_ground()
        world.step(240)
        heights[friction] = world.positions[:, 1].max()
        if friction:
            assert np.linalg.norm(world.velocities, axis=1).mean() < 0.05
    assert heights[1.0] > heights[0.0] * 1.2
