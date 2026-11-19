"""Tests for stepping physics with the render loop."""

import os
import subprocess
import sys

from PIL import Image
import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, window
from fury.physics import Sphere, World


def test_advance_runs_fixed_steps_and_keeps_the_remainder():
    world = World(dt=0.01)
    world.add_bodies([[0, 0, 0]], Sphere(0.5))
    npt.assert_equal(world.advance(elapsed=0.025), 2)
    npt.assert_allclose(world._accumulator, 0.005, atol=1e-12)
    npt.assert_equal(world.advance(elapsed=0.006), 1)


def test_advance_without_elapsed_runs_one_step():
    world = World()
    npt.assert_equal(world.advance(), 1)
    npt.assert_equal(world.step_count, 1)


def test_advance_is_independent_of_frame_rate():
    def run(frame):
        world = World(dt=0.01, max_steps_per_frame=100)
        ball = world.add_bodies([[0, 50.0, 0]], Sphere(0.5))
        for _ in range(int(round(1.0 / frame))):
            world.advance(elapsed=frame)
        return ball.positions[0, 1], world.step_count

    fast, fast_steps = run(0.005)
    slow, slow_steps = run(0.05)
    npt.assert_equal(fast_steps, 100)
    npt.assert_equal(slow_steps, 100)
    npt.assert_allclose(fast, slow)


def test_slow_machine_gets_slow_motion_instead_of_a_backlog():
    world = World(dt=0.01, max_steps_per_frame=2)
    npt.assert_equal(world.advance(elapsed=1.0), 2)
    npt.assert_equal(world._accumulator, 0.0)


def test_negative_elapsed_rejected():
    with pytest.raises(ValueError):
        World().advance(elapsed=-1.0)


def test_invalid_max_steps():
    with pytest.raises(ValueError):
        World(max_steps_per_frame=0)


def test_paused_world_does_not_advance():
    world = World()
    world.pause()
    npt.assert_equal(world.advance(elapsed=1.0), 0)
    npt.assert_equal(world.step_count, 0)
    world.resume()
    assert world.advance() == 1


def test_on_step_runs_before_every_step_and_can_be_removed():
    world = World()
    calls = []

    @world.on_step
    def record(w):
        calls.append(w.step_count)

    world.step(3)
    npt.assert_array_equal(calls, [0, 1, 2])
    world.remove_on_step(record)
    world.step(1)
    npt.assert_equal(len(calls), 3)


def test_interpolated_sync_blends_positions():
    world = World(dt=0.1, gravity=(0, 0, 0), max_steps_per_frame=5)
    ball = actor.sphere(np.zeros((1, 3)), radii=0.5)
    bodies = world.add(ball)
    bodies.velocities = (1.0, 0, 0)
    world.advance(elapsed=0.15)
    world.sync()
    npt.assert_allclose(bodies.positions[0, 0], 0.1)
    npt.assert_allclose(ball.billboard_centers[0, 0], 0.05, atol=1e-6)


def test_interpolation_can_be_disabled():
    world = World(dt=0.1, gravity=(0, 0, 0), interpolate=False)
    ball = actor.sphere(np.zeros((1, 3)), radii=0.5)
    bodies = world.add(ball)
    bodies.velocities = (1.0, 0, 0)
    world.advance(elapsed=0.15)
    world.sync()
    npt.assert_allclose(ball.billboard_centers[0, 0], 0.1, atol=1e-6)


def test_setting_state_discards_stale_interpolation():
    world = World(dt=0.1, gravity=(0, 0, 0))
    ball = actor.sphere(np.zeros((1, 3)), radii=0.5)
    bodies = world.add(ball)
    world.advance(elapsed=0.15)
    bodies.positions = (9, 0, 0)
    world.sync()
    npt.assert_allclose(ball.billboard_centers[0, 0], 9.0)


def _scene():
    scene = window.Scene()
    ball = actor.sphere(np.array([[0.0, 2.0, 0.0]]), radii=0.5, colors=(1, 0, 0))
    scene.add(ball)
    world = World()
    bodies = world.add(ball)
    show = window.ShowManager(scene=scene, size=(64, 64), window_type="offscreen")
    return show, world, bodies, ball


def test_add_physics_rejects_other_objects():
    show, *_ = _scene()
    with pytest.raises(TypeError):
        show.add_physics(object())


def test_draw_advances_one_step_offscreen():
    show, world, bodies, ball = _scene()
    show.add_physics(world)
    show.add_physics(world)
    show._draw_function()
    show._draw_function()
    npt.assert_equal(world.step_count, 2)
    npt.assert_allclose(ball.billboard_centers[0], bodies.positions[0], atol=1e-6)


def test_remove_physics_stops_stepping():
    show, world, *_ = _scene()
    show.add_physics(world)
    show.remove_physics(world)
    show.remove_physics(world)
    show._draw_function()
    npt.assert_equal(world.step_count, 0)


def test_record_physics_writes_a_gif(tmp_path):
    show, world, *_ = _scene()
    show.add_physics(world)
    path = tmp_path / "fall.gif"
    frames = show.record_physics(path, duration=0.5, fps=20)
    npt.assert_equal(frames, 10)
    npt.assert_equal(world.step_count, 30)
    with Image.open(path) as gif:
        npt.assert_equal(gif.n_frames, 10)
        first = np.asarray(gif.convert("RGB")).copy()
        gif.seek(9)
        last = np.asarray(gif.convert("RGB")).copy()
    assert not np.array_equal(first, last)


@pytest.mark.parametrize(
    "fname, kwargs",
    [
        ("x.mp4", {}),
        ("x.gif", {"duration": 0}),
        ("x.gif", {"fps": 0}),
    ],
)
def test_record_physics_invalid_arguments(tmp_path, fname, kwargs):
    show, world, *_ = _scene()
    show.add_physics(world)
    with pytest.raises(ValueError):
        show.record_physics(tmp_path / fname, **kwargs)


def test_record_physics_needs_a_world(tmp_path):
    show, *_ = _scene()
    with pytest.raises(ValueError, match="add_physics"):
        show.record_physics(tmp_path / "x.gif")


CAPTURE_SCRIPT = """
import numpy as np
from fury import actor, window
from fury.physics import World

scene = window.Scene()
ball = actor.sphere(np.array([[0.0, 2.0, 0.0]]), radii=0.5, colors=(1, 0, 0))
scene.add(ball)
world = World()
world.add(ball)
show = window.ShowManager(
    scene=scene, size=(64, 64), window_type=WINDOW, title="physics capture"
)
show.add_physics(world)
show.start()
print("steps", world.step_count)
"""


@pytest.mark.parametrize("window_type", ["offscreen", "default"])
def test_offscreen_capture_records_physics_only_scenes(tmp_path, window_type):
    environment = dict(
        os.environ,
        FURY_OFFSCREEN="1",
        FURY_RECORD_ANIMATION="1",
        FURY_OFFSCREEN_MAX_FRAMES="4",
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    script = f"WINDOW = {window_type!r}\n" + CAPTURE_SCRIPT
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    with Image.open(tmp_path / "physics_capture.gif") as gif:
        npt.assert_equal(gif.n_frames, 4)
    assert "steps 4" in result.stdout


@pytest.mark.parametrize("fps", [20, 25, 30, 100])
def test_record_physics_covers_the_simulated_duration(tmp_path, fps):
    show, world, *_ = _scene()
    show.add_physics(world)
    path = tmp_path / "time.gif"
    show.record_physics(path, duration=1.0, fps=fps)
    npt.assert_allclose(world.time, 1.0, atol=world.dt / 2)
    with Image.open(path) as gif:
        total = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            total += gif.info["duration"]
    npt.assert_allclose(total, 1000, atol=10)


def test_record_physics_leaves_a_paused_world_alone(tmp_path):
    show, world, *_ = _scene()
    show.add_physics(world)
    world.pause()
    show.record_physics(tmp_path / "still.gif", duration=0.5, fps=10)
    npt.assert_equal(world.step_count, 0)


@pytest.mark.parametrize("elapsed", [np.nan, np.inf])
def test_non_finite_elapsed_rejected_without_breaking_the_world(elapsed):
    world = World()
    with pytest.raises(ValueError):
        world.advance(elapsed=elapsed)
    npt.assert_equal(world.advance(elapsed=world.dt), 1)


def test_setting_one_set_keeps_interpolating_the_others():
    world = World(dt=0.1, gravity=(0, 0, 0))
    first = actor.sphere(np.zeros((1, 3)), radii=0.5)
    second = actor.sphere(np.array([[5.0, 0, 0]]), radii=0.5)
    moving = world.add(first, velocities=(1.0, 0, 0))
    pushed = world.add(second)
    world.advance(elapsed=0.15)
    pushed.velocities = (2.0, 0, 0)
    world.sync()
    npt.assert_allclose(first.billboard_centers[0, 0], 0.05, atol=1e-6)
    npt.assert_allclose(second.billboard_centers[0, 0], 5.0, atol=1e-6)
    del moving


def test_setting_velocity_keeps_a_moving_body_smooth():
    world = World(dt=0.1, gravity=(0, 0, 0))
    ball = actor.sphere(np.zeros((1, 3)), radii=0.5)
    bodies = world.add(ball, velocities=(1.0, 0, 0))
    world.advance(elapsed=0.15)
    bodies.velocities = (1.0, 0, 0)
    bodies[0].velocity = (1.0, 0, 0)
    world.sync()
    npt.assert_allclose(ball.billboard_centers[0, 0], 0.05, atol=1e-6)
