"""
=================================
Chaos in a Cloud: Lorenz Ensemble
=================================

Thousands of particles start inside a ball one hundredth of a unit wide and
follow the Lorenz equations, the three-variable weather model in which Edward
Lorenz found chaos in 1963:

.. math::

    \\dot{x} = \\sigma (y - x), \\quad
    \\dot{y} = x (\\rho - z) - y, \\quad
    \\dot{z} = x y - \\beta z

For a while the particles move as one bright blob. Then the tiny differences
between them grow exponentially, and the blob is torn apart and spread over
both wings of the butterfly-shaped attractor. This is sensitive dependence on
initial conditions, which is why weather cannot be forecast far ahead.

Each particle keeps a short fading trail. Colours encode where a particle
started inside the ball, so you can watch neighbours separate. Drag the
:math:`\\rho` slider: below about 24.74 the particles settle onto one of two
fixed points, above it they become chaotic. Press *Reseed* to start a new
ball.

The whole simulation is vectorised NumPy. The actors are created once, and
each frame writes new positions into their GPU buffers in place.
"""

###############################################################################
# Import the modules we need.

import numpy as np

from fury import actor, ui, window

###############################################################################
# Simulation parameters. ``SIGMA`` and ``BETA`` are Lorenz's classic values.
# ``SEED_RADIUS`` is the radius of the starting ball, and each frame advances
# the system by ``STEPS_PER_FRAME * DT`` time units.

N_PARTICLES = 3000
TRAIL_LENGTH = 24
SIGMA = 10.0
BETA = 8.0 / 3.0
INITIAL_RHO = 28.0
DT = 0.005
STEPS_PER_FRAME = 6
SEED_RADIUS = 0.01
CENTER = np.array([0.0, 0.0, 25.0])
BACKGROUND = np.array([0.02, 0.02, 0.05])

rng = np.random.default_rng(1963)

###############################################################################
# The Lorenz vector field and a fourth-order Runge-Kutta step. Both work on an
# array of shape ``(N, 3)``, so all particles advance in a single call.


def lorenz(points, rho):
    """
    Evaluate the Lorenz vector field.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Particle positions.
    rho : float
        The Rayleigh number of the system.

    Returns
    -------
    ndarray, shape (N, 3)
        Time derivatives at each position.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack(
        (SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z),
        axis=1,
    )


def rk4_step(points, rho, dt):
    """
    Advance all particles by one Runge-Kutta step.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Particle positions.
    rho : float
        The Rayleigh number of the system.
    dt : float
        Time step.

    Returns
    -------
    ndarray, shape (N, 3)
        Positions after the step.
    """
    k1 = lorenz(points, rho)
    k2 = lorenz(points + 0.5 * dt * k1, rho)
    k3 = lorenz(points + 0.5 * dt * k2, rho)
    k4 = lorenz(points + dt * k3, rho)
    return points + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


###############################################################################
# Seeds are sampled uniformly inside a small ball around a point that already
# lies on the attractor. A gradient palette, indexed by each seed's offset along
# one direction, colours the particles, so particles that start next to each
# other share a colour.

PALETTE = np.array(
    [
        [0.25, 0.35, 1.00],
        [0.10, 0.85, 1.00],
        [0.55, 1.00, 0.45],
        [1.00, 0.80, 0.20],
        [1.00, 0.30, 0.45],
    ]
)


def palette_colors(values):
    """
    Map values in [0, 1] to colours on the gradient palette.

    Parameters
    ----------
    values : ndarray, shape (N,)
        Values between 0 and 1.

    Returns
    -------
    ndarray, shape (N, 3)
        RGB colours.
    """
    stops = np.linspace(0.0, 1.0, len(PALETTE))
    return np.stack(
        [np.interp(values, stops, PALETTE[:, channel]) for channel in range(3)],
        axis=1,
    )


def sample_ball(center, radius, count):
    """
    Sample points uniformly inside a ball.

    Parameters
    ----------
    center : ndarray, shape (3,)
        Centre of the ball.
    radius : float
        Radius of the ball.
    count : int
        Number of points.

    Returns
    -------
    ndarray, shape (count, 3)
        Sampled points.
    """
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(count) ** (1.0 / 3.0)
    return center + directions * radii[:, None]


def point_on_attractor(rho):
    """
    Find a point on the attractor by integrating one particle for a while.

    Parameters
    ----------
    rho : float
        The Rayleigh number of the system.

    Returns
    -------
    ndarray, shape (3,)
        A point on the attractor.
    """
    point = rng.normal(size=(1, 3)) + np.array([1.0, 1.0, 20.0])
    for _ in range(3000):
        point = rk4_step(point, rho, DT)
    return point[0]


###############################################################################
# All mutable simulation state lives in one dictionary that the callbacks
# share.

state = {"rho": INITIAL_RHO, "time": 0.0}


def reseed():
    """Place every particle in a new small ball and reset its trail."""
    center = point_on_attractor(state["rho"])
    state["points"] = sample_ball(center, SEED_RADIUS, N_PARTICLES)
    state["trails"] = np.repeat(state["points"][:, None, :], TRAIL_LENGTH, axis=1)
    state["time"] = 0.0


reseed()
offsets = state["points"] - state["points"].mean(axis=0)
seed_colors = palette_colors(
    (offsets[:, 0] - offsets[:, 0].min()) / np.ptp(offsets[:, 0])
)

###############################################################################
# Trail colours fade from the particle colour into the background colour. The
# fade depends only on a vertex's age, so the colour buffer is written once and
# only the positions change per frame.

fade = (1.0 - np.linspace(0.0, 1.0, TRAIL_LENGTH)) ** 1.6
trail_colors = (
    fade[None, :, None] * seed_colors[:, None, :]
    + (1.0 - fade)[None, :, None] * BACKGROUND
)

###############################################################################
# Build the scene: one line actor holding every trail, one point actor for the
# particle heads, and a faint guide showing the two fixed points.

scene = window.Scene()
scene.background = tuple(BACKGROUND)

trails_actor = actor.line(
    list(state["trails"] - CENTER), colors=trail_colors, enable_picking=False
)
heads_actor = actor.point(
    state["points"] - CENTER,
    colors=np.clip(seed_colors * 1.25, 0.0, 1.0),
    size=4.0,
    material="gaussian",
    enable_picking=False,
)
scene.add(trails_actor, heads_actor)

###############################################################################
# Write the simulation state into the actors' existing GPU buffers. The trail
# buffer stores each trail followed by a NaN separator, so we fill a padded
# ``(N, TRAIL_LENGTH + 1, 3)`` view and drop the final separator.

trail_buffer = np.full((N_PARTICLES, TRAIL_LENGTH + 1, 3), np.nan, dtype=np.float32)


def upload_positions():
    """Copy particle and trail positions into the actor buffers."""
    trail_buffer[:, :TRAIL_LENGTH] = state["trails"] - CENTER
    trails_actor.geometry.positions.data[:] = trail_buffer.reshape(-1, 3)[:-1]
    trails_actor.geometry.positions.update_full()

    heads_actor.geometry.positions.data[:] = state["points"] - CENTER
    heads_actor.geometry.positions.update_full()


###############################################################################
# Overlay: a title, a live readout of the ensemble spread, the rho slider and
# the reseed button. The spread is the root-mean-square distance of the
# particles from their mean. It grows roughly exponentially until it saturates
# at the size of the attractor.

title = ui.TextBlock2D(
    text="Lorenz ensemble: 3000 particles, one tiny starting ball",
    position=(20, 20),
    font_size=18,
    color=(0.95, 0.95, 1.0),
    bold=True,
    dynamic_bbox=True,
)
readout = ui.TextBlock2D(
    text="",
    position=(20, 50),
    font_size=15,
    color=(0.75, 0.8, 0.95),
    dynamic_bbox=True,
)
rho_slider = ui.LineSlider2D(
    position=(90, 755),
    initial_value=INITIAL_RHO,
    min_value=10.0,
    max_value=40.0,
    length=260,
    text_template="rho {value:.1f}",
)
reseed_button = ui.TextButton2D(
    "Reseed",
    position=(480, 735),
    size=(110, 34),
    font_size=16,
)
scene.add(title, readout, rho_slider, reseed_button)


def on_rho_change(slider):
    """Use the slider value as the new rho."""
    state["rho"] = slider.value


def on_reseed(button):
    """Start again from a new small ball."""
    reseed()
    upload_positions()


rho_slider.on_change = on_rho_change
reseed_button.on_clicked = on_reseed

###############################################################################
# Each frame advances the particles, shifts the trails by one sample, slowly
# orbits the camera, and updates the readout.

show_manager = window.ShowManager(scene=scene, size=(800, 800), title="Lorenz Ensemble")
camera = show_manager.screens[0].camera
orbit = {"angle": 0.6}


def place_camera():
    """Put the camera on a slow circular orbit around the attractor."""
    distance = 85.0
    camera.local.position = (
        distance * np.cos(orbit["angle"]),
        distance * np.sin(orbit["angle"]),
        18.0,
    )
    camera.world.reference_up = (0, 0, 1)
    camera.look_at((0.0, 0.0, 0.0))


def step_simulation():
    """Advance the ensemble by one frame and refresh the view."""
    points = state["points"]
    for _ in range(STEPS_PER_FRAME):
        points = rk4_step(points, state["rho"], DT)
    state["points"] = points
    state["trails"][:, 1:] = state["trails"][:, :-1]
    state["trails"][:, 0] = points
    state["time"] += STEPS_PER_FRAME * DT
    upload_positions()

    spread = np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    readout.message = (
        f"t = {state['time']:5.2f}    spread = {spread:9.2e}    "
        f"rho = {state['rho']:.1f}"
    )

    orbit["angle"] += 0.002
    place_camera()


place_camera()
upload_positions()
show_manager.register_callback(step_simulation, 1 / 60, True, "lorenz_step")
show_manager.start()
