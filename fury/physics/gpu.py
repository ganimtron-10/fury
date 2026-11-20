"""
Sphere physics that runs entirely on the GPU.

The CPU :class:`~fury.physics.World` handles a few thousand mixed bodies with
accurate stacking. :class:`GPUWorld` trades shape variety for scale: it
simulates only spheres, but keeps every sphere's state in GPU buffers and
steps them with WGSL compute kernels, so hundreds of thousands to millions of
spheres move every frame. The renderer draws straight from the same buffers,
so nothing is copied back to the CPU while the simulation runs.

The solver is position-based dynamics: each substep predicts positions from
velocities and forces, sorts spheres into a hashed uniform grid, pushes
overlapping spheres apart with Jacobi iterations, and derives the new velocities
from how far each sphere moved.
"""

import logging
import math

import numpy as np
from scipy.spatial.transform import Rotation

from fury.actor import Mesh
from fury.colormap import normalize_colors
from fury.lib import (
    Buffer,
    BufferUsage,
    Geometry,
    ShaderStage,
    ensure_wgpu_object,
    gfx_wgpu,
    load_wgsl,
    register_wgpu_render_function,
)
from fury.material import BillboardSphereMaterial
from fury.physics.forces import (
    Relaxation,
    _Attractor,
    _Uniform,
    _Vortex,
    _number,
    _vector,
)
from fury.physics.kinematics import _Rock, _Shake, _Slide, _Spin, _Tilt
from fury.shader import GPUSpheresShader

_WORKGROUP_SIZE = 256
_MAX_DISPATCH = 65535
_PARAMS_FLOATS = 48
_UNIFORM_ALIGNMENT = 256
_FORCE_FLOATS = 12
_MAX_FORCES = 64
_MAX_OBSTACLES = 1024
_UNIFORM, _RELAXATION, _ATTRACTOR, _VORTEX, _BLAST = 1, 2, 3, 4, 5
_KERNELS = ("blast", "clear", "predict", "insert", "solve", "apply", "finalize")


def _next_power_of_two(value):
    """
    Return the smallest power of two not below a value.

    Parameters
    ----------
    value : int
        A positive integer.

    Returns
    -------
    int
        The power of two.
    """
    return 1 << max(int(value) - 1, 1).bit_length()


def _per_sphere(value, count, width, name):
    """
    Broadcast one value or one value per sphere to a float array.

    Parameters
    ----------
    value : float or array_like
        A single value or one per sphere.
    count : int
        Number of spheres.
    width : int
        Values per sphere; 1 for scalars.
    name : str
        Parameter name for error messages.

    Returns
    -------
    ndarray, shape (count,) or (count, width)
        The broadcast values.
    """
    array = np.asarray(value, dtype=np.float64)
    shape = (count,) if width == 1 else (count, width)
    try:
        array = np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(
            f"{name} must be one value or one value per sphere."
        ) from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


class GPUSpheres(Mesh):
    """
    Spheres drawn from a :class:`GPUWorld`'s buffers.

    Create these with :meth:`GPUWorld.add_spheres`, not directly.

    Parameters
    ----------
    world : GPUWorld
        The world that owns the sphere records.
    first : int
        Index of the first sphere in the world.
    count : int
        Number of spheres.
    colors : ndarray, shape (count, 4)
        RGBA colors.
    bounds : ndarray, shape (2, 3)
        Lower and upper corners enclosing the initial spheres.
    """

    def __init__(self, world, first, count, colors, bounds):
        super().__init__()
        self.first = first
        self.count = count
        self._bounds = bounds
        self.geometry = Geometry(spheres=world._render_positions(), colors=colors)
        self.material = BillboardSphereMaterial(color_mode="vertex", side="both")

    def get_bounding_box(self):
        """
        Return the bounds of the spheres where they started.

        The live positions stay on the GPU, so the bounds are not updated as
        the spheres move.

        Returns
        -------
        ndarray, shape (2, 3)
            Lower and upper corners.
        """
        return self._bounds

    def _use_positions(self, positions):
        """
        Point this actor at a new world position buffer.

        Parameters
        ----------
        positions : Buffer
            Positions of every sphere in the world.
        """
        self.geometry.spheres = positions


@register_wgpu_render_function(GPUSpheres, BillboardSphereMaterial)
def register_gpu_spheres_render_function(wobject):
    """
    Register the pipeline that draws GPU-simulated spheres.

    Parameters
    ----------
    wobject : GPUSpheres
        The spheres to draw.

    Returns
    -------
    tuple
        The configured :class:`~fury.shader.GPUSpheresShader`.
    """
    return (GPUSpheresShader(wobject),)


class GPUContainer:
    """
    An oriented box that holds the spheres of a :class:`GPUWorld` inside it.

    Create it with :meth:`GPUWorld.set_container`. As with
    :class:`~fury.physics.Container`, the walls act only on spheres whose
    centre is inside the box: spheres outside are not affected, and over an
    open top a sphere is let go once its centre rises above the rim. Moving
    the box uses the same motions as the CPU container, with its pose blended
    across substeps.

    Parameters
    ----------
    center : ndarray, shape (3,)
        Centre of the box.
    size : ndarray, shape (3,)
        Edge lengths along the box's local axes.
    rotation : ndarray, shape (3, 3)
        Rotation from the box's local axes to world axes.
    open_top : bool
        Whether to leave out the wall on the local +Y side.
    """

    def __init__(self, center, size, rotation, open_top):
        self.center = center
        self.size = size
        self.rotation = rotation
        self.open_top = open_top
        self._motions = []
        self._time = 0.0

    @property
    def is_moving(self):
        """
        Return whether a scripted motion is running.

        Returns
        -------
        bool
            Whether a scripted motion is running.
        """
        return bool(self._motions)

    def shake(self, *, amplitude=0.5, frequency=2.0, axis=(1, 0, 0), duration=None):
        """
        Move back and forth along an axis.

        Parameters
        ----------
        amplitude : float, optional
            Largest distance from the start.
        frequency : float, optional
            Cycles per second.
        axis : array_like, shape (3,), optional
            Direction of motion.
        duration : float, optional
            Seconds to shake, rounded up to whole cycles. None shakes until
            stopped.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        return self._begin(_Shake(amplitude, frequency, axis, duration))

    def rock(self, *, axis=(0, 0, 1), angle=0.3, frequency=1.0, duration=None):
        """
        Rotate back and forth about the box centre.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        angle : float, optional
            Largest rotation, in radians.
        frequency : float, optional
            Cycles per second.
        duration : float, optional
            Seconds to rock, rounded up to whole cycles. None runs until
            stopped.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        return self._begin(_Rock(axis, angle, frequency, duration))

    def spin(self, *, axis=(0, 1, 0), speed=1.0, duration=None):
        """
        Rotate about the box centre at a constant rate.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        speed : float, optional
            Angular speed, in radians per second.
        duration : float, optional
            Seconds to spin. None runs until stopped.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        return self._begin(_Spin(axis, speed, None, duration))

    def tilt(self, *, axis=(0, 0, 1), angle=0.5, duration=1.0):
        """
        Rotate smoothly by a fixed angle about the box centre, then stop.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        angle : float, optional
            Total rotation, in radians.
        duration : float, optional
            Seconds to take.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        return self._begin(_Tilt(axis, angle, duration))

    def move_by(self, offset, *, duration=1.0):
        """
        Translate smoothly by an offset, then stop.

        Parameters
        ----------
        offset : array_like, shape (3,)
            Distance to move.
        duration : float, optional
            Seconds to take.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        return self._begin(_Slide(offset, duration))

    def stop(self):
        """
        Cancel all motions.

        Returns
        -------
        GPUContainer
            This container, so calls can be chained.
        """
        self._motions.clear()
        return self

    def _begin(self, motion):
        """
        Start a motion.

        Parameters
        ----------
        motion : Motion
            The motion to start.

        Returns
        -------
        GPUContainer
            This container.
        """
        self._motions.append(motion)
        return self

    def _advance(self, dt):
        """
        Move the box by one step of its running motions.

        Parameters
        ----------
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,)
            Linear velocity during the step.
        angular : ndarray, shape (3,)
            Angular velocity during the step.
        """
        linear = np.zeros(3)
        angular = np.zeros(3)
        running = []
        for motion in self._motions:
            if motion.started is None:
                motion.started = self._time
            start = self._time - motion.started
            end = start + dt
            if motion.duration is not None:
                if start >= motion.duration - 1e-9 * dt:
                    continue
                end = min(end, motion.duration)
            step_linear, step_angular = motion.sample(
                start, end, self.center[np.newaxis], dt
            )
            linear += np.reshape(step_linear, (-1, 3))[0]
            angular += step_angular
            running.append(motion)
        self._motions = running
        self._time += dt
        return linear, angular

    def _pose_records(self, linear, angular, dt, substeps):
        """
        Return the box pose at the end of each substep, then commit the last.

        Parameters
        ----------
        linear : ndarray, shape (3,)
            Linear velocity during the step.
        angular : ndarray, shape (3,)
            Angular velocity during the step.
        dt : float
            Step duration.
        substeps : int
            Number of substeps.

        Returns
        -------
        list of tuple
            ``(center, rotation)`` per substep.
        """
        start_center = self.center.copy()
        start_rotation = Rotation.from_matrix(self.rotation)
        poses = []
        for substep in range(1, substeps + 1):
            fraction = substep / substeps
            turn = Rotation.from_rotvec(angular * dt * fraction)
            poses.append(
                (
                    start_center + linear * dt * fraction,
                    (turn * start_rotation).as_matrix(),
                )
            )
        self.center, self.rotation = poses[-1]
        return poses


class GPUWorld:
    """
    A world of spheres simulated with WGSL compute shaders.

    Add every sphere set with :meth:`add_spheres` before the first step; the
    GPU buffers are sized then. Pass the world to
    :meth:`fury.window.ShowManager.add_physics` like a CPU world.

    Parameters
    ----------
    gravity : array_like, shape (3,), optional
        Acceleration applied to every dynamic sphere.
    dt : float, optional
        Seconds per step.
    substeps : int, optional
        Solver substeps per step. More substeps make piles stiffer and fast
        spheres less likely to pass through each other. Fixed once the world
        first steps.
    iterations : int, optional
        Contact iterations per substep.
    friction : float, optional
        Coulomb friction coefficient, from 0 to 1. Sliding slows by at most
        this fraction of the speed the contacts push with.
    restitution : float, optional
        Bounciness of contacts, from 0 (none) to 1.
    over_relaxation : float, optional
        Scale of each Jacobi contact correction, from 1 to 2. Larger values
        settle piles faster but can jitter.
    max_speed : float, optional
        Speed limit that keeps blasts from tunnelling through walls.
    bucket_size : int, optional
        Spheres one grid cell can hold. Spheres beyond it are missed by the
        contact search and counted in :attr:`overflow`; raise it when radii
        vary a lot. Fixed once the world first steps.
    max_steps_per_frame : int, optional
        Most steps :meth:`advance` runs per call, so a slow machine shows slow
        motion instead of freezing.
    """

    def __init__(
        self,
        *,
        gravity=(0.0, -9.81, 0.0),
        dt=1 / 60,
        substeps=8,
        iterations=2,
        friction=0.3,
        restitution=0.0,
        over_relaxation=1.5,
        max_speed=50.0,
        bucket_size=12,
        max_steps_per_frame=2,
    ):
        self.gravity = _vector(gravity, "gravity")
        self.dt = _number(dt, "dt", positive=True)
        self._gpu = None
        self.substeps = substeps
        self.iterations = iterations
        self.friction = friction
        self.restitution = restitution
        self.over_relaxation = over_relaxation
        self.max_speed = max_speed
        self.bucket_size = bucket_size
        self.max_steps_per_frame = self._count(
            max_steps_per_frame, "max_steps_per_frame"
        )
        self.time = 0.0
        self.paused = False
        self.container = None
        self.ground = None
        self._accumulator = 0.0
        self._records = []
        self._colors = []
        self._actors = []
        self._fields = []
        self._blasts = []
        self._step_callbacks = []
        self._positions_buffer = None

    @property
    def iterations(self):
        """
        Return the contact iterations per substep.

        Returns
        -------
        int
            Iterations per substep.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        """
        Set the contact iterations per substep.

        Parameters
        ----------
        value : int
            Iterations per substep, at least 1.
        """
        self._iterations = self._count(value, "iterations")

    @property
    def friction(self):
        """
        Return the Coulomb friction coefficient.

        Returns
        -------
        float
            Coefficient from 0 to 1.
        """
        return self._friction

    @friction.setter
    def friction(self, value):
        """
        Set the Coulomb friction coefficient.

        Parameters
        ----------
        value : float
            Coefficient from 0 to 1.
        """
        self._friction = self._fraction(value, "friction")

    @property
    def restitution(self):
        """
        Return the bounciness of contacts.

        Returns
        -------
        float
            Restitution from 0 to 1.
        """
        return self._restitution

    @restitution.setter
    def restitution(self, value):
        """
        Set the bounciness of contacts.

        Parameters
        ----------
        value : float
            Restitution from 0 to 1.
        """
        self._restitution = self._fraction(value, "restitution")

    @property
    def over_relaxation(self):
        """
        Return the scale of each Jacobi contact correction.

        Returns
        -------
        float
            Scale from 1 to 2.
        """
        return self._over_relaxation

    @over_relaxation.setter
    def over_relaxation(self, value):
        """
        Set the scale of each Jacobi contact correction.

        Parameters
        ----------
        value : float
            Scale from 1 to 2.
        """
        value = _number(value, "over_relaxation")
        if not 1.0 <= value <= 2.0:
            raise ValueError("over_relaxation must be between 1 and 2.")
        self._over_relaxation = value

    @property
    def max_speed(self):
        """
        Return the speed limit of every sphere.

        Returns
        -------
        float
            Largest speed.
        """
        return self._max_speed

    @max_speed.setter
    def max_speed(self, value):
        """
        Set the speed limit of every sphere.

        Parameters
        ----------
        value : float
            Largest speed, positive.
        """
        self._max_speed = _number(value, "max_speed", positive=True)

    @property
    def substeps(self):
        """
        Return the solver substeps per step.

        Returns
        -------
        int
            Substeps per step.
        """
        return self._substeps

    @substeps.setter
    def substeps(self, value):
        """
        Set the solver substeps per step, before the world first steps.

        Parameters
        ----------
        value : int
            Substeps per step.
        """
        self._check_not_started("substeps")
        self._substeps = self._count(value, "substeps")

    @property
    def bucket_size(self):
        """
        Return how many spheres one grid cell can hold.

        Returns
        -------
        int
            Spheres per grid cell.
        """
        return self._bucket_size

    @bucket_size.setter
    def bucket_size(self, value):
        """
        Set how many spheres one grid cell can hold, before the world first
        steps.

        Parameters
        ----------
        value : int
            Spheres per grid cell.
        """
        self._check_not_started("bucket_size")
        self._bucket_size = self._count(value, "bucket_size")

    def _check_not_started(self, name):
        """
        Reject changes to an option that sizes the GPU buffers.

        Parameters
        ----------
        name : str
            Option name for the error message.
        """
        if self._gpu is not None:
            raise RuntimeError(
                f"{name} sizes the GPU buffers; set it before the world first steps."
            )

    @staticmethod
    def _count(value, name):
        """
        Validate a positive whole number.

        Parameters
        ----------
        value : int
            The value.
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            The value.
        """
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @staticmethod
    def _fraction(value, name):
        """
        Validate a number from 0 to 1.

        Parameters
        ----------
        value : float
            The value.
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            The value.
        """
        value = _number(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
        return value

    @property
    def count(self):
        """
        Return the number of spheres.

        Returns
        -------
        int
            Number of spheres.
        """
        return sum(len(record) for record in self._records)

    def add_spheres(
        self,
        centers,
        *,
        radii=0.1,
        colors=(1.0, 1.0, 1.0),
        velocities=(0.0, 0.0, 0.0),
        masses=1.0,
        static=False,
    ):
        """
        Add spheres and return the actor that draws them.

        Parameters
        ----------
        centers : array_like, shape (N, 3)
            Sphere centres.
        radii : float or array_like, shape (N,), optional
            Sphere radii.
        colors : str, tuple or array_like, optional
            One color or one per sphere, as RGB(A) in [0, 1] or [0, 255], or
            a hex string.
        velocities : array_like, shape (3,) or (N, 3), optional
            Starting velocities.
        masses : float or array_like, shape (N,), optional
            Sphere masses. Heavier spheres are pushed less in contacts.
        static : bool, optional
            Whether the spheres stay fixed and only act as obstacles.

        Returns
        -------
        GPUSpheres
            The actor that draws these spheres; add it to a scene.
        """
        if self._gpu is not None:
            raise RuntimeError("Add every sphere before the world first steps.")
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            raise ValueError("centers must have shape (N, 3) with N > 0.")
        if not np.all(np.isfinite(centers)):
            raise ValueError("centers must be finite.")
        count = len(centers)
        radii = _per_sphere(radii, count, 1, "radii")
        if np.any(radii <= 0):
            raise ValueError("radii must be positive.")
        velocities = _per_sphere(velocities, count, 3, "velocities")
        masses = _per_sphere(masses, count, 1, "masses")
        if np.any(masses <= 0):
            raise ValueError("masses must be positive.")
        inverse_masses = np.zeros(count) if static else 1.0 / masses
        if static:
            velocities = np.zeros((count, 3))

        record = np.zeros((count, 8), dtype=np.float32)
        record[:, :3] = centers
        record[:, 3] = radii
        record[:, 4:7] = velocities
        record[:, 7] = inverse_masses
        rgba = self._colors_for(colors, count)
        first = self.count
        self._records.append(record)
        self._colors.append(rgba)
        self._resize_positions()
        bounds = np.stack(
            ((centers - radii[:, None]).min(0), (centers + radii[:, None]).max(0))
        )
        actor = GPUSpheres(self, first, count, self._all_colors(), bounds)
        self._actors.append(actor)
        for existing in self._actors:
            existing.geometry.colors = actor.geometry.colors
        return actor

    @staticmethod
    def _colors_for(colors, count):
        """
        Return one RGBA color per sphere.

        Parameters
        ----------
        colors : str, tuple or array_like
            One color or one per sphere.
        count : int
            Number of spheres.

        Returns
        -------
        ndarray, shape (count, 4)
            RGBA colors in [0, 1].
        """
        colors = np.atleast_2d(normalize_colors(colors)).astype(np.float32)
        if colors.shape[0] not in (1, count) or colors.shape[1] not in (3, 4):
            raise ValueError("colors must be one color or one per sphere.")
        rgba = np.ones((count, 4), dtype=np.float32)
        rgba[:, : colors.shape[1]] = colors
        return rgba

    def _all_colors(self):
        """
        Return the colors of every sphere in the world.

        Returns
        -------
        ndarray, shape (count, 4)
            RGBA colors.
        """
        return np.concatenate(self._colors)

    def _resize_positions(self):
        """Create a GPU position buffer sized for every sphere so far."""
        count = self.count
        self._positions_buffer = Buffer(
            nbytes=count * 16,
            nitems=count,
            format="4xf4",
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
        )
        for actor in self._actors:
            actor._use_positions(self._positions_buffer)

    def _render_positions(self):
        """
        Return the GPU buffer actors draw sphere positions from.

        Returns
        -------
        Buffer
            Centre and radius of every sphere.
        """
        return self._positions_buffer

    def set_container(
        self,
        *,
        center=(0.0, 0.0, 0.0),
        size=(4.0, 4.0, 4.0),
        rotation=None,
        open_top=False,
    ):
        """
        Hold the spheres inside an oriented box.

        Only spheres whose centre is inside the box are held; see
        :class:`GPUContainer`.

        Parameters
        ----------
        center : array_like, shape (3,), optional
            Centre of the box.
        size : array_like, shape (3,), optional
            Edge lengths along the box's local axes.
        rotation : array_like, shape (3, 3), optional
            Rotation from the box's local axes to world axes.
        open_top : bool, optional
            Whether to leave out the wall on the local +Y side.

        Returns
        -------
        GPUContainer
            The container, which can be shaken, rocked, spun or tilted.
        """
        size = _vector(size, "size")
        if np.any(size <= 0):
            raise ValueError("size must be positive.")
        if rotation is None:
            rotation = np.eye(3)
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6
        ):
            raise ValueError("rotation must be a (3, 3) rotation matrix.")
        self.container = GPUContainer(
            _vector(center, "center"), size, rotation, bool(open_top)
        )
        return self.container

    def remove_container(self):
        """Let spheres leave the box set with :meth:`set_container`."""
        self.container = None

    def set_ground(self, *, normal=(0.0, 1.0, 0.0), offset=0.0):
        """
        Keep spheres on the positive side of a plane.

        Parameters
        ----------
        normal : array_like, shape (3,), optional
            Plane normal, pointing to the side spheres stay on.
        offset : float, optional
            Plane offset along the normal from the origin.
        """
        normal = _vector(normal, "normal")
        length = np.linalg.norm(normal)
        if length < 1e-12:
            raise ValueError("normal must be non-zero.")
        self.ground = (normal / length, _number(offset, "offset"))

    def remove_ground(self):
        """Remove the plane set with :meth:`set_ground`."""
        self.ground = None

    def add_force(self, field):
        """
        Add a force field that acts on every dynamic sphere at each substep.

        Parameters
        ----------
        field : callable
            A field from :func:`~fury.physics.uniform`,
            :func:`~fury.physics.drag`, :func:`~fury.physics.wind`,
            :func:`~fury.physics.attractor` or :func:`~fury.physics.vortex`.
            :func:`~fury.physics.mutual_gravity` and custom callables need the
            CPU world.

        Returns
        -------
        callable
            The same field, for :meth:`remove_force`.
        """
        self._force_record(field)
        if len(self._fields) >= _MAX_FORCES:
            raise ValueError(f"A GPU world holds at most {_MAX_FORCES} fields.")
        self._fields.append(field)
        return field

    def remove_force(self, field):
        """
        Remove a force field.

        Parameters
        ----------
        field : callable
            A field passed to :meth:`add_force`.
        """
        if field not in self._fields:
            raise ValueError("This field was not added to the world.")
        self._fields.remove(field)

    @staticmethod
    def _force_record(field):
        """
        Encode a force field for the compute kernels.

        Parameters
        ----------
        field : callable
            A built-in force field.

        Returns
        -------
        ndarray, shape (12,)
            Kind and values, then two vectors.
        """
        record = np.zeros(_FORCE_FLOATS, dtype=np.float32)
        if isinstance(field, _Uniform):
            record[0] = _UNIFORM
            record[4:7] = field.force
        elif isinstance(field, Relaxation):
            record[0] = _RELAXATION
            record[3] = field.coefficient
            record[4:7] = field.target
        elif isinstance(field, _Attractor):
            record[:3] = (_ATTRACTOR, field.strength, field.softening)
            record[4:7] = field.point
        elif isinstance(field, _Vortex):
            record[:3] = (_VORTEX, field.strength, field.pull)
            record[4:7] = field.center
            record[8:11] = field.axis
        else:
            raise TypeError(
                "The GPU world supports uniform, drag, wind, attractor and vortex "
                "fields."
            )
        return record

    def explode(self, center, *, strength=10.0, radius=5.0):
        """
        Throw spheres away from a point at the start of the next step.

        The speed each sphere gains falls off linearly to zero at ``radius``,
        whatever its mass.

        Parameters
        ----------
        center : array_like, shape (3,)
            Centre of the blast.
        strength : float, optional
            Speed gained by a sphere at the centre.
        radius : float, optional
            Distance beyond which nothing is affected.
        """
        record = np.zeros(_FORCE_FLOATS, dtype=np.float32)
        record[:3] = (
            _BLAST,
            _number(strength, "strength"),
            _number(radius, "radius", positive=True),
        )
        record[4:7] = _vector(center, "center")
        self._blasts.append(record)

    def on_step(self, callback):
        """
        Call a function before every step.

        Parameters
        ----------
        callback : callable
            Called with the world.

        Returns
        -------
        callable
            The same callback, for :meth:`remove_on_step`.
        """
        if not callable(callback):
            raise TypeError("callback must be callable.")
        self._step_callbacks.append(callback)
        return callback

    def remove_on_step(self, callback):
        """
        Stop calling a function registered with :meth:`on_step`.

        Parameters
        ----------
        callback : callable
            A callback passed to :meth:`on_step`.
        """
        self._step_callbacks.remove(callback)

    def pause(self):
        """Stop :meth:`advance` from stepping."""
        self.paused = True

    def resume(self):
        """Let :meth:`advance` step again."""
        self.paused = False

    def advance(self, *, elapsed=None):
        """
        Move the simulation forward with the clock.

        Parameters
        ----------
        elapsed : float, optional
            Seconds since the previous call. None runs exactly one step, which
            makes offscreen rendering reproducible.

        Returns
        -------
        int
            The number of steps run.
        """
        if self.paused:
            return 0
        if elapsed is None:
            self.step(1)
            return 1
        if not 0 <= elapsed < np.inf:
            raise ValueError("elapsed must be finite and not negative.")
        self._accumulator += elapsed
        steps = min(int(self._accumulator / self.dt), self.max_steps_per_frame)
        self.step(steps)
        self._accumulator -= steps * self.dt
        if self._accumulator >= self.dt:
            self._accumulator = 0.0
        return steps

    def sync(self):
        """Upload the spheres on first use; actors already read GPU state."""
        self._ensure_gpu()

    def step(self, count):
        """
        Advance the simulation by whole steps.

        Parameters
        ----------
        count : int
            Number of steps.
        """
        if count <= 0:
            return
        gpu = self._ensure_gpu()
        for _ in range(count):
            for callback in list(self._step_callbacks):
                callback(self)
            self._write_step(gpu)
            gpu.encode_step(self.substeps, self.iterations, bool(self._blasts))
            self._blasts.clear()
            self.time += self.dt

    def _write_step(self, gpu):
        """
        Upload this step's parameters and force fields.

        Parameters
        ----------
        gpu : _GPUState
            The world's GPU resources.
        """
        records = [self._force_record(field) for field in self._fields]
        records += self._blasts
        forces = np.zeros((max(len(records), 1), _FORCE_FLOATS), dtype=np.float32)
        if records:
            forces[: len(records)] = records
        gpu.write_forces(forces)

        poses = None
        if self.container is not None:
            linear, angular = self.container._advance(self.dt)
            poses = self.container._pose_records(
                linear, angular, self.dt, self.substeps
            )
            poses = [(center, rotation, linear, angular) for center, rotation in poses]
        params = np.zeros((self.substeps, _PARAMS_FLOATS), dtype=np.float32)
        for substep in range(self.substeps):
            pose = None if poses is None else poses[substep]
            params[substep] = self._params(gpu, len(records), pose)
        gpu.write_params(params)

    def _params(self, gpu, force_count, pose):
        """
        Pack the uniform record of one substep.

        Parameters
        ----------
        gpu : _GPUState
            The world's GPU resources.
        force_count : int
            Number of encoded force records.
        pose : tuple or None
            Container ``(center, rotation, linear, angular)``: its pose at
            the end of the substep and its velocity during it.

        Returns
        -------
        ndarray, shape (48,)
            The record, with integer fields stored as raw bits.
        """
        record = np.zeros(_PARAMS_FLOATS, dtype=np.float32)
        record[0:3] = self.gravity
        record[3] = self.dt / self.substeps
        record[4] = gpu.cell_size
        record[5] = 1.0 / gpu.cell_size
        record[6] = self.friction
        record[7] = self.max_speed
        record.view(np.uint32)[8:12] = (
            gpu.count,
            gpu.cell_count - 1,
            gpu.bucket_size,
            force_count,
        )
        if pose is not None:
            center, rotation, linear, angular = pose
            record[12:15] = center
            record[15] = 1.0
            record[16:19] = self.container.size / 2.0
            record[19] = float(self.container.open_top)
            record[20:23] = rotation[:, 0]
            record[24:27] = rotation[:, 1]
            record[28:31] = rotation[:, 2]
        if self.ground is not None:
            record[32:35] = self.ground[0]
            record[35] = self.ground[1]
            record[36] = 1.0
        record[37] = self.over_relaxation
        record[38] = self.restitution
        record[43] = gpu.obstacle_count
        if pose is not None:
            record[40:43] = linear
            record[44:47] = angular
        return record

    def _ensure_gpu(self):
        """
        Create and fill the GPU resources on first use.

        Returns
        -------
        _GPUState
            The world's GPU resources.
        """
        if self._gpu is None:
            if not self._records:
                raise RuntimeError("Add spheres before stepping the world.")
            state = np.concatenate(self._records)
            self._gpu = _GPUState(
                state,
                self._positions_buffer,
                bucket_size=self.bucket_size,
                substeps=self.substeps,
            )
        return self._gpu

    @property
    def positions(self):
        """
        Read the sphere centres back from the GPU.

        This waits for the GPU, so call it for checks, not every frame.

        Returns
        -------
        ndarray, shape (N, 3)
            Sphere centres.
        """
        return self._ensure_gpu().read_positions()[:, :3].copy()

    @property
    def velocities(self):
        """
        Read the sphere velocities back from the GPU.

        This waits for the GPU, so call it for checks, not every frame.

        Returns
        -------
        ndarray, shape (N, 3)
            Sphere velocities.
        """
        return self._ensure_gpu().read_velocities()[:, :3].copy()

    @property
    def overflow(self):
        """
        Return how many grid insertions were dropped since the world started.

        A non-zero count means some contacts were missed; create the world
        with a larger ``bucket_size``.

        Returns
        -------
        int
            Dropped insertions.
        """
        return self._ensure_gpu().read_overflow()


class _GPUState:
    """
    GPU buffers, pipelines and dispatch of one :class:`GPUWorld`.

    Parameters
    ----------
    state : ndarray, shape (N, 8)
        Centre, radius, velocity and inverse mass of every sphere.
    positions : Buffer
        Render buffer that receives centre and radius records.
    bucket_size : int
        Spheres one grid cell can hold.
    substeps : int
        Substeps per step; one uniform record is kept for each.
    """

    def __init__(self, state, positions, *, bucket_size, substeps):
        device = gfx_wgpu.get_shared().device
        self.device = device
        self.count = len(state)
        self.cell_size, obstacles = _grid_layout(state)
        self.obstacle_count = len(obstacles)
        self.cell_count = _next_power_of_two(max(2 * self.count, 8192))
        self.bucket_size = bucket_size
        self.substeps = substeps
        limits = device.limits
        largest = self.cell_count * bucket_size * 4
        if largest > limits["max-storage-buffer-binding-size"]:
            raise ValueError(
                "The hashed grid is larger than this GPU allows; lower "
                "bucket_size or the number of spheres."
            )

        storage = BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC
        self.positions = ensure_wgpu_object(positions)
        sphere_bytes = self.count * 16
        self.velocities = device.create_buffer(size=sphere_bytes, usage=storage)
        self.previous = device.create_buffer(size=sphere_bytes, usage=storage)
        self.corrected = device.create_buffer(size=sphere_bytes, usage=storage)
        self.contacts = device.create_buffer(size=sphere_bytes, usage=storage)
        self.surroundings = device.create_buffer(size=sphere_bytes, usage=storage)
        self.cell_counts = device.create_buffer(size=self.cell_count * 4, usage=storage)
        self.cell_items = device.create_buffer(
            size=self.cell_count * bucket_size * 4, usage=storage
        )
        self.stats = device.create_buffer(size=16, usage=storage)
        self.obstacles = device.create_buffer(
            size=max(obstacles.nbytes, 16),
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
        )
        if len(obstacles):
            device.queue.write_buffer(self.obstacles, 0, obstacles)
        self.params = device.create_buffer(
            size=substeps * _UNIFORM_ALIGNMENT,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        )
        self.forces = None
        self.forces_size = 0

        queue = device.queue
        queue.write_buffer(self.positions, 0, np.ascontiguousarray(state[:, :4]))
        queue.write_buffer(self.velocities, 0, np.ascontiguousarray(state[:, 4:]))

        source = load_wgsl("physics_gpu_compute.wgsl", package_name="fury.wgsl")
        source = source.replace("{{ workgroup_size }}", str(_WORKGROUP_SIZE))
        module = device.create_shader_module(code=source)
        self.layout = device.create_bind_group_layout(entries=_layout_entries())
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self.layout]
        )
        self.pipelines = {
            name: device.create_compute_pipeline(
                layout=pipeline_layout,
                compute={"module": module, "entry_point": name},
            )
            for name in _KERNELS
        }
        self.bind_groups = None

    def write_forces(self, forces):
        """
        Upload the force records, growing the buffer when needed.

        Parameters
        ----------
        forces : ndarray, shape (F, 12)
            Encoded force fields and pending blasts.
        """
        data = np.ascontiguousarray(forces, dtype=np.float32)
        if data.nbytes > self.forces_size:
            self.forces_size = max(data.nbytes, 48 * 8)
            self.forces = self.device.create_buffer(
                size=self.forces_size,
                usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            )
            self.bind_groups = None
        self.device.queue.write_buffer(self.forces, 0, data)

    def write_params(self, params):
        """
        Upload one uniform record per substep.

        Parameters
        ----------
        params : ndarray, shape (substeps, 48)
            The records.
        """
        padded = np.zeros((self.substeps, _UNIFORM_ALIGNMENT // 4), dtype=np.float32)
        padded[:, :_PARAMS_FLOATS] = params
        self.device.queue.write_buffer(self.params, 0, padded)

    def _entries(self, substep):
        """
        Return the bind group entries for one substep.

        Parameters
        ----------
        substep : int
            Index of the substep, which selects its uniform record.

        Returns
        -------
        list of dict
            Bind group entries.
        """
        buffers = (
            self.positions,
            self.velocities,
            self.previous,
            self.corrected,
            self.contacts,
            self.cell_counts,
            self.cell_items,
            self.stats,
            self.forces,
            self.obstacles,
            self.surroundings,
        )
        entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self.params,
                    "offset": substep * _UNIFORM_ALIGNMENT,
                    "size": _PARAMS_FLOATS * 4,
                },
            }
        ]
        for binding, buffer in enumerate(buffers, start=1):
            entries.append(
                {
                    "binding": binding,
                    "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
                }
            )
        return entries

    def _make_bind_groups(self):
        """Create one bind group per substep, shared by every kernel."""
        self.bind_groups = [
            self.device.create_bind_group(
                layout=self.layout, entries=self._entries(substep)
            )
            for substep in range(self.substeps)
        ]

    def _dispatch(self, compute_pass, name, substep, items):
        """
        Run one kernel over a number of items.

        Parameters
        ----------
        compute_pass : GPUComputePassEncoder
            The open compute pass.
        name : str
            Kernel name.
        substep : int
            Index of the substep.
        items : int
            Number of invocations needed.
        """
        groups = math.ceil(items / _WORKGROUP_SIZE)
        columns = min(groups, _MAX_DISPATCH)
        rows = math.ceil(groups / columns)
        compute_pass.set_pipeline(self.pipelines[name])
        compute_pass.set_bind_group(0, self.bind_groups[substep])
        compute_pass.dispatch_workgroups(columns, rows, 1)

    def encode_step(self, substeps, iterations, blast):
        """
        Submit one step of kernels.

        Parameters
        ----------
        substeps : int
            Number of substeps.
        iterations : int
            Contact iterations per substep.
        blast : bool
            Whether pending blasts should be applied first.
        """
        if self.bind_groups is None:
            self._make_bind_groups()
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        if blast:
            self._dispatch(compute_pass, "blast", 0, self.count)
        for substep in range(substeps):
            self._dispatch(compute_pass, "predict", substep, self.count)
            self._dispatch(compute_pass, "clear", substep, self.cell_count)
            self._dispatch(compute_pass, "insert", substep, self.count)
            for _ in range(iterations):
                self._dispatch(compute_pass, "solve", substep, self.count)
                self._dispatch(compute_pass, "apply", substep, self.count)
            self._dispatch(compute_pass, "finalize", substep, self.count)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

    def _read(self, buffer):
        """
        Copy a GPU buffer back to the CPU.

        Parameters
        ----------
        buffer : GPUBuffer
            The buffer.

        Returns
        -------
        ndarray
            Its contents as float32 records of four values.
        """
        data = self.device.queue.read_buffer(buffer)
        return np.frombuffer(data, dtype=np.float32).reshape(-1, 4)

    def read_positions(self):
        """
        Return centre and radius records.

        Returns
        -------
        ndarray, shape (N, 4)
            Records.
        """
        return self._read(self.positions)

    def read_velocities(self):
        """
        Return velocity and inverse mass records.

        Returns
        -------
        ndarray, shape (N, 4)
            Records.
        """
        return self._read(self.velocities)

    def read_overflow(self):
        """
        Return the number of dropped grid insertions.

        Returns
        -------
        int
            Dropped insertions.
        """
        data = self.device.queue.read_buffer(self.stats)
        return int(np.frombuffer(data, dtype=np.uint32)[0])


def _layout_entries():
    """
    Return the bind group layout shared by every physics kernel.

    Returns
    -------
    list of dict
        Layout entries: the substep uniform, the read-write sphere and grid
        buffers, and the read-only force and obstacle records.
    """
    compute = ShaderStage.COMPUTE
    entries = [{"binding": 0, "visibility": compute, "buffer": {"type": "uniform"}}]
    for binding in range(1, 12):
        kind = "read-only-storage" if binding in (9, 10) else "storage"
        entries.append(
            {"binding": binding, "visibility": compute, "buffer": {"type": kind}}
        )
    return entries


def _grid_layout(state):
    """
    Choose the grid cell size and the static spheres kept out of the grid.

    Cells fit the largest moving sphere. Static spheres larger than that would
    need larger cells for everything, so they are checked directly instead.

    Parameters
    ----------
    state : ndarray, shape (N, 8)
        Centre, radius, velocity and inverse mass of every sphere.

    Returns
    -------
    cell_size : float
        Edge length of a grid cell.
    obstacles : ndarray, shape (K, 4)
        Centre and radius of each large static sphere.
    """
    radii = state[:, 3]
    moving = state[:, 7] > 0
    largest = radii[moving].max() if moving.any() else radii.max()
    cell_size = float(2.0 * largest)
    threshold = np.float32(cell_size) * np.float32(1.0001)
    obstacles = np.ascontiguousarray(
        state[(~moving) & (np.float32(2.0) * radii > threshold), :4]
    )
    if len(obstacles) > _MAX_OBSTACLES:
        logging.warning(
            "%d static spheres are larger than the moving ones. Each moving "
            "sphere checks all of them every iteration, which is slow; use "
            "fewer, or make them no larger than the moving spheres.",
            len(obstacles),
        )
    if moving.any():
        median = float(np.median(radii[moving]))
        if largest > 4.0 * median:
            logging.warning(
                "The largest moving sphere is %.1f times the median radius. "
                "Grid cells fit the largest one, so small spheres crowd into "
                "cells and contacts may be missed; check GPUWorld.overflow or "
                "raise bucket_size.",
                largest / median,
            )
    return cell_size, obstacles
