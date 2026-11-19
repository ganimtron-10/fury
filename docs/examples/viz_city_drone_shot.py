"""
=============================================
City Drone Shot: Cinematic Flythrough in FURY
=============================================

A cinematic drone fly-through of a procedurally generated 3D city using
FURY's ``fury.motion`` animation system. The camera path is defined via
``CameraAnimation`` keyframes sampled from one smooth closed route, flown at a
gently changing speed. It chases a car, spirals up a skyscraper, loops beside its
roof, dives into a street and barrel-rolls on the way back.

Static city geometry (thousands of road tiles and buildings) is heavily optimized
using NumPy concatenation to batch them into a road actor and a grid of building chunks,
enabling buttery smooth rendering and allowing for high-density animated traffic.

A custom pure-Python/NumPy OBJ parser is used to completely eliminate VTK
dependencies while preserving UV texture coordinates.

The drone never enters a building: its route is baked once at start-up into a
dense path that rises smoothly over anything in the way, so playback stays
smooth with no per-frame collision response.

The models come from Kenney's Car Kit, City Kit (Commercial) and City Kit (Roads)
packs (CC0, https://kenney.nl/assets). ``polyxios`` downloads them from the
``polyxios-data`` catalog on first use and reuses the cached copies after that.
"""

import logging
import os

import numpy as np
import polyxios as px
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d, maximum_filter1d, uniform_filter1d

from fury import actor, window
from fury.lib import DirectionalLight
from fury.motion import (
    Animation,
    CameraAnimation,
    Timeline,
    linear_interpolator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Assets
# ======
# Each Kenney model is published as ``kenney-<name>.obj`` with a companion
# archive that ``polyxios`` unpacks into a ``kenney-<name>`` folder next to it,
# holding the pack's ``colormap.png`` texture.


def model_path(filename):
    """
    Return the local path of a Kenney model, downloading it on first use.

    Parameters
    ----------
    filename : str
        Model file name as named in its Kenney pack, such as ``"sedan.obj"``.

    Returns
    -------
    str
        Path of the cached OBJ file.
    """
    local = _local_model_path(filename)
    if local:
        return local
    return px.fetch(f"kenney-{filename}")


KENNEY_ASSETS = os.environ.get(
    "FURY_KENNEY_ASSETS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kenny_assets"),
)


def _local_model_path(filename):
    """
    Find a Kenney model in the local packs folder.

    The polyxios-data catalog does not publish the Kenney models yet, so the
    demo branch reads the unpacked packs from ``kenny_assets`` (or
    ``FURY_KENNEY_ASSETS``) first.

    Parameters
    ----------
    filename : str
        Model file name as named in its Kenney pack.

    Returns
    -------
    str or None
        Path of the OBJ file, or None when no local pack has it.
    """
    for root, _, files in os.walk(KENNEY_ASSETS):
        if filename in files:
            return os.path.join(root, filename)
    return None


def texture_path(filename):
    """
    Return the local path of the texture a Kenney model uses.

    Parameters
    ----------
    filename : str
        Model file name as named in its Kenney pack.

    Returns
    -------
    str
        Path of the model's ``colormap.png``.
    """
    path = model_path(filename)
    local = os.path.join(os.path.dirname(path), "Textures", "colormap.png")
    if os.path.exists(local):
        return local
    folder = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(path), folder, "colormap.png")


###############################################################################
# Model Loading & Optimization Helpers
# ====================================

_MODEL_CACHE = {}


def load_obj(path):
    """
    Read an OBJ model into flat triangle arrays.

    Parameters
    ----------
    path : str
        Path of the OBJ file.

    Returns
    -------
    tuple of ndarray
        Vertices, triangle indices, and texture coordinates flipped to image
        orientation, or None for the coordinates when the model has none.
    """
    verts, uvs, faces, uv_indices = [], [], [], []

    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.strip().split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                f_verts, f_uvs = [], []
                for p in parts:
                    vals = p.split("/")
                    f_verts.append(int(vals[0]) - 1)
                    if len(vals) > 1 and vals[1]:
                        f_uvs.append(int(vals[1]) - 1)
                    else:
                        f_uvs.append(-1)
                for i in range(1, len(f_verts) - 1):
                    faces.append([f_verts[0], f_verts[i], f_verts[i + 1]])
                    uv_indices.append([f_uvs[0], f_uvs[i], f_uvs[i + 1]])

    verts = np.array(verts, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
    if uvs is not None:
        # OBJ UVs start at the bottom of the image, textures at the top. Without
        # this flip every model samples the wrong row of Kenney's palette.
        uvs[:, 1] = 1.0 - uvs[:, 1]
    faces = np.array(faces, dtype=np.int32)
    uv_indices = np.array(uv_indices, dtype=np.int32)

    flattened_verts = verts[faces.flatten()]
    flattened_faces = np.arange(len(flattened_verts)).reshape(-1, 3)

    if uvs is not None and len(uvs) > 0 and np.all(uv_indices != -1):
        flattened_uvs = uvs[uv_indices.flatten()]
    else:
        flattened_uvs = None

    return flattened_verts, flattened_faces, flattened_uvs


def get_base_mesh(filename):
    """
    Return a model's arrays, reading the file only the first time.

    Parameters
    ----------
    filename : str
        Model file name as named in its Kenney pack.

    Returns
    -------
    tuple of ndarray
        The arrays returned by ``load_obj``.
    """
    if filename not in _MODEL_CACHE:
        _MODEL_CACHE[filename] = load_obj(model_path(filename))
    return _MODEL_CACHE[filename]


def transform_vertices(v, scale, pos, rot_y):
    """
    Return a transformed copy of model vertices.

    Parameters
    ----------
    v : ndarray, shape (N, 3)
        Model-space vertices.
    scale : float or sequence of float
        Uniform scale, or per-axis scale, applied first.
    pos : sequence of float
        World-space translation, applied last.
    rot_y : float
        Rotation about the Y axis in degrees.

    Returns
    -------
    ndarray, shape (N, 3)
        World-space vertices.
    """
    v = v * np.asarray(scale, dtype=np.float32)
    if rot_y != 0.0:
        angle = np.radians(rot_y)
        c_val, s_val = np.cos(angle), np.sin(angle)
        v_x = v[:, 0] * c_val + v[:, 2] * s_val
        v_z = -v[:, 0] * s_val + v[:, 2] * c_val
        v[:, 0] = v_x
        v[:, 2] = v_z
    return v + np.asarray(pos, dtype=np.float32)


def instance_bounds(filename, scale, pos, rot_y):
    """
    Return the world-space bounding box of a placed model.

    Parameters
    ----------
    filename : str
        Model file name.
    scale : float or sequence of float
        Uniform or per-axis scale.
    pos : sequence of float
        World-space translation.
    rot_y : float
        Rotation about the Y axis in degrees.

    Returns
    -------
    ndarray, shape (6,)
        The box as (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    v = transform_vertices(get_base_mesh(filename)[0], scale, pos, rot_y)
    return np.concatenate([v.min(axis=0), v.max(axis=0)])


def create_batched_actor(instances):
    """
    Merge many placed copies of models from one Kenney pack into one actor.

    Every model in a pack shares the same palette texture, so the texture of the
    first model is used for the whole actor.

    Parameters
    ----------
    instances : list of tuple
        ``(filename, scale, position, rotation_y)`` for each copy.

    Returns
    -------
    Mesh
        The merged actor.
    """
    all_verts = []
    all_faces = []
    all_uvs = []
    v_offset = 0

    for filename, scale, pos, rot_y in instances:
        v, f, uv = get_base_mesh(filename)
        v = transform_vertices(v, scale, pos, rot_y)

        all_verts.append(v)
        all_faces.append(f + v_offset)
        if uv is not None:
            all_uvs.append(uv)

        v_offset += len(v)

    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    uvs = np.concatenate(all_uvs) if len(all_uvs) > 0 else None

    tex_path = texture_path(instances[0][0])
    surf = actor.surface(verts, faces, texture=tex_path, texture_coords=uvs)
    return surf


CAR_SCALE = 3.5
CAR_TEMPLATES = {}


def create_car(filename, *, position=(0.0, 0.0, 0.0), rotation_y=0.0):
    """
    Create a car sharing geometry with other cars of that model and heading.

    Scale and heading are baked into the vertices because ``Animation`` replaces
    the actor's transform with a translation-only matrix.

    Parameters
    ----------
    filename : str
        Model file name in the Car Kit.
    position : sequence of float, optional
        World-space position of the car.
    rotation_y : float, optional
        Rotation about the Y axis in degrees.

    Returns
    -------
    Mesh
        The car actor.
    """
    key = (filename, rotation_y)
    if key not in CAR_TEMPLATES:
        v, f, uv = get_base_mesh(filename)
        v = transform_vertices(v, CAR_SCALE, (0.0, 0.0, 0.0), rotation_y)
        template = actor.surface(
            v, f, texture=texture_path(filename), texture_coords=uv
        )
        CAR_TEMPLATES[key] = (template.geometry, template.material)

    geometry, material = CAR_TEMPLATES[key]
    car = actor.create_mesh(geometry, material)
    car.local.position = list(position)
    return car


###############################################################################
# Scene Setup
# ===========

scene = window.Scene()
# Soft daylight sky
scene.background = (0.70, 0.84, 0.93)

# Ground Plane
ground = actor.box(
    centers=np.array([[0.0, -0.25, 0.0]]),
    colors=(0.47, 0.60, 0.40),  # Park green under the city blocks
    scales=(3000, 0.5, 3000),
)
scene.add(ground)

# Warm sunlight from high in the south-west
sunlight = DirectionalLight(color=(1.0, 0.95, 0.85), intensity=1.6)
sunlight.local.position = (-400.0, 700.0, -300.0)
scene.add(sunlight)

###############################################################################
# City Layout Constants

np.random.seed(42)

BLOCK_SIZE = 70.0
ROAD_WIDTH = 14.0
BUILDING_SCALE = 12.0
GRID_SIZE = 8  # Expanded grid size (17x17 roads)
SIDEWALK_WIDTH = 1.0
# Furthest a building edge may reach from its block centre without touching a road
BLOCK_HALF_LIMIT = BLOCK_SIZE / 2.0 - ROAD_WIDTH / 2.0 - SIDEWALK_WIDTH
CAMERA_DEPTH_RANGE = (1.0, 8000.0)
DRONE_CLEARANCE = 6.0  # Minimum gap between the drone and any building
CHUNK_BLOCKS = 2  # City blocks per side of a separately culled building chunk

BUILDING_MODELS = [
    f"building-{c}.obj" for c in ["a", "b", "c", "d", "e", "f", "g", "h"]
]
SKYSCRAPER_MODELS = [f"building-skyscraper-{c}.obj" for c in ["a", "b", "c", "d", "e"]]
CAR_MODELS = [
    "sedan.obj",
    "taxi.obj",
    "suv.obj",
    "truck.obj",
    "police.obj",
    "van.obj",
    "hatchback-sports.obj",
    "ambulance.obj",
]

###############################################################################
# Roads & Buildings Generation (Batched)
# ======================================


def generate_static_city():
    """
    Build the roads and buildings as batched actors and add them to the scene.

    The building batches are split into chunks for culling. Each chunk and its
    bounds are appended to ``building_chunks``, ``chunk_mins`` and ``chunk_maxs``.

    Returns
    -------
    ndarray
        Bounding box ``(min_x, min_y, min_z, max_x, max_y, max_z)`` of every
        building, used to keep the drone clear of them.
    """
    road_instances = []
    building_instances = []
    building_bounds = []

    # 1. Generate Roads
    for ix in range(-GRID_SIZE, GRID_SIZE + 1):
        for iz in range(-GRID_SIZE, GRID_SIZE + 1):
            cx = ix * BLOCK_SIZE
            cz = iz * BLOCK_SIZE

            # Intersection
            road_instances.append(
                ("road-crossroad.obj", ROAD_WIDTH, (cx, 0.0, cz), 0.0)
            )

            # One tile stretched along each street, the kit's tile runs along X
            segment = (BLOCK_SIZE - ROAD_WIDTH, ROAD_WIDTH, ROAD_WIDTH)
            if ix < GRID_SIZE:
                road_instances.append(
                    (
                        "road-straight.obj",
                        segment,
                        (cx + BLOCK_SIZE / 2.0, 0.0, cz),
                        0.0,
                    )
                )
            if iz < GRID_SIZE:
                road_instances.append(
                    (
                        "road-straight.obj",
                        segment,
                        (cx, 0.0, cz + BLOCK_SIZE / 2.0),
                        90.0,
                    )
                )

    # 2. Generate Buildings
    for ix in range(-GRID_SIZE, GRID_SIZE):
        for iz in range(-GRID_SIZE, GRID_SIZE):
            block_cx = ix * BLOCK_SIZE + BLOCK_SIZE / 2.0
            block_cz = iz * BLOCK_SIZE + BLOCK_SIZE / 2.0

            # Sometimes leave a block empty for a park
            if np.random.random() < 0.15:
                continue

            offsets = [
                (-16, -16),
                (0, -16),
                (16, -16),
                (-16, 0),
                (16, 0),
                (-16, 16),
                (0, 16),
                (16, 16),
            ]

            num_buildings = np.random.randint(2, 5)
            chosen_indices = np.random.choice(
                len(offsets), num_buildings, replace=False
            )
            block_bounds = []

            for idx in chosen_indices:
                dx, dz = offsets[idx]
                bx = block_cx + dx
                bz = block_cz + dz

                # Scatter skyscrapers randomly across the entire city (20% chance)
                if np.random.random() < 0.20:
                    model_name = SKYSCRAPER_MODELS[
                        np.random.randint(0, len(SKYSCRAPER_MODELS))
                    ]
                    scale = np.random.uniform(
                        BUILDING_SCALE * 1.2, BUILDING_SCALE * 1.8
                    )
                else:
                    model_name = BUILDING_MODELS[
                        np.random.randint(0, len(BUILDING_MODELS))
                    ]
                    scale = np.random.uniform(
                        BUILDING_SCALE * 0.8, BUILDING_SCALE * 1.2
                    )

                rot_angle = float(np.random.choice([0, 90, 180, 270]))

                # Pull the building inward so it stays off the road
                box = instance_bounds(model_name, scale, (bx, 0.0, bz), rot_angle)
                half_x = (box[3] - box[0]) / 2.0
                half_z = (box[5] - box[2]) / 2.0
                bx = block_cx + float(
                    np.clip(dx, -(BLOCK_HALF_LIMIT - half_x), BLOCK_HALF_LIMIT - half_x)
                )
                bz = block_cz + float(
                    np.clip(dz, -(BLOCK_HALF_LIMIT - half_z), BLOCK_HALF_LIMIT - half_z)
                )
                box = instance_bounds(model_name, scale, (bx, 0.0, bz), rot_angle)

                # Skip it if it would intersect a neighbour in the same block
                if any(
                    box[0] < o[3] and box[3] > o[0] and box[2] < o[5] and box[5] > o[2]
                    for o in block_bounds
                ):
                    continue

                block_bounds.append(box)
                building_bounds.append(box)
                building_instances.append((model_name, scale, (bx, 0.0, bz), rot_angle))

    # Hardcode a "Colossal Skyscraper" specifically for the drone stunt sequence
    colossal_x = 105.0
    colossal_z = 105.0
    # Clear block (1, 1) so nothing intersects the colossal skyscraper
    keep = [
        i
        for i, b in enumerate(building_instances)
        if not (
            abs(b[2][0] - colossal_x) < BLOCK_SIZE / 2.0
            and abs(b[2][2] - colossal_z) < BLOCK_SIZE / 2.0
        )
    ]
    building_instances = [building_instances[i] for i in keep]
    building_bounds = [building_bounds[i] for i in keep]

    # Add the colossal skyscraper right in the middle of block (1, 1)
    colossal_scale = (BUILDING_SCALE * 3.0, BUILDING_SCALE * 6.0, BUILDING_SCALE * 3.0)
    building_instances.append(
        (
            "building-skyscraper-c.obj",
            colossal_scale,
            (colossal_x, 0.0, colossal_z),
            0.0,
        )
    )
    building_bounds.append(
        instance_bounds(
            "building-skyscraper-c.obj",
            colossal_scale,
            (colossal_x, 0.0, colossal_z),
            0.0,
        )
    )

    logger.info(f"Batching {len(road_instances)} road tiles...")
    roads_actor = create_batched_actor(road_instances)
    # Enable backface culling so we don't render the inside of models
    roads_actor.material.side = "front"
    scene.add(roads_actor)

    logger.info(f"Batching {len(building_instances)} buildings...")
    bounds = np.array(building_bounds, dtype=np.float64)

    # Split buildings into chunks so off-screen parts of the city can be skipped
    chunk_span = CHUNK_BLOCKS * BLOCK_SIZE
    grouped = {}
    for instance, box in zip(building_instances, bounds, strict=True):
        key = (
            np.floor(instance[2][0] / chunk_span),
            np.floor(instance[2][2] / chunk_span),
        )
        grouped.setdefault(key, []).append((instance, box))

    material = None
    for members in grouped.values():
        chunk = create_batched_actor([m[0] for m in members])
        if material is None:
            material = chunk.material
            material.side = "front"  # Skip the hollow interiors of the models
        chunk.material = material
        scene.add(chunk)

        boxes = np.array([m[1] for m in members])
        building_chunks.append(chunk)
        chunk_mins.append(boxes[:, :3].min(axis=0))
        chunk_maxs.append(boxes[:, 3:].max(axis=0))

    return bounds


building_chunks, chunk_mins, chunk_maxs = [], [], []
BUILDING_BOUNDS = generate_static_city()
# Keeps chunks visible through the short gap between culling updates
CHUNK_MARGIN = 40.0
CHUNK_MINS = np.array(chunk_mins) - CHUNK_MARGIN
CHUNK_MAXS = np.array(chunk_maxs) + CHUNK_MARGIN


def update_visible_chunks(camera):
    """
    Hide building chunks outside the view frustum to skip drawing them.

    Parameters
    ----------
    camera : Camera
        The camera whose view decides which chunks are drawn.
    """
    m = np.asarray(camera.camera_matrix)
    planes = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] - m[2]])
    # For each plane test the box corner furthest along that plane's normal
    corners = np.where(
        planes[:, :3] >= 0, CHUNK_MAXS[:, None, :], CHUNK_MINS[:, None, :]
    )
    visible = ((corners * planes[:, :3]).sum(axis=-1) + planes[:, 3] >= 0).all(axis=1)
    for chunk, is_visible in zip(building_chunks, visible, strict=True):
        chunk.visible = bool(is_visible)


###############################################################################
# Drone Flight Path
# =================
# The route is a list of waypoints joined into one closed spline. Instead of
# timing each waypoint by hand, the drone flies along the curve at a speed that
# only ramps gently, so it never lurches between waypoints. Where it looks and
# how it rolls are derived from the same curve, which keeps every stunt smooth.

ANIMATION_DURATION = 45.0
SAMPLE_RATE = 30.0  # Camera samples per second of the loop
HERO_LANE = -4.0  # Right-hand lane for traffic heading +Z on the X=0 avenue
CHASE_SPEED = 45.0  # Hero car speed, matched by the drone while it chases
CHASE_GAP = 50.0  # How far the drone trails the hero car
CHASE_START_Z = -560.0  # Where the drone is on the avenue when the loop starts
HERO_START_Z = CHASE_START_Z + CHASE_GAP
LOOK_AHEAD = 40.0  # Distance from the camera to its focal point


def waypoint_line(start, end, count):
    """
    Return evenly spaced waypoints from ``start`` to ``end``, excluding ``end``.

    Parameters
    ----------
    start, end : sequence of float
        End points of the line.
    count : int
        Number of waypoints.

    Returns
    -------
    ndarray, shape (count, 3)
        The waypoints.
    """
    alpha = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
    return np.asarray(start) * (1.0 - alpha) + np.asarray(end) * alpha


def plan_route():
    """
    Lay out the waypoints of the drone's route, one group per shot.

    The shots are a chase behind the hero car, a climb into a panorama, a
    spiral around the colossal skyscraper, a corkscrew loop beside its roof, a
    dive into a street, a low run with a barrel roll, and a swoop back onto the
    avenue behind the hero car.

    Returns
    -------
    list of tuple
        ``(name, waypoints, speed)`` for each shot, in flight order. ``speed``
        is the cruise speed of the shot before the route is fitted to the loop
        duration, or None to hold ``CHASE_SPEED`` exactly.
    dict
        Named points the camera looks at or rolls around.
    """
    tower = BUILDING_BOUNDS[-1]  # The colossal skyscraper is placed last
    centre = np.array([(tower[0] + tower[3]) / 2.0, 0.0, (tower[2] + tower[5]) / 2.0])
    top = tower[4]
    radius = max(tower[3] - tower[0], tower[5] - tower[2]) / 2.0 + 55.0
    cruise = top + 30.0

    # Spiral one and a quarter turns up the tower, counter-clockwise from above
    angles = np.linspace(0.0, 2.5 * np.pi, 26)
    heights = np.interp(angles, [0.0, 2.5 * np.pi], [70.0, cruise])
    spiral = np.stack(
        [
            centre[0] + radius * np.sin(angles),
            heights,
            centre[2] + radius * np.cos(angles),
        ],
        axis=1,
    )
    spiral_end = spiral[-1]  # East of the tower, flying towards -Z

    # A corkscrew loop: up, over and down while drifting sideways, so the exit
    # does not cross the entry
    loop_radius = 38.0
    loop_entry = spiral_end + np.array([0.0, 0.0, -110.0])
    loop_centre = loop_entry + np.array([0.0, loop_radius, 0.0])
    phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    loop = np.stack(
        [
            loop_entry[0] + 30.0 * phi / (2.0 * np.pi),
            loop_centre[1] - loop_radius * np.cos(phi),
            loop_entry[2] - loop_radius * np.sin(phi),
        ],
        axis=1,
    )
    loop_exit = loop_entry + np.array([30.0, 0.0, 0.0])

    # Dive to street level onto the road running along X = street_x
    street_x = BLOCK_SIZE * np.round(loop_exit[0] / BLOCK_SIZE)
    street_z = BLOCK_SIZE * np.round((loop_exit[2] - 140.0) / BLOCK_SIZE)
    dive = np.array(
        [
            loop_exit + np.array([0.0, -10.0, -60.0]),
            [street_x, cruise * 0.55, loop_exit[2] - 130.0],
            [street_x, 40.0, street_z - 60.0],
        ]
    )
    edge = (GRID_SIZE - 1) * BLOCK_SIZE
    street = waypoint_line(
        [street_x, 24.0, street_z - 120.0], [street_x, 24.0, -edge - 60.0], 10
    )

    # Swoop out of the city and back onto the avenue behind the hero car
    swoop = np.array(
        [
            [street_x, 30.0, -edge - 120.0],
            [street_x * 0.75, 45.0, CHASE_START_Z - 170.0],
            [street_x * 0.3, 32.0, CHASE_START_Z - 210.0],
            [0.0, 20.0, CHASE_START_Z - 160.0],
            [0.0, 15.0, CHASE_START_Z - 80.0],
        ]
    )

    chase_end_z = CHASE_START_Z + 330.0
    route = [
        (
            "chase",
            waypoint_line([0.0, 14.0, CHASE_START_Z], [0.0, 14.0, chase_end_z], 8),
            None,
        ),
        (
            "climb",
            np.array(
                [
                    [0.0, 18.0, chase_end_z],
                    [-15.0, 40.0, chase_end_z + 60.0],
                    [-60.0, 90.0, chase_end_z + 120.0],
                ]
            ),
            70.0,
        ),
        (
            "panorama",
            np.array(
                [
                    [-150.0, 140.0, chase_end_z + 220.0],
                    [-190.0, 160.0, chase_end_z + 340.0],
                    [-150.0, 150.0, centre[2] + radius + 40.0],
                    [-40.0, 110.0, centre[2] + radius + 60.0],
                ]
            ),
            90.0,
        ),
        ("spiral", spiral, 75.0),
        ("loop", np.vstack([waypoint_line(spiral_end, loop_entry, 3)[1:], loop]), 70.0),
        ("dive", np.vstack([loop_exit[None], dive]), 115.0),
        ("street", street, 95.0),
        ("swoop", swoop, 85.0),
    ]
    points = {
        "tower": np.array([centre[0], 0.0, centre[2]]),
        "loop_centre": loop_centre,
        "loop_radius": loop_radius,
    }
    return route, points


def lift_clear_of_buildings(path, bounds, clearance, ease_samples):
    """
    Raise path points that would pass through a building.

    Each point inside a building footprint (grown by ``clearance``) gets the
    lift that puts it above that roof. A sliding maximum followed by a Hann
    kernel no wider than that maximum keeps every lift at least as large as
    required while easing it in and out, so the drone climbs ahead of time.
    Only height changes, and buildings stand on the ground, so a lift can never
    push the drone into another building.

    Parameters
    ----------
    path : ndarray, shape (N, 3)
        Points of a closed path.
    bounds : ndarray, shape (M, 6)
        Building boxes as (xmin, ymin, zmin, xmax, ymax, zmax).
    clearance : float
        Minimum gap to keep from any building.
    ease_samples : int
        Number of points over which a lift eases in and out.

    Returns
    -------
    ndarray, shape (N, 3)
        The lifted path.
    """
    inside = (
        (path[:, 0, None] > bounds[:, 0] - clearance)
        & (path[:, 0, None] < bounds[:, 3] + clearance)
        & (path[:, 2, None] > bounds[:, 2] - clearance)
        & (path[:, 2, None] < bounds[:, 5] + clearance)
    )
    required = np.where(inside, bounds[:, 4] + clearance, -np.inf).max(axis=1)
    lift = np.maximum(required - path[:, 1], 0.0)

    radius = max(1, ease_samples)
    lift = maximum_filter1d(lift, size=2 * radius + 1, mode="wrap")
    kernel = np.hanning(2 * radius + 3)[1:-1]
    lifted = path.copy()
    lifted[:, 1] += convolve1d(lift, kernel / kernel.sum(), mode="wrap")
    return lifted


def smooth_periodic(values, width):
    """
    Smooth samples of a closed loop with three passes of a moving average.

    Three box passes approximate a Gaussian, so steps in ``values`` become
    gentle S-shaped ramps with no sudden change of slope.

    Parameters
    ----------
    values : ndarray, shape (N,) or (N, K)
        Samples around the loop.
    width : int
        Width of each moving average, in samples.

    Returns
    -------
    ndarray
        The smoothed samples, same shape as ``values``.
    """
    out = np.asarray(values, dtype=np.float64)
    for _ in range(3):
        out = uniform_filter1d(out, max(1, int(width)), axis=0, mode="wrap")
    return out


def rotate_about(vectors, axes, angles):
    """
    Rotate vectors about unit axes by the given angles (Rodrigues' formula).

    Parameters
    ----------
    vectors, axes : ndarray, shape (N, 3)
        Vectors to rotate and the unit axes to rotate them about.
    angles : ndarray, shape (N,)
        Rotation angles in radians.

    Returns
    -------
    ndarray, shape (N, 3)
        The rotated vectors.
    """
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    dot = np.sum(axes * vectors, axis=1, keepdims=True)
    return (
        vectors * cos_a + np.cross(axes, vectors) * sin_a + axes * dot * (1.0 - cos_a)
    )


def normalize(vectors):
    """
    Return the vectors scaled to unit length.

    Parameters
    ----------
    vectors : ndarray, shape (N, 3)
        Vectors to normalize.

    Returns
    -------
    ndarray, shape (N, 3)
        Unit vectors.
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def grow_mask(mask, reach):
    """
    Return a closed-loop mask grown by ``reach`` samples on both sides.

    Parameters
    ----------
    mask : ndarray of bool, shape (N,)
        Samples around the loop.
    reach : int
        Number of samples to grow by.

    Returns
    -------
    ndarray of bool, shape (N,)
        The grown mask.
    """
    return maximum_filter1d(mask.astype(np.uint8), 2 * reach + 1, mode="wrap") > 0


def time_route(route, duration, sample_rate):
    """
    Fly the route at a gently changing speed and sample it at a fixed rate.

    Parameters
    ----------
    route : list of tuple
        Shots from ``plan_route``.
    duration : float
        Length of the loop in seconds.
    sample_rate : float
        Samples per second.

    Returns
    -------
    times : ndarray, shape (N,)
        Sample times from 0 up to, but excluding, ``duration``.
    positions : ndarray, shape (N, 3)
        Camera position at each time.
    shots : ndarray of int, shape (N,)
        Index of the shot flown at each time.
    """
    waypoints = np.vstack([w for _, w, _ in route])
    shot_of_waypoint = np.concatenate(
        [np.full(len(w), i) for i, (_, w, _) in enumerate(route)]
    )

    # A closed spline through the waypoints, parameterised by chord length
    closed = np.vstack([waypoints, waypoints[:1]])
    knots = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    )
    spline = CubicSpline(knots, closed, bc_type="periodic")

    # Sample it densely, then lift it clear of buildings
    dense_u = np.linspace(0.0, knots[-1], 40000, endpoint=False)
    step = knots[-1] / len(dense_u)
    path = lift_clear_of_buildings(
        spline(dense_u), BUILDING_BOUNDS, DRONE_CLEARANCE, int(60.0 / step)
    )
    shot = shot_of_waypoint[np.searchsorted(knots, dense_u, side="right") - 1]
    seg = np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)

    # Each shot's cruise speed, blended over about 260 units of path. The chase
    # holds CHASE_SPEED exactly, so its neighbours reach that speed before the
    # chase starts and leave it only after the chase ends.
    width = max(1, int(260.0 / 3.0 / np.mean(seg)))
    chase = shot == [name for name, _, _ in route].index("chase")
    near_chase = grow_mask(chase, (3 * width) // 2 + 1)

    def speeds(scale):
        cruise = np.array(
            [CHASE_SPEED if s is None else s * scale for _, _, s in route]
        )
        speed = smooth_periodic(np.where(near_chase, CHASE_SPEED, cruise[shot]), width)
        speed[chase] = CHASE_SPEED
        return speed

    # Scale the cruise speeds so the loop lasts exactly ``duration``
    low, high = 0.1, 10.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if np.sum(seg / speeds(mid)) > duration:
            low = mid
        else:
            high = mid
    if high == 10.0:
        raise ValueError(f"The route cannot be flown in {duration} s.")
    step_time = seg / speeds(high)
    clock = np.concatenate([[0.0], np.cumsum(step_time)[:-1]])
    clock *= duration / step_time.sum()

    times = np.arange(int(round(duration * sample_rate))) / sample_rate
    positions = np.stack(
        [np.interp(times, clock, path[:, k]) for k in range(3)], axis=1
    )
    index = np.searchsorted(clock, times, side="right") - 1
    return times, positions, shot[np.clip(index, 0, len(shot) - 1)]


def aim_camera(route, points, times, positions, shots, hero_wrap, sample_rate):
    """
    Work out where the camera looks and which way is up along the flight.

    Parameters
    ----------
    route : list of tuple
        Shots from ``plan_route``.
    points : dict
        Named points from ``plan_route``.
    times : ndarray, shape (N,)
        Sample times.
    positions : ndarray, shape (N, 3)
        Camera position at each time.
    shots : ndarray of int, shape (N,)
        Index of the shot flown at each time.
    hero_wrap : float
        Time at which the hero car jumps back to the start of the avenue.
    sample_rate : float
        Samples per second.

    Returns
    -------
    focals, ups : ndarray, shape (N, 3)
        Camera focal point and up vector at each time.
    """
    names = [name for name, _, _ in route]
    forward = normalize(np.roll(positions, -1, axis=0) - np.roll(positions, 1, axis=0))

    # Up is the sky, except in the loop, where it points at the loop's centre so
    # the camera turns over with the drone
    up = np.tile([0.0, 1.0, 0.0], (len(times), 1))
    to_centre = np.stack(
        [
            np.zeros(len(times)),
            points["loop_centre"][1] - positions[:, 1],
            points["loop_centre"][2] - positions[:, 2],
        ],
        axis=1,
    )
    on_loop = (shots == names.index("loop")) & (
        np.linalg.norm(to_centre, axis=1) < 1.3 * points["loop_radius"]
    )
    up[on_loop] = normalize(to_centre[on_loop])
    up = normalize(smooth_periodic(up, sample_rate * 0.15))

    # Where each shot looks: at the hero car, ahead along the path, or in at
    # the tower. Blending the choices over time keeps the view from snapping.
    hero_z = HERO_START_Z + CHASE_SPEED * np.where(
        times < hero_wrap, times, times - len(times) / sample_rate
    )
    car = np.stack(
        [np.full_like(times, HERO_LANE), np.full_like(times, 3.0), hero_z], axis=1
    )
    at_car = normalize(car - positions)
    # Tip the view a little below the direction of travel. In the loop, tip it
    # towards the loop's centre as the drone turns over, so the city fills the
    # frame at the top instead of empty sky.
    in_loop = smooth_periodic(on_loop.astype(float), sample_rate * 0.4)
    turned_over = 0.5 * (1.0 - up[:, 1])  # 0 upright, 1 upside down
    tip = -0.25 * (1.0 - in_loop) + 1.2 * turned_over * in_loop
    ahead = normalize(forward + tip[:, None] * up)
    tower_axis = np.stack(
        [
            np.full_like(times, points["tower"][0]),
            positions[:, 1] - 15.0,
            np.full_like(times, points["tower"][2]),
        ],
        axis=1,
    )
    at_tower = normalize(normalize(tower_axis - positions) * 0.8 + forward * 0.2)

    choice = np.array([{"chase": 0, "spiral": 2}.get(names[s], 1) for s in shots])
    # Look at the car a little before and after the chase too, so the blend in
    # and out of it happens outside the chase
    blend_width = int(sample_rate * 1.8)
    in_chase = shots == names.index("chase")
    choice[grow_mask(in_chase, (3 * blend_width) // 2 + 1)] = 0
    weights = np.zeros((len(times), 3))
    weights[np.arange(len(times)), choice] = 1.0
    weights = smooth_periodic(weights, blend_width)
    look = normalize(
        weights[:, :1] * at_car + weights[:, 1:2] * ahead + weights[:, 2:] * at_tower
    )
    look = normalize(smooth_periodic(look, sample_rate * 0.5))

    # A barrel roll through the middle of the street run, eased in and out
    street = times[shots == names.index("street")]
    middle = 0.5 * (street[0] + street[-1])
    half = min(1.8, 0.45 * (street[-1] - street[0]))
    progress = np.clip((times - (middle - half)) / (2.0 * half), 0.0, 1.0)
    roll = 2.0 * np.pi * progress * progress * (3.0 - 2.0 * progress)
    up = rotate_about(up, look, roll)

    # Keep up perpendicular to the view so the roll is exact
    up = normalize(up - look * np.sum(up * look, axis=1, keepdims=True))
    return positions + look * LOOK_AHEAD, up


ROUTE, ROUTE_POINTS = plan_route()
CAMERA_TIMES, CAMERA_POSITIONS, CAMERA_SHOTS = time_route(
    ROUTE, ANIMATION_DURATION, SAMPLE_RATE
)
# The hero car wraps back to the start of the avenue halfway round the tower,
# far out of the drone's view
spiral_times = CAMERA_TIMES[CAMERA_SHOTS == [r[0] for r in ROUTE].index("spiral")]
HERO_WRAP = float(0.5 * (spiral_times[0] + spiral_times[-1]))
CAMERA_FOCALS, CAMERA_UPS = aim_camera(
    ROUTE,
    ROUTE_POINTS,
    CAMERA_TIMES,
    CAMERA_POSITIONS,
    CAMERA_SHOTS,
    HERO_WRAP,
    SAMPLE_RATE,
)


def as_keyframes(values):
    """
    Return camera samples as a closed loop of keyframes.

    Parameters
    ----------
    values : ndarray, shape (N, 3)
        One sample per entry of ``CAMERA_TIMES``.

    Returns
    -------
    dict
        Maps each time to its sample, plus the loop end mapped to the first.
    """
    frames = {float(t): v for t, v in zip(CAMERA_TIMES, values, strict=True)}
    frames[ANIMATION_DURATION] = values[0]
    return frames


# The samples are dense and already smooth, so the default straight lines
# between them follow the flight exactly.
camera_anim = CameraAnimation(loop=True)
camera_anim.set_position_keyframes(as_keyframes(CAMERA_POSITIONS))
camera_anim.set_focal_keyframes(as_keyframes(CAMERA_FOCALS))
camera_anim.set_view_up_keyframes(as_keyframes(CAMERA_UPS))


###############################################################################
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

TRAFFIC_EXTENT = 900.0  # Cars wrap around this far out, well beyond the city edge
NUM_CARS = 75
car_animations = []

# The hero car the drone chases, driving at CHASE_SPEED
hero_car = create_car("hatchback-sports.obj", position=(HERO_LANE, 0.0, HERO_START_Z))
scene.add(hero_car)
hero_anim = Animation(actors=hero_car, loop=True)
hero_anim.set_position(0.0, np.array([HERO_LANE, 0.0, HERO_START_Z]))
hero_anim.set_position(
    HERO_WRAP - 1e-6,
    np.array([HERO_LANE, 0.0, HERO_START_Z + CHASE_SPEED * HERO_WRAP]),
)
hero_anim.set_position(
    HERO_WRAP,
    np.array(
        [HERO_LANE, 0.0, HERO_START_Z - CHASE_SPEED * (ANIMATION_DURATION - HERO_WRAP)]
    ),
)
hero_anim.set_position(ANIMATION_DURATION, np.array([HERO_LANE, 0.0, HERO_START_Z]))
hero_anim.set_position_interpolator(linear_interpolator)
car_animations.append(hero_anim)

# Every lane except the hero's. All cars share one speed and are evenly spaced
# within their lane, so a car can never run into the one in front of it.
lanes = []
for line_index in range(-GRID_SIZE + 1, GRID_SIZE):
    line = line_index * BLOCK_SIZE
    for offset in (-4.0, 4.0):
        along_x = 1 if offset > 0 else -1
        lanes.append((True, line + offset, along_x))
        if not (line_index == 0 and offset == HERO_LANE):
            lanes.append((False, line + offset, -along_x))

np.random.shuffle(lanes)
lane_population = np.bincount(np.arange(NUM_CARS) % len(lanes), minlength=len(lanes))
lane_phase = np.random.uniform(0.0, 1.0, len(lanes))


def lane_point(progress, is_x_axis, lane, direction):
    """
    Return the world position at a given progress along a traffic lane.

    Parameters
    ----------
    progress : float
        Progress along the lane from 0 to 1.
    is_x_axis : bool
        Whether the lane runs along the X axis instead of the Z axis.
    lane : float
        Coordinate of the lane on the other axis.
    direction : int
        Travel direction along the lane, 1 or -1.

    Returns
    -------
    ndarray, shape (3,)
        The world position.
    """
    coord = direction * TRAFFIC_EXTENT * (2.0 * progress - 1.0)
    return np.array([coord, 0.0, lane] if is_x_axis else [lane, 0.0, coord])


car_index = 0
for lane_id, (is_x_axis, lane, direction) in enumerate(lanes):
    for rank in range(lane_population[lane_id]):
        model_name = CAR_MODELS[car_index % len(CAR_MODELS)]
        car_index += 1

        if is_x_axis:
            rot_y = 90.0 if direction > 0 else -90.0
        else:
            rot_y = 0.0 if direction > 0 else 180.0

        # Wrap off-screen part-way through the loop instead of emptying the city
        phase = max(
            (lane_phase[lane_id] + rank / lane_population[lane_id]) % 1.0, 0.001
        )
        wrap_time = (1.0 - phase) * ANIMATION_DURATION

        start = lane_point(phase, is_x_axis, lane, direction)
        car = create_car(model_name, position=start, rotation_y=rot_y)
        scene.add(car)

        car_anim = Animation(actors=car, loop=True)
        car_anim.set_position(0.0, start)
        car_anim.set_position(
            wrap_time - 1e-6, lane_point(1.0, is_x_axis, lane, direction)
        )
        car_anim.set_position(wrap_time, lane_point(0.0, is_x_axis, lane, direction))
        car_anim.set_position(ANIMATION_DURATION, start)
        car_anim.set_position_interpolator(linear_interpolator)
        car_animations.append(car_anim)

logger.info(f"Created {len(car_animations)} animated Kenney cars")


def on_camera_update(time):
    """
    Apply the camera setup that must follow every camera update.

    The callback also runs in FURY's recorder, which renders with its own camera.

    Parameters
    ----------
    time : float
        Current animation time, required by the callback signature.
    """
    camera = camera_anim.camera
    camera.depth_range = CAMERA_DEPTH_RANGE
    update_visible_chunks(camera)


camera_anim.add_update_callback(on_camera_update)


def keep_chunks_culled(show_manager):
    """
    Refresh building culling on every draw, including while the timeline is paused.

    Parameters
    ----------
    show_manager : ShowManager
        The show manager whose camera decides which chunks are drawn.
    """
    camera = show_manager.screens[0].camera
    show_manager.renderer.add_event_handler(
        lambda _event: update_visible_chunks(camera), "before_render"
    )


timeline = Timeline(playback_panel=True, loop=True)
timeline.add_animation(camera_anim)
for ca in car_animations:
    timeline.add_animation(ca)


# Force UI elements to render on top of all 3D geometry
def bring_ui_to_front(element):
    """
    Draw a UI element and its children over the 3D scene.

    Parameters
    ----------
    element : UI
        The element to bring forward.
    """
    if hasattr(element, "actors"):
        for a in element.actors:
            if hasattr(a, "material"):
                a.material.depth_test = False
    if hasattr(element, "_children"):
        for c in element._children:
            bring_ui_to_front(c)


if hasattr(timeline, "playback_panel"):
    bring_ui_to_front(timeline.playback_panel)


if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene,
        size=(1280, 768),
        title="FURY City Drone Shot — Action Stunt Sequence",
    )
    showm.add_animation(timeline)
    keep_chunks_culled(showm)
    showm.start()
