"""
=============================================
City Drone Shot: Cinematic Flythrough in FURY
=============================================

A cinematic drone fly-through of a procedurally generated 3D city using
FURY's ``fury.motion`` animation system. The camera path is defined via
``CameraAnimation`` keyframes with cubic spline interpolation for perfectly
smooth, cinematic motion. It tracks cars, climbs skyscrapers, and performs stunts!

3D models are loaded from Kenney's Car Kit and City Kit (low-poly OBJ files).
Static city geometry (thousands of road tiles and buildings) is heavily optimized
using NumPy concatenation to batch them into a road actor and a grid of building chunks,
enabling buttery smooth rendering and allowing for high-density animated traffic.

A custom pure-Python/NumPy OBJ parser is used to completely eliminate VTK
dependencies while preserving UV texture coordinates.

The drone never enters a building: its spline is baked once at start-up into a
dense path that rises smoothly over anything in the way, so playback stays
perfectly smooth (no per-frame collision response).

Assets: Kenney packs (CC0, https://kenney.nl/assets) are expected in a
``kenny_assets`` folder next to this script. Set ``FURY_KENNEY_ASSETS`` to use
another location.
"""

import logging
import os
import numpy as np
from scipy.ndimage import maximum_filter1d

from fury import actor, window
from fury.lib import DirectionalLight
from fury.motion import (
    Animation,
    CameraAnimation,
    Timeline,
    cubic_spline_interpolator,
    linear_interpolator,
)
from fury.motion.helpers import get_timestamps_from_keyframes, get_values_from_keyframes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Asset Paths
# ===========
ASSETS_DIR = os.environ.get(
    "FURY_KENNEY_ASSETS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kenny_assets"),
)
CAR_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_car-kit/Models/OBJ format")
CITY_KIT_DIR = os.path.join(
    ASSETS_DIR, "kenney_city-kit-commercial_2.1/Models/OBJ format"
)
ROAD_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_city-kit-roads/Models/OBJ format")

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

    with open(path, "r") as f:
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


def get_base_mesh(directory, filename):
    """
    Return a model's arrays, reading the file only the first time.

    Parameters
    ----------
    directory : str
        Folder holding the model.
    filename : str
        OBJ file name.

    Returns
    -------
    tuple of ndarray
        The arrays returned by ``load_obj``.
    """
    key = (directory, filename)
    if key not in _MODEL_CACHE:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Download the Kenney packs (https://kenney.nl/assets) "
                "into the assets folder, or set FURY_KENNEY_ASSETS."
            )
        v, f, uv = load_obj(path)
        _MODEL_CACHE[key] = (v, f, uv)
    return _MODEL_CACHE[key]


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


def instance_bounds(directory, filename, scale, pos, rot_y):
    """
    Return the world-space bounding box of a placed model.

    Parameters
    ----------
    directory : str
        Folder holding the model.
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
    v = transform_vertices(get_base_mesh(directory, filename)[0], scale, pos, rot_y)
    return np.concatenate([v.min(axis=0), v.max(axis=0)])


def create_batched_actor(directory, instances, texture_file=None):
    """
    Merge many placed copies of models into one textured actor.

    Parameters
    ----------
    directory : str
        Folder holding the models.
    instances : list of tuple
        ``(filename, scale, position, rotation_y)`` for each copy.
    texture_file : str, optional
        Texture image path, or None for an untextured actor.

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
        v, f, uv = get_base_mesh(directory, filename)
        v = transform_vertices(v, scale, pos, rot_y)

        all_verts.append(v)
        all_faces.append(f + v_offset)
        if uv is not None:
            all_uvs.append(uv)

        v_offset += len(v)

    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    uvs = np.concatenate(all_uvs) if len(all_uvs) > 0 else None

    tex_path = None
    if texture_file:
        tex_path = os.path.join(directory, "Textures", texture_file)
        if not os.path.exists(tex_path):
            tex_path = os.path.join(
                os.path.dirname(directory), "Textures", texture_file
            )

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
        v, f, uv = get_base_mesh(CAR_KIT_DIR, filename)
        v = transform_vertices(v, CAR_SCALE, (0.0, 0.0, 0.0), rotation_y)
        texture = os.path.join(CAR_KIT_DIR, "Textures", "colormap.png")
        template = actor.surface(v, f, texture=texture, texture_coords=uv)
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
                box = instance_bounds(
                    CITY_KIT_DIR, model_name, scale, (bx, 0.0, bz), rot_angle
                )
                half_x = (box[3] - box[0]) / 2.0
                half_z = (box[5] - box[2]) / 2.0
                bx = block_cx + float(
                    np.clip(dx, -(BLOCK_HALF_LIMIT - half_x), BLOCK_HALF_LIMIT - half_x)
                )
                bz = block_cz + float(
                    np.clip(dz, -(BLOCK_HALF_LIMIT - half_z), BLOCK_HALF_LIMIT - half_z)
                )
                box = instance_bounds(
                    CITY_KIT_DIR, model_name, scale, (bx, 0.0, bz), rot_angle
                )

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
            CITY_KIT_DIR,
            "building-skyscraper-c.obj",
            colossal_scale,
            (colossal_x, 0.0, colossal_z),
            0.0,
        )
    )

    logger.info(f"Batching {len(road_instances)} road tiles...")
    roads_actor = create_batched_actor(
        ROAD_KIT_DIR, road_instances, texture_file="colormap.png"
    )
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
        chunk = create_batched_actor(
            CITY_KIT_DIR, [m[0] for m in members], texture_file="colormap.png"
        )
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
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

ANIMATION_DURATION = 45.0
TRAFFIC_EXTENT = 900.0  # Cars wrap around this far out, well beyond the city edge
NUM_CARS = 75
HERO_LANE = -4.0  # Right-hand lane for traffic heading +Z on the X=0 avenue
car_animations = []

# Add a "Hero Car" for the drone to follow in the first scene
hero_car = create_car("hatchback-sports.obj", position=(HERO_LANE, 0.0, -380.0))
scene.add(hero_car)
hero_anim = Animation(actors=hero_car, loop=True)
# Paces the hero car to the camera's tracking shot
hero_speed = 1985.0
hero_anim.set_position(0.0, np.array([HERO_LANE, 0.0, -380.0]))
hero_anim.set_position(
    ANIMATION_DURATION, np.array([HERO_LANE, 0.0, -380.0 + hero_speed])
)
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

###############################################################################
# Camera Animation: Action Stunt Sequence
# =======================================

camera_anim = CameraAnimation(loop=True)

# Cinematic 45s action sequence mathematically paced for constant speed!
camera_positions = {
    0.0: np.array([0.0, 40.0, -430.0]),  # Behind car, centred on the avenue
    1.5: np.array([0.0, 40.0, -355.0]),  # Hold the avenue centre line
    3.0: np.array([0.0, 40.0, -280.0]),  # Clamp spline to horizontal at loop start
    5.0: np.array([0.0, 40.0, -201.0]),  # Hold the avenue centre line
    6.8: np.array([0.0, 40.0, -130.0]),  # Tracking car
    9.5: np.array([-30.0, 70.0, -40.0]),  # Rise away from the avenue
    14.4: np.array([-140.0, 150.0, 140.0]),  # Panorama
    19.0: np.array(
        [100.0, 140.0, -30.0]
    ),  # Swing south, around the Colossal Skyscraper
    23.6: np.array(
        [250.0, 50.0, 105.0]
    ),  # Approach Skyscraper from a majestic distance
    28.0: np.array([250.0, 240.0, 105.0]),  # Climb alongside it
    30.0: np.array([250.0, 280.0, 140.0]),  # Flip over roof
    37.6: np.array([150.0, 80.0, -150.0]),  # Dive back down
    41.5: np.array([0.0, 42.0, -320.0]),  # Level out of dive above the avenue
    43.25: np.array([0.0, 41.0, -375.0]),  # Hold the avenue centre line
    45.0: np.array([0.0, 40.0, -430.0]),  # Loop back horizontally
}

camera_focals = {
    0.0: np.array([HERO_LANE, 5.0, -380.0]),  # Look directly at Hero Car
    3.0: np.array([HERO_LANE, 5.0, -230.0]),  # Look directly at Hero Car
    6.8: np.array([HERO_LANE, 5.0, -80.0]),  # Track car maintaining 50 units ahead
    14.4: np.array([0.0, 50.0, 70.0]),  # Look down at city
    19.0: np.array([105.0, 90.0, 105.0]),  # Keep the Colossal Skyscraper in frame
    23.6: np.array([105.0, 50.0, 105.0]),  # Look at Colossal Skyscraper base
    28.0: np.array([105.0, 250.0, 105.0]),  # Look at the building center while climbing
    30.0: np.array([105.0, 300.0, 105.0]),  # Look up during flip apex
    37.6: np.array([4.0, 20.0, -250.0]),  # Look down street
    41.5: np.array([4.0, 10.0, -320.0]),  # Look down street
    45.0: np.array([HERO_LANE, 5.0, -380.0]),
}

# The View-Up Vector controls camera tilt and barrel rolls!
# We step through 45-degree angles to create a clean pitch-loop
# and avoid passing through (0,0,0) which crashes linear interpolators.
camera_view_ups = {
    0.0: np.array([0.0, 1.0, 0.0]),
    28.0: np.array([0.0, 1.0, 0.0]),
    # The Pitch-Loop Stunt over the roof!
    28.5: np.array([0.0, 0.707, -0.707]),
    29.0: np.array([0.0, 0.0, -1.0]),
    29.5: np.array([0.0, -0.707, -0.707]),
    30.0: np.array([0.0, -1.0, 0.0]),  # Apex upside down!
    31.9: np.array([0.0, -0.707, 0.707]),
    33.8: np.array([0.0, 0.0, 1.0]),  # Pitching forward
    35.7: np.array([0.0, 0.707, 0.707]),
    37.6: np.array([0.0, 1.0, 0.0]),  # Right side up!
    45.0: np.array([0.0, 1.0, 0.0]),
}


def binary_search_interpolator(keyframes):
    """
    Create a linear interpolator that finds keyframes by binary search.

    Parameters
    ----------
    keyframes : dict
        Keyframe data to interpolate.

    Returns
    -------
    function
        Takes a time and returns the interpolated value at that time.
    """
    timestamps = np.asarray(get_timestamps_from_keyframes(keyframes), dtype=np.float64)
    values = np.asarray(get_values_from_keyframes(keyframes), dtype=np.float64)

    def interpolate(t):
        i = int(
            np.clip(
                np.searchsorted(timestamps, t, side="right") - 1, 0, len(timestamps) - 2
            )
        )
        alpha = np.clip(
            (t - timestamps[i]) / (timestamps[i + 1] - timestamps[i]), 0.0, 1.0
        )
        return values[i] * (1.0 - alpha) + values[i + 1] * alpha

    return interpolate


def bake_collision_free_path(
    keyframes, bounds, clearance, sample_rate=30.0, ease_seconds=1.5
):
    """
    Sample a keyframed path and lift it clear of every building.

    Each sample inside a building footprint (grown by ``clearance``) gets a
    required lift that puts it above that roof. A sliding maximum followed by a
    Hann kernel no wider than that maximum keeps every lift at least as large as
    required, while easing it in and out so the drone climbs ahead of time.
    Only height changes, and buildings stand on the ground, so a lift can never
    push the drone into another building.

    Parameters
    ----------
    keyframes : dict
        Maps timestamp to an (x, y, z) position.
    bounds : ndarray, shape (N, 6)
        Building boxes as (xmin, ymin, zmin, xmax, ymax, zmax).
    clearance : float
        Minimum gap to keep from any building.
    sample_rate : float, optional
        Samples per second of the baked path.
    ease_seconds : float, optional
        Time taken to ease into and out of a lift.

    Returns
    -------
    dict
        Maps timestamp to the collision-free (x, y, z) position.
    """
    raw = CameraAnimation()
    raw.set_position_keyframes(keyframes)
    raw.set_position_interpolator(cubic_spline_interpolator)

    duration = max(keyframes)
    count = int(round(duration * sample_rate))
    times = np.linspace(0.0, duration, count, endpoint=False)
    path = np.array([raw.get_position(t) for t in times], dtype=np.float64)

    inside = (
        (path[:, 0, None] > bounds[:, 0] - clearance)
        & (path[:, 0, None] < bounds[:, 3] + clearance)
        & (path[:, 2, None] > bounds[:, 2] - clearance)
        & (path[:, 2, None] < bounds[:, 5] + clearance)
    )
    required = np.where(inside, bounds[:, 4] + clearance, -np.inf).max(axis=1)
    lift = np.maximum(required - path[:, 1], 0.0)

    radius = max(1, int(round(ease_seconds * sample_rate)))
    lift = maximum_filter1d(lift, size=2 * radius + 1, mode="wrap")
    kernel = np.hanning(2 * radius + 3)[1:-1]
    kernel /= kernel.sum()
    wrapped = np.concatenate([lift[-radius:], lift, lift[:radius]])
    path[:, 1] += np.convolve(wrapped, kernel, mode="valid")

    baked = {float(t): p for t, p in zip(times, path, strict=True)}
    baked[float(duration)] = path[0]  # Close the loop exactly
    return baked


camera_positions = bake_collision_free_path(
    camera_positions, BUILDING_BOUNDS, DRONE_CLEARANCE
)

camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focals)
camera_anim.set_view_up_keyframes(camera_view_ups)

# The baked path is already dense and smooth; linear keeps it exactly on the
# collision-free samples, where a new spline could overshoot into a building.
camera_anim.set_position_interpolator(binary_search_interpolator)
camera_anim.set_focal_interpolator(cubic_spline_interpolator)
# Use linear for view_up to avoid spline duplicate value errors
camera_anim.set_view_up_interpolator(linear_interpolator)


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
