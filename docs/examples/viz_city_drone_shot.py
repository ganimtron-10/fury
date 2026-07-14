"""
=============================================
City Drone Shot: Cinematic Flythrough in FURY
=============================================

A cinematic drone fly-through of a procedurally generated 3D city using
FURY's ``fury.motion`` animation system. The camera path is defined via
``CameraAnimation`` keyframes with cubic spline interpolation, traffic
is animated via ``Animation`` objects, and the entire show is orchestrated
by a ``Timeline`` with an interactive playback panel.

3D models are loaded from Kenney's Car Kit and City Kit (low-poly OBJ files)
via ``fury.io.read_mesh`` + ``actor.surface``, combined with polyxios
tree models for vegetation.

Features:
    - fury.motion Timeline with PlaybackPanel (play/pause/seek/speed)
    - CameraAnimation with cubic spline keyframes through 6 flight phases
    - Kenney city-kit buildings and skyscrapers (.obj)
    - Kenney car-kit vehicles (.obj) animated along roads
    - Polyxios tree.obj for street vegetation
    - Organic random city layout with main boulevard and side streets
"""

import logging
import os

import numpy as np
import polyxios as px

from fury import actor, window
from fury.io import read_mesh
from fury.motion import (
    Animation,
    CameraAnimation,
    Timeline,
    cubic_spline_interpolator,
    linear_interpolator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Asset Paths
# ===========

CAR_KIT_DIR = "/mnt/d/FuryWorkspace/fury-data/kenney_car-kit/Models/OBJ format"
CITY_KIT_DIR = (
    "/mnt/d/FuryWorkspace/fury-data/kenney_city-kit-commercial_2.1/Models/OBJ format"
)

###############################################################################
# Model Loading Helper
# ====================

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


def load_kenney_model(
    directory, filename, texture_file, scale=1.0, position=None, rotation_y=0.0
):
    """Load a Kenney .obj model with textures as a FURY surface actor."""
    import vtkmodules.vtkCommonCore as vtk_core

    vtk_core.vtkObject.GlobalWarningDisplayOff()

    path = os.path.join(directory, filename)
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()

    verts = vtk_to_numpy(polydata.GetPoints().GetData()).copy()

    # Rotate vertices around Y axis if needed (avoids FURY transform quirks for animations)
    if rotation_y != 0.0:
        angle = np.radians(rotation_y)
        c, s = np.cos(angle), np.sin(angle)
        verts_x = verts[:, 0] * c + verts[:, 2] * s
        verts_z = -verts[:, 0] * s + verts[:, 2] * c
        verts[:, 0] = verts_x
        verts[:, 2] = verts_z

    # VTK Polys might contain triangles or quads; assume triangles for simplicity
    cells = vtk_to_numpy(polydata.GetPolys().GetData())
    faces = cells.reshape(-1, 4)[:, 1:4]

    uvs = None
    tcoords = polydata.GetPointData().GetTCoords()
    if tcoords is not None:
        uvs = vtk_to_numpy(tcoords)
    elif polydata.GetPointData().HasArray("colormap"):
        # Some OBJ loaders put texture coords in 'colormap' array
        uvs = vtk_to_numpy(polydata.GetPointData().GetArray("colormap"))

    # Build FURY surface actor with texture
    tex_path = os.path.join(directory, "Textures", texture_file)
    if not os.path.exists(tex_path):
        # Fallback to Models/Textures
        parent_dir = os.path.dirname(directory)
        tex_path = os.path.join(parent_dir, "Textures", texture_file)

    surf = actor.surface(verts, faces, texture=tex_path, texture_coords=uvs)

    surf.local.scale = [scale, scale, scale]
    if position is not None:
        surf.local.position = list(position)

    return surf


def load_polyxios_model(filename, scale=1.0, position=None, color=None):
    """Load a polyxios .obj model, normalize to unit size, return actor."""
    path = px.fetch(filename)
    vertices, faces, colors = read_mesh(path)

    # Normalize to unit bounding box
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices = vertices - center
    extent = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    if extent > 1e-6:
        vertices = vertices / extent

    if color is not None:
        colors = np.tile(np.array(color, dtype=np.float32), (len(vertices), 1))
    elif colors is None:
        colors = np.full((len(vertices), 3), 0.5, dtype=np.float32)

    surf = actor.surface(vertices, faces, colors=colors)
    surf.local.scale = [scale, scale, scale]
    if position is not None:
        surf.local.position = list(position)

    return surf


###############################################################################
# Scene Setup
# ===========

scene = window.Scene()
scene.background = (0.08, 0.06, 0.18)  # Deep twilight

###############################################################################
# Ground Plane

ground = actor.box(
    centers=np.array([[0.0, -0.25, 0.0]]),
    colors=(0.07, 0.07, 0.09),
    scales=(500, 0.5, 500),
)
scene.add(ground)

###############################################################################
# Sun / Atmosphere

sun = actor.sphere(
    centers=np.array([[0.0, 0.0, 0.0]]),
    colors=(1.0, 0.85, 0.5),
    radii=20.0,
)
sun.local.position = [180.0, 160.0, 250.0]
scene.add(sun)

###############################################################################
# City Layout Constants

np.random.seed(42)

CITY_RADIUS = 180.0
BUILDING_SCALE = 12.0  # Kenney models are ~1-4.5 units, scale up

# Available building models from Kenney city kit
BUILDING_MODELS = [
    "building-a.obj",
    "building-b.obj",
    "building-c.obj",
    "building-d.obj",
    "building-e.obj",
    "building-f.obj",
    "building-g.obj",
    "building-h.obj",
]

SKYSCRAPER_MODELS = [
    "building-skyscraper-a.obj",
    "building-skyscraper-b.obj",
    "building-skyscraper-c.obj",
    "building-skyscraper-d.obj",
    "building-skyscraper-e.obj",
]

# Available car models from Kenney car kit
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
# Roads
# =====


def generate_roads():
    """Generate road surfaces and lane markings."""
    # Main boulevard along X axis
    boulevard = actor.box(
        centers=np.array([[0.0, 0.0, 0.0]]),
        colors=(0.14, 0.14, 0.16),
        scales=(CITY_RADIUS * 2.2, 0.1, 14.0),
    )
    boulevard.local.position = [0.0, 0.02, 0.0]
    scene.add(boulevard)

    # Lane markings
    for dx in np.arange(-CITY_RADIUS, CITY_RADIUS, 7.0):
        dash = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=(0.85, 0.85, 0.55),
            scales=(3.5, 0.02, 0.35),
        )
        dash.local.position = [dx, 0.12, 0.0]
        scene.add(dash)

    # Cross road through center along Z axis
    cross_road = actor.box(
        centers=np.array([[0.0, 0.0, 0.0]]),
        colors=(0.14, 0.14, 0.16),
        scales=(14.0, 0.1, CITY_RADIUS * 2.0),
    )
    cross_road.local.position = [0.0, 0.02, 0.0]
    scene.add(cross_road)

    # Cross road lane markings
    for dz in np.arange(-CITY_RADIUS, CITY_RADIUS, 7.0):
        dash = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=(0.85, 0.85, 0.55),
            scales=(0.35, 0.02, 3.5),
        )
        dash.local.position = [0.0, 0.12, dz]
        scene.add(dash)


generate_roads()
logger.info("Roads generated")

###############################################################################
# Buildings (Kenney City Kit)
# ===========================
# Place buildings on a jittered grid to avoid overlaps and keep roads clear.


def generate_buildings():
    """Generate organically placed Kenney buildings on a jittered grid."""
    placed = 0
    grid_step = 35.0

    for gx in np.arange(-CITY_RADIUS + 20, CITY_RADIUS - 20, grid_step):
        for gz in np.arange(-CITY_RADIUS + 20, CITY_RADIUS - 20, grid_step):
            # Avoid the main boulevard and cross road corridors
            if abs(gx) < 22.0 or abs(gz) < 22.0:
                continue

            # Random chance to leave a spot empty for a park
            if np.random.random() < 0.15:
                continue

            # Add jitter to make it look organic
            px_pos = gx + np.random.uniform(-6.0, 6.0)
            pz_pos = gz + np.random.uniform(-6.0, 6.0)

            # Use skyscrapers near center, regular buildings further out
            dist = np.sqrt(px_pos**2 + pz_pos**2)
            if dist < CITY_RADIUS * 0.45 and np.random.random() < 0.5:
                model_name = SKYSCRAPER_MODELS[
                    np.random.randint(0, len(SKYSCRAPER_MODELS))
                ]
                scale = np.random.uniform(BUILDING_SCALE * 1.0, BUILDING_SCALE * 1.5)
            else:
                model_name = BUILDING_MODELS[np.random.randint(0, len(BUILDING_MODELS))]
                scale = np.random.uniform(BUILDING_SCALE * 0.7, BUILDING_SCALE * 1.2)

            rot_angle = np.random.choice([0, 90, 180, 270])

            bldg = load_kenney_model(
                CITY_KIT_DIR,
                model_name,
                "variation-a.png",
                scale=scale,
                position=(px_pos, 0.0, pz_pos),
                rotation_y=rot_angle,
            )
            scene.add(bldg)
            placed += 1

    logger.info(f"Placed {placed} Kenney buildings")


generate_buildings()

###############################################################################
# Street Lights
# =============


def generate_street_lights():
    """Generate street lights along the main boulevard and cross road."""
    positions = []

    # Along boulevard
    for dx in np.arange(-CITY_RADIUS + 10, CITY_RADIUS - 10, 22.0):
        for side_z in [-8.0, 8.0]:
            positions.append((dx, side_z))

    # Along cross road
    for dz in np.arange(-CITY_RADIUS + 20, CITY_RADIUS - 20, 30.0):
        if abs(dz) > 10:  # Skip intersection area
            for side_x in [-7.5, 7.5]:
                positions.append((side_x, dz))

    for lx, lz in positions:
        pole = actor.cylinder(
            centers=np.array([[0.0, 0.0, 0.0]]),
            directions=np.array([[0.0, 1.0, 0.0]]),
            colors=(0.3, 0.3, 0.32),
            height=5.5,
            radii=0.12,
        )
        pole.local.position = [lx, 2.75, lz]
        scene.add(pole)

        bulb = actor.sphere(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=(1.0, 0.9, 0.45),
            radii=0.4,
        )
        bulb.local.position = [lx, 5.7, lz]
        scene.add(bulb)


generate_street_lights()
logger.info("Street lights placed")

###############################################################################
# Trees (Polyxios tree.obj)
# =========================

tree_positions = [
    (-80, 16),
    (-50, 16),
    (-20, -16),
    (15, -16),
    (45, 16),
    (75, -16),
    (105, 16),
    (-105, -16),
    (125, -16),
    (-125, 16),
    (-60, -22),
    (30, 22),
    (80, -22),
    (-40, 26),
    (60, -26),
]

for tx, tz in tree_positions:
    tree_color = (
        0.12 + np.random.uniform(-0.02, 0.02),
        0.40 + np.random.uniform(-0.08, 0.08),
        0.15 + np.random.uniform(-0.02, 0.02),
    )
    scale = np.random.uniform(6.0, 10.0)
    # Unit box trees need to be moved up by scale/2 to rest on the ground
    tree = load_polyxios_model(
        "tree.obj",
        color=tree_color,
        scale=scale,
        position=(tx, scale / 2.0, tz),
    )
    scene.add(tree)

logger.info(f"Placed {len(tree_positions)} trees")

###############################################################################
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

ANIMATION_DURATION = 45.0
car_animations = []

# Car routes: (model_index, start_coord, lane_offset, speed_factor, direction, is_x_axis)
car_routes = [
    (0, -150, 4.0, 1.0, 1, True),  # sedan going right on X
    (1, -120, -4.0, 0.85, 1, True),  # taxi going right on X
    (2, 150, 4.0, 0.9, -1, True),  # suv going left on X
    (3, 130, -4.0, 1.1, -1, True),  # truck going left on X
    (4, -100, 3.0, 0.75, 1, True),  # police going right on X
    (5, 80, -3.0, 1.2, -1, True),  # van going left on X
    (6, -120, 4.0, 0.95, 1, False),  # hatchback going up on Z
    (7, 100, -4.0, 1.1, -1, False),  # ambulance going down on Z
    (0, -80, 3.5, 1.05, 1, False),  # sedan 2 up on Z
    (1, 60, -3.5, 0.78, -1, False),  # taxi 2 down on Z
    (2, -40, 4.0, 0.92, 1, False),  # suv 2 up on Z
]

for mi, start_coord, lane, speed_f, direction, is_x_axis in car_routes:
    model_name = CAR_MODELS[mi % len(CAR_MODELS)]

    # Calculate initial rotation so the model's front (+Z natively) aligns with motion
    if is_x_axis:
        rot_y = 90.0 if direction > 0 else -90.0
        start_pos = (start_coord, 0.0, lane)
    else:
        rot_y = 0.0 if direction > 0 else 180.0
        start_pos = (lane, 0.0, start_coord)

    car = load_kenney_model(
        CAR_KIT_DIR,
        model_name,
        "colormap.png",
        scale=3.5,
        position=start_pos,
        rotation_y=rot_y,
    )
    scene.add(car)

    # Create position keyframes for driving animation
    car_anim = Animation(actors=car, loop=True)
    travel_dist = 280.0 * speed_f
    n_kf = 10

    for ki in range(n_kf + 1):
        t = (ki / n_kf) * ANIMATION_DURATION
        progress = ki / n_kf
        val = start_coord + direction * travel_dist * progress
        # Wrap within bounds
        val = ((val + CITY_RADIUS + 20) % (2 * CITY_RADIUS + 40)) - CITY_RADIUS - 20

        if is_x_axis:
            pos = np.array([val, 0.0, lane])
        else:
            pos = np.array([lane, 0.0, val])

        car_anim.set_position(t, pos)

    car_anim.set_position_interpolator(cubic_spline_interpolator)
    car_animations.append(car_anim)

logger.info(f"Created {len(car_animations)} animated Kenney cars")

###############################################################################
# Camera Animation (CameraAnimation with cubic spline keyframes)
# ==============================================================

camera_anim = CameraAnimation(loop=True)

# Camera path explicitly designed to stay above the roads (X=0 and Z=0 corridors)
camera_positions = {
    # Phase 1: Drive down Boulevard (+X direction)
    0.0: np.array([-140.0, 3.5, 0.0]),
    3.5: np.array([-80.0, 3.5, 0.0]),
    7.0: np.array([-20.0, 4.0, 0.0]),
    # Phase 2: Ascend at intersection
    9.0: np.array([0.0, 25.0, 0.0]),
    11.0: np.array([0.0, 60.0, 0.0]),
    13.0: np.array([0.0, 90.0, 0.0]),
    # Phase 3: Panorama Orbit
    15.0: np.array([30.0, 95.0, 30.0]),
    17.0: np.array([65.0, 95.0, 65.0]),
    20.0: np.array([0.0, 85.0, 90.0]),
    # Phase 4: Dive into Cross Road (-Z direction)
    22.0: np.array([0.0, 65.0, 60.0]),
    24.0: np.array([0.0, 15.0, 30.0]),
    26.0: np.array([0.0, 20.0, 10.0]),
    28.0: np.array([0.0, 25.0, -5.0]),
    # Phase 5: Fly down Cross Road (-Z direction)
    30.0: np.array([0.0, 12.0, -20.0]),
    32.0: np.array([0.0, 12.0, -40.0]),
    34.0: np.array([0.0, 14.0, -60.0]),
    36.0: np.array([0.0, 25.0, -80.0]),
    # Phase 6: Sweep back to start
    38.0: np.array([-60.0, 50.0, -80.0]),
    40.0: np.array([-90.0, 70.0, -60.0]),
    43.0: np.array([-120.0, 80.0, -40.0]),
    45.0: np.array([-140.0, 3.5, 0.0]),  # Loop back to start
}

# Focal targets (what the camera looks at)
camera_focals = {
    0.0: np.array([-80.0, 3.0, 0.0]),
    3.5: np.array([-20.0, 3.0, 0.0]),
    7.0: np.array([0.0, 8.0, 0.0]),
    9.0: np.array([0.0, 25.0, -20.0]),
    11.0: np.array([0.0, 40.0, -30.0]),
    13.0: np.array([0.0, 40.0, -30.0]),
    15.0: np.array([0.0, 40.0, 0.0]),
    17.0: np.array([0.0, 40.0, 0.0]),
    20.0: np.array([0.0, 40.0, 0.0]),
    22.0: np.array([0.0, 10.0, 20.0]),
    24.0: np.array([0.0, 10.0, -20.0]),
    26.0: np.array([0.0, 10.0, -30.0]),
    28.0: np.array([0.0, 10.0, -40.0]),
    30.0: np.array([0.0, 10.0, -60.0]),
    32.0: np.array([0.0, 10.0, -80.0]),
    34.0: np.array([0.0, 10.0, -100.0]),
    36.0: np.array([0.0, 10.0, -120.0]),
    38.0: np.array([0.0, 0.0, 0.0]),
    40.0: np.array([0.0, 0.0, 0.0]),
    43.0: np.array([0.0, 0.0, 0.0]),
    45.0: np.array([-80.0, 3.0, 0.0]),
}

camera_view_ups = {
    0.0: np.array([0.0, 1.0, 0.0]),
    20.0: np.array([0.0, 1.0, 0.0]),
    # Barrel roll during stunt dive
    21.5: np.array([0.0, 1.0, 0.0]),
    22.0: np.array([0.5, 0.87, 0.0]),
    22.5: np.array([1.0, 0.0, 0.0]),
    23.0: np.array([0.0, -1.0, 0.0]),
    23.5: np.array([-1.0, 0.0, 0.0]),
    24.0: np.array([0.0, 1.0, 0.0]),
    45.0: np.array([0.0, 1.0, 0.0]),
}

camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focals)
camera_anim.set_view_up_keyframes(camera_view_ups)

camera_anim.set_position_interpolator(cubic_spline_interpolator)
camera_anim.set_focal_interpolator(linear_interpolator)
camera_anim.set_view_up_interpolator(linear_interpolator)

logger.info("Camera animation keyframes set")

###############################################################################
# Timeline Assembly
# =================

timeline = Timeline(playback_panel=True, loop=True)
timeline.add_animation(camera_anim)

for ca in car_animations:
    timeline.add_animation(ca)

logger.info(
    f"Timeline assembled: camera + {len(car_animations)} car animations, "
    f"duration={ANIMATION_DURATION}s"
)

###############################################################################
# Application Entry Point
# =======================

if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene,
        size=(1280, 768),
        title="FURY City Drone Shot — Kenney Assets + fury.motion",
    )
    showm.add_animation(timeline)
    showm.start()
