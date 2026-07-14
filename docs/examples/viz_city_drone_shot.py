"""
=============================================
City Drone Shot: Cinematic Flythrough in FURY
=============================================

A cinematic drone fly-through of a procedurally generated 3D city using
FURY's ``fury.motion`` animation system. The camera path is defined via
``CameraAnimation`` keyframes with linear interpolation to prevent clipping,
traffic is animated via ``Animation`` objects, and the entire show is 
orchestrated by a ``Timeline`` with an interactive playback panel.

3D models are loaded from Kenney's Car Kit and City Kit (low-poly OBJ files).
Static city geometry (thousands of road tiles and buildings) is heavily optimized
using NumPy concatenation to batch them into just two massive actors,
enabling buttery smooth rendering and allowing for high-density animated traffic
without relying on heavy VTK dependencies.
"""

import logging
import os
import numpy as np

from fury import actor, window
from fury.motion import Animation, CameraAnimation, Timeline, cubic_spline_interpolator, linear_interpolator
from fury.io import read_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Asset Paths
# ===========
ASSETS_DIR = "/mnt/d/FuryWorkspace/fury/docs/examples/kenny_assets"
CAR_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_car-kit/Models/OBJ format")
CITY_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_city-kit-commercial_2.1/Models/OBJ format")
ROAD_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_city-kit-roads/Models/OBJ format")

###############################################################################
# Model Loading & Optimization Helpers
# ====================================

_MODEL_CACHE = {}

def get_base_mesh(directory, filename):
    """Cache and return the base NumPy arrays for a model to avoid disk I/O."""
    key = (directory, filename)
    if key not in _MODEL_CACHE:
        path = os.path.join(directory, filename)
        v, f, c = read_mesh(path)
        _MODEL_CACHE[key] = (v, f, c)
    return _MODEL_CACHE[key]

def create_batched_actor(directory, instances, default_color=None, extract_colors=True):
    """
    Batch thousands of meshes using NumPy concatenation.
    Returns a single FURY actor for extreme performance.
    """
    all_verts = []
    all_faces = []
    all_colors = []
    v_offset = 0
    
    for filename, scale, pos, rot_y in instances:
        v, f, c = get_base_mesh(directory, filename)
        v = v.copy()
        
        # Apply scale
        v = v * scale
        
        # Apply Y-rotation
        if rot_y != 0.0:
            angle = np.radians(rot_y)
            c_val, s_val = np.cos(angle), np.sin(angle)
            v_x = v[:, 0] * c_val + v[:, 2] * s_val
            v_z = -v[:, 0] * s_val + v[:, 2] * c_val
            v[:, 0] = v_x
            v[:, 2] = v_z
            
        # Apply translation
        v = v + pos
        
        all_verts.append(v)
        all_faces.append(f + v_offset)
        
        if extract_colors and c is not None and c.shape == v.shape and not (c.max() <= 1.0 and c.min() >= -1.0 and np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=0.1)):
            # Valid vertex colors found (ignoring normal-mapped colors which lie on unit sphere)
            all_colors.append(c)
        else:
            # Fallback color
            color = default_color if default_color is not None else (0.8, 0.8, 0.8)
            all_colors.append(np.full((len(v), 3), color, dtype=np.float32))
            
        v_offset += len(v)
        
    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    colors = np.concatenate(all_colors)
    
    # Ensure colors are RGBA float32 for pygfx
    colors_rgba = np.ones((len(colors), 4), dtype=np.float32)
    colors_rgba[:, :3] = colors
    
    surf = actor.surface(verts, faces, colors=colors_rgba)
    return surf

def load_kenney_model(directory, filename, scale=1.0, position=None, rotation_y=0.0, color=None):
    """Load a single Kenney model dynamically (used for animated cars)."""
    v, f, c = get_base_mesh(directory, filename)
    v = v.copy()
    
    # Pre-rotate vertices for dynamic objects so they drive forward natively
    if rotation_y != 0.0:
        angle = np.radians(rotation_y)
        c_val, s_val = np.cos(angle), np.sin(angle)
        v_x = v[:, 0] * c_val + v[:, 2] * s_val
        v_z = -v[:, 0] * s_val + v[:, 2] * c_val
        v[:, 0] = v_x
        v[:, 2] = v_z

    if c is not None and c.shape == v.shape and not (c.max() <= 1.0 and c.min() >= -1.0 and np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=0.1)):
        # Valid vertex colors
        colors = c
    else:
        fallback = color if color is not None else (0.5, 0.5, 0.5)
        colors = np.full((len(v), 3), fallback, dtype=np.float32)
        
    colors_rgba = np.ones((len(colors), 4), dtype=np.float32)
    colors_rgba[:, :3] = colors

    surf = actor.surface(v, f, colors=colors_rgba)
    surf.local.scale = [scale, scale, scale]
    if position is not None:
        surf.local.position = list(position)

    return surf

###############################################################################
# Scene Setup
# ===========

scene = window.Scene()
# Sunny day sky blue background
scene.background = (0.53, 0.81, 0.92)  

# Ground Plane
ground = actor.box(
    centers=np.array([[0.0, -0.25, 0.0]]),
    colors=(0.4, 0.45, 0.4),  # Grass color for the ground under the city
    scales=(3000, 0.5, 3000),
)
scene.add(ground)

# Sun
sun = actor.sphere(
    centers=np.array([[0.0, 0.0, 0.0]]),
    colors=(1.0, 1.0, 0.8),
    radii=60.0,
)
sun.local.position = [400.0, 300.0, 500.0]
scene.add(sun)

###############################################################################
# City Layout Constants

np.random.seed(42)

BLOCK_SIZE = 70.0
ROAD_WIDTH = 14.0
BUILDING_SCALE = 12.0
GRID_SIZE = 8  # Expanded grid size (17x17 roads)

BUILDING_MODELS = [
    f"building-{c}.obj" for c in ['a','b','c','d','e','f','g','h']
]
SKYSCRAPER_MODELS = [
    f"building-skyscraper-{c}.obj" for c in ['a','b','c','d','e']
]
CAR_MODELS = [
    "sedan.obj", "taxi.obj", "suv.obj", "truck.obj",
    "police.obj", "van.obj", "hatchback-sports.obj", "ambulance.obj"
]

###############################################################################
# Roads & Buildings Generation (Batched)
# ======================================

def generate_static_city():
    """Generate dense NYC grid roads and buildings into optimized batched actors."""
    road_instances = []
    building_instances = []

    # 1. Generate Roads
    for ix in range(-GRID_SIZE, GRID_SIZE + 1):
        for iz in range(-GRID_SIZE, GRID_SIZE + 1):
            cx = ix * BLOCK_SIZE
            cz = iz * BLOCK_SIZE
            
            # Intersection
            road_instances.append(
                ("road-crossroad.obj", ROAD_WIDTH, (cx, 0.0, cz), 0.0)
            )

            # Straight roads to the right (if not at edge)
            if ix < GRID_SIZE:
                for s in range(1, 5):  # 4 tiles between intersections
                    road_x = cx + s * ROAD_WIDTH
                    road_instances.append(
                        ("road-straight.obj", ROAD_WIDTH, (road_x, 0.0, cz), 90.0)
                    )
            
            # Straight roads down (if not at edge)
            if iz < GRID_SIZE:
                for s in range(1, 5):
                    road_z = cz + s * ROAD_WIDTH
                    road_instances.append(
                        ("road-straight.obj", ROAD_WIDTH, (cx, 0.0, road_z), 0.0)
                    )

    # 2. Generate Buildings
    for ix in range(-GRID_SIZE, GRID_SIZE):
        for iz in range(-GRID_SIZE, GRID_SIZE):
            block_cx = ix * BLOCK_SIZE + BLOCK_SIZE / 2.0
            block_cz = iz * BLOCK_SIZE + BLOCK_SIZE / 2.0
            
            # Determine if this is a downtown block (skyscrapers)
            dist_to_center = np.sqrt(block_cx**2 + block_cz**2)
            is_downtown = dist_to_center < 100.0

            # Sometimes leave a block empty for a park
            if not is_downtown and np.random.random() < 0.15:
                continue

            offsets = [
                (-16, -16), (0, -16), (16, -16),
                (-16, 0),             (16, 0),
                (-16, 16),  (0, 16),  (16, 16)
            ]
            
            # Randomly pick 2 to 4 buildings per block to spread them out and reduce congestion
            num_buildings = np.random.randint(2, 5)
            chosen_indices = np.random.choice(len(offsets), num_buildings, replace=False)
            
            for idx in chosen_indices:
                dx, dz = offsets[idx]
                bx = block_cx + dx
                bz = block_cz + dz
                
                if is_downtown and np.random.random() < 0.6:
                    model_name = SKYSCRAPER_MODELS[np.random.randint(0, len(SKYSCRAPER_MODELS))]
                    scale = np.random.uniform(BUILDING_SCALE * 1.2, BUILDING_SCALE * 1.8)
                else:
                    model_name = BUILDING_MODELS[np.random.randint(0, len(BUILDING_MODELS))]
                    scale = np.random.uniform(BUILDING_SCALE * 0.8, BUILDING_SCALE * 1.2)
                
                rot_angle = float(np.random.choice([0, 90, 180, 270]))
                building_instances.append(
                    (model_name, scale, (bx, 0.0, bz), rot_angle)
                )

    logger.info(f"Batching {len(road_instances)} road tiles...")
    roads_actor = create_batched_actor(ROAD_KIT_DIR, road_instances, extract_colors=True)
    scene.add(roads_actor)

    logger.info(f"Batching {len(building_instances)} buildings...")
    buildings_actor = create_batched_actor(CITY_KIT_DIR, building_instances, default_color=(0.85, 0.85, 0.85), extract_colors=False)
    scene.add(buildings_actor)

generate_static_city()

###############################################################################
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

ANIMATION_DURATION = 45.0
car_animations = []

# Generate cars on random road segments (Massively increased traffic!)
NUM_CARS = 75
car_routes = []
for i in range(NUM_CARS):
    is_x_axis = np.random.choice([True, False])
    # Pick a random road line
    line = np.random.randint(-GRID_SIZE + 1, GRID_SIZE) * BLOCK_SIZE
    lane = line + np.random.choice([-4.0, 4.0])
    direction = 1 if lane < line else -1
    speed_f = np.random.uniform(0.7, 1.3)
    start_coord = np.random.uniform(-GRID_SIZE * BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE)
    car_routes.append((i, start_coord, lane, speed_f, direction, is_x_axis))

for mi, start_coord, lane, speed_f, direction, is_x_axis in car_routes:
    model_name = CAR_MODELS[mi % len(CAR_MODELS)]
    
    if is_x_axis:
        rot_y = 90.0 if direction > 0 else -90.0
        start_pos = (start_coord, 0.0, lane)
    else:
        rot_y = 0.0 if direction > 0 else 180.0
        start_pos = (lane, 0.0, start_coord)

    # Random vibrant colors for cars
    random_car_color = (np.random.uniform(0.2, 1.0), np.random.uniform(0.2, 1.0), np.random.uniform(0.2, 1.0))
    
    car = load_kenney_model(
        CAR_KIT_DIR, model_name, scale=3.5, 
        position=start_pos, rotation_y=rot_y,
        color=random_car_color
    )
    scene.add(car)

    car_anim = Animation(actors=car, loop=True)
    travel_dist = 280.0 * speed_f
    n_kf = 10
    
    bound = GRID_SIZE * BLOCK_SIZE + 20

    for ki in range(n_kf + 1):
        t = (ki / n_kf) * ANIMATION_DURATION
        progress = ki / n_kf
        val = start_coord + direction * travel_dist * progress
        # Wrap within bounds
        val = ((val + bound) % (2 * bound)) - bound
        
        if is_x_axis:
            pos = np.array([val, 0.0, lane])
        else:
            pos = np.array([lane, 0.0, val])
            
        car_anim.set_position(t, pos)

    car_anim.set_position_interpolator(cubic_spline_interpolator)
    car_animations.append(car_anim)

logger.info(f"Created {len(car_animations)} animated Kenney cars")

###############################################################################
# Camera Animation
# ================

camera_anim = CameraAnimation(loop=True)

# Path navigating the dense grid
camera_positions = {
    # Start high above a road
    0.0: np.array([-210.0, 25.0, 0.0]),
    4.0: np.array([-140.0, 25.0, 0.0]),
    8.0: np.array([-70.0, 25.0, 0.0]),
    # Ascend
    10.0: np.array([0.0, 45.0, 0.0]),
    14.0: np.array([0.0, 150.0, 0.0]),
    # High altitude panorama
    18.0: np.array([150.0, 180.0, 150.0]),
    22.0: np.array([-50.0, 170.0, 220.0]),
    26.0: np.array([-180.0, 150.0, 100.0]),
    # Dive into a cross street
    28.0: np.array([-70.0, 70.0, 70.0]),
    30.0: np.array([-70.0, 25.0, 0.0]),
    # Zip down street
    34.0: np.array([-70.0, 25.0, -140.0]),
    # Ascend and fly back to start
    38.0: np.array([-140.0, 70.0, -140.0]),
    42.0: np.array([-210.0, 100.0, -70.0]),
    45.0: np.array([-210.0, 25.0, 0.0]),
}

camera_focals = {
    0.0: np.array([-140.0, 15.0, 0.0]),
    4.0: np.array([-70.0, 15.0, 0.0]),
    8.0: np.array([0.0, 15.0, 0.0]),
    10.0: np.array([0.0, 45.0, 50.0]),
    14.0: np.array([0.0, 40.0, -50.0]),
    18.0: np.array([0.0, 40.0, 0.0]),
    22.0: np.array([0.0, 40.0, 0.0]),
    26.0: np.array([0.0, 40.0, 0.0]),
    28.0: np.array([-70.0, 20.0, 0.0]),
    30.0: np.array([-70.0, 15.0, -70.0]),
    34.0: np.array([-70.0, 15.0, -210.0]),
    38.0: np.array([0.0, 15.0, 0.0]),
    42.0: np.array([0.0, 15.0, 0.0]),
    45.0: np.array([-140.0, 15.0, 0.0]),
}

camera_view_ups = {
    0.0: np.array([0.0, 1.0, 0.0]),
    20.0: np.array([0.0, 1.0, 0.0]),
    # Stunt barrel roll dive
    27.0: np.array([0.0, 1.0, 0.0]),
    27.5: np.array([1.0, 0.0, 0.0]),
    28.0: np.array([0.0, -1.0, 0.0]),
    28.5: np.array([-1.0, 0.0, 0.0]),
    29.0: np.array([0.0, 1.0, 0.0]),
    45.0: np.array([0.0, 1.0, 0.0]),
}

# Linear interpolator to strictly follow the path and prevent over-shooting into buildings
camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focals)
camera_anim.set_view_up_keyframes(camera_view_ups)

camera_anim.set_position_interpolator(linear_interpolator)
camera_anim.set_focal_interpolator(linear_interpolator)
camera_anim.set_view_up_interpolator(linear_interpolator)

timeline = Timeline(playback_panel=True, loop=True)
timeline.add_animation(camera_anim)
for ca in car_animations:
    timeline.add_animation(ca)

if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene,
        size=(1280, 768),
        title="FURY City Drone Shot — Optimized Pure NumPy Backend",
    )
    showm.add_animation(timeline)
    showm.start()
