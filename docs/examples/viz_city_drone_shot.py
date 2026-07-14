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
via FURY and VTK, mapped to FURY actors with textures.
"""

import logging
import os
import numpy as np

from fury import actor, window
from fury.motion import Animation, CameraAnimation, Timeline, cubic_spline_interpolator, linear_interpolator
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

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
# Model Loading Helper
# ====================

def load_kenney_model(directory, filename, texture_file, scale=1.0, position=None, rotation_y=0.0):
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

    cells = vtk_to_numpy(polydata.GetPolys().GetData())
    faces = cells.reshape(-1, 4)[:, 1:4]
    
    uvs = None
    tcoords = polydata.GetPointData().GetTCoords()
    if tcoords is not None:
        uvs = vtk_to_numpy(tcoords)
    elif polydata.GetPointData().HasArray('colormap'):
        uvs = vtk_to_numpy(polydata.GetPointData().GetArray('colormap'))

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
    scales=(2000, 0.5, 2000),
)
scene.add(ground)

# Sun
sun = actor.sphere(
    centers=np.array([[0.0, 0.0, 0.0]]),
    colors=(1.0, 1.0, 0.8),
    radii=40.0,
)
sun.local.position = [400.0, 300.0, 500.0]
scene.add(sun)

###############################################################################
# City Layout Constants

np.random.seed(42)

BLOCK_SIZE = 70.0
ROAD_WIDTH = 14.0
BUILDING_SCALE = 12.0
GRID_SIZE = 6  # Grid goes from -GRID_SIZE to GRID_SIZE

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
# Roads (Kenney City Kit Roads)
# =============================

def generate_roads():
    """Generate dense NYC grid roads using Kenney road models."""
    road_tex = "colormap.png"
    placed = 0

    for ix in range(-GRID_SIZE, GRID_SIZE + 1):
        for iz in range(-GRID_SIZE, GRID_SIZE + 1):
            cx = ix * BLOCK_SIZE
            cz = iz * BLOCK_SIZE
            
            # Intersection
            inter = load_kenney_model(
                ROAD_KIT_DIR, "road-crossroad.obj", road_tex,
                scale=ROAD_WIDTH, position=(cx, 0.0, cz)
            )
            scene.add(inter)
            placed += 1

            # Straight roads to the right (if not at edge)
            if ix < GRID_SIZE:
                for s in range(1, 5):  # 4 tiles between intersections
                    road_x = cx + s * ROAD_WIDTH
                    road = load_kenney_model(
                        ROAD_KIT_DIR, "road-straight.obj", road_tex,
                        scale=ROAD_WIDTH, position=(road_x, 0.0, cz),
                        rotation_y=90.0
                    )
                    scene.add(road)
                    placed += 1
            
            # Straight roads down (if not at edge)
            if iz < GRID_SIZE:
                for s in range(1, 5):
                    road_z = cz + s * ROAD_WIDTH
                    road = load_kenney_model(
                        ROAD_KIT_DIR, "road-straight.obj", road_tex,
                        scale=ROAD_WIDTH, position=(cx, 0.0, road_z),
                        rotation_y=0.0
                    )
                    scene.add(road)
                    placed += 1

    logger.info(f"Generated {placed} road tiles")

generate_roads()

###############################################################################
# Buildings (Kenney City Kit, White Variation)
# ============================================

def generate_buildings():
    """Generate tightly packed white buildings inside the city blocks."""
    placed = 0
    building_tex = "variation-b.png"  # Default white building texture
    
    # Iterate over blocks (between the roads)
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

            # Tightly pack buildings in a 3x3 grid inside the 56x56 block
            # But only on the perimeter so we don't overlap inside 
            # Perimeter of 3x3 means the outer 8 cells
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
                
                rot_angle = np.random.choice([0, 90, 180, 270])
                bldg = load_kenney_model(
                    CITY_KIT_DIR, model_name, building_tex, 
                    scale=scale, position=(bx, 0.0, bz), rotation_y=rot_angle
                )
                scene.add(bldg)
                placed += 1

    logger.info(f"Placed {placed} Kenney buildings")

generate_buildings()

###############################################################################
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

ANIMATION_DURATION = 45.0
car_animations = []

# Generate cars on random road segments
car_routes = []
for i in range(25):
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

    car = load_kenney_model(
        CAR_KIT_DIR, model_name, "colormap.png", scale=3.5, 
        position=start_pos, rotation_y=rot_y
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
    # Start above a road
    0.0: np.array([-210.0, 5.0, 0.0]),
    4.0: np.array([-140.0, 5.0, 0.0]),
    8.0: np.array([-70.0, 5.0, 0.0]),
    # Ascend
    10.0: np.array([0.0, 25.0, 0.0]),
    14.0: np.array([0.0, 120.0, 0.0]),
    # High altitude panorama
    18.0: np.array([120.0, 150.0, 120.0]),
    22.0: np.array([-50.0, 140.0, 180.0]),
    26.0: np.array([-150.0, 120.0, 80.0]),
    # Dive into a cross street
    28.0: np.array([-70.0, 50.0, 70.0]),
    30.0: np.array([-70.0, 8.0, 0.0]),
    # Zip down street
    34.0: np.array([-70.0, 8.0, -140.0]),
    # Ascend and fly back to start
    38.0: np.array([-140.0, 50.0, -140.0]),
    42.0: np.array([-210.0, 80.0, -70.0]),
    45.0: np.array([-210.0, 5.0, 0.0]),
}

camera_focals = {
    0.0: np.array([-140.0, 4.0, 0.0]),
    4.0: np.array([-70.0, 4.0, 0.0]),
    8.0: np.array([0.0, 4.0, 0.0]),
    10.0: np.array([0.0, 25.0, 50.0]),
    14.0: np.array([0.0, 40.0, -50.0]),
    18.0: np.array([0.0, 40.0, 0.0]),
    22.0: np.array([0.0, 40.0, 0.0]),
    26.0: np.array([0.0, 40.0, 0.0]),
    28.0: np.array([-70.0, 10.0, 0.0]),
    30.0: np.array([-70.0, 8.0, -70.0]),
    34.0: np.array([-70.0, 8.0, -210.0]),
    38.0: np.array([0.0, 0.0, 0.0]),
    42.0: np.array([0.0, 0.0, 0.0]),
    45.0: np.array([-140.0, 4.0, 0.0]),
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

camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focals)
camera_anim.set_view_up_keyframes(camera_view_ups)

camera_anim.set_position_interpolator(cubic_spline_interpolator)
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
        title="FURY City Drone Shot — NYC Grid Overhaul",
    )
    showm.add_animation(timeline)
    showm.start()
