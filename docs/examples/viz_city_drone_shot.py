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
using NumPy concatenation to batch them into just two massive actors,
enabling buttery smooth rendering and allowing for high-density animated traffic.

A custom pure-Python/NumPy OBJ parser is used to completely eliminate VTK 
dependencies while preserving UV texture coordinates.
"""

import logging
import os
import numpy as np

from fury import actor, window
from fury.motion import Animation, CameraAnimation, Timeline, cubic_spline_interpolator, linear_interpolator

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

def load_obj(path):
    """
    Custom zero-dependency OBJ parser optimized for Kenney low-poly models.
    Extracts flattened vertices, faces, and UVs perfectly without VTK.
    """
    verts, uvs, faces, uv_indices = [], [], [], []
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.strip().split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                f_verts, f_uvs = [], []
                for p in parts:
                    vals = p.split('/')
                    f_verts.append(int(vals[0]) - 1)
                    if len(vals) > 1 and vals[1]:
                        f_uvs.append(int(vals[1]) - 1)
                    else:
                        f_uvs.append(-1)
                for i in range(1, len(f_verts) - 1):
                    faces.append([f_verts[0], f_verts[i], f_verts[i+1]])
                    uv_indices.append([f_uvs[0], f_uvs[i], f_uvs[i+1]])
                    
    verts = np.array(verts, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
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
    """Cache and return the base NumPy arrays for a model to avoid disk I/O."""
    key = (directory, filename)
    if key not in _MODEL_CACHE:
        path = os.path.join(directory, filename)
        v, f, uv = load_obj(path)
        _MODEL_CACHE[key] = (v, f, uv)
    return _MODEL_CACHE[key]

def create_batched_actor(directory, instances, texture_file=None):
    """
    Batch thousands of meshes using NumPy concatenation.
    Returns a single textured FURY actor for extreme performance.
    """
    all_verts = []
    all_faces = []
    all_uvs = []
    v_offset = 0
    
    for filename, scale, pos, rot_y in instances:
        v, f, uv = get_base_mesh(directory, filename)
        v = v.copy()
        
        # Apply scale
        if isinstance(scale, (list, tuple, np.ndarray)):
            v = v * np.array(scale)
        else:
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
            tex_path = os.path.join(os.path.dirname(directory), "Textures", texture_file)
    
    surf = actor.surface(verts, faces, texture=tex_path, texture_coords=uvs)
    return surf

def load_kenney_model(directory, filename, texture_file, scale=1.0, position=None, rotation_y=0.0):
    """Load a single Kenney model dynamically (used for animated cars)."""
    v, f, uv = get_base_mesh(directory, filename)
    v = v.copy()
    
    if rotation_y != 0.0:
        angle = np.radians(rotation_y)
        c_val, s_val = np.cos(angle), np.sin(angle)
        v_x = v[:, 0] * c_val + v[:, 2] * s_val
        v_z = -v[:, 0] * s_val + v[:, 2] * c_val
        v[:, 0] = v_x
        v[:, 2] = v_z

    tex_path = None
    if texture_file:
        tex_path = os.path.join(directory, "Textures", texture_file)
        if not os.path.exists(tex_path):
            tex_path = os.path.join(os.path.dirname(directory), "Textures", texture_file)

    surf = actor.surface(v, f, texture=tex_path, texture_coords=uv)
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
            
            # Sometimes leave a block empty for a park
            if np.random.random() < 0.15:
                continue

            offsets = [
                (-16, -16), (0, -16), (16, -16),
                (-16, 0),             (16, 0),
                (-16, 16),  (0, 16),  (16, 16)
            ]
            
            num_buildings = np.random.randint(2, 5)
            chosen_indices = np.random.choice(len(offsets), num_buildings, replace=False)
            
            for idx in chosen_indices:
                dx, dz = offsets[idx]
                bx = block_cx + dx
                bz = block_cz + dz
                
                # Scatter skyscrapers randomly across the entire city (20% chance)
                if np.random.random() < 0.20:
                    model_name = SKYSCRAPER_MODELS[np.random.randint(0, len(SKYSCRAPER_MODELS))]
                    scale = np.random.uniform(BUILDING_SCALE * 1.2, BUILDING_SCALE * 1.8)
                else:
                    model_name = BUILDING_MODELS[np.random.randint(0, len(BUILDING_MODELS))]
                    scale = np.random.uniform(BUILDING_SCALE * 0.8, BUILDING_SCALE * 1.2)
                
                rot_angle = float(np.random.choice([0, 90, 180, 270]))
                building_instances.append(
                    (model_name, scale, (bx, 0.0, bz), rot_angle)
                )

    # Hardcode a "Colossal Skyscraper" specifically for the drone stunt sequence
    colossal_x = 70.0
    colossal_z = 70.0
    # Remove any existing buildings near the colossal spot to prevent overlap
    building_instances = [b for b in building_instances if not (abs(b[2][0] - colossal_x) < 40 and abs(b[2][2] - colossal_z) < 40)]
    
    # Add the colossal skyscraper right in the middle of block (1, 1)
    building_instances.append(
        ("building-skyscraper-c.obj", (BUILDING_SCALE * 3.0, BUILDING_SCALE * 6.0, BUILDING_SCALE * 3.0), (colossal_x + 35.0, 0.0, colossal_z + 35.0), 0.0)
    )

    logger.info(f"Batching {len(road_instances)} road tiles...")
    roads_actor = create_batched_actor(ROAD_KIT_DIR, road_instances, texture_file="colormap.png")
    # Enable backface culling so we don't render the inside of models
    roads_actor.material.side = "front"
    scene.add(roads_actor)

    logger.info(f"Batching {len(building_instances)} buildings...")
    # Use variation-a.png for standard white buildings as requested
    buildings_actor = create_batched_actor(CITY_KIT_DIR, building_instances, texture_file="variation-a.png")
    # Enable backface culling to ensure hollow interiors are invisible if the camera gets too close
    buildings_actor.material.side = "front"
    scene.add(buildings_actor)

generate_static_city()

###############################################################################
# Animated Traffic (Kenney Car Kit + fury.motion Animation)
# =========================================================

ANIMATION_DURATION = 45.0
car_animations = []
NUM_CARS = 75

# Add a "Hero Car" for the drone to follow in the first scene
# It starts at Z=-400, drives along X=0, lane=4.0
hero_car = load_kenney_model(CAR_KIT_DIR, "hatchback-sports.obj", "colormap.png", scale=3.5, position=(0.0, 0.0, -400.0), rotation_y=0.0)
scene.add(hero_car)
hero_anim = Animation(actors=hero_car, loop=True)
hero_speed = 1000.0  # Needs to travel far in 30s
for ki in range(11):
    t = (ki / 10.0) * ANIMATION_DURATION
    progress = ki / 10.0
    z_pos = -400.0 + (hero_speed * progress)
    hero_anim.set_position(t, np.array([4.0, 0.0, z_pos]))
hero_anim.set_position_interpolator(cubic_spline_interpolator)
car_animations.append(hero_anim)


car_routes = []
for i in range(NUM_CARS):
    is_x_axis = np.random.choice([True, False])
    line = np.random.randint(-GRID_SIZE + 1, GRID_SIZE) * BLOCK_SIZE
    lane = line + np.random.choice([-4.0, 4.0])
    direction = 1 if lane < line else -1
    speed_f = np.random.uniform(0.7, 1.3)
    # Start cars FAR outside the city and drive completely through it 
    # to avoid wrapping glitches!
    start_coord = -900.0 * direction
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
    travel_dist = 1800.0 * speed_f  # Travel all the way across the city grid
    n_kf = 10
    
    for ki in range(n_kf + 1):
        t = (ki / n_kf) * ANIMATION_DURATION
        progress = ki / n_kf
        val = start_coord + direction * travel_dist * progress
        
        if is_x_axis:
            pos = np.array([val, 0.0, lane])
        else:
            pos = np.array([lane, 0.0, val])
            
        car_anim.set_position(t, pos)

    car_anim.set_position_interpolator(cubic_spline_interpolator)
    car_animations.append(car_anim)

logger.info(f"Created {len(car_animations)} animated Kenney cars")

###############################################################################
# Camera Animation: Action Stunt Sequence
# =======================================

camera_anim = CameraAnimation(loop=True)

# Cinematic 45s action sequence with carefully clamped spline curves
camera_positions = {
    # 0-8s: Following the Hero Car down the main avenue
    0.0: np.array([4.0, 15.0, -450.0]),
    4.0: np.array([4.0, 15.0, -200.0]),
    8.0: np.array([4.0, 15.0, 50.0]),
    
    # 8-16s: Break away and ascend into a beautiful panorama
    12.0: np.array([-70.0, 80.0, 100.0]), # Intermediate step to prevent spline dip
    16.0: np.array([-140.0, 150.0, 140.0]),
    
    # 16-24s: Approach the Colossal Skyscraper at Block (1,1)
    20.0: np.array([-20.0, 80.0, 70.0]),
    24.0: np.array([105.0, 20.0, 0.0]),
    
    # 24-30s: The Climb (fly alongside the glass looking up, offset by 45 units to avoid FOV clip)
    27.0: np.array([150.0, 150.0, 105.0]),
    30.0: np.array([150.0, 240.0, 105.0]),
    
    # 30-36s: The Stunt (Flip over the roof)
    36.0: np.array([140.0, 280.0, 140.0]),
    
    # 36-45s: Plummet back down into a street and catch up to the start loop
    41.0: np.array([70.0, 80.0, -150.0]), # Brake point to prevent ground crash!
    45.0: np.array([4.0, 15.0, -450.0]),
}

camera_focals = {
    # Looking at the Hero Car
    0.0: np.array([4.0, 5.0, -380.0]),
    4.0: np.array([4.0, 5.0, -130.0]),
    8.0: np.array([4.0, 5.0, 120.0]),
    
    # Looking down at the city during panorama
    12.0: np.array([-40.0, 50.0, 90.0]),
    16.0: np.array([0.0, 50.0, 70.0]),
    
    # Looking AT the Colossal Skyscraper's base
    20.0: np.array([40.0, 40.0, 40.0]),
    24.0: np.array([105.0, 50.0, 105.0]),
    
    # Looking at the building center while climbing alongside it
    27.0: np.array([105.0, 250.0, 105.0]),
    30.0: np.array([105.0, 300.0, 105.0]),
    
    # Looking over the roof during the flip stunt
    36.0: np.array([70.0, 200.0, 70.0]),
    
    # Looking down the street to loop
    41.0: np.array([20.0, 25.0, -250.0]),
    45.0: np.array([4.0, 5.0, -380.0]),
}

# The View-Up Vector controls camera tilt and barrel rolls!
# We step through 45-degree angles to create a clean pitch-loop 
# and avoid passing through (0,0,0) which crashes linear interpolators.
camera_view_ups = {
    0.0: np.array([0.0, 1.0, 0.0]),
    24.0: np.array([0.0, 1.0, 0.0]),
    
    # Tilt slightly back while climbing
    27.0: np.array([0.0, 0.707, -0.707]),
    28.5: np.array([0.0, 0.0, -1.0]),
    
    # The Pitch-Loop Stunt over the roof!
    30.0: np.array([0.0, -0.707, -0.707]),
    32.0: np.array([0.0, -1.0, 0.0]),      # Upside down!
    34.0: np.array([0.0, -0.707, 0.707]),
    35.0: np.array([0.0, 0.0, 1.0]),       # Pitching forward
    36.0: np.array([0.0, 0.707, 0.707]),
    38.0: np.array([0.0, 1.0, 0.0]),        # Right side up!
    
    45.0: np.array([0.0, 1.0, 0.0]),
}

camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focals)
camera_anim.set_view_up_keyframes(camera_view_ups)

def physics_aware_camera_spline(keyframes, **kwargs):
    """
    Custom wrapper to evaluate the spline and apply continuous Raycast 
    Collision detection, ensuring the camera acts as a physical object.
    """
    base_eval = cubic_spline_interpolator(keyframes, **kwargs)
    
    def evaluator(t):
        pos = base_eval(t).copy()
        
        # 1. Ground Collision (Altitude clamp to prevent underground dips)
        if pos[1] < 15.0:
            pos[1] = 15.0
            
        # 2. Skyscraper Raycast Collision
        # Colossal Skyscraper Center: X=105, Z=105.
        # Safe collision radius: 35.0 (This also handles Near Clip Adjust by preventing FOV slicing!)
        dx = pos[0] - 105.0
        dz = pos[2] - 105.0
        dist = np.sqrt(dx**2 + dz**2)
        if dist < 35.0 and pos[1] < 300.0: # Only collide if we aren't above the roof!
            # Snap the camera safely outside the geometry bounding box
            pos[0] = 105.0 + (dx/dist) * 35.0
            pos[2] = 105.0 + (dz/dist) * 35.0
            
        return pos
    return evaluator

# Apply physical collision wrapper for camera position
camera_anim.set_position_interpolator(physics_aware_camera_spline)
camera_anim.set_focal_interpolator(cubic_spline_interpolator)
# Use linear for view_up to avoid spline duplicate value errors
camera_anim.set_view_up_interpolator(linear_interpolator)

timeline = Timeline(playback_panel=True, loop=True)
timeline.add_animation(camera_anim)
for ca in car_animations:
    timeline.add_animation(ca)

if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene,
        size=(1280, 768),
        title="FURY City Drone Shot — Action Stunt Sequence",
    )
    showm.add_animation(timeline)
    showm.start()
