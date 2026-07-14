"""
=============================================
City Drone Shot: Cinematic Flythrough in FURY
=============================================

A cinematic drone fly-through of a procedurally generated 3D city.
The camera follows a scripted path that starts at street level, ascends
to reveal the skyline panorama, performs aerial stunts (barrel roll, dive),
swoops between buildings through moving traffic, and finishes with a
sweeping landscape shot. Demonstrates FURY's animation capabilities
including timer callbacks, manual camera control, quaternion-based rotations,
and procedural scene construction from primitive actors.

Controls:
    The demo is fully automated — just sit back and enjoy the flight.
    Close the window to exit.
"""

import numpy as np
from fury import actor, ui, window

###############################################################################
# Mathematical Helpers
# ====================
# Quaternion utilities for smooth camera rotation and barrel-roll stunts.
# Catmull-Rom spline for buttery camera path interpolation between keyframes.


def axis_angle_to_quat(axis, angle_deg):
    """Convert axis + angle (degrees) to unit quaternion [x,y,z,w]."""
    angle_rad = np.radians(angle_deg)
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


def quat_mult(q1, q2):
    """Multiply two quaternions q1 * q2, returns unit quaternion."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([x, y, z, w])
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def rotate_vector(quat, vec):
    """Rotate vec by quaternion quat using the sandwich product."""
    q_vec = quat[:3]
    q_w = quat[3]
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (q_w * uv + uuv)


def catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom spline interpolation between p1 and p2 at parameter t in [0,1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def evaluate_spline_path(keyframes, time_val):
    """
    Evaluate a catmull-rom path through keyframes at a given time.

    params: keyframes - list of (time, np.array(position)) sorted by time
    params: time_val - float time to evaluate at
    returns: np.array interpolated position
    """
    if time_val <= keyframes[0][0]:
        return keyframes[0][1].copy()
    if time_val >= keyframes[-1][0]:
        return keyframes[-1][1].copy()

    # Find the segment
    seg = 0
    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= time_val <= keyframes[i + 1][0]:
            seg = i
            break

    t0 = keyframes[seg][0]
    t1 = keyframes[seg + 1][0]
    t_local = (time_val - t0) / (t1 - t0) if (t1 - t0) > 1e-8 else 0.0

    # Get surrounding control points (clamped)
    idx_prev = max(seg - 1, 0)
    idx_next2 = min(seg + 2, len(keyframes) - 1)

    p0 = keyframes[idx_prev][1]
    p1 = keyframes[seg][1]
    p2 = keyframes[seg + 1][1]
    p3 = keyframes[idx_next2][1]

    return catmull_rom(p0, p1, p2, p3, t_local)


def smoothstep(edge0, edge1, x):
    """Hermite smoothstep for easing transitions."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


###############################################################################
# City Generation Parameters
# ==========================

CITY_GRID = 8  # NxN blocks
BLOCK_SIZE = 40.0  # Size of each city block
ROAD_WIDTH = 10.0  # Width of roads between blocks
CITY_EXTENT = CITY_GRID * (BLOCK_SIZE + ROAD_WIDTH) / 2.0

# Derived
CELL_SIZE = BLOCK_SIZE + ROAD_WIDTH

# Animation timing
TOTAL_DURATION = 42.0  # Total animation duration in seconds

###############################################################################
# Scene Setup
# ===========

scene = window.Scene()
scene.background = (0.12, 0.10, 0.22)  # Deep twilight purple

###############################################################################
# Ground Plane
# ============
# A large dark asphalt slab under the entire city.

ground = actor.box(
    centers=np.array([[0.0, -0.5, 0.0]]),
    colors=(0.08, 0.08, 0.10),
    scales=(600, 1.0, 600),
)
scene.add(ground)

###############################################################################
# Sun / Moon
# ==========
# A warm glowing sphere in the distance for atmosphere.

sun = actor.sphere(
    centers=np.array([[0.0, 0.0, 0.0]]),
    colors=(1.0, 0.85, 0.5),
    radii=25.0,
)
sun.local.position = [200.0, 180.0, 300.0]
scene.add(sun)

###############################################################################
# Road Grid Generation
# ====================
# Creates a grid of roads (horizontal and vertical) with lane markings.


def generate_roads():
    """Generate road surfaces and lane dashes for the city grid."""
    half = CITY_EXTENT
    road_actors = []

    for i in range(CITY_GRID + 1):
        # Road center position along x/z
        coord = -half + i * CELL_SIZE

        # Horizontal road (along X axis)
        h_road = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=(0.15, 0.15, 0.17),
            scales=(CITY_GRID * CELL_SIZE + ROAD_WIDTH, 0.1, ROAD_WIDTH),
        )
        h_road.local.position = [0.0, 0.02, coord]
        scene.add(h_road)
        road_actors.append(h_road)

        # Vertical road (along Z axis)
        v_road = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=(0.15, 0.15, 0.17),
            scales=(ROAD_WIDTH, 0.1, CITY_GRID * CELL_SIZE + ROAD_WIDTH),
        )
        v_road.local.position = [coord, 0.02, 0.0]
        scene.add(v_road)
        road_actors.append(v_road)

        # Lane dashes on horizontal roads
        for dx in np.arange(-CITY_GRID * CELL_SIZE / 2, CITY_GRID * CELL_SIZE / 2, 8.0):
            dash = actor.box(
                centers=np.array([[0.0, 0.0, 0.0]]),
                colors=(0.85, 0.85, 0.6),
                scales=(4.0, 0.02, 0.4),
            )
            dash.local.position = [dx, 0.12, coord]
            scene.add(dash)

        # Lane dashes on vertical roads
        for dz in np.arange(-CITY_GRID * CELL_SIZE / 2, CITY_GRID * CELL_SIZE / 2, 8.0):
            dash = actor.box(
                centers=np.array([[0.0, 0.0, 0.0]]),
                colors=(0.85, 0.85, 0.6),
                scales=(0.4, 0.02, 4.0),
            )
            dash.local.position = [coord, 0.12, dz]
            scene.add(dash)

    return road_actors


roads = generate_roads()

###############################################################################
# Building Generation
# ===================
# Procedural buildings of varying height, width, and color placed in city
# blocks. Each building gets small window-glow boxes on its faces.

np.random.seed(42)  # Reproducible city layout

# Color palettes for buildings (glass, concrete, modern tones)
BUILDING_PALETTES = [
    (0.18, 0.22, 0.35),  # Steel blue
    (0.25, 0.25, 0.28),  # Concrete gray
    (0.30, 0.25, 0.18),  # Warm tan
    (0.15, 0.20, 0.30),  # Dark navy
    (0.22, 0.18, 0.25),  # Plum
    (0.20, 0.28, 0.25),  # Teal gray
    (0.35, 0.30, 0.22),  # Sandstone
    (0.12, 0.15, 0.22),  # Midnight
]

WINDOW_COLORS = [
    (1.0, 0.95, 0.6),  # Warm yellow
    (0.9, 0.85, 0.5),  # Soft gold
    (0.7, 0.85, 1.0),  # Cool white
    (1.0, 0.8, 0.4),  # Amber
]

buildings = []  # List of dicts with actor and metadata


def generate_buildings():
    """Generate procedural buildings placed in each city block."""
    half = CITY_EXTENT

    for gx in range(CITY_GRID):
        for gz in range(CITY_GRID):
            # Block center
            bx = -half + ROAD_WIDTH / 2 + gx * CELL_SIZE + BLOCK_SIZE / 2
            bz = -half + ROAD_WIDTH / 2 + gz * CELL_SIZE + BLOCK_SIZE / 2

            # 1-3 buildings per block
            n_buildings = np.random.randint(1, 4)
            for _ in range(n_buildings):
                height = np.random.uniform(12.0, 80.0)
                width_x = np.random.uniform(6.0, min(18.0, BLOCK_SIZE * 0.4))
                width_z = np.random.uniform(6.0, min(18.0, BLOCK_SIZE * 0.4))

                # Random offset within the block
                ox = np.random.uniform(-BLOCK_SIZE * 0.25, BLOCK_SIZE * 0.25)
                oz = np.random.uniform(-BLOCK_SIZE * 0.25, BLOCK_SIZE * 0.25)

                px = bx + ox
                pz = bz + oz

                color = BUILDING_PALETTES[np.random.randint(0, len(BUILDING_PALETTES))]

                bldg = actor.box(
                    centers=np.array([[0.0, 0.0, 0.0]]),
                    colors=color,
                    scales=(width_x, height, width_z),
                )
                bldg.local.position = [px, height / 2.0, pz]
                scene.add(bldg)

                buildings.append(
                    {
                        "actor": bldg,
                        "pos": np.array([px, 0.0, pz]),
                        "height": height,
                        "width_x": width_x,
                        "width_z": width_z,
                    }
                )

                # Windows on two visible faces (X-facing and Z-facing)
                _add_windows(px, pz, height, width_x, width_z)


def _add_windows(px, pz, height, width_x, width_z):
    """Add small glowing window boxes to building faces."""
    win_spacing_y = 4.0
    win_spacing_h = 3.5
    n_floors = max(1, int(height / win_spacing_y) - 1)
    n_win_x = max(1, int(width_x / win_spacing_h) - 1)
    n_win_z = max(1, int(width_z / win_spacing_h) - 1)

    # Limit total windows per building to keep performance reasonable
    max_windows = 8
    count = 0

    for floor in range(1, min(n_floors + 1, 6)):
        wy = floor * win_spacing_y
        if wy > height - 2.0:
            break

        # X-facing front
        for wi in range(min(n_win_x, 3)):
            if count >= max_windows:
                return
            wx = px - width_x * 0.35 + wi * win_spacing_h
            wz_pos = pz + width_z / 2.0 + 0.05

            win_color = WINDOW_COLORS[np.random.randint(0, len(WINDOW_COLORS))]
            # Randomly dim some windows
            if np.random.random() > 0.6:
                win_color = tuple(c * 0.15 for c in win_color)

            win = actor.box(
                centers=np.array([[0.0, 0.0, 0.0]]),
                colors=win_color,
                scales=(1.2, 1.5, 0.1),
            )
            win.local.position = [wx, wy, wz_pos]
            scene.add(win)
            count += 1

        # Z-facing side
        for wi in range(min(n_win_z, 3)):
            if count >= max_windows:
                return
            wx_pos = px + width_x / 2.0 + 0.05
            wz = pz - width_z * 0.35 + wi * win_spacing_h

            win_color = WINDOW_COLORS[np.random.randint(0, len(WINDOW_COLORS))]
            if np.random.random() > 0.6:
                win_color = tuple(c * 0.15 for c in win_color)

            win = actor.box(
                centers=np.array([[0.0, 0.0, 0.0]]),
                colors=win_color,
                scales=(0.1, 1.5, 1.2),
            )
            win.local.position = [wx_pos, wy, wz]
            scene.add(win)
            count += 1


generate_buildings()

###############################################################################
# Street Lights
# =============
# Poles with glowing sphere tops along the major roads.


def generate_street_lights():
    """Generate street light poles with glowing sphere tops along roads."""
    half = CITY_EXTENT
    lights = []

    for i in range(CITY_GRID + 1):
        coord = -half + i * CELL_SIZE

        # Lights along horizontal roads
        for dx in np.arange(-half, half, 25.0):
            for side in [-1, 1]:
                pole = actor.cylinder(
                    centers=np.array([[0.0, 0.0, 0.0]]),
                    directions=np.array([[0.0, 1.0, 0.0]]),
                    colors=(0.3, 0.3, 0.32),
                    height=6.0,
                    radii=0.15,
                )
                pole.local.position = [
                    dx,
                    3.0,
                    coord + side * (ROAD_WIDTH / 2.0 - 0.5),
                ]
                scene.add(pole)

                bulb = actor.sphere(
                    centers=np.array([[0.0, 0.0, 0.0]]),
                    colors=(1.0, 0.9, 0.5),
                    radii=0.5,
                )
                bulb.local.position = [
                    dx,
                    6.2,
                    coord + side * (ROAD_WIDTH / 2.0 - 0.5),
                ]
                scene.add(bulb)
                lights.append((pole, bulb))

    return lights


street_lights = generate_street_lights()

###############################################################################
# Traffic (Animated Cars)
# =======================
# Small colored boxes that move along road segments to create a living city.

CAR_COLORS = [
    (0.85, 0.15, 0.15),  # Red
    (0.15, 0.15, 0.85),  # Blue
    (0.9, 0.9, 0.2),  # Yellow
    (0.9, 0.9, 0.9),  # White
    (0.1, 0.1, 0.1),  # Black
    (0.2, 0.7, 0.2),  # Green
    (0.9, 0.5, 0.1),  # Orange
]


def generate_traffic(n_cars=30):
    """Generate animated car actors on the road grid."""
    half = CITY_EXTENT
    cars = []

    road_coords = [-half + i * CELL_SIZE for i in range(CITY_GRID + 1)]

    for _ in range(n_cars):
        color = CAR_COLORS[np.random.randint(0, len(CAR_COLORS))]

        # Randomly choose horizontal or vertical road
        is_horizontal = np.random.random() > 0.5
        road_idx = np.random.randint(0, len(road_coords))
        road_coord = road_coords[road_idx]

        if is_horizontal:
            # Car moves along X
            start_x = np.random.uniform(-half, half)
            offset_z = np.random.uniform(-2.0, 2.0)
            pos = np.array([start_x, 0.7, road_coord + offset_z])
            speed = np.random.uniform(8.0, 20.0)
            direction = 1.0 if np.random.random() > 0.5 else -1.0
            vel = np.array([speed * direction, 0.0, 0.0])
            car_scale = (3.0, 1.2, 1.5)
        else:
            # Car moves along Z
            start_z = np.random.uniform(-half, half)
            offset_x = np.random.uniform(-2.0, 2.0)
            pos = np.array([road_coord + offset_x, 0.7, start_z])
            speed = np.random.uniform(8.0, 20.0)
            direction = 1.0 if np.random.random() > 0.5 else -1.0
            vel = np.array([0.0, 0.0, speed * direction])
            car_scale = (1.5, 1.2, 3.0)

        car_actor = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]),
            colors=color,
            scales=car_scale,
        )
        car_actor.local.position = pos.tolist()
        scene.add(car_actor)

        cars.append(
            {
                "actor": car_actor,
                "pos": pos,
                "vel": vel,
                "is_horizontal": is_horizontal,
            }
        )

    return cars


traffic = generate_traffic(30)

###############################################################################
# HUD Overlay
# ===========
# Title and phase indicator text.

hud_title = ui.TextBlock2D(
    text="FURY City Drone Shot",
    position=(30, 720),
    size=(400, 30),
    font_size=22,
    color=(1.0, 0.9, 0.4),
    bold=True,
)
scene.add(hud_title)

hud_phase = ui.TextBlock2D(
    text="Phase: Street Level",
    position=(30, 690),
    size=(400, 30),
    font_size=16,
    color=(0.8, 0.8, 0.9),
)
scene.add(hud_phase)

###############################################################################
# Drone Camera Path Definition
# =============================
# Keyframe positions and focal targets for each phase of the cinematic flight.
# The camera smoothly interpolates through these using Catmull-Rom splines.

# We pick a prominent road for the street-level start
half = CITY_EXTENT
main_road_z = -half + 2 * CELL_SIZE  # 3rd horizontal road

# Camera position keyframes: (time, np.array([x, y, z]))
camera_position_keyframes = [
    # Phase 1: Street Level (0-6s) - moving along a road
    (0.0, np.array([-half + 20.0, 3.0, main_road_z])),
    (3.0, np.array([-half + 80.0, 3.5, main_road_z])),
    (6.0, np.array([-half + 130.0, 4.0, main_road_z + 2.0])),
    # Phase 2: Ascent (6-12s) - rising upward
    (8.0, np.array([-half + 140.0, 30.0, main_road_z + 10.0])),
    (10.0, np.array([-half + 120.0, 70.0, main_road_z + 30.0])),
    (12.0, np.array([0.0, 120.0, main_road_z + 60.0])),
    # Phase 3: Landscape Panorama (12-18s) - orbiting high
    (14.0, np.array([60.0, 130.0, 30.0])),
    (16.0, np.array([80.0, 120.0, -60.0])),
    (18.0, np.array([40.0, 110.0, -100.0])),
    # Phase 4: Drone Stunts (18-26s) - dive and weave
    (20.0, np.array([0.0, 90.0, -80.0])),
    (22.0, np.array([-30.0, 25.0, -40.0])),  # Sharp dive
    (24.0, np.array([-60.0, 40.0, 0.0])),  # Pull up + weave
    (26.0, np.array([-30.0, 35.0, 40.0])),  # S-curve
    # Phase 5: Building Flyby (26-34s) - sideways past buildings
    (28.0, np.array([0.0, 25.0, 60.0])),
    (30.0, np.array([40.0, 20.0, 30.0])),
    (32.0, np.array([70.0, 18.0, -10.0])),
    (34.0, np.array([50.0, 22.0, -50.0])),
    # Phase 6: Final Sweep (34-42s) - pull back up, wide shot
    (36.0, np.array([20.0, 60.0, -80.0])),
    (38.0, np.array([-30.0, 100.0, -60.0])),
    (40.0, np.array([-60.0, 140.0, 20.0])),
    (42.0, np.array([-half + 20.0, 3.0, main_road_z])),  # Loop back to start
]

# Camera focal point keyframes (what the camera looks at)
camera_focal_keyframes = [
    # Phase 1: Looking down the road
    (0.0, np.array([-half + 100.0, 3.0, main_road_z])),
    (3.0, np.array([-half + 140.0, 5.0, main_road_z])),
    (6.0, np.array([-half + 160.0, 10.0, main_road_z])),
    # Phase 2: Looking at the city center as we rise
    (8.0, np.array([0.0, 20.0, 0.0])),
    (10.0, np.array([0.0, 10.0, 0.0])),
    (12.0, np.array([0.0, 0.0, 0.0])),
    # Phase 3: Panning across the city
    (14.0, np.array([0.0, 0.0, 0.0])),
    (16.0, np.array([-20.0, 0.0, 20.0])),
    (18.0, np.array([0.0, 10.0, 0.0])),
    # Phase 4: Fast targets during stunts
    (20.0, np.array([0.0, 0.0, -40.0])),
    (22.0, np.array([-30.0, 0.0, -10.0])),
    (24.0, np.array([-40.0, 10.0, 20.0])),
    (26.0, np.array([0.0, 15.0, 60.0])),
    # Phase 5: Looking sideways at buildings
    (28.0, np.array([30.0, 15.0, 50.0])),
    (30.0, np.array([60.0, 10.0, 10.0])),
    (32.0, np.array([50.0, 8.0, -30.0])),
    (34.0, np.array([20.0, 10.0, -40.0])),
    # Phase 6: Looking down at the whole city
    (36.0, np.array([0.0, 0.0, 0.0])),
    (38.0, np.array([0.0, 0.0, 0.0])),
    (40.0, np.array([0.0, 0.0, 0.0])),
    (42.0, np.array([-half + 100.0, 3.0, main_road_z])),
]

###############################################################################
# Phase Labels

PHASE_LABELS = [
    (0.0, "Street Level"),
    (6.0, "Ascent"),
    (12.0, "Landscape Panorama"),
    (18.0, "Drone Stunts"),
    (26.0, "Building Flyby"),
    (34.0, "Final Sweep"),
]

###############################################################################
# Animation State
# ===============

state = {
    "time": 0.0,
    "dt": 0.016,
}

###############################################################################
# Core Animation Loop
# ====================
# Called every ~16ms. Updates traffic, evaluates camera spline, applies
# barrel-roll rotation during stunts, and renders the frame.


def get_current_phase(t):
    """Get the label of the current camera phase."""
    label = PHASE_LABELS[0][1]
    for pt, pl in PHASE_LABELS:
        if t >= pt:
            label = pl
    return label


def animation_tick(showm):
    """Main animation callback driving traffic and camera each frame."""
    dt = state["dt"]
    state["time"] += dt

    # Loop the animation
    t = state["time"] % TOTAL_DURATION

    # --- Update Traffic ---
    for car in traffic:
        car["pos"] += car["vel"] * dt

        # Wrap cars around when they leave the city bounds
        for axis in [0, 2]:
            if car["pos"][axis] > CITY_EXTENT + 20.0:
                car["pos"][axis] = -CITY_EXTENT - 20.0
            elif car["pos"][axis] < -CITY_EXTENT - 20.0:
                car["pos"][axis] = CITY_EXTENT + 20.0

        car["actor"].local.position = car["pos"].tolist()

    # --- Evaluate Camera Path ---
    cam_pos = evaluate_spline_path(camera_position_keyframes, t)
    cam_focal = evaluate_spline_path(camera_focal_keyframes, t)

    # --- Camera Setup ---
    camera = showm.screens[0].camera
    camera.local.position = cam_pos.tolist()

    # Compute the look direction for reference_up calculation
    look_dir = cam_focal - cam_pos
    look_dist = np.linalg.norm(look_dir)
    if look_dist > 1e-5:
        look_dir = look_dir / look_dist

    # Default up vector
    up = np.array([0.0, 1.0, 0.0])

    # --- Barrel Roll during Stunt Phase (t ~ 20-22s) ---
    barrel_roll_start = 19.5
    barrel_roll_end = 22.5
    if barrel_roll_start < t < barrel_roll_end:
        roll_progress = (t - barrel_roll_start) / (barrel_roll_end - barrel_roll_start)
        roll_angle = roll_progress * 360.0  # Full 360 barrel roll
        roll_quat = axis_angle_to_quat(look_dir, roll_angle)
        up = rotate_vector(roll_quat, up)

    # --- Gentle banking during S-weave (t ~ 24-26s) ---
    weave_start = 23.5
    weave_end = 26.5
    if weave_start < t < weave_end:
        weave_progress = (t - weave_start) / (weave_end - weave_start)
        bank_angle = 30.0 * np.sin(weave_progress * 2.0 * np.pi)
        bank_quat = axis_angle_to_quat(look_dir, bank_angle)
        up = rotate_vector(bank_quat, up)

    # --- Gentle banking during building flyby (t ~ 28-34s) ---
    flyby_start = 27.0
    flyby_end = 34.0
    if flyby_start < t < flyby_end:
        flyby_progress = (t - flyby_start) / (flyby_end - flyby_start)
        tilt_angle = 15.0 * np.sin(flyby_progress * 3.0 * np.pi)
        tilt_quat = axis_angle_to_quat(look_dir, tilt_angle)
        up = rotate_vector(tilt_quat, up)

    camera.look_at(cam_focal.tolist())
    camera.reference_up = up.tolist()

    # --- Update HUD ---
    phase_label = get_current_phase(t)
    hud_phase.message = f"Phase: {phase_label}"

    showm.render()


###############################################################################
# Application Entry Point
# =======================
# Set up ShowManager, disable orbit controller, register animation callback.

if __name__ == "__main__":
    showm = window.ShowManager(
        scene=scene,
        size=(1280, 768),
        title="FURY City Drone Shot",
    )

    # Disable default orbit controller — we drive the camera manually
    showm.screens[0].controller.enabled = False

    # Set initial camera
    camera = showm.screens[0].camera
    camera.local.position = camera_position_keyframes[0][1].tolist()
    camera.look_at(camera_focal_keyframes[0][1].tolist())

    # Register the animation loop at ~60fps
    showm.register_callback(animation_tick, 0.016, True, "DroneShotLoop", showm)

    showm.start()
