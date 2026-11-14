"""
====================================
Jungle Ecosystem Survival Simulation
====================================

An interactive 3D jungle ecosystem survival simulation containing Lions, Elephants,
and Deers navigating land, lakes, rivers, and vegetation patches.
Features survival states (thirst, hunger, aging, mating, hunting, and death),
a WASD free fly camera, and diagnostic control interfaces.

The animals are the elephant, lion and deer from Kenney's Cube Pets pack.

Assets: Kenney packs (CC0, https://kenney.nl/assets) are expected in a
``kenny_assets`` folder next to this script. Set ``FURY_KENNEY_ASSETS`` to use
another location.
"""

import os
from time import perf_counter

import numpy as np
from fury import actor, ui, window
from fury.lib import AmbientLight
from fury.window import EventType

# Simulation dimensions and constants
JUNGLE_SIZE = 600.0
LAKE_RADIUS = 55.0
LAKE_CENTER = np.array([0.0, 0.0, 0.0])

# Species characteristics
MAX_HEALTH = {"lion": 150.0, "elephant": 500.0, "deer": 100.0}
MAX_AGE = {"lion": 90.0, "elephant": 140.0, "deer": 75.0}
BASE_SPEED = {"lion": 7.0, "elephant": 3.5, "deer": 10.0}
RUN_SPEED = {"lion": 15.5, "elephant": 8.5, "deer": 14.0}
MIN_SPEED = 2.0
HUNGER_RATE = {"lion": 0.5, "elephant": 0.45, "deer": 0.9}
THIRST_RATE = {"lion": 1.0, "elephant": 0.55, "deer": 1.1}
NEED_START = {"food": 40.0, "water": 45.0}
NEED_MET = 10.0
NEED_URGENT = 85.0
STARVATION_DAMAGE = 1.5
LION_FLEE_RANGE = 40.0
LION_HUNT_FLEE_RANGE = 15.0
DEER_FLEE_RANGE = 32.0
LION_FINISH_RANGE = 15.0
NURSING_RANGE = 15.0
ELEPHANT_CHARGE_RANGE = 12.0
ELEPHANT_DAMAGE = 12.0
MAX_POPULATION = 150
SPECIES_LIMIT = {"lion": 25, "elephant": 25, "deer": 100}
# Simulated time per real second; matches the original 0.16 per tick at 60 FPS
SIM_TIME_SCALE = 10.0
MAX_SIM_STEP = 0.17
MAX_SIM_STEPS_PER_FRAME = 12

# Kenney Cube Pets models (feet at y=0, facing +Z)
ASSETS_DIR = os.environ.get(
    "FURY_KENNEY_ASSETS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kenny_assets"),
)
ANIMAL_KIT_DIR = os.path.join(ASSETS_DIR, "kenney_cube-pets/Models/OBJ format")
ANIMAL_MODELS = {
    "lion": "animal-lion.obj",
    "elephant": "animal-elephant.obj",
    "deer": "animal-deer.obj",
}
ANIMAL_SCALE = {"lion": 3.0, "elephant": 5.5, "deer": 2.6}
ANIMAL_HEIGHT = {"lion": 1.75, "elephant": 1.43, "deer": 1.99}

# Grouping / Flocking coefficients per species
GROUP_COEFFS = {
    "lion": {"cohesion": 1.0, "alignment": 0.8, "separation": 1.2},
    "elephant": {"cohesion": 2.2, "alignment": 0.2, "separation": 1.8},
    "deer": {"cohesion": 1.8, "alignment": 1.2, "separation": 1.5},
}

# Vegetation patch centers
veg_patches = [
    np.array([-150.0, 0.0, -150.0]),
    np.array([160.0, 0.0, 160.0]),
    np.array([-180.0, 0.0, 120.0]),
    np.array([180.0, 0.0, -150.0]),
    np.array([0.0, 0.0, -220.0]),
]
VEG_PATCH_CENTERS = np.array(veg_patches, dtype=np.float64)


# Global simulation state
state = {
    "selected_animal": None,
    "screen_size": None,
    "layout_due": [],
    "last_tick": None,
    "clock": 0.0,
    "next_label_update": 0.0,
    "cam_pos": None,
    "animals": [],
    "next_animal_id": 0,
    # Camera states
    "keys": set(),
    "cam_yaw": 0.0,
    "cam_pitch": -np.radians(45.0),
    "is_dragging_cam": False,
    "last_mouse": None,
    # Simulation factors and rates
    "sim_speed": 1.0,
    "cohesion_mult": 1.0,
    "separation_mult": 1.0,
    # Global multipliers
    "global_speed_factor": 1.0,
    "global_hunger_factor": 1.0,
    "global_thirst_factor": 1.0,
    "global_age_factor": 1.0,
    "global_mating_factor": 1.0,
    "global_hunting_factor": 1.0,
    # Species factors
    "lion_speed_factor": 1.0,
    "lion_hunger_factor": 1.0,
    "lion_thirst_factor": 1.0,
    "lion_age_factor": 1.0,
    "lion_mating_factor": 1.0,
    "lion_hunting_factor": 1.0,
    "elephant_speed_factor": 1.0,
    "elephant_hunger_factor": 1.0,
    "elephant_thirst_factor": 1.0,
    "elephant_age_factor": 1.0,
    "elephant_mating_factor": 1.0,
    "elephant_hunting_factor": 1.0,
    "deer_speed_factor": 1.0,
    "deer_hunger_factor": 1.0,
    "deer_thirst_factor": 1.0,
    "deer_age_factor": 1.0,
    "deer_mating_factor": 1.0,
    "deer_hunting_factor": 1.0,
}

# Setup Scene
scene = window.Scene()
scene.background = (0.62, 0.82, 0.95)


# Height map generator for rolling hills and valleys terrain
def get_terrain_height(x, z):
    # Base height function (sine/cosine waves)
    y = 12.0 * np.sin(x / 45.0) * np.cos(z / 45.0) + 6.0 * np.cos(x / 90.0)

    # Flatten lake
    d_lake = np.sqrt(x**2 + z**2)
    lake_factor = np.clip((d_lake - LAKE_RADIUS) / 25.0, 0.0, 1.0)

    # Bank crossing water level exactly at the shoreline, for a clean round edge
    bank = np.clip((d_lake - LAKE_RADIUS) / 8.0, -1.0, 1.0) * 1.5
    return y * lake_factor + bank


# Procedural Terrain Generation
def generate_terrain(grid_size=120, size=600.0):
    N = grid_size + 1
    axis = np.linspace(-size / 2, size / 2, N)
    xs, zs = np.meshgrid(axis, axis, indexing="ij")
    ys = get_terrain_height(xs, zs)
    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]).astype(np.float32)

    # Color mapping by elevation, blending to sand around the lake
    h_norm = np.clip((ys.ravel() + 15.0) / 30.0, 0.0, 1.0)[:, None]
    grass = np.array([0.05, 0.20, 0.05]) + np.array([0.13, 0.14, 0.03]) * h_norm
    shore = (
        1.0
        - np.clip((np.hypot(xs, zs).ravel() - LAKE_RADIUS) / 10.0, 0.0, 1.0)[:, None]
    )
    rgb = grass * (1.0 - shore) + np.array([0.62, 0.52, 0.32]) * shore
    colors = np.column_stack([rgb, np.ones(len(rgb))]).astype(np.float32)

    r, c = np.meshgrid(np.arange(N - 1), np.arange(N - 1), indexing="ij")
    idx0 = (r * N + c).ravel()
    idx1 = ((r + 1) * N + c).ravel()
    idx2 = (r * N + c + 1).ravel()
    idx3 = ((r + 1) * N + c + 1).ravel()
    faces = np.concatenate(
        [np.column_stack([idx0, idx1, idx2]), np.column_stack([idx1, idx3, idx2])]
    ).astype(np.int32)
    return vertices, faces, colors


# Generate Ground surface
verts, faces, cols = generate_terrain()
ground = actor.surface(verts, faces=faces, colors=cols)
scene.add(ground)

# Lake surface
lake = actor.cylinder(
    centers=np.array([[LAKE_CENTER[0], 0.02, LAKE_CENTER[2]]]),
    directions=np.array([[0.0, 1.0, 0.0]]),
    colors=(0.10, 0.45, 0.70),
    height=0.08,
    radii=LAKE_RADIUS,
    sectors=128,
)
scene.add(lake)


# Vegetation patches, batched into one actor
bush_centers, bush_heights = [], []
for vp in veg_patches:
    for _ in range(12):
        wx = vp[0] + np.random.uniform(-15.0, 15.0)
        wz = vp[2] + np.random.uniform(-15.0, 15.0)
        sh = np.random.uniform(4.0, 8.0)
        bush_centers.append([wx, get_terrain_height(wx, wz) + sh * 0.4, wz])
        bush_heights.append(sh)

bush_heights = np.array(bush_heights)
bushes = actor.cone(
    centers=np.array(bush_centers),
    directions=np.array([[0.0, 1.0, 0.0]] * len(bush_centers)),
    colors=(0.10, 0.42, 0.14),
    height=bush_heights,
    radii=bush_heights * 0.4,
)
scene.add(bushes)

# Lights
ambient = AmbientLight(color=(0.55, 0.6, 0.55), intensity=1.8)
scene.add(ambient)


# Quaternion helpers
def axis_angle_to_quat(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotate_vector(quat, vec):
    q_vec = quat[:3]
    q_w = quat[3]
    uv = np.cross(q_vec, vec)
    uv_cross = np.cross(q_vec, uv)
    return vec + 2.0 * (q_w * uv + uv_cross)


def disable_depth_testing(world_object):
    if hasattr(world_object, "material") and world_object.material is not None:
        world_object.material.depth_test = False
        world_object.material.depth_write = False
    if hasattr(world_object, "children"):
        for child in world_object.children:
            disable_depth_testing(child)


def load_obj(path):
    """
    Read a triangulated, flat-shaded mesh and its UVs from a Kenney OBJ file.

    Parameters
    ----------
    path : str
        Path of the OBJ file.

    Returns
    -------
    tuple of ndarray
        Vertices (N, 3), triangle indices (M, 3) and UV coordinates (N, 2).
    """
    verts, uvs, faces, uv_indices = [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(p) for p in parts[1:4]])
            elif parts[0] == "vt":
                uvs.append([float(p) for p in parts[1:3]])
            elif parts[0] == "f":
                ids = [p.split("/") for p in parts[1:]]
                for i in range(1, len(ids) - 1):
                    tri = (ids[0], ids[i], ids[i + 1])
                    faces.append([int(p[0]) - 1 for p in tri])
                    uv_indices.append([int(p[1]) - 1 for p in tri])

    verts = np.array(verts, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    # OBJ UVs start at the bottom of the image, textures at the top
    uvs[:, 1] = 1.0 - uvs[:, 1]

    flat_verts = verts[np.array(faces).ravel()]
    flat_uvs = uvs[np.array(uv_indices).ravel()]
    flat_faces = np.arange(len(flat_verts)).reshape(-1, 3)
    return flat_verts, flat_faces, flat_uvs


_animal_templates = {}


def create_animal_actor(species):
    """
    Create a mesh for a species, sharing geometry and texture per species.

    Parameters
    ----------
    species : str
        One of "lion", "elephant" or "deer".

    Returns
    -------
    Mesh
        The animal actor.
    """
    if species not in _animal_templates:
        path = os.path.join(ANIMAL_KIT_DIR, ANIMAL_MODELS[species])
        texture = os.path.join(ANIMAL_KIT_DIR, "Textures", "colormap.png")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Download Kenney's Cube Pets pack "
                "(https://kenney.nl/assets/cube-pets) into the assets folder, "
                "or set FURY_KENNEY_ASSETS."
            )
        v, f, uv = load_obj(path)
        template = actor.surface(v, f, texture=texture, texture_coords=uv)
        template.material.pick_write = True
        _animal_templates[species] = (template.geometry, template.material)

    geometry, material = _animal_templates[species]
    return actor.create_mesh(geometry, material)


# Gold marker that hovers over the selected animal
selection_marker = actor.cone(
    centers=np.array([[0.0, 0.0, 0.0]]),
    directions=np.array([[0.0, -1.0, 0.0]]),
    colors=(1.0, 0.8, 0.1),
    height=3.0,
    radii=1.4,
)
selection_marker.visible = False
scene.add(selection_marker)


# Animal Spawn Helper
def spawn_animal(species, position, is_child=False):
    """
    Add an animal to the scene and the simulation.

    Parameters
    ----------
    species : str
        One of "lion", "elephant" or "deer".
    position : ndarray
        Ground position; its height is replaced by the terrain height.
    is_child : bool, optional
        Whether the animal is born young and still growing.

    Returns
    -------
    dict
        The new animal's simulation record.
    """
    position[1] = get_terrain_height(position[0], position[2])

    art = create_animal_actor(species)
    base_scale = ANIMAL_SCALE[species]
    art.local.scale = [base_scale, base_scale, base_scale]
    art.local.position = position
    scene.add(art)

    animal = {
        "id": state["next_animal_id"],
        "species": species,
        "pos": np.array(position, dtype=np.float32),
        "vel": np.random.randn(3).astype(np.float32),
        "actor": art,
        "age": np.random.uniform(16.0, 45.0) if not is_child else 0.0,
        "health": MAX_HEALTH[species],
        "hunger": np.random.uniform(10.0, 30.0),
        "thirst": np.random.uniform(10.0, 30.0),
        "is_child": is_child,
        "gender": np.random.choice(["M", "F"]),
        "cooldown": np.random.uniform(5.0, 15.0),
        "hunt_target_id": None,
        "goal": None,
    }
    animal["vel"][1] = 0.0
    speed = BASE_SPEED[species]
    animal["vel"] = (animal["vel"] / (np.linalg.norm(animal["vel"]) + 1e-5)) * speed

    animal["heading"] = animal["vel"] / (np.linalg.norm(animal["vel"]) + 1e-5)

    if is_child:
        child_scale = base_scale * 0.4
        art.local.scale = [child_scale, child_scale, child_scale]

    art.agent_idx = animal["id"]
    state["animals"].append(animal)
    state["next_animal_id"] += 1
    return animal


# Initial Population Spawning
for _ in range(55):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("deer", np.array([rx, 0.0, rz]))

for _ in range(12):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("lion", np.array([rx, 0.0, rz]))

for _ in range(10):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("elephant", np.array([rx, 0.0, rz]))


# UI Redesign - Info Panel (Height expanded to 280 to hold two-column status)
info_panel = ui.Panel2D(
    size=(320, 280), color=(0.06, 0.09, 0.06), has_border=True, border_width=2
)
info_panel.set_position((15, 15))

lbl_title = ui.TextBlock2D(
    text="JUNGLE ECOSYSTEM",
    position=(20, 15),
    font_size=16,
    color=(0.9, 0.75, 0.2),
    bold=True,
    dynamic_bbox=True,
)
info_panel.add_element(lbl_title, (20, 15))

lbl_legend = ui.TextBlock2D(
    text="Lions: 0 | Elephants: 0 | Deers: 0",
    position=(20, 42),
    font_size=13,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_legend, (20, 42))

# Snug fit Animal focus buttons placed inside Info panel below legend
btn_states_lion = {
    "hover": {"text": "LION", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "LION", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "LION", "color": (0.15, 0.25, 0.15)},
}
btn_lion = ui.TextButton2D(
    label="LION",
    size=(80, 26),
    position=(20, 68),
    font_size=14,
    states=btn_states_lion,
)

btn_states_ele = {
    "hover": {"text": "ELEPHANT", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "ELEPHANT", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "ELEPHANT", "color": (0.15, 0.25, 0.15)},
}
btn_ele = ui.TextButton2D(
    label="ELEPHANT",
    size=(90, 26),
    position=(110, 68),
    font_size=14,
    states=btn_states_ele,
)

btn_states_deer = {
    "hover": {"text": "DEER", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "DEER", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "DEER", "color": (0.15, 0.25, 0.15)},
}
btn_deer = ui.TextButton2D(
    label="DEER",
    size=(80, 26),
    position=(210, 68),
    font_size=14,
    states=btn_states_deer,
)


def select_closest_animal_by_species(species):
    candidates = [a for a in state["animals"] if a["species"] == species]
    if len(candidates) > 0:
        target = min(
            candidates, key=lambda a: np.linalg.norm(a["pos"] - state["cam_pos"])
        )
        state["selected_animal"] = target["id"]
        print(f"Selected Species Snapping focus to animal ID #{target['id']}")


btn_lion.on_clicked = lambda event: select_closest_animal_by_species("lion")
btn_ele.on_clicked = lambda event: select_closest_animal_by_species("elephant")
btn_deer.on_clicked = lambda event: select_closest_animal_by_species("deer")

info_panel.add_element(btn_lion, (20, 68))
info_panel.add_element(btn_ele, (110, 68))
info_panel.add_element(btn_deer, (210, 68))

lbl_card_title = ui.TextBlock2D(
    text="SELECTED ANIMAL STATUS",
    position=(20, 108),
    font_size=13,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
info_panel.add_element(lbl_card_title, (20, 108))

# Two columns TextBlock2D info
lbl_animal_info_left = ui.TextBlock2D(
    text="Click an animal\nto monitor.",
    position=(20, 132),
    font_size=14,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_animal_info_left, (20, 132))

lbl_animal_info_right = ui.TextBlock2D(
    text="",
    position=(165, 132),
    font_size=14,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_animal_info_right, (165, 132))

scene.add(info_panel)


# Controls Panel - Now a TabUI container separating factors
control_panel = ui.TabUI(
    position=(15, 235),
    size=(340, 700),
    tab_titles=["Global", "Lion", "Elephant", "Deer"],
    startup_tab_id=0,
    font_size=16,
    active_color=(0.2, 0.5, 0.8),
    inactive_color=(0.3, 0.3, 0.3),
)

# ----------------- TAB 0: GLOBAL FACTORS -----------------
lbl_g_title = ui.TextBlock2D(
    text="GLOBAL TUNING FACTORS",
    position=(20, 15),
    font_size=15,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
control_panel.add_element(0, lbl_g_title, (20, 15))

start_y = 70
y_step = 45
current_y = start_y

slider_g_speed = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Speed Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_speed, (20, current_y))
current_y += y_step

slider_g_hunger = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Hunger Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_hunger, (20, current_y))
current_y += y_step

slider_g_thirst = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Thirst Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_thirst, (20, current_y))
current_y += y_step

slider_g_age = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Age Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_age, (20, current_y))
current_y += y_step

slider_g_mating = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Mating Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_mating, (20, current_y))
current_y += y_step

slider_g_hunting = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Hunting Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_hunting, (20, current_y))
current_y += y_step

lbl_g_sim = ui.TextBlock2D(
    text="SIMULATION RATE CONTROLS",
    position=(20, current_y),
    font_size=15,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
control_panel.add_element(0, lbl_g_sim, (20, current_y))
current_y += 70

slider_global_coh = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Cohesion Mult: {value:.1f}",
)
control_panel.add_element(0, slider_global_coh, (20, current_y))
current_y += y_step

slider_global_sep = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Separation Mult: {value:.1f}",
)
control_panel.add_element(0, slider_global_sep, (20, current_y))
current_y += y_step

slider_global_speed = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=1.0,
    max_value=10.0,
    length=180,
    text_template="Sim Speed: {value:.1f}x",
)
control_panel.add_element(0, slider_global_speed, (20, current_y))
current_y += y_step

# Reset button
btn_states_reset = {
    "hover": {"text": "RESET ALL FACTORS", "color": (0.5, 0.2, 0.2)},
    "pressed": {"text": "RESETTING...", "color": (0.3, 0.1, 0.1)},
    "default": {"text": "RESET ALL FACTORS", "color": (0.4, 0.15, 0.15)},
}
btn_reset = ui.TextButton2D(
    label="RESET ALL FACTORS",
    size=(250, 25),
    position=(20, current_y),
    font_size=14,
    states=btn_states_reset,
)
control_panel.add_element(0, btn_reset, (20, current_y))


# Global callbacks
def on_g_speed(slider):
    state["global_speed_factor"] = slider.value


def on_g_hunger(slider):
    state["global_hunger_factor"] = slider.value


def on_g_thirst(slider):
    state["global_thirst_factor"] = slider.value


def on_g_age(slider):
    state["global_age_factor"] = slider.value


def on_g_mating(slider):
    state["global_mating_factor"] = slider.value


def on_g_hunting(slider):
    state["global_hunting_factor"] = slider.value


def on_global_coh_change(slider):
    state["cohesion_mult"] = slider.value


def on_global_sep_change(slider):
    state["separation_mult"] = slider.value


def on_global_speed_change(slider):
    state["sim_speed"] = slider.value


slider_g_speed.on_change = on_g_speed
slider_g_hunger.on_change = on_g_hunger
slider_g_thirst.on_change = on_g_thirst
slider_g_age.on_change = on_g_age
slider_g_mating.on_change = on_g_mating
slider_g_hunting.on_change = on_g_hunting
slider_global_coh.on_change = on_global_coh_change
slider_global_sep.on_change = on_global_sep_change
slider_global_speed.on_change = on_global_speed_change

# Keep references to species specific sliders so we can reset their values
species_sliders = []


# Helper to populate species specific factor sliders
def build_species_tab(tab_idx, species_name):
    lbl_title = ui.TextBlock2D(
        text=f"{species_name.upper()} FACTOR ADJUSTERS",
        position=(20, 15),
        font_size=15,
        color=(0.95, 0.8, 0.2),
        bold=True,
        dynamic_bbox=True,
    )
    control_panel.add_element(tab_idx, lbl_title, (20, 15))

    start_y = 70
    y_step = 45
    current_y = start_y

    s_speed = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Speed Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_speed, (20, current_y))
    current_y += y_step

    s_hunger = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Hunger Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_hunger, (20, current_y))
    current_y += y_step

    s_thirst = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Thirst Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_thirst, (20, current_y))
    current_y += y_step

    s_age = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Age Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_age, (20, current_y))
    current_y += y_step

    s_mating = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Mating Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_mating, (20, current_y))
    current_y += y_step

    s_hunting = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Hunting Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_hunting, (20, current_y))

    # Event bindings
    s_speed.on_change = lambda sl: state.update(
        {f"{species_name}_speed_factor": sl.value}
    )
    s_hunger.on_change = lambda sl: state.update(
        {f"{species_name}_hunger_factor": sl.value}
    )
    s_thirst.on_change = lambda sl: state.update(
        {f"{species_name}_thirst_factor": sl.value}
    )
    s_age.on_change = lambda sl: state.update({f"{species_name}_age_factor": sl.value})
    s_mating.on_change = lambda sl: state.update(
        {f"{species_name}_mating_factor": sl.value}
    )
    s_hunting.on_change = lambda sl: state.update(
        {f"{species_name}_hunting_factor": sl.value}
    )

    # Track for reset action
    species_sliders.append(s_speed)
    species_sliders.append(s_hunger)
    species_sliders.append(s_thirst)
    species_sliders.append(s_age)
    species_sliders.append(s_mating)
    species_sliders.append(s_hunting)


# Reset All Factors Callback
def reset_all_factors(event):
    state["global_speed_factor"] = 1.0
    state["global_hunger_factor"] = 1.0
    state["global_thirst_factor"] = 1.0
    state["global_age_factor"] = 1.0
    state["global_mating_factor"] = 1.0
    state["global_hunting_factor"] = 1.0

    state["cohesion_mult"] = 1.0
    state["separation_mult"] = 1.0
    state["sim_speed"] = 1.0

    for name in ["lion", "elephant", "deer"]:
        state[f"{name}_speed_factor"] = 1.0
        state[f"{name}_hunger_factor"] = 1.0
        state[f"{name}_thirst_factor"] = 1.0
        state[f"{name}_age_factor"] = 1.0
        state[f"{name}_mating_factor"] = 1.0
        state[f"{name}_hunting_factor"] = 1.0

    # Sync UI Sliders
    slider_g_speed.value = 1.0
    slider_g_hunger.value = 1.0
    slider_g_thirst.value = 1.0
    slider_g_age.value = 1.0
    slider_g_mating.value = 1.0
    slider_g_hunting.value = 1.0
    slider_global_coh.value = 1.0
    slider_global_sep.value = 1.0
    slider_global_speed.value = 1.0

    for sl in species_sliders:
        sl.value = 1.0


btn_reset.on_clicked = reset_all_factors

# Build Lion, Elephant, Deer Tabs
build_species_tab(1, "lion")
build_species_tab(2, "elephant")
build_species_tab(3, "deer")


def on_tab_change(panel):
    """
    Schedule a layout so the newly shown tab's text is positioned correctly.

    Parameters
    ----------
    panel : TabUI
        The panel whose tab changed, required by the callback signature.
    """
    state["layout_due"] = [state["clock"], state["clock"] + 0.3]


control_panel.on_change = on_tab_change

scene.add(control_panel)


# Click selection logic
def on_click(event):
    """
    Select the clicked animal, or clear the selection on the ground or lake.

    Parameters
    ----------
    event : PointerEvent
        The pointer press.
    """
    target = event.target
    if hasattr(target, "agent_idx"):
        state["selected_animal"] = target.agent_idx
        print(f"Selected Animal ID #{target.agent_idx}")
    elif target is ground or target is lake:
        state["selected_animal"] = None


# Keyboard input handlers for flying
def on_key_down(event):
    state["keys"].add(event.key.lower())


def on_key_up(event):
    if event.key.lower() in state["keys"]:
        state["keys"].remove(event.key.lower())


# Drag input handlers for free look
def is_ui_target(target):
    """
    Return whether an event target belongs to the 2D UI overlay.

    Parameters
    ----------
    target : WorldObject or None
        The target of a pointer event.

    Returns
    -------
    bool
        True when the target is part of the UI overlay.
    """
    while target is not None:
        if target is scene.ui_scene:
            return True
        target = getattr(target, "parent", None)
    return False


def on_pointer_down(event):
    if event.button == 1:
        # Dragging a slider must not also swing the camera
        if not hasattr(event.target, "agent_idx") and not is_ui_target(event.target):
            state["is_dragging_cam"] = True
            state["last_mouse"] = (event.x, event.y)


def on_pointer_move(event):
    if state["is_dragging_cam"] and state["selected_animal"] is None:
        if state["last_mouse"] is not None:
            dx = event.x - state["last_mouse"][0]
            dy = event.y - state["last_mouse"][1]
            state["last_mouse"] = (event.x, event.y)

            # Update orientation
            state["cam_yaw"] -= dx * 0.003
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] - dy * 0.003, -np.pi / 2.2, np.pi / 2.2
            )


def on_pointer_up(event):
    state["is_dragging_cam"] = False
    state["last_mouse"] = None


# Main simulation loop callback
def sim_tick(showm):
    """
    Advance the simulation by the real time since the previous tick.

    Parameters
    ----------
    showm : ShowManager
        The show manager running the simulation.
    """
    now = perf_counter()
    last_tick = state["last_tick"]
    real_dt = 1.0 / 60.0 if last_tick is None else min(now - last_tick, 0.1)
    state["last_tick"] = now
    advance_simulation(
        showm.screens[0].camera, tuple(showm.renderer.logical_size), real_dt
    )
    showm.render()


def choose_goal(animal):
    """
    Choose which need an animal works on.

    An animal keeps its goal until that need is met, so it does not turn back
    halfway between the lake and its food. It switches early only when the other
    need becomes urgent.

    Parameters
    ----------
    animal : dict
        The animal, with its ``hunger``, ``thirst`` and current ``goal``.

    Returns
    -------
    str or None
        ``"food"``, ``"water"``, or None when the animal has nothing to do.
    """
    level = {"food": animal["hunger"], "water": animal["thirst"]}
    goal = animal["goal"]
    if goal is not None:
        other = "water" if goal == "food" else "food"
        if level[other] >= NEED_URGENT and level[goal] < NEED_URGENT:
            return other
        if level[goal] > NEED_MET:
            return goal
    due = [need for need in level if level[need] > NEED_START[need]]
    if not due:
        return None
    return max(due, key=lambda need: level[need])


def step_ecosystem(dt):
    """
    Advance every animal by one fixed simulation step.

    Parameters
    ----------
    dt : float
        Simulated time covered by this step.
    """
    animals = state["animals"]

    # 1. Animal Needs & Aging Update loop
    for a in animals:
        # An animal killed in the previous step must not heal before it is removed
        if a["health"] <= 0.0:
            continue
        in_water = np.hypot(a["pos"][0], a["pos"][2]) <= LAKE_RADIUS

        sp = a["species"]
        a["age"] += dt * 0.008 * state["global_age_factor"] * state[f"{sp}_age_factor"]

        h_rate = HUNGER_RATE[sp]
        t_rate = THIRST_RATE[sp]

        # Quench thirst if in water, otherwise build up thirst
        if in_water:
            a["thirst"] = max(0.0, a["thirst"] - dt * 15.0)
        else:
            a["thirst"] += (
                dt
                * t_rate
                * state["global_thirst_factor"]
                * state[f"{sp}_thirst_factor"]
            )

        # Build up hunger for all species (reduced later via grazing or hunting)
        a["hunger"] += (
            dt * h_rate * state["global_hunger_factor"] * state[f"{sp}_hunger_factor"]
        )

        a["hunger"] = min(a["hunger"], 100.0)
        a["thirst"] = min(a["thirst"], 100.0)

        # Update mating cooldown
        if a["cooldown"] > 0.0:
            a["cooldown"] = max(0.0, a["cooldown"] - dt)

        # Growth scaling for children
        if a["is_child"]:
            scale = ANIMAL_SCALE[sp] * (0.4 + min(0.6, a["age"] * 0.05))
            a["actor"].local.scale = [scale, scale, scale]
            if a["age"] > 12.0:
                a["is_child"] = False

        # Apply starvation/dehydration damage
        if a["hunger"] >= 100.0 or a["thirst"] >= 100.0:
            a["health"] -= dt * STARVATION_DAMAGE
        else:
            a["health"] = min(MAX_HEALTH[sp], a["health"] + dt * 1.5)

    # Filter dead animals
    dead_list = []
    alive_list = []

    for a in animals:
        limit_age = MAX_AGE[a["species"]]
        if a["health"] <= 0.0 or a["age"] >= limit_age:
            dead_list.append(a)
        else:
            alive_list.append(a)

    # Remove dead actors
    for a in dead_list:
        scene.remove(a["actor"])
        if state["selected_animal"] == a["id"]:
            state["selected_animal"] = None

    state["animals"] = alive_list
    animals = state["animals"]

    # 2. Ecosystem Behaviors, Steer Forces & Movement
    new_spawns = []

    # One shared ground-plane distance table replaces per-animal scans of every animal
    positions = np.array([a["pos"] for a in animals], dtype=np.float64).reshape(-1, 3)
    positions[:, 1] = 0.0
    velocities = np.array([a["vel"] for a in animals], dtype=np.float64).reshape(-1, 3)
    offsets = positions[:, None, :] - positions[None, :, :]  # offsets[i, j] = i - j
    distances = np.sqrt(np.sum(offsets**2, axis=-1))
    species_index = {
        name: np.array(
            [i for i, a in enumerate(animals) if a["species"] == name], dtype=int
        )
        for name in ANIMAL_MODELS
    }
    index_by_id = {a["id"]: i for i, a in enumerate(animals)}

    def nearest(i, candidates):
        """
        Return the candidate closest to an animal and its distance.

        Parameters
        ----------
        i : int
            Index of the animal.
        candidates : ndarray
            Indices of the animals to search.

        Returns
        -------
        tuple
            The closest candidate index and its distance, or None and infinity
            when there are no candidates.
        """
        candidates = candidates[[animals[j]["health"] > 0.0 for j in candidates]]
        if candidates.size == 0:
            return None, np.inf
        best = candidates[np.argmin(distances[i, candidates])]
        return best, distances[i, best]

    for i, a in enumerate(animals):
        # Animals killed earlier in this step neither move nor breed
        if a["health"] <= 0.0:
            continue
        sp = a["species"]
        # Base movement speed scaled by speed factors
        current_max_speed = (
            BASE_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]
        )
        run_speed_val = (
            RUN_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]
        )

        if a["is_child"]:
            current_max_speed *= 0.6
            run_speed_val *= 0.6

        a["goal"] = choose_goal(a)
        seek_water = a["goal"] == "water"
        seek_food = a["goal"] == "food"

        # Species Grouping / Flocking dynamics
        same_species = species_index[sp]
        same_species = same_species[same_species != i]

        flock_cohesion = np.zeros(3)
        flock_alignment = np.zeros(3)
        flock_separation = np.zeros(3)

        if same_species.size > 0:
            diffs = offsets[i, same_species]
            dists_sq = distances[i, same_species] ** 2
            neighbors = same_species[dists_sq < 60.0**2]
            sep_mask = dists_sq < 12.0**2

            if neighbors.size > 0:
                flock_cohesion = positions[neighbors].mean(axis=0) - positions[i]
                flock_alignment = velocities[neighbors].mean(axis=0) - a["vel"]

            if np.any(sep_mask):
                flock_separation = np.sum(
                    diffs[sep_mask] / (dists_sq[sep_mask][:, None] + 1e-5), axis=0
                )

        flock_weight = 1.0
        if a["goal"] is not None:
            flock_weight = 0.1

        coeffs = GROUP_COEFFS[sp]
        steer = (
            flock_cohesion * coeffs["cohesion"] * 0.1 * state["cohesion_mult"]
            + flock_alignment * coeffs["alignment"] * 0.1
        ) * flock_weight + flock_separation * coeffs["separation"] * 0.3 * state[
            "separation_mult"
        ]

        # Lake Seeking (Thirst)
        if seek_water:
            d_lake = np.linalg.norm(positions[i])
            if d_lake > LAKE_RADIUS:
                steer -= (positions[i] / (d_lake + 1e-5)) * 2.8

        # Grazing (Deers and Elephants feeding on vegetation patches)
        if sp in ["deer", "elephant"] and seek_food:
            to_patches = VEG_PATCH_CENTERS - positions[i]
            patch_dists = np.linalg.norm(to_patches, axis=1)
            closest_patch = np.argmin(patch_dists)
            min_d = patch_dists[closest_patch]
            steer += (to_patches[closest_patch] / (min_d + 1e-5)) * 2.2
            if min_d < 25.0:
                a["hunger"] = max(0.0, a["hunger"] - dt * 15.0)

        # Young animals feed from any adult of their species close by
        if a["is_child"]:
            near = same_species[distances[i, same_species] < NURSING_RANGE]
            if any(
                not animals[j]["is_child"] and animals[j]["health"] > 0.0 for j in near
            ):
                a["hunger"] = max(0.0, a["hunger"] - dt * 15.0)
            elif sp == "lion" and seek_food:
                # Cubs cannot hunt, so a hungry cub goes back to the pride
                adults = same_species[
                    [not animals[j]["is_child"] for j in same_species]
                ]
                adult, dist = nearest(i, adults)
                if adult is not None:
                    steer -= (offsets[i, adult] / (dist + 1e-5)) * 2.2

        # Hunting state & pack coordination (Lions targeting Deers)
        a_target = None
        own = index_by_id.get(a["hunt_target_id"])
        if (
            sp == "lion"
            and own is not None
            and animals[own]["species"] == "deer"
            and animals[own]["health"] > 0.0
            and distances[i, own] < LION_FINISH_RANGE
        ):
            # A lion close to its prey finishes the kill even once fed
            a_target = own
        elif sp == "lion" and seek_food and not a["is_child"]:
            # Follow the first nearby packmate that is already hunting live prey
            for j in same_species[distances[i, same_species] < 40.0]:
                candidate = index_by_id.get(animals[j]["hunt_target_id"])
                if (
                    candidate is not None
                    and animals[candidate]["species"] == "deer"
                    and animals[candidate]["health"] > 0.0
                ):
                    a_target = candidate
                    break

            if a_target is None:
                a_target, _ = nearest(i, species_index["deer"])

        if a_target is not None:
            target_deer = animals[a_target]
            min_d = distances[i, a_target]
            a["hunt_target_id"] = target_deer["id"]
            current_max_speed = run_speed_val
            steer -= (offsets[i, a_target] / (min_d + 1e-5)) * 3.8

            # Attack/Deal damage over time
            if min_d < 3.5:
                damage = (
                    dt
                    * 45.0
                    * state["global_hunting_factor"]
                    * state["lion_hunting_factor"]
                )
                target_deer["health"] -= damage
                # Reduce hunger while eating
                a["hunger"] = max(0.0, a["hunger"] - damage * 1.5)
                if target_deer["health"] <= 0.0:
                    # The pride shares the kill with every lion on the hunt
                    pride = same_species[distances[i, same_species] < LION_FINISH_RANGE]
                    for j in pride:
                        if animals[j]["hunt_target_id"] == target_deer["id"]:
                            animals[j]["hunger"] = 0.0
                            animals[j]["hunt_target_id"] = None
                    a["hunger"] = 0.0
                    a["hunt_target_id"] = None
        else:
            a["hunt_target_id"] = None

        # Predator Avoidance (Deers running from Lions)
        if sp == "deer":
            closest_lion, min_d = nearest(i, species_index["lion"])
            if closest_lion is not None and min_d < DEER_FLEE_RANGE:
                current_max_speed = run_speed_val
                steer += (offsets[i, closest_lion] / (min_d + 1e-5)) * 3.8

        # Lions avoid Elephants
        if sp == "lion":
            close = species_index["elephant"]
            # A healthy lion on a hunt braves elephants, a wounded one keeps away
            bold = a_target is not None and a["health"] > 0.6 * MAX_HEALTH["lion"]
            flee_range = LION_HUNT_FLEE_RANGE if bold else LION_FLEE_RANGE
            close = close[distances[i, close] < flee_range]
            if close.size > 0:
                current_max_speed = run_speed_val
                steer += (
                    np.sum(
                        offsets[i, close] / (distances[i, close][:, None] + 1e-5),
                        axis=0,
                    )
                    * 3.8
                )

        # Elephant Family Defense & Retaliation against Lions
        if sp == "elephant" and not a["is_child"]:
            lions = species_index["lion"]
            adult_lions = lions[[not animals[j]["is_child"] for j in lions]]
            target_lion, dist = nearest(i, adult_lions)
            if target_lion is not None and dist < ELEPHANT_CHARGE_RANGE:
                current_max_speed = run_speed_val
                steer -= (offsets[i, target_lion] / (dist + 1e-5)) * 4.2
                if dist < 4.5:
                    animals[target_lion]["health"] -= dt * ELEPHANT_DAMAGE

        is_satiated = a["goal"] is None and a_target is None
        if is_satiated:
            current_max_speed *= 0.3  # chill out / wander slowly

        # Breeding Logic
        if (
            is_satiated
            and a["cooldown"] <= 0.0
            and not a["is_child"]
            and len(animals) + len(new_spawns) < MAX_POPULATION
            and species_index[sp].size + sum(s == sp for s, _ in new_spawns)
            < SPECIES_LIMIT[sp]
        ):
            for j in same_species[distances[i, same_species] < 12.0]:
                partner = animals[j]
                if (
                    partner["gender"] != a["gender"]
                    and partner["health"] > 0.0
                    and partner["cooldown"] <= 0.0
                    and not partner["is_child"]
                ):
                    cooldown_val = 35.0 / (
                        state["global_mating_factor"] * state[f"{sp}_mating_factor"]
                        + 1e-5
                    )
                    a["cooldown"] = cooldown_val
                    partner["cooldown"] = cooldown_val
                    new_spawns.append((sp, (positions[i] + positions[j]) * 0.5))
                    break

        # Map Boundaries Constraint
        dist_from_origin = np.linalg.norm(positions[i])
        if dist_from_origin > 280.0:
            steer -= (positions[i] / (dist_from_origin + 1e-5)) * 3.5

        # Update velocity & limit speed
        a["vel"] += steer * dt * 22.0
        a["vel"][1] = 0.0  # Animals steer on the ground plane only
        speed = np.linalg.norm(a["vel"])
        if speed > current_max_speed:
            a["vel"] = (a["vel"] / speed) * current_max_speed
        elif speed < MIN_SPEED:
            a["vel"] = (a["vel"] / (speed + 1e-5)) * MIN_SPEED

        # Position update
        a["pos"] += a["vel"] * dt

        # Hard Boundary constraint: keep inside boundary radius of 290
        d_origin = np.hypot(a["pos"][0], a["pos"][2])
        if d_origin > 290.0:
            inward = np.array([-a["pos"][0], 0.0, -a["pos"][2]]) / d_origin
            a["pos"][0] *= 290.0 / d_origin
            a["pos"][2] *= 290.0 / d_origin
            a["vel"] = (inward * np.linalg.norm(a["vel"])).astype(np.float32)

        # Project animal pos dynamically to follow terrain elevation changes
        a["pos"][1] = get_terrain_height(a["pos"][0], a["pos"][2])

        speed = np.linalg.norm(a["vel"])
        if speed > 1e-3:
            a["heading"] = a["vel"] / speed

    # Spawn new babies
    for species, position in new_spawns:
        spawn_animal(species, position, is_child=True)


def advance_simulation(camera, screen_size, real_dt):
    """
    Advance the ecosystem, the UI and the camera by one step.

    Parameters
    ----------
    camera : Camera
        The camera to move.
    screen_size : tuple of float
        Logical window size used to lay out the UI.
    real_dt : float
        Real time covered by this step in seconds.
    """
    state["clock"] += real_dt
    now = state["clock"]

    # FURY reframes the camera on resize, so restore where this demo left it
    camera.local.position = state["cam_pos"]

    # Laying out the UI is expensive, so never do it every frame. Text sizes are only
    # known once drawn, so repeat it shortly after startup or a window resize.
    if screen_size != state["screen_size"]:
        state["screen_size"] = screen_size
        state["layout_due"] = [now, now + 0.3, now + 1.0]
    if state["layout_due"] and now >= state["layout_due"][0]:
        state["layout_due"].pop(0)
        info_panel.set_position((15, 15))
        control_panel.set_position((screen_size[0] - 335.0, 15.0))

    # Large steps let animals jump past each other, so split the time into small
    # steps. A slow machine runs the ecosystem slower rather than less accurately.
    sim_time = real_dt * SIM_TIME_SCALE * state["sim_speed"]
    steps = max(1, int(np.ceil(sim_time / MAX_SIM_STEP - 1e-9)))
    steps = min(steps, MAX_SIM_STEPS_PER_FRAME)
    for _ in range(steps):
        step_ecosystem(min(sim_time / steps, MAX_SIM_STEP))
    animals = state["animals"]
    # Only the last sub-step is drawn, so move the actors once per frame
    for a in animals:
        a["actor"].local.position = a["pos"]
        angle = np.arctan2(a["heading"][0], a["heading"][2])
        a["actor"].local.rotation = [0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)]
    num_lions = sum(a["species"] == "lion" for a in animals)
    num_elephants = sum(a["species"] == "elephant" for a in animals)
    num_deers = len(animals) - num_lions - num_elephants

    # 3. UI Status updates
    # Text layout is costly and unreadable at frame rate, so refresh 5 times a second
    refresh_labels = now >= state["next_label_update"]
    if refresh_labels:
        state["next_label_update"] = now + 0.2
        lbl_legend.message = (
            f"Lions: {num_lions} | Elephants: {num_elephants} | Deers: {num_deers}"
        )

    selected = state["selected_animal"]

    # Hover the selection marker above the selected animal
    selection_marker.visible = False
    for a in animals:
        if a["id"] == selected:
            top = ANIMAL_HEIGHT[a["species"]] * a["actor"].local.scale[1] + 3.0
            selection_marker.local.position = a["pos"] + np.array([0.0, top, 0.0])
            selection_marker.visible = True
            break

    # 4. Camera Control (Focus vs Free look Fly mode)
    if selected is not None:
        selected_a = None
        for a in animals:
            if a["id"] == selected:
                selected_a = a
                break

        if selected_a is not None:
            gender_txt = "Male" if selected_a["gender"] == "M" else "Female"
            type_txt = "Calf" if selected_a["is_child"] else "Adult"
            if selected_a["species"] == "deer":
                type_txt = "Fawn" if selected_a["is_child"] else "Adult"
            elif selected_a["species"] == "lion":
                type_txt = "Cub" if selected_a["is_child"] else "Adult"

            # Display stats on details card (Two columns)
            if refresh_labels:
                lbl_animal_info_left.message = (
                    f"Species: {selected_a['species'].upper()}\n"
                    f"Class: {type_txt}\n"
                    f"Sex: {gender_txt}\n"
                    f"Health: {selected_a['health']:.1f}"
                )
                lbl_animal_info_right.message = (
                    f"Hunger: {selected_a['hunger']:.1f}%\n"
                    f"Thirst: {selected_a['thirst']:.1f}%\n"
                    f"Age: {selected_a['age']:.1f}\n"
                    f"CD: {selected_a['cooldown']:.1f}s"
                )

            # Chase camera positioning
            b_pos = selected_a["pos"]
            h = selected_a["heading"]

            target_cam_pos = b_pos - h * 25.0 + np.array([0.0, 12.0, 0.0])
            follow = 1.0 - np.exp(-6.0 * real_dt)  # Same feel at any frame rate
            camera.local.position = (
                camera.local.position
                + (target_cam_pos - camera.local.position) * follow
            )
            camera.look_at(b_pos + h * 4.0)

            # Update camera look rotation states to match
            diff = b_pos - camera.local.position
            # Free-look forward is (-sin yaw, -cos yaw) on the ground plane
            state["cam_yaw"] = np.arctan2(-diff[0], -diff[2])
            state["cam_pitch"] = np.arcsin(
                np.clip(diff[1] / (np.linalg.norm(diff) + 1e-5), -0.99, 0.99)
            )
        else:
            state["selected_animal"] = None
    else:
        if refresh_labels:
            lbl_animal_info_left.message = "Click an animal\nto monitor."
            lbl_animal_info_right.message = ""

        keys = state["keys"]

        # Keyboard Camera Rotation
        if "arrowleft" in keys or "left" in keys:
            state["cam_yaw"] += 1.5 * real_dt
        if "arrowright" in keys or "right" in keys:
            state["cam_yaw"] -= 1.5 * real_dt
        if "arrowup" in keys or "up" in keys:
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] + 1.5 * real_dt, -np.pi / 2.2, np.pi / 2.2
            )
        if "arrowdown" in keys or "down" in keys:
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] - 1.5 * real_dt, -np.pi / 2.2, np.pi / 2.2
            )

        # Counterstrike style WASD fly camera mode
        qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
        qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
        camera.local.rotation = quat_mult(qy, qx)

        fwd = rotate_vector(camera.local.rotation, np.array([0.0, 0.0, -1.0]))
        right = rotate_vector(camera.local.rotation, np.array([1.0, 0.0, 0.0]))

        fly_speed = 150.0
        move = np.zeros(3)

        if "w" in keys:
            move += fwd
        if "s" in keys:
            move -= fwd
        if "a" in keys:
            move -= right
        if "d" in keys:
            move += right
        if "q" in keys:
            move += np.array([0.0, 1.0, 0.0])
        if "e" in keys:
            move -= np.array([0.0, 1.0, 0.0])

        if np.any(move):
            move_dir = move / np.linalg.norm(move)
            camera.local.position = (
                camera.local.position + move_dir * fly_speed * real_dt
            )

    # Enforce camera does not move below ground
    cx, cy, cz = camera.local.position
    cam_terrain_h = get_terrain_height(cx, cz)
    cy_new = max(cy, cam_terrain_h + 5.0)
    camera.local.position = np.array([cx, cy_new, cz])
    state["cam_pos"] = np.array(camera.local.position)


def place_initial_camera(camera):
    """
    Place the camera above the map, looking at the lake.

    Parameters
    ----------
    camera : Camera
        The camera to place.
    """
    camera.local.position = (0.0, 170.0, 240.0)
    state["cam_pos"] = np.array(camera.local.position)
    state["cam_yaw"] = 0.0
    state["cam_pitch"] = -np.radians(40.0)
    qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
    qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
    camera.local.rotation = quat_mult(qy, qx)


if __name__ == "__main__":
    show_manager = window.ShowManager(
        scene=scene, size=(1024, 768), title="Jungle Ecosystem Survival Simulator"
    )

    # Disable default camera controller to allow our WASD mouse free-look camera
    show_manager.screens[0].controller.enabled = False

    # Bind pointer clicks and keyboard fly controls
    show_manager.renderer.add_event_handler(on_click, EventType.POINTER_DOWN)
    show_manager.renderer.add_event_handler(on_key_down, EventType.KEY_DOWN)
    show_manager.renderer.add_event_handler(on_key_up, EventType.KEY_UP)
    show_manager.renderer.add_event_handler(on_pointer_down, EventType.POINTER_DOWN)
    show_manager.renderer.add_event_handler(on_pointer_move, EventType.POINTER_MOVE)
    show_manager.renderer.add_event_handler(on_pointer_up, EventType.POINTER_UP)

    place_initial_camera(show_manager.screens[0].camera)

    # Ensure UI elements always render on top
    disable_depth_testing(scene.ui_scene)

    # Start simulation
    show_manager.register_callback(sim_tick, 0.016, True, "JungleLoop", show_manager)
    show_manager.start()
