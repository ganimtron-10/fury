"""
=============================================
Interactive 3D Jungle Survival Flocking Simulation
=============================================

An interactive 3D jungle ecosystem simulation containing Lions, Elephants,
and Deers navigating land, lakes, rivers, and vegetation patches.
Features survival states (thirst, hunger, aging, mating, hunting, and death),
a dynamic minimap, WASD free fly camera, and diagnostic control interfaces.
"""

import numpy as np
import pygfx as gfx
from fury import actor, ui, window
from fury.window import EventType

# Simulation dimensions and constants
JUNGLE_SIZE = 600.0
LAKE_RADIUS = 55.0
LAKE_CENTER = np.array([0.0, 0.0, 0.0])

# Species characteristics
MAX_HEALTH = {"lion": 150.0, "elephant": 500.0, "deer": 100.0}
MAX_AGE = {"lion": 90.0, "elephant": 140.0, "deer": 75.0}
BASE_SPEED = {"lion": 7.0, "elephant": 3.5, "deer": 10.0}
RUN_SPEED = {"lion": 12.0, "elephant": 8.5, "deer": 14.0}
MIN_SPEED = 2.0
ANIMAL_COLOR = {
    "lion": (0.9, 0.45, 0.1),
    "elephant": (0.45, 0.45, 0.5),
    "deer": (0.55, 0.35, 0.2),
}

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

# Winding River path
river_pts = [
    np.array([-300.0, 0.1, -120.0]),
    np.array([-120.0, 0.1, -20.0]),
    np.array([20.0, 0.1, -80.0]),
    np.array([150.0, 0.1, 80.0]),
    np.array([300.0, 0.1, 150.0]),
]

# Global simulation state
state = {
    "selected_animal": None,
    "screen_size": (1024.0, 768.0),
    "animals": [],
    "next_animal_id": 0,
    "minimap_dots_actor": None,
    "minimap_selected_actor": None,
    # Camera states
    "keys": set(),
    "cam_yaw": 0.0,
    "cam_pitch": -np.radians(45.0),
    "is_dragging_cam": False,
    "last_mouse": None,
    # Params
    "cohesion_mult": 1.0,
    "separation_mult": 1.0,
}

# Setup Scene
scene = window.Scene()
scene.background = (0.04, 0.08, 0.04)

# Ground
ground = actor.box(
    centers=np.array([[0.0, -0.5, 0.0]]),
    colors=(0.06, 0.12, 0.06),
    scales=(JUNGLE_SIZE, 1.0, JUNGLE_SIZE),
)
scene.add(ground)

# Lake
lake = actor.cylinder(
    centers=np.array([[LAKE_CENTER[0], 0.02, LAKE_CENTER[2]]]),
    directions=np.array([[0.0, 1.0, 0.0]]),
    colors=(0.1, 0.4, 0.75),
    height=0.08,
    radii=LAKE_RADIUS,
)
scene.add(lake)

# Winding River
river = actor.line([np.array(river_pts)], colors=(0.12, 0.45, 0.8), material="basic")
scene.add(river)

# Vegetation patches (represented by small green cones)
for vp in veg_patches:
    for _ in range(12):
        offset_x = np.random.uniform(-15.0, 15.0)
        offset_z = np.random.uniform(-15.0, 15.0)
        c_pos = vp + np.array([offset_x, 0.5, offset_z])
        sh = np.random.uniform(1.0, 2.0)
        bush = actor.cone(
            centers=np.array([c_pos]),
            directions=np.array([[0.0, 1.0, 0.0]]),
            colors=(0.15, 0.5, 0.2),
            height=sh,
            radii=sh * 0.4,
        )
        scene.add(bush)

# Lights
ambient = gfx.AmbientLight(color=(0.55, 0.6, 0.55), intensity=1.8)
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
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (q_w * uv + uuv)


# Animal Spawn Helper
def spawn_animal(species, position, is_child=False):
    color = ANIMAL_COLOR[species]
    if species == "lion":
        art = actor.ellipsoid(
            centers=np.array([[0.0, 0.6, 0.0]]),
            lengths=(2.0, 1.0, 1.0),
            colors=color,
        )
    elif species == "elephant":
        art = actor.box(
            centers=np.array([[0.0, 1.0, 0.0]]), colors=color, scales=(3.2, 1.8, 1.6)
        )
    else:
        art = actor.cone(
            centers=np.array([[0.0, 0.5, 0.0]]),
            directions=np.array([[0.0, 0.0, 1.0]]),
            colors=color,
            height=1.5,
            radii=0.45,
        )

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
    }
    animal["vel"][1] = 0.0
    speed = BASE_SPEED[species]
    animal["vel"] = (animal["vel"] / (np.linalg.norm(animal["vel"]) + 1e-5)) * speed

    if is_child:
        art.local.scale = [0.4, 0.4, 0.4]

    art.agent_idx = animal["id"]
    state["animals"].append(animal)
    state["next_animal_id"] += 1
    return animal


# Initial Population Spawning
for _ in range(55):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("deer", np.array([rx, 0.5, rz]))

for _ in range(12):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("lion", np.array([rx, 0.6, rz]))

for _ in range(10):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("elephant", np.array([rx, 1.0, rz]))


# Project 3D to Screen Coordinates
def world_to_screen(world_pos, camera, screen_size):
    p4 = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    v = camera.view_matrix
    p = camera.projection_matrix
    clip = p @ (v @ p4)
    if clip[3] == 0:
        return np.array([0.0, 0.0])
    ndc = clip[:3] / clip[3]
    w, h = screen_size
    screen_x = (ndc[0] + 1.0) * 0.5 * w
    screen_y = (1.0 - ndc[1]) * 0.5 * h
    return np.array([screen_x, screen_y])


# UI Control Panel setup (Left side) - Height expanded to 540 to fit all controls
panel = ui.Panel2D(
    size=(320, 540), color=(0.08, 0.12, 0.08), has_border=True, border_width=2
)
panel.set_position((15, 15))

lbl_title = ui.TextBlock2D(
    text="JUNGLE SIMULATOR",
    position=(20, 20),
    font_size=14,
    color=(0.9, 0.75, 0.2),
    bold=True,
    dynamic_bbox=True,
)
panel.add_element(lbl_title, (20, 20))

lbl_legend = ui.TextBlock2D(
    text="Lions: 0 | Elephants: 0 | Deers: 0",
    position=(20, 50),
    font_size=11,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
panel.add_element(lbl_legend, (20, 50))

# Card for selected animal details
lbl_card_title = ui.TextBlock2D(
    text="SELECTED ANIMAL STATUS",
    position=(20, 80),
    font_size=11,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
panel.add_element(lbl_card_title, (20, 80))

lbl_animal_info = ui.TextBlock2D(
    text="Click an animal to monitor it.",
    position=(20, 105),
    font_size=10,
    color=(0.8, 0.8, 0.8),
    dynamic_bbox=True,
)
panel.add_element(lbl_animal_info, (20, 105))

# Dynamic stats editing sliders
slider_hunger = ui.LineSlider2D(
    position=(20, 175),
    initial_value=0.0,
    min_value=0.0,
    max_value=100.0,
    length=180,
    text_template="Edit Hunger: {value:.0f}%",
)
panel.add_element(slider_hunger, (20, 175))

slider_thirst = ui.LineSlider2D(
    position=(20, 220),
    initial_value=0.0,
    min_value=0.0,
    max_value=100.0,
    length=180,
    text_template="Edit Thirst: {value:.0f}%",
)
panel.add_element(slider_thirst, (20, 220))

slider_health = ui.LineSlider2D(
    position=(20, 265),
    initial_value=100.0,
    min_value=0.0,
    max_value=100.0,
    length=180,
    text_template="Edit Health: {value:.0f}%",
)
panel.add_element(slider_health, (20, 265))

slider_age = ui.LineSlider2D(
    position=(20, 310),
    initial_value=0.0,
    min_value=0.0,
    max_value=140.0,
    length=180,
    text_template="Edit Age: {value:.1f}",
)
panel.add_element(slider_age, (20, 310))


def on_hunger_slide(slider):
    sel = state["selected_animal"]
    if sel is not None:
        for a in state["animals"]:
            if a["id"] == sel:
                a["hunger"] = slider.value
                break


def on_thirst_slide(slider):
    sel = state["selected_animal"]
    if sel is not None:
        for a in state["animals"]:
            if a["id"] == sel:
                a["thirst"] = slider.value
                break


def on_health_slide(slider):
    sel = state["selected_animal"]
    if sel is not None:
        for a in state["animals"]:
            if a["id"] == sel:
                # Map percentage to species max health
                mx = MAX_HEALTH[a["species"]]
                a["health"] = (slider.value / 100.0) * mx
                break


def on_age_slide(slider):
    sel = state["selected_animal"]
    if sel is not None:
        for a in state["animals"]:
            if a["id"] == sel:
                a["age"] = slider.value
                break


slider_hunger.on_change = on_hunger_slide
slider_thirst.on_change = on_thirst_slide
slider_health.on_change = on_health_slide
slider_age.on_change = on_age_slide

# Animal highlight focus buttons
btn_states_lion = {
    "hover": {"text": "LION", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "LION", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "LION", "color": (0.15, 0.25, 0.15)},
}
btn_lion = ui.TextButton2D(
    label="LION", size=(80, 25), position=(20, 370), states=btn_states_lion
)

btn_states_ele = {
    "hover": {"text": "ELEPHANT", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "ELEPHANT", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "ELEPHANT", "color": (0.15, 0.25, 0.15)},
}
btn_ele = ui.TextButton2D(
    label="ELEPHANT", size=(90, 25), position=(110, 370), states=btn_states_ele
)

btn_states_deer = {
    "hover": {"text": "DEER", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "DEER", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "DEER", "color": (0.15, 0.25, 0.15)},
}
btn_deer = ui.TextButton2D(
    label="DEER", size=(80, 25), position=(210, 370), states=btn_states_deer
)


def select_closest_animal_by_species(species):
    # Find one of this species to select
    candidates = [a for a in state["animals"] if a["species"] == species]
    if len(candidates) > 0:
        target = candidates[0]
        # Highlight old selected
        if state["selected_animal"] is not None:
            for a in state["animals"]:
                if a["id"] == state["selected_animal"]:
                    a["actor"].color = ANIMAL_COLOR[a["species"]]
                    break
        state["selected_animal"] = target["id"]
        target["actor"].color = (1.0, 0.2, 0.1)


btn_lion.on_clicked = lambda event: select_closest_animal_by_species("lion")
btn_ele.on_clicked = lambda event: select_closest_animal_by_species("elephant")
btn_deer.on_clicked = lambda event: select_closest_animal_by_species("deer")

panel.add_element(btn_lion, (20, 370))
panel.add_element(btn_ele, (110, 370))
panel.add_element(btn_deer, (210, 370))

# Global cohesion and separation sliders
slider_global_coh = ui.LineSlider2D(
    position=(20, 430),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Cohesion Mult: {value:.1f}",
)
panel.add_element(slider_global_coh, (20, 430))

slider_global_sep = ui.LineSlider2D(
    position=(20, 485),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Separation Mult: {value:.1f}",
)
panel.add_element(slider_global_sep, (20, 485))


def on_global_coh_change(slider):
    state["cohesion_mult"] = slider.value


def on_global_sep_change(slider):
    state["separation_mult"] = slider.value


slider_global_coh.on_change = on_global_coh_change
slider_global_sep.on_change = on_global_sep_change

scene.add(panel)

# Minimap Configuration (Top-Right)
minimap_panel = ui.Panel2D(
    size=(180, 180), color=(0.04, 0.07, 0.04), has_border=True, border_width=2
)
scene.add(minimap_panel)

minimap_group = gfx.Group()
scene.ui_scene.add(minimap_group)

# Local coordinate map scaling helper
map_w, map_h = 180.0, 180.0


def get_map_coords(wx, wz):
    mx = ((wx - (-300.0)) / 600.0) * map_w
    my = ((wz - (-300.0)) / 600.0) * map_h
    return mx, my


# Draw River on minimap
map_river_pts = []
for rp in river_pts:
    rx, ry = get_map_coords(rp[0], rp[2])
    map_river_pts.append([rx, ry, 0.0])
map_river_actor = actor.line(
    [np.array(map_river_pts)], colors=(0.2, 0.5, 1.0), material="basic"
)
minimap_group.add(map_river_actor)

# Draw Lake on minimap
lx, ly = get_map_coords(LAKE_CENTER[0], LAKE_CENTER[2])
map_lake_circ = actor.sphere(
    centers=np.array([[lx, ly, 0.0]]), colors=(0.15, 0.45, 0.8), radii=16.0
)
minimap_group.add(map_lake_circ)

# Draw static vegetation patches on minimap
for vp in veg_patches:
    vx, vy = get_map_coords(vp[0], vp[2])
    map_veg = actor.sphere(
        centers=np.array([[vx, vy, 0.0]]), colors=(0.15, 0.5, 0.2), radii=4.0
    )
    minimap_group.add(map_veg)


# Click selection logic
def on_click(event):
    target = event.target
    if hasattr(target, "agent_idx"):
        # Highlight new selected
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
def on_pointer_down(event):
    if event.button == 1:
        if not hasattr(event.target, "agent_idx") and event.target is not panel:
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
    global pos, vel
    dt = 0.016

    animals = state["animals"]
    state["screen_size"] = showm.renderer.logical_size

    # Position minimap dynamically relative to window size
    mx_pos = state["screen_size"][0] - 195.0
    my_pos = 15.0
    minimap_panel.set_position((mx_pos, my_pos))
    minimap_group.local.position = (mx_pos, my_pos, 0.0)

    # 1. Animal Needs & Aging Update loop
    for a in animals:
        a["age"] += dt * 0.08
        a["hunger"] += dt * 1.6
        a["thirst"] += dt * 2.2
        if a["cooldown"] > 0:
            a["cooldown"] -= dt

        # Growth scaling for children
        if a["is_child"]:
            scale = 0.4 + min(0.6, a["age"] * 0.05)
            a["actor"].local.scale = [scale, scale, scale]
            if a["age"] > 12.0:
                a["is_child"] = False

        # Apply starvation/dehydration damage
        if a["hunger"] >= 100.0 or a["thirst"] >= 100.0:
            a["health"] -= dt * 15.0
        else:
            a["health"] = min(MAX_HEALTH[a["species"]], a["health"] + dt * 1.5)

    # Filter dead animals
    dead_list = []
    alive_list = []
    num_lions = 0
    num_elephants = 0
    num_deers = 0

    for a in animals:
        limit_age = MAX_AGE[a["species"]]
        if a["health"] <= 0.0 or a["age"] >= limit_age:
            dead_list.append(a)
        else:
            alive_list.append(a)
            if a["species"] == "lion":
                num_lions += 1
            elif a["species"] == "elephant":
                num_elephants += 1
            else:
                num_deers += 1

    # Remove dead actors
    for a in dead_list:
        scene.remove(a["actor"])
        if state["selected_animal"] == a["id"]:
            state["selected_animal"] = None

    state["animals"] = alive_list
    animals = state["animals"]

    # Limit population explosion
    if len(animals) > 200:
        for a in animals:
            a["cooldown"] = 12.0

    # 2. Ecosystem Behaviors, Steer Forces & Movement
    new_spawns = []

    for a in animals:
        # Base movement speed
        current_max_speed = BASE_SPEED[a["species"]]
        if a["is_child"]:
            current_max_speed *= 0.6

        # Determine priorities
        is_thirsty = a["thirst"] > 45.0
        is_hungry = a["hunger"] > 40.0

        # Species Grouping / Flocking dynamics
        same_species = [
            other
            for other in animals
            if other["species"] == a["species"] and other["id"] != a["id"]
        ]

        flock_cohesion = np.zeros(3)
        flock_alignment = np.zeros(3)
        flock_separation = np.zeros(3)

        if len(same_species) > 0:
            positions = np.array([other["pos"] for other in same_species])
            velocities = np.array([other["vel"] for other in same_species])
            diffs = a["pos"] - positions
            dists_sq = np.sum(diffs**2, axis=-1)

            # Neighbor masks
            neigh_mask = dists_sq < 60.0**2
            sep_mask = dists_sq < 12.0**2

            n_neighbors = np.sum(neigh_mask)
            n_sep = np.sum(sep_mask)

            if n_neighbors > 0:
                avg_pos = np.mean(positions[neigh_mask], axis=0)
                flock_cohesion = avg_pos - a["pos"]
                flock_alignment = np.mean(velocities[neigh_mask], axis=0) - a["vel"]

            if n_sep > 0:
                flock_separation = np.sum(
                    diffs[sep_mask] / (dists_sq[sep_mask][:, None] + 1e-5), axis=0
                )

        coeffs = GROUP_COEFFS[a["species"]]
        steer = (
            flock_cohesion * coeffs["cohesion"] * 0.1 * state["cohesion_mult"]
            + flock_alignment * coeffs["alignment"] * 0.1
            + flock_separation * coeffs["separation"] * 0.3 * state["separation_mult"]
        )

        # Lake Seeking (Thirst)
        if is_thirsty:
            to_lake = LAKE_CENTER - a["pos"]
            d_lake = np.linalg.norm(to_lake)
            if d_lake > LAKE_RADIUS - 10.0:
                steer += (to_lake / (d_lake + 1e-5)) * 1.5
            else:
                a["thirst"] = max(0.0, a["thirst"] - dt * 25.0)

        # Grazing (Deers feeding on vegetation)
        if a["species"] == "deer" and is_hungry:
            best_patch = veg_patches[0]
            min_d = np.linalg.norm(a["pos"] - best_patch)
            for vp in veg_patches[1:]:
                dist = np.linalg.norm(a["pos"] - vp)
                if dist < min_d:
                    min_d = dist
                    best_patch = vp

            to_patch = best_patch - a["pos"]
            to_patch[1] = 0.0
            steer += (to_patch / (min_d + 1e-5)) * 1.2
            if min_d < 25.0:
                a["hunger"] = max(0.0, a["hunger"] - dt * 35.0)

        # Hunting state & pack coordination (Lions targeting Deers)
        if a["species"] == "lion" and is_hungry:
            target_deer = None
            min_d = 9999.0

            pack_target_deer = None
            pack_members = [
                other
                for other in same_species
                if np.linalg.norm(other["pos"] - a["pos"]) < 40.0
                and other["hunt_target_id"] is not None
            ]

            if len(pack_members) > 0:
                shared_id = pack_members[0]["hunt_target_id"]
                for target in animals:
                    if target["id"] == shared_id and target["species"] == "deer":
                        pack_target_deer = target
                        min_d = np.linalg.norm(a["pos"] - target["pos"])
                        break

            if pack_target_deer is not None:
                target_deer = pack_target_deer
            else:
                for target in animals:
                    if target["species"] == "deer":
                        dist = np.linalg.norm(a["pos"] - target["pos"])
                        if dist < min_d:
                            min_d = dist
                            target_deer = target

            if target_deer is not None:
                a["hunt_target_id"] = target_deer["id"]
                current_max_speed = RUN_SPEED["lion"]
                to_prey = target_deer["pos"] - a["pos"]
                to_prey[1] = 0.0
                steer += (to_prey / (min_d + 1e-5)) * 2.2

                # Attack/Deal damage over time instead of instant death
                if min_d < 3.5:
                    target_deer["health"] -= dt * 45.0
                    if target_deer["health"] <= 0.0:
                        a["hunger"] = 0.0
                        a["hunt_target_id"] = None
            else:
                a["hunt_target_id"] = None
        else:
            a["hunt_target_id"] = None

        # Predator Avoidance (Deers running from Lions)
        if a["species"] == "deer":
            closest_lion = None
            min_d = 9999.0
            for target in animals:
                if target["species"] == "lion":
                    dist = np.linalg.norm(target["pos"] - a["pos"])
                    if dist < min_d:
                        min_d = dist
                        closest_lion = target

            if closest_lion is not None and min_d < 45.0:
                current_max_speed = RUN_SPEED["deer"]
                to_pred = closest_lion["pos"] - a["pos"]
                to_pred[1] = 0.0
                steer -= (to_pred / (min_d + 1e-5)) * 3.8

        # Elephant Family Defense & Retaliation against Lions
        if a["species"] == "elephant":
            is_rage = False
            target_lion = None
            if not a["is_child"]:
                for calf in animals:
                    if calf["species"] == "elephant" and calf["is_child"]:
                        for lion in animals:
                            if lion["species"] == "lion":
                                d_lion_calf = np.linalg.norm(lion["pos"] - calf["pos"])
                                if d_lion_calf < 25.0:
                                    is_rage = True
                                    target_lion = lion
                                    break
                        if is_rage:
                            break

            if is_rage and target_lion is not None:
                current_max_speed = RUN_SPEED["elephant"]
                to_lion = target_lion["pos"] - a["pos"]
                to_lion[1] = 0.0
                dist = np.linalg.norm(to_lion)
                steer += (to_lion / (dist + 1e-5)) * 4.2
                if dist < 4.5:
                    target_lion["health"] -= dt * 80.0

        # Breeding Logic
        if a["cooldown"] <= 0.0 and not a["is_child"]:
            for partner in animals:
                if (
                    partner["species"] == a["species"]
                    and partner["id"] != a["id"]
                    and partner["gender"] != a["gender"]
                    and partner["cooldown"] <= 0.0
                    and not partner["is_child"]
                ):
                    dist = np.linalg.norm(a["pos"] - partner["pos"])
                    if dist < 12.0:
                        a["cooldown"] = 35.0
                        partner["cooldown"] = 35.0
                        mid_pos = (a["pos"] + partner["pos"]) * 0.5
                        mid_pos[1] = 0.6 if a["species"] == "lion" else 0.5
                        if a["species"] == "elephant":
                            mid_pos[1] = 1.0
                        new_spawns.append((a["species"], mid_pos))
                        break

        # Map Boundaries Constraint
        dist_from_origin = np.linalg.norm(a["pos"])
        if dist_from_origin > 280.0:
            steer -= (a["pos"] / (dist_from_origin + 1e-5)) * 3.5

        # Update velocity & limit speed
        a["vel"] += steer * dt * 22.0
        speed = np.linalg.norm(a["vel"])
        if speed > current_max_speed:
            a["vel"] = (a["vel"] / speed) * current_max_speed
        elif speed < MIN_SPEED:
            a["vel"] = (a["vel"] / (speed + 1e-5)) * MIN_SPEED

        # Position update
        a["pos"] += a["vel"] * dt

        # Hard Boundary constraint: keep inside boundary radius of 290
        d_origin = np.linalg.norm(a["pos"])
        if d_origin > 290.0:
            a["pos"] = (a["pos"] / (d_origin + 1e-5)) * 290.0
            a["vel"] = -(a["pos"] / 290.0) * np.linalg.norm(a["vel"])

        a["pos"][1] = 0.5
        if a["species"] == "lion":
            a["pos"][1] = 0.6
        elif a["species"] == "elephant":
            a["pos"][1] = 1.0

        # Update actor transforms
        a["actor"].local.position = a["pos"]
        h = a["vel"] / (np.linalg.norm(a["vel"]) + 1e-5)
        angle = np.arctan2(h[0], h[2])
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        a["actor"].local.rotation = [0.0, s, 0.0, c]

    # Spawn new babies
    for species, position in new_spawns:
        spawn_animal(species, position, is_child=True)

    # 3. UI Status updates
    lbl_legend.message = (
        f"Lions: {num_lions} | Elephants: {num_elephants} | Deers: {num_deers}"
    )

    selected = state["selected_animal"]
    camera = showm.screens[0].camera

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

            # Display stats on details card
            lbl_animal_info.message = (
                f"Species: {selected_a['species'].upper()}\n"
                f"Class: {type_txt} ({gender_txt})\n"
                f"Health: {selected_a['health']:.1f} / "
                f"{MAX_HEALTH[selected_a['species']]}\n"
                f"Hunger: {selected_a['hunger']:.1f}%\n"
                f"Thirst: {selected_a['thirst']:.1f}%"
            )

            # Update sliders value
            slider_hunger.value = selected_a["hunger"]
            slider_thirst.value = selected_a["thirst"]
            slider_health.value = (
                selected_a["health"] / MAX_HEALTH[selected_a["species"]]
            ) * 100.0
            slider_age.value = selected_a["age"]

            # Chase camera positioning
            b_pos = selected_a["pos"]
            b_vel = selected_a["vel"]
            spd = np.linalg.norm(b_vel)
            h = b_vel / (spd + 1e-5)

            target_cam_pos = b_pos - h * 25.0 + np.array([0.0, 12.0, 0.0])
            camera.local.position = (
                camera.local.position + (target_cam_pos - camera.local.position) * 0.1
            )
            camera.look_at(b_pos + h * 4.0)

            # Update camera look rotation states to match
            diff = b_pos - camera.local.position
            state["cam_yaw"] = np.arctan2(diff[0], diff[2])
            state["cam_pitch"] = np.arcsin(
                np.clip(diff[1] / (np.linalg.norm(diff) + 1e-5), -0.99, 0.99)
            )
        else:
            state["selected_animal"] = None
            lbl_animal_info.message = "Click an animal to monitor it."
    else:
        lbl_animal_info.message = "Click an animal to monitor it."

        # Counterstrike style WASD fly camera mode
        qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
        qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
        camera.local.rotation = quat_mult(qy, qx)

        fwd = rotate_vector(camera.local.rotation, np.array([0.0, 0.0, -1.0]))
        right = rotate_vector(camera.local.rotation, np.array([1.0, 0.0, 0.0]))

        fly_speed = 70.0
        keys = state["keys"]
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
            camera.local.position = camera.local.position + move_dir * fly_speed * dt

    # 5. Minimap Dynamic Dots rendering
    if state["minimap_dots_actor"] is not None:
        minimap_group.remove(state["minimap_dots_actor"])

    if len(animals) > 0:
        bx = np.zeros(len(animals))
        by = np.zeros(len(animals))
        colors = []

        for idx, a in enumerate(animals):
            mx, my = get_map_coords(a["pos"][0], a["pos"][2])
            bx[idx] = mx
            by[idx] = my
            colors.append(ANIMAL_COLOR[a["species"]])

        m_centers = np.stack([bx, by, np.zeros(len(animals))], axis=-1)
        state["minimap_dots_actor"] = actor.sphere(
            centers=m_centers, colors=colors, radii=1.2
        )
        minimap_group.add(state["minimap_dots_actor"])

    # Update selected marker on minimap
    if state["minimap_selected_actor"] is not None:
        minimap_group.remove(state["minimap_selected_actor"])
        state["minimap_selected_actor"] = None

    if selected is not None:
        selected_a = None
        for a in animals:
            if a["id"] == selected:
                selected_a = a
                break
        if selected_a is not None:
            sbx, sby = get_map_coords(selected_a["pos"][0], selected_a["pos"][2])
            state["minimap_selected_actor"] = actor.sphere(
                centers=np.array([[sbx, sby, 0.0]]),
                colors=(1.0, 0.1, 0.1),
                radii=2.8,
            )
            minimap_group.add(state["minimap_selected_actor"])

    showm.render()


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

    # Initial camera placement
    camera = show_manager.screens[0].camera
    camera.local.position = (0.0, 200.0, -250.0)

    # Set initial camera look rotation
    state["cam_yaw"] = 0.0
    state["cam_pitch"] = -np.radians(40.0)
    qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
    qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
    camera.local.rotation = quat_mult(qy, qx)

    # Start simulation
    show_manager.register_callback(sim_tick, 0.016, True, "JungleLoop", show_manager)
    show_manager.start()
