"""
=============================================
Interactive 3D Jungle Survival Flocking Simulation
=============================================

An interactive 3D jungle ecosystem simulation containing Lions, Elephants,
and Deers navigating land, lakes, rivers, hills, and vegetation patches.
Features survival states (thirst, hunger, aging, mating, hunting, and death),
a dynamic minimap, and diagnostic control interfaces.
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
MAX_SPEED = {"lion": 9.0, "elephant": 5.0, "deer": 11.0}
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

# Hills
hills = [
    np.array([-200.0, 0.0, 50.0]),
    np.array([200.0, 0.0, -50.0]),
    np.array([-60.0, 0.0, 200.0]),
    np.array([80.0, 0.0, -180.0]),
]
for h in hills:
    hill_scale = np.random.uniform(18.0, 30.0, size=3)
    hill_scale[1] *= 0.4
    mound = actor.ellipsoid(
        centers=np.array([[h[0], hill_scale[1] / 2.0, h[2]]]),
        lengths=tuple(hill_scale),
        colors=(0.2, 0.18, 0.15),
    )
    scene.add(mound)

# Lights
ambient = gfx.AmbientLight(color=(0.55, 0.6, 0.55), intensity=1.8)
scene.add(ambient)


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


# UI Control Panel setup (Left side)
panel = ui.Panel2D(
    size=(310, 360), color=(0.08, 0.12, 0.08), has_border=True, border_width=2
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
    position=(20, 55),
    font_size=11,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
panel.add_element(lbl_legend, (20, 55))

# Card for selected animal details
lbl_card_title = ui.TextBlock2D(
    text="SELECTED ANIMAL STATUS",
    position=(20, 90),
    font_size=11,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
panel.add_element(lbl_card_title, (20, 90))

lbl_animal_info = ui.TextBlock2D(
    text="Click an animal to monitor it.",
    position=(20, 115),
    font_size=10,
    color=(0.8, 0.8, 0.8),
    dynamic_bbox=True,
)
panel.add_element(lbl_animal_info, (20, 115))

# Sliders to manipulate selected animal
slider_hunger = ui.LineSlider2D(
    position=(20, 210),
    initial_value=0.0,
    min_value=0.0,
    max_value=100.0,
    length=180,
    text_template="Edit Hunger: {value:.0f}%",
)
panel.add_element(slider_hunger, (20, 210))

slider_thirst = ui.LineSlider2D(
    position=(20, 270),
    initial_value=0.0,
    min_value=0.0,
    max_value=100.0,
    length=180,
    text_template="Edit Thirst: {value:.0f}%",
)
panel.add_element(slider_thirst, (20, 270))


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


slider_hunger.on_change = on_hunger_slide
slider_thirst.on_change = on_thirst_slide

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
        state["selected_animal"] = target.agent_idx
        print(f"Selected Animal ID #{target.agent_idx}")
    elif target is ground or target is lake:
        state["selected_animal"] = None
        slider_hunger.value = 0.0
        slider_thirst.value = 0.0


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
        # Compute same-species neighbors to calculate cohesion, alignment, separation
        same_species = [
            other
            for other in animals
            if other["species"] == a["species"] and other["id"] != a["id"]
        ]

        flock_cohesion = np.zeros(3)
        flock_alignment = np.zeros(3)
        flock_separation = np.zeros(3)

        if len(same_species) > 0:
            # Calculate centers and velocities of same-species neighbors
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
            flock_cohesion * coeffs["cohesion"] * 0.1
            + flock_alignment * coeffs["alignment"] * 0.1
            + flock_separation * coeffs["separation"] * 0.3
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
            # Hunt individually or coordinate targets with nearby pack members
            target_deer = None
            min_d = 9999.0

            # Scan for nearby pack members that already have a target
            pack_target_deer = None
            pack_members = [
                other
                for other in same_species
                if np.linalg.norm(other["pos"] - a["pos"]) < 40.0
                and other["hunt_target_id"] is not None
            ]

            if len(pack_members) > 0:
                # Share the target (pack hunting of 2-5 lions)
                shared_id = pack_members[0]["hunt_target_id"]
                for target in animals:
                    if target["id"] == shared_id and target["species"] == "deer":
                        pack_target_deer = target
                        min_d = np.linalg.norm(a["pos"] - target["pos"])
                        break

            if pack_target_deer is not None:
                target_deer = pack_target_deer
            else:
                # Find closest Deer
                for target in animals:
                    if target["species"] == "deer":
                        dist = np.linalg.norm(a["pos"] - target["pos"])
                        if dist < min_d:
                            min_d = dist
                            target_deer = target

            if target_deer is not None:
                a["hunt_target_id"] = target_deer["id"]
                current_max_speed = RUN_SPEED["lion"]  # Run speed in chase
                to_prey = target_deer["pos"] - a["pos"]
                to_prey[1] = 0.0
                steer += (to_prey / (min_d + 1e-5)) * 2.2

                # Attack/Deal damage over time instead of instant death
                if min_d < 3.5:
                    target_deer["health"] -= dt * 45.0  # Deal bite damage
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
                current_max_speed = RUN_SPEED["deer"]  # Sprint in fear
                to_pred = closest_lion["pos"] - a["pos"]
                to_pred[1] = 0.0
                steer -= (to_pred / (min_d + 1e-5)) * 3.8

        # Elephant Family Defense & Retaliation against Lions
        if a["species"] == "elephant":
            is_rage = False
            target_lion = None
            if not a["is_child"]:
                # Defend nearby calves
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
                current_max_speed = RUN_SPEED["elephant"]  # Charge speed!
                to_lion = target_lion["pos"] - a["pos"]
                to_lion[1] = 0.0
                dist = np.linalg.norm(to_lion)
                steer += (to_lion / (dist + 1e-5)) * 4.2
                # Trample damage
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
        # Keep ground level Y height stable
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

    selected = state["selected_agent"] = state["selected_animal"]
    camera = showm.screens[0].camera

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

            lbl_animal_info.message = (
                f"Species: {selected_a['species'].upper()}\n"
                f"Class: {type_txt} ({gender_txt})\n"
                f"Health: {selected_a['health']:.1f}%\n"
                f"Hunger: {selected_a['hunger']:.1f}%\n"
                f"Thirst: {selected_a['thirst']:.1f}%"
            )

            # Match sliders to dynamic value
            slider_hunger.value = selected_a["hunger"]
            slider_thirst.value = selected_a["thirst"]

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
        else:
            state["selected_animal"] = None
            lbl_animal_info.message = "Click an animal to monitor it."
    else:
        lbl_animal_info.message = "Click an animal to monitor it."
        # Global overview camera (Framing the whole map)
        target_cam_pos = np.array([0.0, 240.0, -320.0])
        camera.local.position = (
            camera.local.position + (target_cam_pos - camera.local.position) * 0.04
        )
        camera.look_at(np.array([0.0, 0.0, -20.0]))

    # 4. Minimap Dynamic Dots rendering
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

    # Bind pointer clicks
    show_manager.renderer.add_event_handler(on_click, EventType.POINTER_DOWN)

    # Initial overview camera placement
    camera = show_manager.screens[0].camera
    camera.local.position = (0.0, 240.0, -320.0)
    camera.look_at((0.0, 0.0, -20.0))

    # Start simulation
    show_manager.register_callback(sim_tick, 0.016, True, "JungleLoop", show_manager)
    show_manager.start()
