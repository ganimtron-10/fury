"""
=============================================
Interactive 2D Jungle Survival Flocking Simulation
=============================================

An interactive 2D jungle ecosystem simulation containing Lions, Elephants,
and Deers navigating land, lakes, and vegetation patches using 2D UI elements.
"""

import numpy as np
from fury import ui, window
from fury.window import EventType

# Simulation dimensions and constants
JUNGLE_SIZE = 600.0
LAKE_RADIUS = 55.0
LAKE_CENTER = np.array([0.0, 0.0]) # 2D center

# Window size mapping
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SIM_OFFSET_X = WINDOW_WIDTH / 2
SIM_OFFSET_Y = WINDOW_HEIGHT / 2

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

# Vegetation patch centers in 2D
veg_patches = [
    np.array([-150.0, -150.0]),
    np.array([160.0, 160.0]),
    np.array([-180.0, 120.0]),
    np.array([180.0, -150.0]),
    np.array([0.0, -220.0]),
]

# Global simulation state
state = {
    "selected_animal": None,
    "animals": [],
    "next_animal_id": 0,
    "screen_size": (1200.0, 800.0),
    "sim_speed": 1.0,
    "cohesion_mult": 1.0,
    "separation_mult": 1.0,
    "global_speed_factor": 1.0,
    "global_hunger_factor": 1.0,
    "global_thirst_factor": 1.0,
    "global_age_factor": 1.0,
    "global_mating_factor": 1.0,
    "global_hunting_factor": 1.0,
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
scene.background = (0.2, 0.3, 0.2)

def world_to_screen(pos):
    sx = state["screen_size"][0] / 2
    sy = state["screen_size"][1] / 2
    return (pos[0] + sx, pos[1] + sy)

# Lake surface
lake_screen = world_to_screen(LAKE_CENTER)
lake = ui.Disk2D(outer_radius=LAKE_RADIUS, center=lake_screen, color=(0.1, 0.4, 0.75))
lake.z_order = -3
scene.add(lake)
state["lake"] = lake

# Vegetation patches
bushes = []
for vp in veg_patches:
    vp_screen = world_to_screen(vp)
    bush = ui.Disk2D(outer_radius=25, center=vp_screen, color=(0.15, 0.5, 0.2))
    bush.z_order = -2
    scene.add(bush)
    bushes.append(bush)
state["bushes"] = bushes

# Animal Spawn Helper
def spawn_animal(species, position, is_child=False):
    color = ANIMAL_COLOR[species]
    size = 10 if species == "lion" else 15
    if species == "elephant":
        size = 12

    screen_pos = world_to_screen(position)

    out_color = (1.0, 1.0, 1.0)
    if species == "lion":
        art_out = ui.Disk2D(outer_radius=size+1.5, center=screen_pos, color=out_color)
        art = ui.Disk2D(outer_radius=size, center=screen_pos, color=color)
    elif species == "elephant":
        art_out = ui.Rectangle2D(size=(size*2+3, size*2+3), position=(screen_pos[0]-size-1.5, screen_pos[1]-size-1.5), color=out_color)
        art = ui.Rectangle2D(size=(size*2, size*2), position=(screen_pos[0]-size, screen_pos[1]-size), color=color)
    else:
        art_out = ui.Rectangle2D(size=(size*1.5+3, size*0.4+3), position=(screen_pos[0]-size*0.75-1.5, screen_pos[1]-size*0.2-1.5), color=out_color)
        art = ui.Rectangle2D(size=(size*1.5, size*0.4), position=(screen_pos[0]-size*0.75, screen_pos[1]-size*0.2), color=color)

    art_out.z_order = 0
    art.z_order = 1
    scene.add(art_out)
    scene.add(art)

    animal = {
        "id": state["next_animal_id"],
        "species": species,
        "pos": np.array(position, dtype=np.float32),
        "vel": np.random.randn(2).astype(np.float32),
        "actor": art,
        "actor_out": art_out,
        "age": np.random.uniform(16.0, 45.0) if not is_child else 0.0,
        "health": MAX_HEALTH[species],
        "hunger": np.random.uniform(10.0, 30.0),
        "thirst": np.random.uniform(10.0, 30.0),
        "is_child": is_child,
        "gender": np.random.choice(["M", "F"]),
        "cooldown": np.random.uniform(5.0, 15.0),
        "hunt_target_id": None,
    }
    speed = BASE_SPEED[species]
    animal["vel"] = (animal["vel"] / (np.linalg.norm(animal["vel"]) + 1e-5)) * speed

    if is_child:
        if species == "lion":
            art.outer_radius = size * 0.4
            art_out.outer_radius = size * 0.4 + 1.5
        elif species == "elephant":
            art.resize((size*2 * 0.4, size*2 * 0.4))
            art_out.resize((size*2 * 0.4 + 3, size*2 * 0.4 + 3))
        else:
            art.resize((size*1.5 * 0.4, size*0.4 * 0.4))
            art_out.resize((size*1.5 * 0.4 + 3, size*0.4 * 0.4 + 3))

    art.agent_idx = animal["id"]
    state["animals"].append(animal)
    state["next_animal_id"] += 1
    return animal

# Initial Population Spawning
for _ in range(55):
    rx = np.random.uniform(-380.0, 380.0)
    ry = np.random.uniform(-380.0, 380.0)
    spawn_animal("deer", np.array([rx, ry]))

for _ in range(12):
    rx = np.random.uniform(-380.0, 380.0)
    ry = np.random.uniform(-380.0, 380.0)
    spawn_animal("lion", np.array([rx, ry]))

for _ in range(10):
    rx = np.random.uniform(-380.0, 380.0)
    ry = np.random.uniform(-380.0, 380.0)
    spawn_animal("elephant", np.array([rx, ry]))

# UI Redesign - Info Panel
info_panel = ui.Panel2D(size=(360, 280), color=(0.06, 0.09, 0.06), has_border=True, border_width=2)
info_panel.set_position((15, 15))

lbl_title = ui.TextBlock2D(text="2D JUNGLE ECOSYSTEM", position=(20, 15), font_size=20, color=(0.9, 0.75, 0.2), bold=True, dynamic_bbox=True)
info_panel.add_element(lbl_title, (20, 15))

lbl_legend = ui.TextBlock2D(text="Lions: 0 | Elephants: 0 | Deers: 0", position=(20, 42), font_size=16, color=(0.85, 0.85, 0.85), dynamic_bbox=True)
info_panel.add_element(lbl_legend, (20, 42))

btn_states_lion = {"hover": {"text": "LION", "color": (0.3, 0.4, 0.3)}, "pressed": {"text": "LION", "color": (0.1, 0.2, 0.1)}, "default": {"text": "LION", "color": (0.15, 0.25, 0.15)}}
btn_lion = ui.TextButton2D(label="LION", size=(80, 26), position=(20, 68), font_size=16, states=btn_states_lion)

btn_states_ele = {"hover": {"text": "ELEPHANT", "color": (0.3, 0.4, 0.3)}, "pressed": {"text": "ELEPHANT", "color": (0.1, 0.2, 0.1)}, "default": {"text": "ELEPHANT", "color": (0.15, 0.25, 0.15)}}
btn_ele = ui.TextButton2D(label="ELEPHANT", size=(90, 26), position=(110, 68), font_size=16, states=btn_states_ele)

btn_states_deer = {"hover": {"text": "DEER", "color": (0.3, 0.4, 0.3)}, "pressed": {"text": "DEER", "color": (0.1, 0.2, 0.1)}, "default": {"text": "DEER", "color": (0.15, 0.25, 0.15)}}
btn_deer = ui.TextButton2D(label="DEER", size=(80, 26), position=(210, 68), font_size=16, states=btn_states_deer)

def select_closest_animal_by_species(species):
    candidates = [a for a in state["animals"] if a["species"] == species]
    if len(candidates) > 0:
        target = candidates[0]
        state["selected_animal"] = target["id"]

btn_lion.on_clicked = lambda event: select_closest_animal_by_species("lion")
btn_ele.on_clicked = lambda event: select_closest_animal_by_species("elephant")
btn_deer.on_clicked = lambda event: select_closest_animal_by_species("deer")

info_panel.add_element(btn_lion, (20, 68))
info_panel.add_element(btn_ele, (110, 68))
info_panel.add_element(btn_deer, (210, 68))

lbl_card_title = ui.TextBlock2D(text="SELECTED ANIMAL STATUS", position=(20, 108), font_size=16, color=(0.95, 0.8, 0.2), bold=True, dynamic_bbox=True)
info_panel.add_element(lbl_card_title, (20, 108))

lbl_animal_info_left = ui.TextBlock2D(text="Click an animal\nto monitor.", position=(20, 132), font_size=16, color=(0.85, 0.85, 0.85), dynamic_bbox=True)
info_panel.add_element(lbl_animal_info_left, (20, 132))

lbl_animal_info_right = ui.TextBlock2D(text="", position=(165, 132), font_size=16, color=(0.85, 0.85, 0.85), dynamic_bbox=True)
info_panel.add_element(lbl_animal_info_right, (165, 132))

scene.add(info_panel)

# Controls Panel
control_panel = ui.TabUI(position=(WINDOW_WIDTH - 360, 15), size=(340, 560), tab_titles=["Global", "Lion", "Elephant", "Deer"], startup_tab_id=0, font_size=16, active_color=(0.2, 0.6, 0.2), inactive_color=(0.04, 0.06, 0.04))

lbl_g_title = ui.TextBlock2D(text="GLOBAL TUNING FACTORS", position=(20, 15), font_size=16, color=(0.95, 0.8, 0.2), bold=True, dynamic_bbox=True)
control_panel.add_element(0, lbl_g_title, (20, 15))

slider_g_speed = ui.LineSlider2D(position=(20, 50), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Speed Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_speed, (20, 50))

slider_g_hunger = ui.LineSlider2D(position=(20, 100), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Hunger Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_hunger, (20, 100))

slider_g_thirst = ui.LineSlider2D(position=(20, 150), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Thirst Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_thirst, (20, 150))

slider_g_age = ui.LineSlider2D(position=(20, 200), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Age Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_age, (20, 200))

slider_g_mating = ui.LineSlider2D(position=(20, 250), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Mating Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_mating, (20, 250))

slider_g_hunting = ui.LineSlider2D(position=(20, 300), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Hunting Factor: {value:.1f}x")
control_panel.add_element(0, slider_g_hunting, (20, 300))

slider_global_coh = ui.LineSlider2D(position=(20, 350), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Cohesion Mult: {value:.1f}")
control_panel.add_element(0, slider_global_coh, (20, 350))

slider_global_sep = ui.LineSlider2D(position=(20, 400), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Separation Mult: {value:.1f}")
control_panel.add_element(0, slider_global_sep, (20, 400))

slider_global_speed = ui.LineSlider2D(position=(20, 450), initial_value=1.0, min_value=1.0, max_value=20.0, length=180, text_template="Sim Speed: {value:.1f}x")
control_panel.add_element(0, slider_global_speed, (20, 450))

btn_reset = ui.TextButton2D(label="RESET ALL FACTORS", size=(180, 25), position=(20, 500))
control_panel.add_element(0, btn_reset, (20, 500))

def on_g_speed(slider): state["global_speed_factor"] = slider.value
def on_g_hunger(slider): state["global_hunger_factor"] = slider.value
def on_g_thirst(slider): state["global_thirst_factor"] = slider.value
def on_g_age(slider): state["global_age_factor"] = slider.value
def on_g_mating(slider): state["global_mating_factor"] = slider.value
def on_g_hunting(slider): state["global_hunting_factor"] = slider.value
def on_global_coh_change(slider): state["cohesion_mult"] = slider.value
def on_global_sep_change(slider): state["separation_mult"] = slider.value
def on_global_speed_change(slider): state["sim_speed"] = slider.value

slider_g_speed.on_change = on_g_speed
slider_g_hunger.on_change = on_g_hunger
slider_g_thirst.on_change = on_g_thirst
slider_g_age.on_change = on_g_age
slider_g_mating.on_change = on_g_mating
slider_g_hunting.on_change = on_g_hunting
slider_global_coh.on_change = on_global_coh_change
slider_global_sep.on_change = on_global_sep_change
slider_global_speed.on_change = on_global_speed_change

species_sliders = []
def build_species_tab(tab_idx, species_name):
    lbl_title = ui.TextBlock2D(text=f"{species_name.upper()} FACTOR ADJUSTERS", position=(20, 15), font_size=16, color=(0.95, 0.8, 0.2), bold=True, dynamic_bbox=True)
    control_panel.add_element(tab_idx, lbl_title, (20, 15))
    s_speed = ui.LineSlider2D(position=(20, 50), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Speed Factor: {value:.1f}x")
    s_hunger = ui.LineSlider2D(position=(20, 100), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Hunger Factor: {value:.1f}x")
    s_thirst = ui.LineSlider2D(position=(20, 150), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Thirst Factor: {value:.1f}x")
    s_age = ui.LineSlider2D(position=(20, 200), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Age Factor: {value:.1f}x")
    s_mating = ui.LineSlider2D(position=(20, 250), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Mating Factor: {value:.1f}x")
    s_hunting = ui.LineSlider2D(position=(20, 300), initial_value=1.0, min_value=0.0, max_value=3.0, length=180, text_template="Hunting Factor: {value:.1f}x")
    
    control_panel.add_element(tab_idx, s_speed, (20, 50))
    control_panel.add_element(tab_idx, s_hunger, (20, 100))
    control_panel.add_element(tab_idx, s_thirst, (20, 150))
    control_panel.add_element(tab_idx, s_age, (20, 200))
    control_panel.add_element(tab_idx, s_mating, (20, 250))
    control_panel.add_element(tab_idx, s_hunting, (20, 300))
    
    s_speed.on_change = lambda sl: state.update({f"{species_name}_speed_factor": sl.value})
    s_hunger.on_change = lambda sl: state.update({f"{species_name}_hunger_factor": sl.value})
    s_thirst.on_change = lambda sl: state.update({f"{species_name}_thirst_factor": sl.value})
    s_age.on_change = lambda sl: state.update({f"{species_name}_age_factor": sl.value})
    s_mating.on_change = lambda sl: state.update({f"{species_name}_mating_factor": sl.value})
    s_hunting.on_change = lambda sl: state.update({f"{species_name}_hunting_factor": sl.value})
    
    species_sliders.extend([s_speed, s_hunger, s_thirst, s_age, s_mating, s_hunting])

build_species_tab(1, "lion")
build_species_tab(2, "elephant")
build_species_tab(3, "deer")

scene.add(control_panel)

def on_click(event):
    target = event.target
    if hasattr(target, "agent_idx"):
        state["selected_animal"] = target.agent_idx
    else:
        state["selected_animal"] = None

def sim_tick(showm):
    dt = 0.016 * state["sim_speed"]
    animals = state["animals"]
    
    # Handle fullscreen/resize dynamically
    state["screen_size"] = showm.renderer.logical_size
    state["lake"].set_position(world_to_screen(LAKE_CENTER))
    for i, bush in enumerate(state["bushes"]):
        bush.set_position(world_to_screen(veg_patches[i]))
    
    # Update TabUI position dynamically to stick to the right
    tx_pos = state["screen_size"][0] - 360.0
    control_panel.set_position((tx_pos, 15.0))

    
    # 1. Animal Needs & Aging
    for a in animals:
        d_lake = np.linalg.norm(a["pos"])
        in_water = d_lake <= LAKE_RADIUS

        sp = a["species"]
        a["age"] += dt * 0.008 * state["global_age_factor"] * state[f"{sp}_age_factor"]
        h_rate = 0.8 if sp == "lion" else 1.6
        t_rate = 1.2 if sp == "lion" else 2.2

        if in_water:
            a["thirst"] = max(0.0, a["thirst"] - dt * 50.0)
        else:
            a["thirst"] += dt * t_rate * state["global_thirst_factor"] * state[f"{sp}_thirst_factor"]

        if sp in ["deer", "elephant"]:
            if a["hunger"] > 40.0 and not in_water:
                a["hunger"] = max(0.0, a["hunger"] - dt * 25.0)
            else:
                a["hunger"] += dt * h_rate * state["global_hunger_factor"] * state[f"{sp}_hunger_factor"]
        else:
            a["hunger"] += dt * h_rate * state["global_hunger_factor"] * state[f"{sp}_hunger_factor"]

        if a["cooldown"] > 0.0:
            a["cooldown"] -= dt

        if a["is_child"]:
            scale = 0.4 + min(0.6, a["age"] * 0.05)
            # Update UI scale based on type
            size = 10 if sp == "lion" else 15
            if sp == "elephant": size = 12
            
            if sp == "lion":
                a["actor"].outer_radius = size * scale
                a["actor_out"].outer_radius = size * scale + 1.5
            elif sp == "elephant":
                a["actor"].resize((size*2 * scale, size*2 * scale))
                a["actor_out"].resize((size*2 * scale + 3, size*2 * scale + 3))
            else:
                a["actor"].resize((size*1.5 * scale, size*0.4 * scale))
                a["actor_out"].resize((size*1.5 * scale + 3, size*0.4 * scale + 3))

            if a["age"] > 12.0:
                a["is_child"] = False

        if a["hunger"] >= 100.0 or a["thirst"] >= 100.0:
            a["health"] -= dt * 4.0
        else:
            a["health"] = min(MAX_HEALTH[sp], a["health"] + dt * 1.5)

    dead_list = []
    alive_list = []
    num_lions, num_elephants, num_deers = 0, 0, 0

    for a in animals:
        limit_age = MAX_AGE[a["species"]]
        if a["health"] <= 0.0 or a["age"] >= limit_age:
            dead_list.append(a)
        else:
            alive_list.append(a)
            if a["species"] == "lion": num_lions += 1
            elif a["species"] == "elephant": num_elephants += 1
            else: num_deers += 1

    for a in dead_list:
        scene.remove(a["actor"])
        scene.remove(a["actor_out"])
        if state["selected_animal"] == a["id"]:
            state["selected_animal"] = None

    state["animals"] = alive_list
    animals = state["animals"]

    if len(animals) > 200:
        for a in animals:
            a["cooldown"] = 12.0

    # 2. Ecosystem Behaviors, Steer Forces & Movement
    new_spawns = []
    for a in animals:
        sp = a["species"]
        current_max_speed = BASE_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]
        run_speed_val = RUN_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]

        if a["is_child"]:
            current_max_speed *= 0.6
            run_speed_val *= 0.6

        is_thirsty = a["thirst"] > 45.0
        is_hungry = a["hunger"] > 40.0

        same_species = [other for other in animals if other["species"] == sp and other["id"] != a["id"]]

        flock_cohesion = np.zeros(2)
        flock_alignment = np.zeros(2)
        flock_separation = np.zeros(2)

        if len(same_species) > 0:
            positions = np.array([other["pos"] for other in same_species])
            velocities = np.array([other["vel"] for other in same_species])
            diffs = a["pos"] - positions
            dists_sq = np.sum(diffs**2, axis=-1)

            neigh_mask = dists_sq < 60.0**2
            sep_mask = dists_sq < 12.0**2

            if np.sum(neigh_mask) > 0:
                avg_pos = np.mean(positions[neigh_mask], axis=0)
                flock_cohesion = avg_pos - a["pos"]
                flock_alignment = np.mean(velocities[neigh_mask], axis=0) - a["vel"]

            if np.sum(sep_mask) > 0:
                flock_separation = np.sum(diffs[sep_mask] / (dists_sq[sep_mask][:, None] + 1e-5), axis=0)

        flock_weight = 1.0
        if is_thirsty or is_hungry:
            flock_weight = 0.1

        coeffs = GROUP_COEFFS[sp]
        steer = (flock_cohesion * coeffs["cohesion"] * 0.1 * state["cohesion_mult"]
                 + flock_alignment * coeffs["alignment"] * 0.1) * flock_weight + flock_separation * coeffs["separation"] * 0.3 * state["separation_mult"]

        if is_thirsty:
            to_lake = LAKE_CENTER - a["pos"]
            d_lake = np.linalg.norm(to_lake)
            if d_lake > LAKE_RADIUS:
                steer += (to_lake / (d_lake + 1e-5)) * 2.8

        if sp in ["deer", "elephant"] and is_hungry:
            best_patch = veg_patches[0]
            min_d = np.linalg.norm(a["pos"] - best_patch)
            for vp in veg_patches[1:]:
                dist = np.linalg.norm(a["pos"] - vp)
                if dist < min_d:
                    min_d = dist
                    best_patch = vp

            to_patch = best_patch - a["pos"]
            steer += (to_patch / (min_d + 1e-5)) * 2.2
            if min_d < 25.0:
                a["hunger"] = max(0.0, a["hunger"] - dt * 15.0)

        if sp == "lion" and is_hungry:
            target_deer = None
            min_d = 9999.0
            for target in animals:
                if target["species"] == "deer":
                    dist = np.linalg.norm(a["pos"] - target["pos"])
                    if dist < min_d:
                        min_d = dist
                        target_deer = target

            if target_deer is not None:
                a["hunt_target_id"] = target_deer["id"]
                current_max_speed = run_speed_val
                to_prey = target_deer["pos"] - a["pos"]
                steer += (to_prey / (min_d + 1e-5)) * 3.8

                if min_d < 3.5:
                    damage = dt * 45.0 * state["global_hunting_factor"] * state["lion_hunting_factor"]
                    target_deer["health"] -= damage
                    a["hunger"] = max(0.0, a["hunger"] - damage * 1.5)
                    if target_deer["health"] <= 0.0:
                        a["hunger"] = 0.0
                        a["hunt_target_id"] = None
            else:
                a["hunt_target_id"] = None
        else:
            a["hunt_target_id"] = None

        if sp == "deer":
            closest_lion = None
            min_d = 9999.0
            for target in animals:
                if target["species"] == "lion":
                    dist = np.linalg.norm(target["pos"] - a["pos"])
                    if dist < min_d:
                        min_d = dist
                        closest_lion = target

            if closest_lion is not None and min_d < 45.0:
                current_max_speed = run_speed_val
                to_pred = closest_lion["pos"] - a["pos"]
                steer -= (to_pred / (min_d + 1e-5)) * 3.8

        if sp == "lion":
            for target in animals:
                if target["species"] == "elephant":
                    dist = np.linalg.norm(target["pos"] - a["pos"])
                    if dist < 28.0:
                        steer -= (target["pos"] - a["pos"]) / (dist + 1e-5) * 2.5

        if sp == "elephant":
            is_rage = False
            target_lion = None
            if not a["is_child"]:
                for lion in animals:
                    if lion["species"] == "lion":
                        dist_lion = np.linalg.norm(lion["pos"] - a["pos"])
                        if dist_lion < 35.0:
                            is_rage = True
                            target_lion = lion
                            break

            if is_rage and target_lion is not None:
                current_max_speed = run_speed_val
                to_lion = target_lion["pos"] - a["pos"]
                dist = np.linalg.norm(to_lion)
                steer += (to_lion / (dist + 1e-5)) * 4.2
                if dist < 4.5:
                    target_lion["health"] -= dt * 90.0

        if a["cooldown"] <= 0.0 and not a["is_child"]:
            for partner in animals:
                if (partner["species"] == sp and partner["id"] != a["id"] and partner["gender"] != a["gender"]
                        and partner["cooldown"] <= 0.0 and not partner["is_child"]):
                    dist = np.linalg.norm(a["pos"] - partner["pos"])
                    if dist < 12.0:
                        cooldown_val = 35.0 / (state["global_mating_factor"] * state[f"{sp}_mating_factor"] + 1e-5)
                        a["cooldown"] = cooldown_val
                        partner["cooldown"] = cooldown_val
                        mid_pos = (a["pos"] + partner["pos"]) * 0.5
                        new_spawns.append((sp, mid_pos))
                        break

        dist_from_origin = np.linalg.norm(a["pos"])
        if dist_from_origin > 440.0:
            steer -= (a["pos"] / (dist_from_origin + 1e-5)) * 3.5

        a["vel"] += steer * dt * 22.0
        speed = np.linalg.norm(a["vel"])
        if speed > current_max_speed:
            a["vel"] = (a["vel"] / speed) * current_max_speed
        elif speed < MIN_SPEED:
            a["vel"] = (a["vel"] / (speed + 1e-5)) * MIN_SPEED

        a["pos"] += a["vel"] * dt

        d_origin = np.linalg.norm(a["pos"])
        if d_origin > 450.0:
            a["pos"] = (a["pos"] / (d_origin + 1e-5)) * 290.0
            a["vel"] = -(a["pos"] / 450.0) * np.linalg.norm(a["vel"])

        # Update actor UI position
        screen_pos = world_to_screen(a["pos"])
        
        # Center adjustment depending on shape
        size = 10 if sp == "lion" else 15
        if sp == "elephant": size = 12
        scale = 0.4 + min(0.6, a["age"] * 0.05) if a["is_child"] else 1.0
        
        if sp == "elephant":
            a["actor"].set_position((screen_pos[0] - size*scale, screen_pos[1] - size*scale))
            a["actor_out"].set_position((screen_pos[0] - size*scale - 1.5, screen_pos[1] - size*scale - 1.5))
        elif sp == "deer":
            a["actor"].set_position((screen_pos[0] - size*0.75*scale, screen_pos[1] - size*0.2*scale))
            a["actor_out"].set_position((screen_pos[0] - size*0.75*scale - 1.5, screen_pos[1] - size*0.2*scale - 1.5))
        else:
            a["actor"].set_position(screen_pos)
            a["actor_out"].set_position(screen_pos)

    for species, position in new_spawns:
        spawn_animal(species, position, is_child=True)

    lbl_legend.message = f"Lions: {num_lions} | Elephants: {num_elephants} | Deers: {num_deers}"

    selected = state["selected_animal"]
    if selected is not None:
        for a in animals:
            if a["id"] == selected:
                a["actor"].color = ANIMAL_COLOR[a["species"]]
                
                gender_txt = "Male" if a["gender"] == "M" else "Female"
                type_txt = "Calf" if a["is_child"] else "Adult"
                if a["species"] == "deer": type_txt = "Fawn" if a["is_child"] else "Adult"
                elif a["species"] == "lion": type_txt = "Cub" if a["is_child"] else "Adult"

                lbl_animal_info_left.message = (
                    f"Species: {a['species'].upper()}\nClass: {type_txt}\nSex: {gender_txt}\nHealth: {a['health']:.1f}"
                )
                lbl_animal_info_right.message = (
                    f"Hunger: {a['hunger']:.1f}%\nThirst: {a['thirst']:.1f}%\nAge: {a['age']:.1f}\nCD: {a['cooldown']:.1f}s"
                )
            else:
                a["actor"].color = tuple(np.array(ANIMAL_COLOR[a["species"]]) * 0.3)
    else:
        for a in animals:
            a["actor"].color = ANIMAL_COLOR[a["species"]]
        lbl_animal_info_left.message = "Click an animal\nto monitor."
        lbl_animal_info_right.message = ""

    showm.render()

if __name__ == "__main__":
    show_manager = window.ShowManager(scene=scene, size=(WINDOW_WIDTH, WINDOW_HEIGHT), title="2D Jungle Ecosystem Simulator")
    show_manager.renderer.add_event_handler(on_click, EventType.POINTER_DOWN)
    show_manager.register_callback(sim_tick, 0.016, True, "JungleLoop2D", show_manager)
    show_manager.start()
