"""Run the biweekly meeting demos one by one.

List them::

    python demos/run.py

Run one, a range, a whole section, or everything in order (close each window to
move to the next)::

    python demos/run.py 7
    python demos/run.py 1-6
    python demos/run.py physics
    python demos/run.py all

Run from the root of this checkout, with FURY installed from it
(``pip install -e .``) or ``PYTHONPATH`` pointing at it.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "docs", "examples")


def _demo(script, *args):
    return (os.path.join(HERE, script), *args)


def _example(name):
    return (os.path.join(EXAMPLES, name),)


# (section, title, command). The order is the suggested meeting order.
DEMOS = [
    ("physics", "Core: 140 balls rain into a bowl of static spheres", _demo("physics_demos.py", "01-core")),
    ("physics", "Boxes: crates and balls tumble onto a box stack", _demo("physics_demos.py", "02-boxes")),
    ("physics", "Containers: a tank is settled, shaken, then tilted", _demo("physics_demos.py", "03-containers")),
    ("physics", "Actors as bodies: a spinning drum driven by its actor", _demo("physics_demos.py", "04-actors")),
    ("physics", "Render loop: a domino run stepped with the frames", _demo("physics_demos.py", "05-loop")),
    ("physics", "Force fields: a vortex, then an explosion", _demo("physics_demos.py", "06-forces")),
    ("sims", "Jungle ecosystem: lions, elephants, deer (WASD camera, click animals)", _example("viz_jungle_sim.py")),
    ("sims", "City drone shot: cinematic flythrough of a Kenney city", _example("viz_city_drone_shot.py")),
    ("layout", "Responsive page: sidebar and cards reflow as the window shrinks", _demo("layout_demos.py", "responsive")),
    ("layout", "grow(weight=...): leftover space shared by weight, with a max", _demo("layout_demos.py", "grow")),
    ("layout", "Shrink: fit/grow children shrink toward their min", _demo("layout_demos.py", "shrink")),
    ("layout", "Alignment: align_main and align_cross", _demo("layout_demos.py", "align")),
    ("layout", "Spacing: padding and gap", _demo("layout_demos.py", "spacing")),
    ("layout", "Wrap: rows break onto new lines", _demo("layout_demos.py", "wrap")),
    ("layout", "Text reflow: text blocks measured by the solver, no frame-2 relayout", _demo("layout_demos.py", "text_reflow")),
    ("layout", "Playground: click to cycle each layout setting, see the code", _demo("playground.py")),
    ("widgets", "Settings form built from real widgets", _demo("widget_demos.py", "settings")),
    ("widgets", "Viewer: 3D view with a side panel", _demo("widget_demos.py", "viewer")),
    ("widgets", "Toolbar", _demo("widget_demos.py", "toolbar")),
    ("widgets", "Thumbnail gallery that wraps", _demo("widget_demos.py", "gallery")),
    ("widgets", "Full app: menu, tree, tabs, list, loader", _demo("widget_demos.py", "app")),
    ("widgets", "Tutorial: layout basics", _example("viz_layout_basics.py")),
    ("widgets", "Tutorial: layout sizing", _example("viz_layout_sizing.py")),
    ("widgets", "Tutorial: layout settings demo", _example("viz_layout_settings_demo.py")),
    ("widgets", "TabUI on the layout engine", _example("viz_tab_ui.py")),
    ("widgets", "Tree2D: FURY UI breakdown", _example("viz_tree_ui.py")),
    ("widgets", "Tree2D: actor modifier", _example("viz_tree_actor_modifier.py")),
    ("themes", "Themes: global switch and theme_scope", _example("viz_ui_themes.py")),
    ("themes", "Graphite and Graphite Light: hover, pressed, disabled, focus", _example("viz_ui_graphite.py")),
    ("themes", "Theme gallery: Hairline, Soft Pill, Frost, Ember, compact density", _example("viz_ui_theme_gallery.py")),
    ("extras", "Tutorial: animate NumPy data by updating buffers in place", _example("viz_update_buffers.py")),
    ("extras", "Lorenz ensemble showcase (paused WIP)", _example("viz_lorenz_ensemble.py")),
]


def _print_menu():
    section = None
    for number, (name, title, _) in enumerate(DEMOS, 1):
        if name != section:
            section = name
            print(f"\n[{name}]")
        print(f"  {number:2d}. {title}")
    print("\nusage: python demos/run.py <number | a-b | section | all>")


def _select(arg):
    if arg == "all":
        return list(range(len(DEMOS)))
    if arg in {section for section, _, _ in DEMOS}:
        return [i for i, (section, _, _) in enumerate(DEMOS) if section == arg]
    if "-" in arg:
        first, last = (int(part) for part in arg.split("-"))
        return list(range(first - 1, last))
    return [int(arg) - 1]


def main(args):
    if not args:
        _print_menu()
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [ROOT, env.get("PYTHONPATH")]))
    for index in _select(args[0]):
        section, title, command = DEMOS[index]
        print(f"\n=== {index + 1}. [{section}] {title}")
        script = command[0]
        result = subprocess.run(
            [sys.executable, *command], cwd=os.path.dirname(script), env=env
        )
        if result.returncode:
            print(f"    exited with {result.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
