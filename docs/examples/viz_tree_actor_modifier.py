"""
==============================
Actor Modifier using a Tree UI
==============================

This example shows how to create an actor modifier using a Tree UI.
The parameters that will be modified are the colors, position, and
rotation of the actors. Selecting an actor's node shows that actor.
"""

###############################################################################
# Adding the imports

import numpy as np

from fury import actor, ui, window

###############################################################################
# Defining the UI Structure
#
# Node labels must be unique, so each actor gets its own position and color
# nodes.

actor_names = ["Cube", "Cylinder", "Cone"]
structure = [{name: [f"{name} position", f"{name} color"]} for name in actor_names]

###############################################################################
# Next, we instantiate the ``Tree2D`` object using the structure defined above.
# With ``multiselect=False``, selecting one actor's node deselects the others.

tree = ui.Tree2D(
    structure=structure,
    tree_name="Actor Modifier",
    size=(460, 700),
    position=(20, 20),
    color=(0.8, 0.4, 0.2),
    opacity=1,
    button_position="end",
    multiselect=False,
)

###############################################################################
# Creating the Actors
#
# The actors share one center, to the right of the scene's origin, and only
# the selected one is shown. Axes mark the origin, so the camera frames the
# actors beside the tree.

center = np.array([4.0, 0.0, 0.0])
initial_colors = {
    "Cube": (0.0, 0.0, 1.0),
    "Cylinder": (0.0, 1.0, 0.0),
    "Cone": (1.0, 0.0, 0.0),
}

actors = {
    "Cube": actor.box(
        centers=np.zeros((1, 3)),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([initial_colors["Cube"]]),
        scales=np.array([[2, 2, 2]]),
    ),
    "Cylinder": actor.cylinder(
        centers=np.zeros((1, 3)),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([initial_colors["Cylinder"]]),
        radii=1.0,
        height=2.0,
    ),
    "Cone": actor.cone(
        centers=np.zeros((1, 3)),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([initial_colors["Cone"]]),
        radii=1.0,
        height=2.0,
    ),
}

for actor_obj in actors.values():
    actor_obj.local.position = center
    actor_obj.visible = False

axes = actor.axes(scale=(4, 4, 4))

###############################################################################
# Building the Slider Controls
#
# Each actor gets sliders for its position and rotation, and one slider per
# color channel. The sliders start at the actor's current values.


def setup_actor_controls(actor_obj, name):
    color = list(initial_colors[name])
    offset = [0.0, 0.0]

    ring_slider = ui.RingSlider2D(initial_value=0, text_template="{angle:5.1f}°")
    slider_x = ui.LineSlider2D(
        initial_value=0, min_value=-3, max_value=3, orientation="horizontal"
    )
    slider_y = ui.LineSlider2D(
        initial_value=0, min_value=-3, max_value=3, orientation="vertical"
    )
    color_sliders = [
        ui.LineSlider2D(
            initial_value=value,
            min_value=0,
            max_value=1,
            length=150,
            orientation="vertical",
        )
        for value in color
    ]

    def rotate(slider):
        actor_obj.local.euler_x = np.radians(slider.value)

    def move(axis):
        def callback(slider):
            offset[axis] = slider.value
            actor_obj.local.position = center + (offset[0], offset[1], 0)

        return callback

    def change_color(channel):
        def callback(slider):
            color[channel] = slider.value
            actor_obj.material.color_mode = "uniform"
            actor_obj.material.color = (*color, 1.0)

        return callback

    ring_slider.on_change = rotate
    slider_x.on_change = move(0)
    slider_y.on_change = move(1)
    for channel, slider in enumerate(color_sliders):
        slider.on_change = change_color(channel)

    tree.add_content(f"{name} position", slider_x, coords=(0, 30))
    tree.add_content(f"{name} position", slider_y, coords=(20, 80))
    tree.add_content(f"{name} position", ring_slider, coords=(110, 80))
    for channel, slider in enumerate(color_sliders):
        tree.add_content(f"{name} color", slider, coords=(20 + 110 * channel, 10))

    ###########################################################################
    # Show the actor only while its node is selected.

    # The control rows are not selectable, so clicking them opens them
    # without deselecting the actor.

    for part in ("position", "color"):
        tree.get_node(f"{name} {part}").selectable = False

    node = tree.get_node(name)
    node.on_node_select = lambda node: setattr(actor_obj, "visible", True)
    node.on_node_deselect = lambda node: setattr(actor_obj, "visible", False)


for name, actor_obj in actors.items():
    setup_actor_controls(actor_obj, name)

###############################################################################
# Start with the cube selected and its controls open.

tree.get_node("Cube").selected = True
tree.get_node("Cube").toggle_view()
tree.get_node("Cube position").toggle_view()

###############################################################################
# Finally, we configure the scene and add the ``Tree2D`` UI, the axes and our
# three actors to it.

scene = window.Scene()
scene.add(tree, axes, *actors.values())

show_manager = window.ShowManager(
    scene=scene, size=(1200, 800), title="FURY Tree2D Actor Modifier"
)

show_manager.start()
