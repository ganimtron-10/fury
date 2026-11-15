"""
================================
Switches and Segmented Controls
================================

This example shows two compact settings widgets. ``Switch2D`` turns an option
on or off, and ``SegmentedControl2D`` picks one of a few mutually exclusive
options. Both take their look from the theme in use; here a Graphite panel
changes how a volume slice is shown in the 3D scene.
"""

##############################################################################
# First, a bunch of imports.

import numpy as np

from fury import actor, ui, window

##############################################################################
# The 3D scene: a small synthetic volume shown one slice at a time, the
# edges of its bounding box, and the coordinate axes.

shape = (64, 64, 64)
grid = np.indices(shape).astype(float) - np.array(shape)[:, None, None, None] / 2
distance = np.linalg.norm(grid, axis=0)
volume = np.cos(distance / 3) * np.exp(-distance / 24)

center = np.array(shape) / 2
slicer = actor.data_slicer(volume)
side = shape[0]
corners = np.array(
    [[x, y, z] for x in (0, side) for y in (0, side) for z in (0, side)], dtype=float
)
edges = [
    (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
]
outline = actor.line(
    [corners[[a, b]] for a, b in edges],
    colors=(0.6, 0.65, 0.7),
)
axes = actor.axes(scale=(48, 48, 48))

##############################################################################
# The settings panel. Only texts are given: sizes, colors, fonts and corner
# radii come from the Graphite theme.

with ui.theme_scope("graphite"):
    panel = ui.Panel2D(size=(240, 100), position=(20, 20))
    panel.set_layout(
        direction=ui.Direction.COLUMN, height=ui.fit(), align_cross=ui.Align.STRETCH
    )
    title = ui.TextBlock2D(text="Slice view", font_weight=600, dynamic_bbox=True)
    outline_switch = ui.Switch2D(label="Show outline", checked=True)
    axes_switch = ui.Switch2D(label="Show axes")
    plane_label = ui.TextBlock2D(text="Plane", muted=True, dynamic_bbox=True)
    plane = ui.SegmentedControl2D(["Axial", "Coronal", "Sagittal"])

for element in (title, outline_switch, axes_switch, plane_label, plane):
    panel.add(element)

##############################################################################
# Callbacks connect the widgets to the scene. Each one receives the widget
# that changed.


def show_slice(name):
    """
    Show only the slice of one plane.

    Parameters
    ----------
    name : str
        ``"Axial"``, ``"Coronal"`` or ``"Sagittal"``. The slicer holds the
        sagittal (x), coronal (y) and axial (z) slices, in that order.
    """
    visible = [name == axis_name for axis_name in ("Sagittal", "Coronal", "Axial")]
    actor.set_group_visibility(slicer, visible)


outline_switch.on_change = lambda switch: setattr(outline, "visible", switch.checked)
axes_switch.on_change = lambda switch: setattr(axes, "visible", switch.checked)
plane.on_change = lambda control: show_slice(control.selected)
axes.visible = axes_switch.checked
show_slice(plane.selected)

##############################################################################
# Show everything.

scene = window.Scene(background=(0.08, 0.1, 0.13, 1))
scene.add(slicer, outline, axes, panel)

show_manager = window.ShowManager(
    scene=scene, size=(640, 480), title="FURY Switch and Segmented Control Example"
)
camera = show_manager.screens[0].camera
camera.world.reference_up = (0, 0, 1)
camera.local.position = center + np.array([-110.0, -170.0, 90.0])
camera.look_at(center + np.array([-22.0, 8.0, 0.0]))
show_manager.start()
