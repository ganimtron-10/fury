"""
=====================================
UI Playground & Workbench with ImGui
=====================================

This example runs the interactive **FURY UI Workbench**, a debugger for the
2D UI layer.

The scene below is deliberately built out of nested containers so that the
workbench has a real hierarchy to show. Every component in it, down to the
internal parts each widget builds for itself (a panel border, a slider handle,
the background of a text block), is listed, selectable and editable while the
application runs.

What the side panel gives you:

- **Hierarchy**: the live scene tree, refreshed every frame. Content and
  internal parts are told apart, parts are named after the attribute that
  holds them (``handle``, ``borders[top]``, ...), and the tree can be
  filtered. Selecting a node outlines it in the viewport.
- **Inspector**: every property the selected class declares, its plain
  instance attributes, the pygfx actors behind it and the callbacks it
  receives, all live editable.
- **Create**: instantiate any FURY UI component into the scene root or into
  any panel, plus a few composite presets.
- **Experiments**: randomize colors, sizes and positions, run boundary stress
  tests and reparenting loops that check for actor leaks.
- **Monitor**: the global :class:`fury.ui.UIContext` state, scene counters and
  a rolling event log.

Clicking a component in the viewport selects it in the panel. Components that
have no drag behaviour of their own can be moved with the mouse, while widgets
that already use dragging (panels, sliders) keep their normal behaviour.
"""

from fury import ui, window

###############################################################################
# Create the scene that the workbench will inspect.

scene = window.Scene()

###############################################################################
# A parent container. Its children are added *before* the panel joins the
# scene: ``Scene.add`` walks the component tree once, so anything attached
# afterwards would have no actor in the render list.

main_panel = ui.Panel2D(
    size=(360, 300),
    position=(60, 60),
    color=(0.14, 0.16, 0.22),
    opacity=0.92,
    has_border=True,
    border_width=2,
    border_color=(0.3, 0.6, 0.9),
)

panel_title = ui.TextBlock2D(
    text="Parent Container",
    font_size=18,
    color=(1, 1, 1),
    bg_color=(0.22, 0.26, 0.38),
    size=(330, 32),
)
main_panel.add_element(panel_title, (15, 15))

demo_slider = ui.LineSlider2D(
    initial_value=60,
    min_value=0,
    max_value=100,
    length=300,
    text_template="Level: {value:.0f}%",
)
main_panel.add_element(demo_slider, (30, 100))

demo_button = ui.TextButton2D(label="Click Me", size=(130, 36), font_size=20)
main_panel.add_element(demo_button, (15, 180))

demo_checkbox = ui.Checkbox(
    labels=["Shadows", "Bloom"],
    checked_labels=["Bloom"],
    font_size=16,
)
main_panel.add_element(demo_checkbox, (15, 235))

scene.add(main_panel)

###############################################################################
# A second container, so that parenting and reparenting have somewhere to go.

side_panel = ui.Panel2D(
    size=(300, 190),
    position=(60, 400),
    color=(0.10, 0.18, 0.16),
    opacity=0.92,
    has_border=True,
    border_width=2,
    border_color=(0.2, 0.7, 0.5),
)

side_title = ui.TextBlock2D(
    text="Drop Target",
    font_size=18,
    color=(1, 1, 1),
    bg_color=(0.16, 0.28, 0.24),
    size=(270, 30),
)
side_panel.add_element(side_title, (15, 15))

side_ring = ui.RingSlider2D(
    initial_value=35,
    min_value=0,
    max_value=100,
    slider_inner_radius=28,
    slider_outer_radius=42,
    font_size=14,
)
side_panel.add_element(side_ring, (170, 60))

scene.add(side_panel)

###############################################################################
# Free standing components. These have no drag behaviour of their own, so the
# workbench makes them draggable and they can be dropped into either panel.

disk = ui.Disk2D(
    outer_radius=40,
    inner_radius=15,
    center=(520, 120),
    color=(0.2, 0.8, 0.5),
    opacity=0.9,
)
scene.add(disk)

rounded_rect = ui.RoundedRectangle2D(
    size=(170, 95),
    position=(460, 220),
    color=(0.85, 0.35, 0.25),
    opacity=0.9,
    corner_radius=15.0,
)
scene.add(rounded_rect)

options = ui.Checkbox(
    labels=["High Dynamic Range", "Anti-Aliasing", "Motion Blur"],
    checked_labels=["Anti-Aliasing"],
    position=(460, 360),
)
scene.add(options)

notes = ui.TextBlock2D(
    text=(
        "FURY UI Workbench\n"
        "Click a component to select it.\n"
        "Drag the free components around.\n"
        "Parent, reparent or merge them into a panel.\n"
        "Edit every property from the Inspector tab."
    ),
    font_size=16,
    color=(0.95, 0.95, 0.95),
    bg_color=(0.1, 0.1, 0.12),
    size=(400, 120),
    position=(60, 630),
)
scene.add(notes)

###############################################################################
# Attach the workbench and open the window with ImGui enabled.

workbench = ui.UIWorkbench(
    scene=scene,
    panel_width=460,
    enable_universal_drag=True,
)

show_manager = window.ShowManager(
    scene=scene,
    size=(1400, 820),
    title="FURY UI Playground & Workbench",
    window_type="default",
    imgui=True,
    imgui_draw_function=workbench.render,
)

workbench.set_show_manager(show_manager)

if __name__ == "__main__":
    show_manager.start()
