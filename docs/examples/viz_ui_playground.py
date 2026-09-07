"""
====================================
UI Playground & Workbench with ImGui
====================================

This example showcases the interactive **FURY UI Workbench & Debugger**.

Features:
- **ImGui Side Debug Panel**: Live interactive sidebar with Hierarchy,
  Widget Creation / Builder, Live Property Inspector, Fuzzy Testing, and
  Diagnostics / Event Monitor.
- **Dynamic Hierarchy**: Parent any UI element to a Panel2D container,
  reparent between containers, unparent to root scene, or merge multiple
  elements into a newly created panel with calculated relative offsets.
- **Universal Dragging**: Drag and drop UI elements in real time with
  synchronized coordinates and parent offset preservation.
- **Live Property Inspector**: Modify position, size, z_order, colors,
  opacities, borders, corner radii, typography, and slider values live.
- **Fuzzy Experiments**: Stress-test layout, boundary clipping, reparenting
  loops, and jitter to debug rendering and interaction quirks.

"""


from fury import ui, window

# 1. Initialize Scene
scene = window.Scene()

# 2. Add an initial parent container panel with child elements
main_panel = ui.Panel2D(
    size=(340, 240),
    position=(50, 60),
    color=(0.14, 0.16, 0.22),
    opacity=0.92,
    has_border=True,
    border_width=2,
    border_color=(0.3, 0.6, 0.9),
)
scene.add(main_panel)

panel_title = ui.TextBlock2D(
    text="Parent Container",
    font_size=18,
    color=(1, 1, 1),
    bg_color=(0.22, 0.26, 0.38),
    size=(310, 32),
)
main_panel.add_element(panel_title, (15, 15))

demo_slider = ui.LineSlider2D(
    initial_value=60,
    min_value=0,
    max_value=100,
    length=300,
    text_template="Level: {value:.0f}%",
)
main_panel.add_element(demo_slider, (15, 75))

demo_button = ui.TextButton2D(label="Click Me", size=(120, 35))
main_panel.add_element(demo_button, (15, 150))

# 3. Add standalone elements ready for parenting, reparenting, or merging
disk = ui.Disk2D(
    outer_radius=40,
    inner_radius=15,
    center=(430, 80),
    color=(0.2, 0.8, 0.5),
    opacity=0.9,
)
scene.add(disk)

rounded_rect = ui.RoundedRectangle2D(
    size=(160, 90),
    position=(430, 190),
    color=(0.85, 0.35, 0.25),
    opacity=0.9,
    corner_radius=15.0,
)
scene.add(rounded_rect)

options = ui.Checkbox(
    labels=["High Dynamic Range", "Bloom", "Anti-Aliasing"],
    checked_labels=["Bloom"],
    position=(50, 330),
)
scene.add(options)

notes = ui.TextBlock2D(
    text=(
        "FURY UI Workbench & Debugger\n"
        "• Select any element in the ImGui side panel.\n"
        "• Drag elements freely in the viewport.\n"
        "• Parent / Reparent / Merge into panels.\n"
        "• Live edit properties or run fuzzy experiments."
    ),
    font_size=16,
    color=(0.95, 0.95, 0.95),
    bg_color=(0.1, 0.1, 0.12),
    size=(360, 100),
    position=(50, 460),
)
scene.add(notes)

# 4. Instantiate the UI Workbench
workbench = ui.UIWorkbench(
    scene=scene,
    panel_width=440,
    enable_universal_drag=True,
)

# 5. Create ShowManager with ImGui enabled
show_manager = window.ShowManager(
    scene=scene,
    size=(1280, 800),
    title="FURY UI Playground & Workbench",
    window_type="default",
    imgui=True,
    imgui_draw_function=workbench.render,
)

workbench.set_show_manager(show_manager)

# 6. Start the interactive application
if __name__ == "__main__":
    show_manager.start()
