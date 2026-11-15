"""
=============================
FURY UI Breakdown with Tree2D
=============================

This example visualizes the different types of UI elements that are available
in FURY's UI sub-module. We will use a ``Tree2D`` UI element to organize and
display them hierarchically, following the modules they come from: core,
elements and containers.

First, some imports.
"""

from fury import ui, window
from fury.data import fetch_viz_icons, read_viz_icons

fetch_viz_icons()

###############################################################################
# Defining the UI Hierarchy
#
# Each key of the structure is a node, and its list holds the labels of its
# child nodes. Labels must be unique, because nodes are looked up by label.

structure = [
    {"Core": ["Rectangle", "RoundedRectangle", "Disk", "TextBlock"]},
    {
        "Elements": [
            "TextButton",
            "TexturedButton",
            "TextBox",
            "Checkbox",
            "RadioButton",
            "ComboBox",
            "ListBox",
            "LineSlider",
            "LineDoubleSlider",
            "RingSlider",
            "RangeSlider",
            "SpinBox",
            "PlaybackPanel",
            "Card",
        ]
    },
    {"Containers": ["Panel", "ImageContainer"]},
]

tree = ui.Tree2D(
    structure=structure,
    tree_name="FURY UI Breakdown",
    size=(600, 900),
    position=(20, 20),
    color=(0.2, 0.2, 0.25),
    opacity=1.0,
)

###############################################################################
# Now, let's create UI elements for the Core node and add each one to its
# node. ``coords`` is the offset in pixels from the top-left corner of the
# node's content area.

tree.add_content("Rectangle", ui.Rectangle2D(size=(100, 60), color=(0.8, 0.4, 0.7)))
tree.add_content(
    "RoundedRectangle",
    ui.RoundedRectangle2D(size=(120, 60), color=(0.2, 0.6, 0.9), corner_radius=15),
)
tree.add_content("Disk", ui.Disk2D(outer_radius=30, color=(0.6, 0.2, 0.8)))
tree.add_content(
    "TextBlock",
    ui.TextBlock2D(
        text="Some text", font_size=20, size=(150, 30), bg_color=(0.3, 0.3, 0.3)
    ),
    coords=(0, 5),
)

###############################################################################
# Next, the UI elements for the Elements node.

icons = {
    "default": read_viz_icons(fname="circle-right.png"),
    "pressed": read_viz_icons(fname="circle-down.png"),
}

tree.add_content("TextButton", ui.TextButton2D(label="Click Me!", size=(150, 40)))
tree.add_content("TexturedButton", ui.TexturedButton2D(states=icons, size=(40, 40)))
tree.add_content("TextBox", ui.TextBox2D(width=20, height=1, text="Type here"))
tree.add_content("Checkbox", ui.Checkbox(labels=["One", "Two"], font_size=18))
tree.add_content(
    "RadioButton",
    ui.RadioButton(labels=["Yes", "No"], checked_labels=["Yes"], font_size=18),
)
tree.add_content(
    "ComboBox",
    ui.ComboBox2D(items=["Option A", "Option B", "Option C"], size=(250, 150)),
)
tree.add_content(
    "ListBox",
    ui.ListBox2D(
        values=["First", "Second", "Third", "Fourth", "Fifth", "Sixth"],
        size=(250, 150),
    ),
)
tree.add_content(
    "LineSlider", ui.LineSlider2D(length=200, orientation="vertical"), coords=(40, 0)
)
tree.add_content("LineDoubleSlider", ui.LineDoubleSlider2D(length=200), coords=(0, 30))
tree.add_content("RingSlider", ui.RingSlider2D())
tree.add_content(
    "RangeSlider",
    ui.RangeSlider(range_slider_center=(0, 60), value_slider_center=(0, 0), length=200),
    coords=(0, 30),
)
tree.add_content("SpinBox", ui.SpinBox(min_val=0, max_val=10, initial_val=5))
tree.add_content("PlaybackPanel", ui.PlaybackPanel(width=400))
tree.add_content(
    "Card",
    ui.Card2D(
        read_viz_icons(fname="home3.png"),
        title_text="Card",
        body_text="A card with an image, a title and a body.",
        size=(200, 220),
    ),
)

###############################################################################
# Finally, the containers. A ``Panel2D`` can hold other UI elements, and
# ``ImageContainer2D`` shows an image from a file or a NumPy array.

panel = ui.Panel2D(size=(200, 100), color=(0.5, 0.7, 0.3))
panel.add_element(ui.Disk2D(outer_radius=20, color=(0.9, 0.9, 0.2)), (20, 30))
tree.add_content("Panel", panel)

tree.add_content(
    "ImageContainer",
    ui.ImageContainer2D(img_path=read_viz_icons(fname="home3.png"), size=(80, 80)),
)

###############################################################################
# Expand the module nodes so their children are listed at start. Use the mouse
# wheel or the scroll bar to move through the rows.

for module in ("Core", "Elements", "Containers"):
    tree.get_node(module).toggle_view()

###############################################################################
# Finally, we add the tree to our scene and start the interactive window.

scene = window.Scene()
scene.add(tree)

show_manager = window.ShowManager(
    scene=scene, size=(1000, 1000), title="FURY Tree2D Example"
)

show_manager.start()
