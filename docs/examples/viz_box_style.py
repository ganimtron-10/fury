"""
==========
Box Styles
==========

This example shows how to style a ``RoundedRectangle2D`` with a border,
a different radius for each corner and a soft drop shadow. The shadow is
drawn outside the box but is not part of it: the layout, the size and mouse
picking only see the box itself.
"""

##############################################################################
# First, let's import the necessary modules.

from fury.ui import Column, RoundedRectangle2D, Row, TextBlock2D
from fury.window import Scene, ShowManager

##############################################################################
# Create a dark scene so the borders and shadows stand out.

scene = Scene()
scene.background = "#2A2C31"

##############################################################################
# **A border.** ``border_width`` is in pixels and the border is drawn inside
# the box edge, following the rounded corners.

bordered = RoundedRectangle2D(
    size=(220, 140),
    color="#2F3440",
    corner_radius=14,
    border_color="#4C8BF5",
    border_width=2,
)

##############################################################################
# **Mixed corner radii.** Four values give the top-left, top-right,
# bottom-right and bottom-left radii.

mixed = RoundedRectangle2D(
    size=(220, 140),
    color="#3D7A5C",
    corner_radius=(36, 4, 36, 4),
)

##############################################################################
# **A soft shadow.** ``shadow_blur`` softens the edge, ``shadow_offset``
# moves the shadow (positive y points down) and ``shadow_spread`` grows it.

shadowed = RoundedRectangle2D(
    size=(220, 140),
    color="#E8E9EC",
    corner_radius=18,
    shadow_color=(0.0, 0.0, 0.0, 0.6),
    shadow_offset=(8, 14),
    shadow_blur=24,
    shadow_spread=2,
)

##############################################################################
# **A graphite card.** A dark card with a hairline border and a large, faint
# shadow, a common style for panels in dark interfaces.

graphite = RoundedRectangle2D(
    size=(220, 140),
    color="#1C1D20",
    corner_radius=10,
    border_color="#2C2E33",
    border_width=1,
    shadow_color=(0.0, 0.0, 0.0, 0.35),
    shadow_offset=(0, 10),
    shadow_blur=30,
)

##############################################################################
# Place the boxes in a grid. The gaps are measured from the boxes, so the
# shadows do not push their neighbors aside.

cards = [bordered, mixed, shadowed, graphite]
grid = Column(
    Row(bordered, mixed, gap=60),
    Row(shadowed, graphite, gap=60),
    gap=60,
    position=(70, 70),
)
grid.update_layout()

##############################################################################
# Label each box.

labels = []
for card, text, color in zip(
    cards,
    ["Border", "Corner radii", "Shadow", "Graphite"],
    [(1, 1, 1), (1, 1, 1), (0.1, 0.1, 0.12), (0.85, 0.86, 0.9)],
    strict=True,
):
    x, y = card.get_position()
    label = TextBlock2D(
        text=text,
        font_size=18,
        color=color,
        position=(x + 16, y + 14),
        dynamic_bbox=True,
    )
    label.z_order = 1
    labels.append(label)

scene.add(grid, *labels)

###############################################################################
# Create and start the ShowManager.

current_size = (640, 480)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY Box Style Example",
)
show_manager.start()
