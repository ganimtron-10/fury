"""
=============
Layout Basics
=============

This tutorial shows how to arrange UI elements without computing any pixel
positions. :class:`fury.ui.Column` stacks its children from top to bottom and
:class:`fury.ui.Row` places them from left to right.

The layout engine is part of FURY and needs no extra dependency.
"""

from fury import ui, window

##############################################################################
# A column of labels. ``padding`` is the space inside the column and ``gap`` is
# the space between its children.

labels = ui.Column(
    ui.TextBlock2D(text="First", dynamic_bbox=True, bg_color=(0.2, 0.4, 0.8)),
    ui.TextBlock2D(text="Second label", dynamic_bbox=True, bg_color=(0.2, 0.4, 0.8)),
    ui.TextBlock2D(text="Third", dynamic_bbox=True, bg_color=(0.2, 0.4, 0.8)),
    position=(30, 30),
    padding=10,
    gap=6,
)

##############################################################################
# A row of coloured boxes. ``margin`` adds space around one element, and
# ``align_cross`` centers elements of different heights.

red = ui.Rectangle2D(size=(60, 40), color=(0.9, 0.3, 0.3))
green = ui.Rectangle2D(size=(60, 80), color=(0.3, 0.8, 0.4))
blue = ui.Rectangle2D(size=(60, 60), color=(0.3, 0.5, 0.9))
green.set_layout(margin=(0, 20))

boxes = ui.Row(red, green, blue, position=(30, 200), align_cross=ui.Align.CENTER)

##############################################################################
# A :class:`fury.ui.Panel2D` draws a background and can lay out its children
# too. Use ``add`` for automatic placement, and ``add_element`` when you want
# to place something by hand.

panel = ui.Panel2D((260, 150), position=(320, 30), color=(0.15, 0.15, 0.2), opacity=1)
panel.set_layout(direction=ui.Direction.COLUMN, padding=12, gap=8)
panel.add(ui.TextBlock2D(text="Inside a panel", dynamic_bbox=True))
panel.add(ui.TextButton2D(label="A button", size=(120, 32)))

##############################################################################
# Show everything in a window.

scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(labels, boxes, panel)

show_manager = window.ShowManager(
    scene=scene, size=(620, 340), title="FURY Layout Basics"
)

show_manager.start()
