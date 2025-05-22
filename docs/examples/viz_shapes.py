# -*- coding: utf-8 -*-
"""
==============
Simple Shapes
==============

This example shows how to use the UI API. We will demonstrate how to draw
some geometric shapes from FURY UI elements.

First, a bunch of imports.
"""

from fury.ui import Rectangle2D, Disk2D, Panel2D
from fury.window import Scene, ShowManager
from fury.data import fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

# fetch_viz_icons()

##############################################################################
# Creating a Scene

scene = Scene()

###############################################################################
# Let's draw some simple shapes. First, a rectangle.

rect = Rectangle2D(size=(100, 100), position=(400, 400), color=(1, 0, 1))

###############################################################################
# Then we can draw a solid circle, or disk.

disk = Disk2D(outer_radius=50, center=(400, 200), color=(1, 1, 0))

###############################################################################
# Add an inner radius to make a ring.

# ring = ui.Disk2D(outer_radius=50, inner_radius=45, center=(500, 600), color=(0, 1, 1))


###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

panel = Panel2D(size=(100, 100), position=(500, 500), color=(0, 0, 1))

scene.add(rect)
scene.add(disk)
scene.add(panel)

if __name__ == "__main__":
    current_size = (800, 800)
    show_manager = ShowManager(
        scene=scene, size=current_size, title="FURY 2.0: Shapes Example"
    )
    panel.canvas = show_manager.window
    show_manager.start()
