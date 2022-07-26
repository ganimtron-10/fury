"""
============
UI Animation
============

This example shows how to use animation on the UI elements.

First, a bunch of imports.
"""

from fury import ui, window
from fury.animation.timeline import Timeline
import numpy as np

###############################################################################
# Creating a UI element.
rect = ui.Rectangle2D((50, 50))

###############################################################################
# Creating a Timeline
timeline = Timeline()

timeline.add_actor(rect.actor)

timeline.set_position(0, np.array([0, 0, 0]))
timeline.set_position(2, np.array([10, 10, 10]))
timeline.set_position(5, np.array([-10, 16, 0]))
timeline.set_position(9, np.array([10, 0, 20]))

###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY UI Animation Example")

show_manager.scene.add(timeline)

###############################################################################
# making a function to update the animation and render the scene.


def timer_callback(_obj, _event):
    timeline.update_animation()
    show_manager.render()


show_manager.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size, out_path="viz_ui_animation.png")
