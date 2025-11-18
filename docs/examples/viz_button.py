"""
========
Button2D
========
"""

##############################################################################
# First, a bunch of imports

from fury.ui import Button2D
from fury.window import (
    Scene,
    ShowManager,
)
from fury.data import fetch_viz_icons, read_viz_icons

##############################################################################
# Fetch icons that are included in FURY.

fetch_viz_icons()

##############################################################################
# Creating a Scene

scene = Scene()

#############################################################################
# Creating a Button with multiple icons

icon_files = []
icon_files.append(("down", read_viz_icons(fname="circle-down.png")))
icon_files.append(("left", read_viz_icons(fname="circle-left.png")))
icon_files.append(("up", read_viz_icons(fname="circle-up.png")))
icon_files.append(("right", read_viz_icons(fname="circle-right.png")))

btn = Button2D(icon_fnames=icon_files, size=(100, 100))
btn.set_position(coords=(350, 350), x_anchor="CENTER", y_anchor="CENTER")


def btn_left_button_pressed(event):
    """Change button icon when pressed"""
    btn.next_icon()


btn.on_left_mouse_button_pressed = btn_left_button_pressed

scene.add(btn)

###############################################################################
# Starting the ShowManager

if __name__ == "__main__":
    current_size = (700, 700)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Button2D Example",
    )
    show_manager.start()
