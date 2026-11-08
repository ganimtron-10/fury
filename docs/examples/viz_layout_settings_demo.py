"""
====================
Layout Settings Demo
====================

A small settings window built only from layout containers. There is no pixel
arithmetic: the widgets adapt when the text changes or the window is resized.
"""

from fury import ui, window

##############################################################################
# The window is a panel that fills the screen height next to a preview area.

root = ui.Row(position=(0, 0), width=ui.grow(), height=ui.grow())

sidebar = ui.Panel2D((300, 100), color=(0.14, 0.15, 0.2), opacity=1)
sidebar.set_layout(
    direction=ui.Direction.COLUMN,
    height=ui.grow(),
    padding=16,
    gap=12,
    align_cross=ui.Align.STRETCH,
)
root.add(sidebar)

preview = ui.Panel2D((100, 100), color=(0.08, 0.09, 0.12), opacity=1)
preview.set_layout(
    width=ui.grow(),
    height=ui.grow(),
    align_main=ui.Align.CENTER,
    align_cross=ui.Align.CENTER,
)
root.add(preview)

status = ui.TextBlock2D(text="Quality: 50", font_size=26, dynamic_bbox=True)
preview.add(status)

##############################################################################
# The sidebar holds a heading, a row with a label and a spin box, some check
# boxes and a dropdown.

sidebar.add(ui.TextBlock2D(text="Render settings", font_size=22, dynamic_bbox=True))

quality_row = ui.Row(align_cross=ui.Align.CENTER, gap=8)
quality_row.add(
    ui.TextBlock2D(text="Quality", dynamic_bbox=True),
    width=ui.grow(),
)
quality = ui.SpinBox(size=(150, 40), position=(0, 0))
quality_row.add(quality)
sidebar.add(quality_row)

sidebar.add(ui.Checkbox(labels=["Shadows", "Reflections", "Bloom"], gap=6))

mode = ui.ComboBox2D(
    items=["Low", "Medium", "High"], placeholder="Mode", size=(268, 160)
)
sidebar.add(mode)

##############################################################################
# A spacer pushes the buttons to the bottom of the sidebar.

sidebar.add(ui.Row(height=ui.grow()))

buttons = ui.Row(gap=8, align_main=ui.Align.END)
buttons.add(ui.TextButton2D(label="Reset", size=(90, 34)))
buttons.add(ui.TextButton2D(label="Apply", size=(90, 34)))
sidebar.add(buttons)

##############################################################################
# Changing the text marks the layout dirty, and the next frame re-centers it.


def on_quality_change(spinbox):
    """
    Show the new quality value in the preview.

    Parameters
    ----------
    spinbox : SpinBox
        The widget that changed.
    """
    status.message = f"Quality: {spinbox.value}"


quality.on_change = on_quality_change

##############################################################################
# Show the demo.

scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(root)

show_manager = window.ShowManager(
    scene=scene, size=(860, 560), title="FURY Layout Settings Demo"
)

show_manager.start()
