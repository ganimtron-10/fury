"""
=============
Layout Sizing
=============

Each axis of an element has a sizing rule:

* ``fixed(px)`` is an exact size.
* ``fit()`` follows the content.
* ``grow(weight)`` takes a share of the leftover space.
* ``percent(p)`` is a fraction of the parent.

This tutorial builds one row per rule. Resize the window to see them react.
"""

from fury import ui, window

BAR = (0.25, 0.55, 0.9)
OTHER = (0.9, 0.55, 0.25)


def bar(color=BAR):
    """
    Create a small coloured box.

    Parameters
    ----------
    color : tuple
        RGB color of the box.

    Returns
    -------
    Rectangle2D
        The box.
    """
    return ui.Rectangle2D(size=(30, 30), color=color)


##############################################################################
# The column grows with the window, and every row stretches across it.

column = ui.Column(
    position=(0, 0),
    width=ui.grow(),
    padding=16,
    gap=12,
    align_cross=ui.Align.STRETCH,
)

##############################################################################
# ``grow`` shares the leftover space by weight: the second box gets three times
# more than the first.

row = ui.Row(gap=8)
row.add(bar(), width=ui.grow(1))
row.add(bar(OTHER), width=ui.grow(3))
column.add(row)

##############################################################################
# ``percent`` is a fraction of the parent's width.

row = ui.Row(gap=8)
row.add(bar(), width=ui.percent(0.25))
row.add(bar(OTHER), width=ui.percent(0.75))
column.add(row)

##############################################################################
# ``fixed`` and ``grow`` work together for a sidebar next to a flexible area.

row = ui.Row(gap=8)
row.add(bar(), width=ui.fixed(120))
row.add(bar(OTHER), width=ui.grow())
column.add(row)

##############################################################################
# ``min`` and ``max`` limit a rule. This box stops growing at 200 pixels and
# the other box takes the rest.

row = ui.Row(gap=8)
row.add(bar(), width=ui.grow(max=200))
row.add(bar(OTHER), width=ui.grow())
column.add(row)

##############################################################################
# Text follows the width it is given and wraps when it has to.

column.add(
    ui.TextBlock2D(
        text="This sentence wraps when the window gets narrower, "
        "and the rows below it move out of the way.",
        dynamic_bbox=True,
        bg_color=(0.2, 0.2, 0.28),
    ).set_layout(padding=(6, 10))
)

row = ui.Row(gap=8, align_main=ui.Align.END)
row.add(ui.TextButton2D(label="Cancel", size=(90, 32)))
row.add(ui.TextButton2D(label="Apply", size=(90, 32)))
column.add(row)

##############################################################################
# Show the result.

scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(column)

show_manager = window.ShowManager(
    scene=scene, size=(640, 380), title="FURY Layout Sizing"
)

show_manager.start()
