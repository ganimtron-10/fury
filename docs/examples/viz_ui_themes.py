"""
=========
UI Themes
=========

This example shows how themes set the default colors and fonts of FURY UI
elements. A theme is a set of items, such as the ``color`` of a
:class:`fury.ui.Panel2D` or the ``track_color`` of a slider. Elements read
their defaults from the theme in use when they are created, and an argument
passed to a constructor always wins over the theme.
"""

##############################################################################
# First, a bunch of imports.

from fury import ui
from fury.window import Scene, ShowManager

##############################################################################
# A small settings panel. Only sizes and positions are given, so every color
# and font comes from the theme.


def settings_panel(title, position):
    """
    Build a panel with a title, a slider and a button.

    Parameters
    ----------
    title : str
        Text shown at the top of the panel.
    position : tuple of int
        Position of the panel's top-left corner.

    Returns
    -------
    Panel2D
        The panel with its elements.
    """
    panel = ui.Panel2D(size=(190, 200), position=position)
    heading = ui.TextBlock2D(text=title, size=(160, 30))
    slider = ui.LineSlider2D(length=140, initial_value=75)
    button = ui.TextButton2D("Apply", size=(100, 36))
    panel.add_element(heading, (15, 12))
    panel.add_element(slider, (25, 90))
    panel.add_element(button, (45, 145))
    return panel


##############################################################################
# The built-in ``classic`` theme is the global theme by default. It keeps the
# original look of FURY widgets.

classic_panel = settings_panel("Classic", (15, 40))

##############################################################################
# Derive a theme to change only some items. Each item belongs to a UI class,
# and subclasses inherit it: items set for ``Slider2D`` apply to every
# slider. ``set_theme`` makes it the global theme for elements created
# afterwards.

graphite = ui.builtin_theme("classic").derive(
    name="graphite",
    items={
        "UI": {"font_family": "Noto Sans"},
        "Panel2D": {"color": "#17191E", "opacity": 1.0},
        "TextBlock2D": {"color": "#E6E8EB", "font_size": 16},
        "TextButton2D": {
            "font_size": 15,
            "text_color": "#0B1020",
            "states": {
                "default": "#8AB4FF",
                "hover": "#A5C5FF",
                "pressed": "#6E96E0",
                "disabled": "#3A3F4A",
            },
        },
        "Slider2D": {
            "track_color": "#3A3F4A",
            "handle_color": "#E6E8EB",
            "handle_active_color": "#8AB4FF",
            "font_size": 13,
        },
    },
)
ui.set_theme(graphite)
graphite_panel = settings_panel("Graphite", (225, 40))

##############################################################################
# ``theme_scope`` applies a theme only to elements created inside the
# ``with`` block, like a temporary style in an immediate mode UI. Sub-widgets
# built by a composite widget use the same theme.

ember = graphite.derive(
    name="ember",
    items={
        "Slider2D": {"handle_active_color": "#FF8A3D", "track_color": "#5A3420"},
        "TextButton2D": {
            "states": {
                "default": "#FF8A3D",
                "hover": "#FFA466",
                "pressed": "#D96E26",
                "disabled": "#3A3F4A",
            },
        },
    },
)
with ui.theme_scope(ember):
    ember_panel = settings_panel("Ember", (435, 40))

##############################################################################
# ``Theme.save`` and ``Theme.load`` store a theme in a JSON file, which makes
# it easy to share. Restore the classic theme so later code is not affected.

ui.set_theme("classic")

##############################################################################
# Add the panels to a scene and show it.

scene = Scene()
scene.add(classic_panel, graphite_panel, ember_panel)

show_manager = ShowManager(scene=scene, size=(640, 300), title="FURY UI Themes Example")
show_manager.start()
