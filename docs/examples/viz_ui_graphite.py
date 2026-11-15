"""
===============
Graphite Themes
===============

This example shows the built-in Graphite themes. Graphite is a neutral,
monochrome design: flat panels with a hairline border, rounded corners and a
soft shadow, semibold titles and a single accent color, so nothing competes
with the data in the scene. Graphite Light is the same design with light
colors.

The same settings panel is built twice, once in each theme, over a dark and a
light part of the scene.
"""

##############################################################################
# First, a bunch of imports.

from fury import ui, window

##############################################################################
# The settings panel. Only sizes and texts are given: colors, fonts, corner
# radii, borders, shadows and the spacing between the rows come from the
# theme in use. The panel stacks its rows with the layout engine, using the
# theme padding and gap.


def settings_panel(position):
    """
    Build a settings panel with a title, toggles, sliders and buttons.

    Parameters
    ----------
    position : tuple of int
        Position of the panel's top-left corner.

    Returns
    -------
    Panel2D
        The panel with its elements.
    """
    panel = ui.Panel2D(size=(260, 100), position=position)
    panel.set_layout(
        direction=ui.Direction.COLUMN,
        height=ui.fit(),
        align_cross=ui.Align.STRETCH,
    )

    title = ui.TextBlock2D(text="Rendering", font_weight=600, dynamic_bbox=True)
    subtitle = ui.TextBlock2D(
        text="Volume · T1", font_size=11, muted=True, dynamic_bbox=True
    )
    header = ui.Row(align_main=ui.Align.SPACE_BETWEEN, align_cross=ui.Align.CENTER)
    header.add(title)
    header.add(subtitle)
    panel.add(header)

    panel.add(ui.Checkbox(["Shading", "Show axes"], checked_labels=["Shading"], gap=8))

    panel.add(
        ui.LineSlider2D(
            length=236,
            min_value=0,
            max_value=1,
            initial_value=0.75,
            label="Opacity",
            text_template="{value:.2f}",
        )
    )
    panel.add(
        ui.LineSlider2D(
            length=236,
            min_value=0,
            max_value=255,
            initial_value=96,
            label="Threshold",
            text_template="{value:.0f}",
        )
    )

    panel.add(
        ui.RadioButton(
            ["Axial", "Coronal", "Sagittal"],
            checked_labels=["Axial"],
            orientation="horizontal",
            gap=16,
        )
    )

    panel.add(ui.TextBox2D(36, 1, text="T1 weighted"))

    buttons = ui.Row(gap=8, align_main=ui.Align.END)
    buttons.add(ui.TextButton2D("Reset", size=(72, 28), variant="secondary"))
    buttons.add(ui.TextButton2D("Apply", size=(72, 28)))
    panel.add(buttons)
    return panel


##############################################################################
# ``theme_scope`` builds each panel in its own theme. Use
# ``ui.set_theme("graphite")`` instead to make Graphite the theme of every
# element created afterwards; it also restyles existing elements that follow
# the global theme.

with ui.theme_scope("graphite"):
    dark_panel = settings_panel((30, 60))

with ui.theme_scope("graphite_light"):
    light_panel = settings_panel((350, 60))

##############################################################################
# A light rectangle covers the right half of the dark scene, so each panel is
# shown over a background that suits it.

light_side = ui.Rectangle2D(size=(320, 480), color="#E6E9EE")
light_side.z_order = -1
light_side.set_position((320, 0))

scene = window.Scene(background=(0.08, 0.1, 0.13, 1))
scene.add(light_side, dark_panel, light_panel)

show_manager = window.ShowManager(
    scene=scene, size=(640, 480), title="FURY Graphite Themes Example"
)
show_manager.start()
