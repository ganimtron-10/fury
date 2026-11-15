"""
================
UI Theme Gallery
================

This example shows the same small settings panel in each built-in theme:
Graphite and Graphite Light, Hairline (square, 1 pixel lines, a translucent
panel), Soft Pill (round panels and pill shaped controls), Frost (a light
translucent panel for light scenes) and Ember (Graphite with a warm accent).
``ui.theme_names()`` lists every built-in theme, including ``classic``, the
original look of FURY widgets.

Density is an option of any theme rather than a theme of its own. The last
panel uses Graphite with ``density="compact"``, which scales font sizes,
padding, gaps, corner radii and slider sizes down.
"""

##############################################################################
# First, a bunch of imports.

from fury import ui, window

print("Built-in themes:", ", ".join(ui.theme_names()))

##############################################################################
# The settings panel. Colors, fonts, corner radii, borders, shadows, slider
# shapes and the spacing between the rows come from the theme in use. Sizes
# given explicitly, such as the button size, are kept as they are, so the
# compact panel passes smaller buttons itself.

PANEL_WIDTH = 240


def settings_panel(name, position, *, button_height=28):
    """
    Build a settings panel with a title, toggles, sliders and buttons.

    Parameters
    ----------
    name : str
        Name shown next to the panel title.
    position : tuple of int
        Position of the panel's top-left corner.
    button_height : int, optional
        Height of the buttons, in pixels.

    Returns
    -------
    Panel2D
        The panel with its elements.
    """
    panel = ui.Panel2D(size=(PANEL_WIDTH, 100), position=position)
    panel.set_layout(
        direction=ui.Direction.COLUMN,
        height=ui.fit(),
        align_cross=ui.Align.STRETCH,
    )
    inner_width = PANEL_WIDTH - 24

    title = ui.TextBlock2D(text="Rendering", font_weight=600, dynamic_bbox=True)
    subtitle = ui.TextBlock2D(text=name, font_size=11, muted=True, dynamic_bbox=True)
    header = ui.Row(align_main=ui.Align.SPACE_BETWEEN, align_cross=ui.Align.CENTER)
    header.add(title)
    header.add(subtitle)
    panel.add(header)

    panel.add(ui.Checkbox(["Shading", "Show axes"], checked_labels=["Shading"], gap=8))
    panel.add(
        ui.LineSlider2D(
            length=inner_width,
            min_value=0,
            max_value=1,
            initial_value=0.75,
            label="Opacity",
            text_template="{value:.2f}",
        )
    )
    panel.add(
        ui.RadioButton(
            ["Axial", "Cor", "Sag"],
            checked_labels=["Axial"],
            orientation="horizontal",
            gap=16,
        )
    )
    panel.add(ui.TextBox2D(30, 1, text="viridis"))

    buttons = ui.Row(gap=8, align_main=ui.Align.END)
    size = (72, button_height)
    buttons.add(ui.TextButton2D("Reset", size=size, variant="secondary"))
    buttons.add(ui.TextButton2D("Apply", size=size))
    panel.add(buttons)
    return panel


##############################################################################
# ``theme_scope`` builds each panel in its own theme. ``ui.set_theme("frost")``
# would instead make Frost the theme of every element created afterwards and
# restyle the existing elements that follow the global theme.

COLUMN = 270
TOP, BOTTOM = 30, 310

panels = []
for column, (theme, label) in enumerate(
    [
        ("graphite", "Graphite"),
        ("hairline", "Hairline"),
        ("soft_pill", "Soft Pill"),
        ("ember", "Ember"),
    ]
):
    with ui.theme_scope(theme):
        panels.append(settings_panel(label, (15 + column * COLUMN, TOP)))

##############################################################################
# Compact is a density, not a theme. ``derive(density="compact")`` keeps the
# colors and shapes of a theme and scales its metrics; ``theme_scope`` takes
# the same ``density`` argument. To change the density of the global theme,
# and restyle the existing elements, use ``ui.set_theme(density="compact")``.

with ui.theme_scope("graphite", density="compact"):
    panels.append(settings_panel("Compact", (15, BOTTOM), button_height=22))

##############################################################################
# The light themes are shown over a light part of the scene. Two colored
# disks lie behind the Frost panel, which lets the scene show through.

with ui.theme_scope("graphite_light"):
    panels.append(settings_panel("Graphite Light", (15 + COLUMN, BOTTOM)))
with ui.theme_scope("frost"):
    panels.append(settings_panel("Frost", (15 + 2 * COLUMN, BOTTOM)))

light_side = ui.Rectangle2D(size=(2 * COLUMN + 20, 300), color="#E6E9EE")
light_side.set_position((COLUMN - 5, BOTTOM - 20))
light_side.z_order = -2
blue_disk = ui.Disk2D(26, center=(3 * COLUMN - 70, BOTTOM + 60), color="#1D4ED8")
orange_disk = ui.Disk2D(15, center=(3 * COLUMN - 47, BOTTOM + 150), color="#EA580C")
blue_disk.z_order = orange_disk.z_order = -1
backdrop = [light_side, blue_disk, orange_disk]

##############################################################################
# Add everything to a dark scene and show it.

scene = window.Scene(background=(0.08, 0.1, 0.13, 1))
scene.add(*backdrop, *panels)

show_manager = window.ShowManager(
    scene=scene, size=(4 * COLUMN + 10, 600), title="FURY UI Theme Gallery"
)
show_manager.start()
