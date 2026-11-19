"""Interactive playground for the FURY UI layout engine.

Run it from a checkout of the ``ui-layout/integration`` branch::

    python playground.py

Click a button on the left to cycle that setting. The boxes on the right are
solved again on the next frame, and the text at the bottom of the sidebar shows
the code for the current settings. Resize the window as well.
"""

from fury import ui, window

BOX_COLORS = [
    (0.25, 0.55, 0.9),
    (0.9, 0.55, 0.25),
    (0.35, 0.75, 0.5),
    (0.8, 0.4, 0.7),
    (0.85, 0.75, 0.3),
    (0.45, 0.45, 0.9),
    (0.9, 0.4, 0.4),
    (0.4, 0.75, 0.8),
]

DIRECTIONS = {"row": ui.Direction.ROW, "column": ui.Direction.COLUMN}
MAIN = {
    "start": ui.Align.START,
    "center": ui.Align.CENTER,
    "end": ui.Align.END,
    "space_between": ui.Align.SPACE_BETWEEN,
}
CROSS = {
    "start": ui.Align.START,
    "center": ui.Align.CENTER,
    "end": ui.Align.END,
    "stretch": ui.Align.STRETCH,
}
SIZES = {
    "fixed(110)": ui.fixed(110),
    "grow(1)": ui.grow(weight=1),
    "grow(3)": ui.grow(weight=3),
    "percent(0.5)": ui.percent(0.5),
    "grow(max=160)": ui.grow(max=160),
}

# Each setting is a name and its values. The first value is the starting one.
# The engine only supports wrap on rows, so wrap is ignored for columns.
SETTINGS = {
    "direction": list(DIRECTIONS),
    "align_main": list(MAIN),
    "align_cross": list(CROSS),
    "wrap": ["off", "on"],
    "gap": [8, 0, 16, 32],
    "padding": [16, 0, 32, 64],
    "boxes": [4, 2, 6, 8],
    "box 2": list(SIZES),
}
current = dict.fromkeys(SETTINGS, 0)


def value_of(name):
    """Return the selected value of a setting."""
    return SETTINGS[name][current[name]]


##############################################################################
# The window: a fixed sidebar next to a stage that takes the remaining space.

root = ui.Row(position=(0, 0), width=ui.grow(), height=ui.grow())

sidebar = ui.Panel2D((320, 100), color=(0.14, 0.15, 0.2), opacity=1)
sidebar.set_layout(
    direction=ui.Direction.COLUMN,
    height=ui.grow(),
    padding=14,
    gap=8,
    align_cross=ui.Align.STRETCH,
)
root.add(sidebar)

stage = ui.Panel2D((100, 100), color=(0.08, 0.09, 0.12), opacity=1)
stage.set_layout(width=ui.grow(), height=ui.grow())
root.add(stage)

##############################################################################
# The boxes being laid out. Hidden boxes are taken out of the layout.

boxes = []
for number, color in enumerate(BOX_COLORS, start=1):
    box = ui.Panel2D((110, 70), color=color, opacity=1)
    box.set_layout(
        width=ui.fixed(110),
        height=ui.fixed(70),
        align_main=ui.Align.CENTER,
        align_cross=ui.Align.CENTER,
    )
    box.add(ui.TextBlock2D(text=str(number), font_size=20, dynamic_bbox=True))
    boxes.append(box)

active = []


def apply_settings():
    """Push the selected settings into the layout and update the code text."""
    wanted = value_of("boxes")
    while len(active) < wanted:
        box = boxes[len(active)]
        stage.add(box)
        box.set_visibility(True)
        active.append(box)
    while len(active) > wanted:
        box = active.pop()
        stage.remove_element(box)
        box.set_visibility(False)

    stage.set_layout(
        direction=DIRECTIONS[value_of("direction")],
        align_main=MAIN[value_of("align_main")],
        align_cross=CROSS[value_of("align_cross")],
        wrap=value_of("wrap") == "on" and value_of("direction") == "row",
        gap=value_of("gap"),
        padding=value_of("padding"),
    )
    size = SIZES[value_of("box 2")]
    if value_of("direction") == "row":
        boxes[1].set_layout(width=size, height=ui.fixed(70))
    else:
        boxes[1].set_layout(width=ui.fixed(110), height=size)

    wrapped = value_of("wrap") == "on" and value_of("direction") == "row"
    wrap = ", wrap=True" if wrapped else ""
    code.message = (
        f"{value_of('direction')}, gap={value_of('gap')}, "
        f"padding={value_of('padding')}{wrap}\n"
        f"main={value_of('align_main')}\n"
        f"cross={value_of('align_cross')}\n"
        f"box 2: {value_of('box 2')}"
    )


##############################################################################
# One button per setting. A click moves on to the next value.

buttons = {}


def label_for(name):
    """Return the text shown on the button of a setting."""
    return f"{name}: {value_of(name)}"


def make_button(name):
    """Create the button that cycles one setting."""
    button = ui.TextButton2D(label=label_for(name), size=(292, 34))

    def on_click(_):
        current[name] = (current[name] + 1) % len(SETTINGS[name])
        button.default_label = label_for(name)
        button.update_visual_state()
        apply_settings()

    button.on_clicked = on_click
    return button


sidebar.add(ui.TextBlock2D(text="Layout playground", font_size=22, dynamic_bbox=True))
for setting in SETTINGS:
    buttons[setting] = make_button(setting)
    sidebar.add(buttons[setting])

sidebar.add(ui.Row(height=ui.grow()))

code = ui.TextBlock2D(
    text="", font_size=15, dynamic_bbox=True, color=(0.7, 0.85, 1.0)
)
sidebar.add(code)

apply_settings()

scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(root)
for hidden in boxes[len(active) :]:
    # Keep spare boxes in the scene so they can be shown later.
    scene.add(hidden)
    hidden.set_visibility(False)

show_manager = window.ShowManager(
    scene=scene, size=(1000, 620), title="FURY Layout Playground"
)

if __name__ == "__main__":
    show_manager.start()
