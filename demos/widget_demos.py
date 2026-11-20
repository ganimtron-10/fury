"""Layout demos built from real FURY widgets.

Open one in a window, then use the widgets and resize the window::

    python widget_demos.py app

Record GIFs with FURY's own offscreen capture::

    python widget_demos.py --record            # all demos
    python widget_demos.py --record gallery    # one demo

While recording, each demo plays a short script: the root width animates, or
the script clicks widgets through real picking.
"""

import math
import os
from pathlib import Path
import sys
import tempfile

RECORD = "--record" in sys.argv
if RECORD:
    # FURY reads these in ShowManager.start() and writes <title>.gif.
    os.environ["FURY_OFFSCREEN"] = "1"
    os.environ["FURY_RECORD_ANIMATION"] = "1"

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fury import actor, ui, window  # noqa: E402
from fury.lib import EventType  # noqa: E402

# Page color behind the Graphite panels, as in viz_ui_graphite.py.
PAGE = (0.08, 0.1, 0.13)


def wave(frame, frames):
    """Return a smooth 0 to 1 and back to 0 value over ``frames`` frames."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * frame / frames)


def lerp(start, end, amount):
    """Return the whole number ``amount`` of the way from start to end."""
    return round(start + (end - start) * amount)


def heading(text, size=15):
    """Create a semibold heading whose size follows its text."""
    return ui.TextBlock2D(text=text, font_size=size, font_weight=600, dynamic_bbox=True)


def field(label, widget):
    """Create a form row with a growing label and a widget on the right."""
    row = ui.Row(align_cross=ui.Align.CENTER, gap=10)
    row.add(ui.TextBlock2D(text=label, dynamic_bbox=True), width=ui.grow())
    row.add(widget)
    return row


def card_panel(width=None, **layout):
    """Create a panel that stacks its children vertically."""
    panel = ui.Panel2D((width or 100, 100))
    layout.setdefault("height", ui.grow())
    layout.setdefault("width", ui.fixed(width) if width else ui.grow())
    panel.set_layout(
        direction=ui.Direction.COLUMN, align_cross=ui.Align.STRETCH, **layout
    )
    return panel


def thumbnails(count):
    """Write small flat images for the cards and return their paths."""
    folder = Path(tempfile.mkdtemp(prefix="fury_layout_demo"))
    y, x = np.mgrid[0:120, 0:200]
    paths = []
    for index in range(count):
        hue = index * 2 * math.pi / count
        base = 0.45 + 0.35 * np.sin(hue + np.array([0.0, 2.1, 4.2]))
        image = np.ones((120, 200, 3)) * base
        disc = (x - 100) ** 2 + (y - 60) ** 2 < 38**2
        image[disc] = np.clip(base + 0.3, 0, 1)
        path = folder / f"thumb_{index}.png"
        Image.fromarray((image * 255).astype(np.uint8)).save(path)
        paths.append(str(path))
    return paths


class Demo:
    """What a demo needs in order to be shown or recorded."""

    def __init__(self, root, *, size=(900, 560), frames=150, step=None, clicks=None):
        self.root = root
        self.size = size
        self.frames = frames
        self.step = step
        self.clicks = clicks
        self.extras = []
        self.on_frame = None


def shrink_step(root, width, height, narrow, short=None):
    """Return a step that animates the root between two sizes."""

    def step(frame, frames):
        amount = wave(frame, frames)
        root.set_layout(
            width=ui.fixed(lerp(width, narrow, amount)),
            height=ui.fixed(lerp(height, short or height, amount)),
        )

    return step


def page_root():
    """Create the root row, which follows the window."""
    return ui.Row(
        position=(0, 0),
        width=ui.grow(),
        height=ui.grow(),
        padding=16,
        gap=16,
        align_cross=ui.Align.STRETCH,
    )


def settings():
    """Show a settings form next to a preview that takes the rest."""
    width, height = 900, 560
    root = page_root()

    form = card_panel(340)
    form.add(heading("Render settings", 16))
    form.add(field("Samples", ui.SpinBox(size=(120, 30), initial_val=4)))
    form.add(
        ui.LineSlider2D(
            length=300,
            initial_value=40,
            label="Exposure",
            text_template="{value:.0f}%",
        )
    )
    form.add(ui.Checkbox(labels=["Shadows", "Reflections", "Bloom"], gap=6))
    form.add(heading("Quality", 13))
    form.add(
        ui.RadioButton(
            labels=["Draft", "Balanced", "Final"],
            checked_labels=["Balanced"],
            orientation="horizontal",
            gap=14,
        )
    )
    form.add(ui.Row(height=ui.grow()))
    buttons = ui.Row(gap=8, align_main=ui.Align.END)
    buttons.add(ui.TextButton2D(label="Reset", size=(72, 28), variant="secondary"))
    buttons.add(ui.TextButton2D(label="Apply", size=(72, 28)))
    form.add(buttons)
    root.add(form)

    preview = card_panel()
    preview.set_layout(align_main=ui.Align.CENTER, align_cross=ui.Align.CENTER)
    preview.add(heading("Preview", 18))
    preview.add(
        ui.TextBlock2D(
            text="This area takes all the space the form leaves over.",
            muted=True,
            dynamic_bbox=True,
        )
    )
    root.add(preview)

    return Demo(
        root,
        size=(width, height),
        step=shrink_step(root, width, height, 560, 470),
        frames=170,
    )


def viewer():
    """Float a layout of widgets over a live 3D scene."""
    width, height = 960, 600
    root = ui.Column(
        position=(0, 0),
        width=ui.grow(),
        height=ui.grow(),
        padding=12,
        gap=12,
        align_cross=ui.Align.STRETCH,
    )

    toolbar = ui.Panel2D((100, 52))
    toolbar.set_layout(
        width=ui.grow(), align_cross=ui.Align.CENTER, padding=(0, 12), gap=8
    )
    toolbar.add(heading("Molecule viewer", 16), width=ui.grow())
    for name in ("Open", "Export"):
        toolbar.add(ui.TextButton2D(label=name, size=(72, 28), variant="secondary"))
    toolbar.add(ui.TextButton2D(label="Help", size=(72, 28)))
    root.add(toolbar)

    middle = ui.Row(height=ui.grow(), align_cross=ui.Align.STRETCH)
    left = card_panel(250)
    left.add(heading("Display"))
    left.add(field("Atoms", ui.SpinBox(size=(110, 30), initial_val=64)))
    left.add(ui.Checkbox(labels=["Bonds", "Labels", "Axes"], gap=6))
    left.add(ui.RadioButton(labels=["Ball", "Stick"], checked_labels=["Ball"], gap=6))
    middle.add(left)
    middle.add(ui.Row(width=ui.grow()))
    right = card_panel(220)
    right.add(heading("Chains"))
    right.add(
        ui.ListBox2D(
            values=[f"Chain {letter}" for letter in "ABCDEFGHIJKLMNOPQRST"],
            size=(190, 440),
        ).set_layout(width=ui.grow())
    )
    middle.add(right)
    root.add(middle)

    demo = Demo(
        root,
        size=(width, height),
        step=shrink_step(root, width, height, 680),
        frames=110,
    )
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(64, 3)) * 18
    spheres = actor.sphere(
        centers=centers,
        colors=rng.uniform(0.3, 1, size=(64, 3)),
        radii=4,
    )
    demo.extras.append(spheres)

    def spin(frame):
        spheres.local.euler_y = frame * 0.03
        spheres.local.euler_x = frame * 0.012

    demo.on_frame = spin
    return demo


def toolbar():
    """Cycle the alignment and spacing of a row of real controls."""
    width, height = 900, 400
    hold = 36
    combos = [
        ("START", 8),
        ("CENTER", 8),
        ("END", 8),
        ("SPACE_BETWEEN", 8),
        ("START", 28),
        ("CENTER", 28),
    ]
    root = ui.Column(
        position=(0, 0),
        width=ui.grow(),
        height=ui.grow(),
        padding=16,
        gap=14,
        align_cross=ui.Align.STRETCH,
    )
    note = ui.TextBlock2D(text="", muted=True, dynamic_bbox=True)
    root.add(heading("A toolbar of real controls", 18))
    root.add(note)

    bar = ui.Panel2D((100, 60))
    bar.set_layout(width=ui.grow(), align_cross=ui.Align.CENTER)
    bar.add(ui.TextButton2D(label="Run", size=(72, 28)))
    bar.add(ui.SpinBox(size=(110, 30), initial_val=3))
    bar.add(ui.TextButton2D(label="Stop", size=(72, 28), variant="secondary"))
    bar.add(
        ui.Checkbox(labels=["Loop"], orientation="horizontal"),
    )
    root.add(bar)

    form = card_panel()
    form.add(field("Iterations", ui.SpinBox(size=(120, 30), initial_val=10)))
    form.add(field("Step size", ui.SpinBox(size=(120, 30), initial_val=2)))
    form.add(
        field(
            "Method",
            ui.RadioButton(
                labels=["Fast", "Exact"],
                checked_labels=["Fast"],
                orientation="horizontal",
                gap=14,
            ),
        )
    )
    root.add(form)

    def step(frame, frames):
        main, gap = combos[frame // hold]
        bar.set_layout(align_main=ui.Align[main], gap=gap)
        note.message = f"toolbar: align_main={main}, gap={gap}"

    return Demo(root, size=(width, height), frames=hold * len(combos), step=step)


def gallery():
    """Wrap a row of image cards as the space gets narrower."""
    width, height = 940, 680
    root = ui.Column(
        position=(0, 0),
        width=ui.grow(),
        height=ui.grow(),
        padding=16,
        gap=12,
        align_cross=ui.Align.STRETCH,
    )
    root.add(heading("Card gallery with wrap", 18))
    area = ui.Panel2D((100, 100))
    area.set_layout(
        width=ui.grow(), height=ui.grow(), padding=12, gap=12, wrap=True
    )
    names = ["Brain", "Cells", "Galaxy", "Protein", "Fibers", "Flow", "Bones", "Stars"]
    for name, path in zip(names, thumbnails(len(names)), strict=True):
        area.add(
            ui.Card2D(
                image_path=path,
                title_text=name,
                body_text="Sample data",
                size=(160, 180),
            )
        )
    root.add(area)
    return Demo(
        root,
        size=(width, height),
        step=shrink_step(root, width, height, 570),
        frames=180,
    )


def app():
    """Use a small application: every click below is a real click."""
    width, height = 900, 580
    root = page_root()

    side = card_panel(230)
    side.add(heading("Datasets"))
    datasets = ui.ListBox2D(
        values=["Brain", "Cells", "Galaxy", "Protein", "Fibers", "Flow"],
        size=(200, 300),
    )
    datasets.set_layout(width=ui.grow(), height=ui.grow())
    side.add(datasets)
    root.add(side)

    main = card_panel()
    title = heading("Select a dataset", 18)
    main.add(title)

    tabs = ui.TabUI(
        size=(400, 300), tab_titles=["Render", "About"], startup_tab_id=0
    )
    tabs.set_layout(width=ui.grow(), height=ui.grow())
    main.add(tabs)
    root.add(main)

    render = tabs.tabs[0].content_panel
    render.set_layout(
        direction=ui.Direction.COLUMN,
        padding=14,
        gap=12,
        align_cross=ui.Align.STRETCH,
    )
    status = heading("Samples: 1", 13)
    spin = ui.SpinBox(size=(120, 30), initial_val=1)
    spin.on_change = lambda box: setattr(status, "message", f"Samples: {box.value}")
    checks = ui.Checkbox(labels=["Shadows", "Bloom"], gap=6)
    combo = ui.ComboBox2D(
        items=["Low", "Medium", "High"], placeholder="Mode", size=(240, 150)
    )
    render.add(status)
    render.add(field("Samples", spin))
    render.add(checks)
    # A stretched ComboBox2D also stretches its arrow icon, so keep its width.
    render.add(combo, width=ui.fixed(240))

    about = tabs.tabs[1].content_panel
    about.set_layout(direction=ui.Direction.COLUMN, padding=14, gap=8)
    about.add(
        ui.TextBlock2D(
            text="Every widget here is positioned by the layout engine, "
            "including this text, which wraps to the width of the tab.",
            dynamic_bbox=True,
        ).set_layout(width=ui.grow())
    )

    # Children added to a hidden tab are visible until the tab is hidden again.
    about.set_visibility(False)

    def pick_dataset():
        title.message = f"Dataset: {datasets.selected[0]}"

    datasets.on_change = pick_dataset

    clicks = [
        lambda: datasets.slots[2],
        lambda: spin.increment_button,
        lambda: spin.increment_button,
        lambda: spin.increment_button,
        lambda: checks.options["Bloom"][0],
        lambda: combo.drop_down_button,
        lambda: combo.drop_down_menu.slots[2],
        lambda: datasets.slots[4],
        lambda: tabs.tabs[1].panel,
        lambda: tabs.tabs[0].panel,
    ]
    return Demo(root, size=(width, height), clicks=clicks)


DEMOS = {
    function.__name__: function for function in (settings, viewer, toolbar, gallery, app)
}
MOVE, HOLD = 10, 26


def center_of(element):
    """Return the on-screen center of an element."""
    x, y = element.get_position()
    width, height = element.size
    return int(x + width / 2), int(y + height / 2)


def click(show_manager, x, y):
    """Press and release the left button on whatever is drawn at a point."""
    target = show_manager.renderer.get_pick_info((x, y))["world_object"]
    if target is None:
        raise RuntimeError(f"Nothing is drawn at ({x}, {y})")
    for kind in (EventType.POINTER_DOWN, EventType.POINTER_UP):
        target.handle_event(
            window.PointerEvent(x=x, y=y, type=kind, button=1, target=target)
        )


def run(name):
    """Show or record one demo."""
    demo = DEMOS[name]()
    scene = window.Scene(background=PAGE)
    scene.add(demo.root)
    for extra in demo.extras:
        scene.add(extra)

    frames = demo.frames
    cursor = None
    if RECORD and demo.clicks:
        frames = len(demo.clicks) * (MOVE + HOLD)
        cursor = ui.Rectangle2D(size=(12, 12), color=(1, 0.9, 0.2), position=(450, 300))
        cursor.z_order = 90
        scene.add(cursor)

    show_manager = window.ShowManager(
        scene=scene,
        size=demo.size,
        pixel_ratio=1 if RECORD else None,
        title=f"widgets_{name}",
        window_type="offscreen" if RECORD else "default",
    )
    state = {"frame": 0, "from": (450, 300)}

    def tick():
        frame = state["frame"]
        state["frame"] += 1
        if demo.on_frame:
            demo.on_frame(frame)
        if not RECORD:
            return
        if demo.step:
            demo.step(frame % frames, frames)
        if cursor is not None:
            index, phase = divmod(frame, MOVE + HOLD)
            if index >= len(demo.clicks):
                return
            x, y = center_of(demo.clicks[index]())
            start = state["from"]
            if phase < MOVE:
                amount = (phase + 1) / MOVE
                cursor.set_position(
                    (
                        start[0] + (x + 14 - start[0]) * amount,
                        start[1] + (y + 6 - start[1]) * amount,
                    )
                )
            elif phase == MOVE:
                click(show_manager, x, y)
                state["from"] = (x + 14, y + 6)

    if RECORD or demo.on_frame:
        show_manager.register_callback(tick, 1 / 30, True, "demo")
    if RECORD:
        os.environ["FURY_OFFSCREEN_MAX_FRAMES"] = str(frames)
    show_manager.start()


if __name__ == "__main__":
    names = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if RECORD:
        for demo_name in names or DEMOS:
            run(demo_name)
    elif len(names) == 1 and names[0] in DEMOS:
        run(names[0])
    else:
        sys.exit(f"Choose one of: {', '.join(DEMOS)}")
