"""Animated demos of the FURY UI layout engine.

Open one in a window (it loops until you close it)::

    python layout_demos.py grow

Record GIFs with FURY's own offscreen capture::

    python layout_demos.py --record            # all demos
    python layout_demos.py --record wrap       # one demo

Every frame only changes layout properties. The positions and sizes on screen
all come from the solver.
"""

import math
import os
import sys

RECORD = "--record" in sys.argv
if RECORD:
    # FURY reads these in ShowManager.start() and writes <title>.gif.
    os.environ["FURY_OFFSCREEN"] = "1"
    os.environ["FURY_RECORD_ANIMATION"] = "1"

from fury import ui, window  # noqa: E402

BLUE = (0.25, 0.55, 0.9)
ORANGE = (0.9, 0.55, 0.25)
GREEN = (0.35, 0.75, 0.5)
PINK = (0.8, 0.4, 0.7)
YELLOW = (0.85, 0.7, 0.3)
PANEL = (0.14, 0.15, 0.2)
DARK = (0.08, 0.09, 0.12)
COLORS = [BLUE, ORANGE, GREEN, PINK, YELLOW]


def wave(frame, frames):
    """Return a smooth 0 to 1 and back to 0 value over ``frames`` frames."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * frame / frames)


def lerp(start, end, amount):
    """Return the whole number ``amount`` of the way from start to end."""
    return round(start + (end - start) * amount)


def box(text, color=BLUE, size=(100, 60), **layout):
    """Create a coloured panel with centered text."""
    panel = ui.Panel2D(size, color=color, opacity=1)
    layout.setdefault("width", ui.fixed(size[0]))
    layout.setdefault("height", ui.fixed(size[1]))
    panel.set_layout(align_main=ui.Align.CENTER, align_cross=ui.Align.CENTER, **layout)
    panel.label = ui.TextBlock2D(text=text, font_size=16, dynamic_bbox=True)
    panel.add(panel.label)
    return panel


def caption(text=""):
    """Create the line of text that explains the current frame."""
    return ui.TextBlock2D(
        text=text, font_size=19, dynamic_bbox=True, color=(0.75, 0.87, 1.0)
    )


def page(title, width, height, *children):
    """Create a window-sized column holding a heading and the children."""
    root = ui.Panel2D((width, height), position=(0, 0), color=DARK, opacity=1)
    root.set_layout(
        direction=ui.Direction.COLUMN,
        width=ui.fixed(width),
        height=ui.fixed(height),
        padding=18,
        gap=12,
        align_cross=ui.Align.STRETCH,
    )
    root.add(ui.TextBlock2D(text=title, font_size=24, dynamic_bbox=True))
    for child in children:
        root.add(child)
    return root


def responsive():
    """Show a whole page following the size of its container."""
    frames, width, height = 160, 820, 480
    root = ui.Panel2D((width, height), position=(0, 0), color=DARK, opacity=1)
    root.set_layout(direction=ui.Direction.COLUMN, align_cross=ui.Align.STRETCH)

    header = ui.Panel2D((100, 54), color=PANEL, opacity=1)
    header.set_layout(
        width=ui.grow(), align_cross=ui.Align.CENTER, padding=(0, 16), gap=10
    )
    header.add(
        ui.TextBlock2D(text="Responsive page", font_size=22, dynamic_bbox=True),
        width=ui.grow(),
    )
    header.add(ui.TextButton2D(label="Save", size=(80, 32)))
    root.add(header)

    body = ui.Row(height=ui.grow(), align_cross=ui.Align.STRETCH)
    sidebar = ui.Column(
        width=ui.fixed(170), padding=14, gap=10, align_cross=ui.Align.STRETCH
    )
    for name in ("Render", "Camera", "Export"):
        sidebar.add(ui.TextButton2D(label=name, size=(140, 34)))
    body.add(sidebar)

    content = ui.Column(
        width=ui.grow(), padding=16, gap=12, align_cross=ui.Align.STRETCH
    )
    content.add(
        ui.TextBlock2D(
            text="This paragraph wraps as its column gets narrower, and "
            "everything under it moves out of the way without any "
            "pixel arithmetic in the code.",
            font_size=17,
            dynamic_bbox=True,
            bg_color=(0.18, 0.19, 0.26),
        ).set_layout(padding=10)
    )
    cards = ui.Row(gap=10, height=ui.grow(), align_cross=ui.Align.STRETCH)
    for index in range(3):
        cards.add(box(f"grow {index + 1}", COLORS[index], width=ui.grow(), height=ui.grow()))
    content.add(cards)
    body.add(content)
    root.add(body)

    def step(frame):
        amount = wave(frame, frames)
        root.set_layout(
            width=ui.fixed(lerp(width, 470, amount)),
            height=ui.fixed(lerp(height, 400, amount)),
        )

    return root, (width, height), frames, step


def grow():
    """Show leftover space shared by weight, with a limit on one box."""
    frames, width, height = 180, 760, 330
    side = box("fixed(140)", ORANGE, (140, 70))
    one = box("grow(1)", BLUE, (20, 70), width=ui.grow(weight=1))
    many = box("grow(1)", GREEN, (20, 70), width=ui.grow(weight=1))
    row = ui.Row(side, one, many, gap=8, align_cross=ui.Align.STRETCH)

    capped = box("grow(max=200)", PINK, (20, 70), width=ui.grow(max=200))
    rest = box("grow()", YELLOW, (20, 70), width=ui.grow())
    limited = ui.Row(capped, rest, gap=8, align_cross=ui.Align.STRETCH)

    root = page("Grow shares leftover space", width, height, row, limited)

    def step(frame):
        weight = 1 + 4 * wave(frame, frames)
        many.set_layout(width=ui.grow(weight=weight))
        many.label.message = f"grow({weight:.1f})"
        root.set_layout(width=ui.fixed(lerp(width, 520, wave(frame, frames))))

    return root, (width, height), frames, step


def shrink():
    """Show text boxes giving back overflow according to ``shrink``."""
    frames, width, height = 180, 860, 300

    def paragraph(name, color, amount):
        panel = ui.Panel2D((10, 10), color=color, opacity=1)
        # Without a minimum a box can shrink narrower than its longest word.
        panel.set_layout(
            width=ui.fit(min=110), height=ui.fit(), padding=8, shrink=amount
        )
        panel.add(
            ui.TextBlock2D(
                text=f"shrink={amount}: {name} long line of text",
                font_size=16,
                dynamic_bbox=True,
            )
        )
        return panel

    row = ui.Row(
        paragraph("a stiff", ORANGE, 0),
        paragraph("a normal", BLUE, 1),
        paragraph("an eager", GREEN, 3),
        gap=8,
        width=ui.grow(),
    )
    root = page("Shrink gives space back, and text wraps", width, height, row)

    def step(frame):
        root.set_layout(width=ui.fixed(lerp(width, 480, wave(frame, frames))))

    return root, (width, height), frames, step


def align():
    """Cycle through main and cross axis alignment."""
    hold, width, height = 34, 760, 400
    combos = [
        ("START", "START"),
        ("CENTER", "START"),
        ("END", "START"),
        ("SPACE_BETWEEN", "START"),
        ("SPACE_BETWEEN", "CENTER"),
        ("SPACE_BETWEEN", "END"),
        ("CENTER", "CENTER"),
        ("START", "STRETCH"),
    ]
    frames = hold * len(combos)
    area = ui.Panel2D((100, 100), color=PANEL, opacity=1)
    area.set_layout(width=ui.grow(), height=ui.grow(), padding=14, gap=10)
    # A fit height with a minimum keeps the box sizes but allows STRETCH.
    for index, (w, h) in enumerate([(100, 60), (140, 90), (80, 120)]):
        area.add(box(str(index + 1), COLORS[index], (w, h), height=ui.fit(min=h)))
    text = caption()
    root = page("Alignment", width, height, text, area)

    def step(frame):
        main, cross = combos[frame // hold]
        area.set_layout(
            align_main=ui.Align[main],
            align_cross=ui.Align[cross],
        )
        text.message = f"align_main={main}   align_cross={cross}"

    return root, (width, height), frames, step


def spacing():
    """Animate padding, then gap, then margin."""
    phase, width, height = 80, 760, 360
    frames = phase * 3
    area = ui.Panel2D((100, 100), color=PANEL, opacity=1)
    area.set_layout(width=ui.grow(), height=ui.grow(), gap=8, padding=8)
    boxes = [box(str(i + 1), COLORS[i], (110, 80)) for i in range(4)]
    for item in boxes:
        area.add(item)
    text = caption()
    root = page("Padding, gap and margin", width, height, text, area)

    def step(frame):
        kind = frame // phase
        value = lerp(0, 40, wave(frame % phase, phase))
        padding, gap, margin = 8, 8, 0
        if kind == 0:
            padding = 8 + value
            text.message = f"container: padding={padding}"
        elif kind == 1:
            gap = 8 + value
            text.message = f"container: gap={gap}"
        else:
            margin = value
            text.message = f"box 2 only: margin={margin}"
        area.set_layout(padding=padding, gap=gap)
        boxes[1].set_layout(margin=margin)

    return root, (width, height), frames, step


def wrap():
    """Show a row of chips flowing onto new lines."""
    frames, width, height = 180, 760, 400
    area = ui.Panel2D((100, 100), color=PANEL, opacity=1)
    area.set_layout(
        width=ui.grow(), height=ui.grow(), padding=12, gap=8, wrap=True
    )
    widths = [110, 70, 150, 90, 130, 60, 120, 100, 80, 140]
    for index, chip_width in enumerate(widths):
        area.add(
            box(str(index + 1), COLORS[index % len(COLORS)], (chip_width, 44)),
        )
    root = page("Row with wrap=True", width, height, area)

    def step(frame):
        root.set_layout(width=ui.fixed(lerp(width, 300, wave(frame, frames))))

    return root, (width, height), frames, step


def text_reflow():
    """Show neighbours and centering following text as it changes."""
    frames, width, height = 170, 760, 260
    words = ["OK", "Loading", "A much longer status message", "Done", "Rendering"]
    hold = frames // len(words)
    label = ui.TextBlock2D(
        text="", font_size=20, dynamic_bbox=True, bg_color=BLUE
    ).set_layout(padding=(8, 12))
    follower = box("I follow the text", ORANGE, (170, 42))
    row = ui.Row(label, follower, gap=10, align_cross=ui.Align.CENTER)
    centered = ui.TextBlock2D(text="", font_size=20, dynamic_bbox=True)
    middle = ui.Panel2D((100, 60), color=PANEL, opacity=1)
    middle.set_layout(
        width=ui.grow(),
        height=ui.grow(),
        align_main=ui.Align.CENTER,
        align_cross=ui.Align.CENTER,
    )
    middle.add(centered)
    root = page("Text is measured by the layout", width, height, row, middle)

    def step(frame):
        word = words[frame // hold % len(words)]
        label.message = word
        centered.message = f"Centered: {word}"

    return root, (width, height), frames, step


DEMOS = {
    function.__name__: function
    for function in (responsive, grow, shrink, align, spacing, wrap, text_reflow)
}


def run(name):
    """Show or record one demo."""
    root, size, frames, step = DEMOS[name]()
    scene = window.Scene(background=DARK)
    scene.add(root)
    show_manager = window.ShowManager(
        scene=scene,
        size=size,
        pixel_ratio=1,
        title=f"layout_{name}",
        window_type="offscreen" if RECORD else "default",
    )
    state = {"frame": 0}

    def tick():
        step(state["frame"] % frames)
        state["frame"] += 1

    show_manager.register_callback(tick, 1 / 30, True, "demo")
    if RECORD:
        os.environ["FURY_OFFSCREEN_MAX_FRAMES"] = str(frames)
    show_manager.start()


if __name__ == "__main__":
    names = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if RECORD:
        for demo in names or DEMOS:
            run(demo)
    elif len(names) == 1 and names[0] in DEMOS:
        run(names[0])
    else:
        sys.exit(f"Choose one of: {', '.join(DEMOS)}")
