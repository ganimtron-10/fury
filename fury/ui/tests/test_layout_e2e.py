"""End-to-end tests of the layout engine in a real render loop."""

import os
from pathlib import Path
import runpy

import numpy.testing as npt
import pytest

from fury import ui, window
from fury.data import fetch_viz_icons, read_viz_icons
from fury.lib import EventType

EXAMPLES = Path(__file__).parents[3] / "docs" / "examples"
LAYOUT_TUTORIALS = [
    "viz_layout_basics.py",
    "viz_layout_sizing.py",
    "viz_layout_settings_demo.py",
]


@pytest.fixture
def offscreen_run_dir(tmp_path):
    """
    Run inside ``tmp_path`` with ``FURY_OFFSCREEN`` set, then restore both.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty per-test directory that receives the files saved by ``start``.

    Yields
    ------
    pathlib.Path
        The directory the test runs in.
    """
    previous_env = os.environ.get("FURY_OFFSCREEN")
    previous_dir = os.getcwd()
    os.environ["FURY_OFFSCREEN"] = "true"
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous_dir)
        if previous_env is None:
            os.environ.pop("FURY_OFFSCREEN", None)
        else:
            os.environ["FURY_OFFSCREEN"] = previous_env


def _show(*elements, size=(640, 480)):
    """Open an offscreen window showing the elements and draw one frame."""
    scene = window.Scene()
    scene.add(*elements)
    show_m = window.ShowManager(
        scene=scene, size=size, pixel_ratio=1, window_type="offscreen"
    )
    _draw(show_m)
    return show_m


def _draw(show_m):
    """Draw a frame through the real render path."""
    show_m.render()
    show_m.window.draw()


def _center(element):
    """Return the on-screen center of a laid-out element."""
    x, y = element.get_position()
    width, height = element.size
    return int(x + width / 2), int(y + height / 2)


def _click(show_m, x, y):
    """Press and release the left button on whatever is drawn at a point."""
    target = _pick(show_m, x, y)
    _pointer(show_m, EventType.POINTER_DOWN, x, y, target)
    _pointer(show_m, EventType.POINTER_UP, x, y, target)
    return target


def _pick(show_m, x, y):
    """Return the object drawn at a point, failing if there is none."""
    target = show_m.renderer.get_pick_info((x, y))["world_object"]
    assert target is not None, f"Nothing is drawn at ({x}, {y})"
    return target


def _pointer(show_m, event_type, x, y, target):
    """Send one left-button pointer event through the renderer."""
    event = window.PointerEvent(x=x, y=y, type=event_type, button=1, target=target)
    show_m.renderer.dispatch_event(event)


def _resize(show_m, size):
    """Resize the canvas and draw, so the resize event runs its real route."""
    show_m.window.set_logical_size(*size)
    _draw(show_m)


@pytest.mark.skipif(
    not EXAMPLES.is_dir(), reason="docs/examples is only in a source checkout"
)
@pytest.mark.parametrize("example", LAYOUT_TUTORIALS)
def test_layout_tutorials_run(example, offscreen_run_dir):
    """Every layout tutorial runs through ``start`` and saves its frame."""
    namespace = runpy.run_path(str(EXAMPLES / example), run_name="example")
    assert isinstance(namespace["show_manager"], window.ShowManager)
    assert len(ui.UIContext._dirty_layouts) == 0
    saved = list(offscreen_run_dir.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


def test_click_reaches_a_button_placed_by_the_layout():
    """A button is clickable where the layout put it, also after a resize."""
    clicks = []
    button = ui.TextButton2D(label="Go", size=(80, 30))
    button.on_clicked = lambda btn: clicks.append(btn)
    row = ui.Row(width=ui.grow(), align_main=ui.Align.END, padding=10)
    row.add(button)
    show_m = _show(row)

    npt.assert_array_equal(button.get_position(), (550, 10))
    _click(show_m, *_center(button))
    assert len(clicks) == 1

    _resize(show_m, (400, 480))
    npt.assert_array_equal(button.get_position(), (310, 10))
    _click(show_m, *_center(button))
    assert len(clicks) == 2
    show_m.close()


def test_text_change_moves_siblings_on_the_next_frame():
    """Longer text pushes the next element aside without any manual update."""
    label = ui.TextBlock2D(text="a", dynamic_bbox=True)
    follower = ui.Rectangle2D(size=(20, 20))
    row = ui.Row(label, follower, gap=4, position=(10, 10))
    show_m = _show(row)
    before = follower.get_position()[0]

    label.message = "a much longer label"
    assert row._layout_node.dirty
    _draw(show_m)
    assert not row._layout_node.dirty
    assert follower.get_position()[0] > before + 50
    show_m.close()


def test_spinbox_in_a_layout_updates_a_centered_label():
    """A widget event can change text that the layout then re-centers."""
    label = ui.TextBlock2D(text="1", dynamic_bbox=True)
    area = ui.Panel2D((300, 100), position=(0, 100))
    area.set_layout(align_main=ui.Align.CENTER, align_cross=ui.Align.CENTER)
    area.add(label)
    spin = ui.SpinBox(size=(160, 40), initial_val=1)
    spin.on_change = lambda box: setattr(label, "message", f"value is {box.value}")
    row = ui.Row(spin, position=(0, 0))
    show_m = _show(row, area)
    start_x = label.get_position()[0]

    _click(show_m, *_center(spin.increment_button))
    npt.assert_equal(spin.value, 2)
    _draw(show_m)
    npt.assert_equal(label.message, "value is 2")
    assert label.get_position()[0] < start_x
    width = label.size[0]
    npt.assert_allclose(label.get_position()[0] + width / 2, 150, atol=1)
    show_m.close()


def test_combobox_in_a_column_opens_and_selects():
    """A dropdown keeps working when a layout positions it."""
    combo = ui.ComboBox2D(items=["one", "two"], size=(200, 150))
    column = ui.Column(ui.Rectangle2D(size=(200, 30)), combo, position=(20, 300))
    show_m = _show(column)

    _click(show_m, *_center(combo.drop_down_button))
    assert combo._menu_visibility
    _draw(show_m)

    _click(show_m, *_center(combo.drop_down_menu.slots[1]))
    npt.assert_equal(combo.selected_text, "two")
    assert not combo._menu_visibility
    show_m.close()


def test_dragging_a_panel_carries_laid_out_children():
    """Children stay attached while a laid-out panel is dragged."""
    panel = ui.Panel2D((200, 100), position=(10, 10))
    panel.set_layout(padding=8)
    child = panel.add(ui.Rectangle2D(size=(50, 50), color=(1, 0, 0)))
    show_m = _show(panel)

    target = _pick(show_m, 15, 15)
    _pointer(show_m, EventType.POINTER_DOWN, 15, 15, target)
    _pointer(show_m, EventType.POINTER_MOVE, 115, 65, target)
    _pointer(show_m, EventType.POINTER_UP, 115, 65, target)
    _draw(show_m)
    npt.assert_array_equal(child.get_position(), (118, 68))
    target = show_m.renderer.get_pick_info(_center(child))["world_object"]
    assert target is child.actor
    show_m.close()


def _spinbox():
    return ui.SpinBox(size=(120, 40))


def _card():
    fetch_viz_icons()
    return ui.Card2D(
        image_path=read_viz_icons(fname="play3.png"),
        title_text="T",
        body_text="B",
        size=(120, 200),
    )


def _listbox():
    return ui.ListBox2D(values=[str(i) for i in range(20)], size=(120, 200))


def _combobox():
    return ui.ComboBox2D(items=["a", "b"], size=(120, 200))


def _tabui():
    return ui.TabUI(size=(120, 200), tab_titles=["a", "b"], startup_tab_id=0)


@pytest.mark.parametrize("factory", [_spinbox, _card, _listbox, _combobox, _tabui])
def test_ported_widgets_follow_a_growing_parent(factory):
    """Each ported widget takes the size given to it and stays in its slot."""
    widget = factory()
    row = ui.Row(
        ui.Rectangle2D(size=(40, 40)),
        widget,
        width=ui.grow(),
        height=ui.fixed(300),
        align_cross=ui.Align.STRETCH,
        gap=10,
    )
    widget.set_layout(width=ui.grow())
    show_m = _show(row)

    npt.assert_array_equal(widget.size, (590, 300))
    npt.assert_array_equal(widget.get_position(), (50, 0))

    _resize(show_m, (400, 480))
    npt.assert_array_equal(widget.size, (350, 300))
    show_m.close()
