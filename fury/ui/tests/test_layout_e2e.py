"""End-to-end tests of the layout engine in a real render loop."""

from pathlib import Path
import runpy

import numpy.testing as npt
import pytest

from fury import ui, window
from fury.data import fetch_viz_icons, read_viz_icons
from fury.lib import EventType

EXAMPLES = Path(__file__).parents[3] / "docs" / "examples"


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
    target = show_m.renderer.get_pick_info((x, y))["world_object"]
    assert target is not None, f"Nothing is drawn at ({x}, {y})"
    for event_type in (EventType.POINTER_DOWN, EventType.POINTER_UP):
        event = window.PointerEvent(x=x, y=y, type=event_type, button=1, target=target)
        target.handle_event(event)
    return target


@pytest.mark.parametrize(
    "example",
    ["viz_layout_basics.py", "viz_layout_sizing.py", "viz_layout_settings_demo.py"],
)
def test_layout_tutorials_run(example, monkeypatch):
    """Every layout tutorial runs and draws from start to finish."""
    original = window.ShowManager.__init__

    def offscreen(self, *args, **kwargs):
        kwargs["window_type"] = "offscreen"
        original(self, *args, **kwargs)

    monkeypatch.setattr(window.ShowManager, "__init__", offscreen)
    monkeypatch.setattr(window.ShowManager, "start", lambda self: None)
    namespace = runpy.run_path(str(EXAMPLES / example), run_name="example")
    show_m = namespace["show_manager"]
    _draw(show_m)
    assert len(ui.UIContext._dirty_layouts) == 0
    show_m.close()


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

    show_m._resize((400, 480))
    _draw(show_m)
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

    combo.drop_down_menu.select(combo.drop_down_menu.slots[1])
    combo.select_option_callback()
    npt.assert_equal(combo.selected_text, "two")
    assert not combo._menu_visibility
    show_m.close()


def test_dragging_a_panel_carries_laid_out_children():
    """Children stay attached while a laid-out panel is dragged."""
    panel = ui.Panel2D((200, 100), position=(10, 10))
    panel.set_layout(padding=8)
    child = panel.add(ui.Rectangle2D(size=(50, 50), color=(1, 0, 0)))
    show_m = _show(panel)

    panel.left_button_pressed(window.PointerEvent(x=15, y=15, type="pointer_down"))
    panel.left_button_dragged(window.PointerEvent(x=115, y=65, type="pointer_move"))
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

    show_m._resize((400, 480))
    _draw(show_m)
    npt.assert_array_equal(widget.size, (350, 300))
    show_m.close()


def _all_actors(element):
    """Return the actors of an element and all its descendants."""
    actors = list(element.actors)
    for child in element._children:
        actors.extend(_all_actors(child))
    return actors


def _drawn(show_m, element):
    """Return the actors of an element tree that are in the UI scene."""
    ui_scene = show_m.screens[0].scene.ui_scene
    return [actor for actor in _all_actors(element) if actor.parent is ui_scene]


def _picked(show_m, x, y):
    """Return the world object drawn at a point."""
    return show_m.renderer.get_pick_info((x, y))["world_object"]


def _add_to_row(container, element):
    return container.add(element)


def _remove_from_row(container, element):
    container.remove(element)


def _add_to_panel(container, element):
    return container.add(element)


def _remove_from_panel(container, element):
    container.remove_element(element)


CONTAINERS = {
    "row": (lambda: ui.Row(position=(0, 0), gap=10), _add_to_row, _remove_from_row),
    "panel": (
        lambda: ui.Panel2D((400, 100), position=(0, 0), color=(0, 0, 0)).set_layout(
            gap=10
        ),
        _add_to_panel,
        _remove_from_panel,
    ),
}


@pytest.mark.parametrize("kind", CONTAINERS)
def test_child_added_after_the_container_is_shown_is_drawn_and_clickable(kind):
    """A child added at runtime is drawn where the layout puts it."""
    make, add, _ = CONTAINERS[kind]
    container = make()
    add(container, ui.Rectangle2D(size=(40, 40), color=(0, 1, 0)))
    show_m = _show(container)

    clicks = []
    button = ui.TextButton2D(label="Go", size=(80, 30))
    button.on_clicked = lambda btn: clicks.append(btn)
    add(container, button)
    _draw(show_m)

    npt.assert_array_equal(button.get_position(), (50, 0))
    assert _picked(show_m, *_center(button)) in _all_actors(button)
    _click(show_m, *_center(button))
    assert clicks == [button]
    show_m.close()


@pytest.mark.parametrize("kind", CONTAINERS)
def test_removed_child_is_no_longer_drawn(kind):
    """Removing a child takes it off screen while its sibling moves over."""
    make, add, remove = CONTAINERS[kind]
    container = make()
    first = add(container, ui.Rectangle2D(size=(40, 40), color=(1, 0, 0)))
    second = add(container, ui.Rectangle2D(size=(40, 40), color=(0, 0, 1)))
    last = add(container, ui.Rectangle2D(size=(40, 40), color=(0, 1, 0)))
    show_m = _show(container)
    last_center = _center(last)

    remove(container, last)
    _draw(show_m)
    assert _picked(show_m, *last_center) is not last.actor

    remove(container, first)
    _draw(show_m)
    assert _drawn(show_m, first) == []
    assert _drawn(show_m, last) == []
    npt.assert_array_equal(second.get_position(), (0, 0))
    assert _picked(show_m, *_center(second)) is second.actor
    show_m.close()


@pytest.mark.parametrize("kind", CONTAINERS)
def test_removing_a_container_removes_children_changed_at_runtime(kind):
    """Scene removal also takes away children added after the container."""
    make, add, remove = CONTAINERS[kind]
    container = make()
    kept = add(container, ui.Rectangle2D(size=(40, 40)))
    dropped = add(container, ui.Rectangle2D(size=(40, 40)))
    show_m = _show(container)
    late = add(container, ui.Rectangle2D(size=(40, 40)))
    remove(container, dropped)
    _draw(show_m)

    show_m.screens[0].scene.remove(container)
    _draw(show_m)

    assert _drawn(show_m, container) == []
    assert _drawn(show_m, dropped) == []
    for element in (kept, late):
        assert element.actor.parent is None
    show_m.close()


def test_nested_container_added_at_runtime_draws_its_children():
    """Children of a nested container added later are drawn too."""
    row = ui.Row(ui.Rectangle2D(size=(40, 40)), gap=10)
    show_m = _show(row)
    column = row.add(ui.Column(gap=5))
    inner = column.add(ui.Rectangle2D(size=(30, 30), color=(1, 1, 0)))
    _draw(show_m)

    npt.assert_array_equal(inner.get_position(), (50, 0))
    assert _picked(show_m, *_center(inner)) is inner.actor

    row.remove(column)
    _draw(show_m)
    assert inner.actor.parent is None
    show_m.close()


def test_scene_clear_ends_membership_of_its_ui():
    """A cleared scene does not receive children added later."""
    row = ui.Row(ui.Rectangle2D(size=(40, 40)))
    show_m = _show(row)
    show_m.screens[0].scene.clear()

    late = row.add(ui.Rectangle2D(size=(40, 40)))
    _draw(show_m)
    assert late.actor.parent is None
    show_m.close()


def _visible_slots(listbox):
    return [slot for slot in listbox.slots if slot.element is not None]


def _assert_slots_drawn_and_clickable(show_m, listbox):
    """Every visible slot is drawn and a click selects its own value."""
    for slot in _visible_slots(listbox):
        x, y = _center(slot)
        target = _picked(show_m, x, y)
        assert target in _all_actors(slot), f"slot {slot.element!r} is not drawn"
    last = _visible_slots(listbox)[-1]
    _click(show_m, *_center(last))
    assert listbox.selected == [last.element]


def test_listbox_slots_follow_a_growing_and_shrinking_layout():
    """Slots appear and disappear with the size the layout gives the list."""
    values = [f"item {i}" for i in range(40)]
    listbox = ui.ListBox2D(values=values, size=(200, 100))
    listbox.set_layout(height=ui.grow(), width=ui.grow())
    column = ui.Column(listbox, width=ui.fixed(200), height=ui.fixed(120))
    show_m = _show(column)
    small = len(listbox.slots)
    _assert_slots_drawn_and_clickable(show_m, listbox)

    column.set_layout(height=ui.fixed(400))
    _draw(show_m)
    assert len(listbox.slots) > small
    npt.assert_array_equal(listbox.size, (200, 400))
    _assert_slots_drawn_and_clickable(show_m, listbox)

    removed = listbox.slots[small:]
    column.set_layout(height=ui.fixed(120))
    _draw(show_m)
    assert len(listbox.slots) == small
    for slot in removed:
        assert _drawn(show_m, slot) == []
    below = int(listbox.get_position()[1] + listbox.size[1] + 20)
    assert _picked(show_m, 100, below) not in [
        actor for slot in removed for actor in _all_actors(slot)
    ]
    _assert_slots_drawn_and_clickable(show_m, listbox)
    show_m.close()


def test_listbox_squeezed_below_its_padding_shows_no_slots():
    """A list given less height than its padding has no slots, no crash."""
    listbox = ui.ListBox2D(values=list("abcde"), size=(100, 100))
    listbox.set_layout(height=ui.grow())
    column = ui.Column(height=ui.fixed(60), width=ui.fixed(100))
    column.add(ui.Rectangle2D(size=(100, 50)), height=ui.fixed(50))
    column.add(listbox)
    show_m = _show(column)

    npt.assert_array_equal(listbox.size, (100, 10))
    assert listbox.nb_slots == 0
    assert listbox.slots == []

    column.set_layout(height=ui.fixed(200))
    _draw(show_m)
    assert listbox.nb_slots > 0
    _assert_slots_drawn_and_clickable(show_m, listbox)
    show_m.close()
