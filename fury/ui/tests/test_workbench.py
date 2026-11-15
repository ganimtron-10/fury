"""Test UI Workbench and Playground module."""

import numpy.testing as npt
import pytest

from fury import ui, window
from fury.ui.workbench import UIWorkbench, UIWorkbenchItem


def test_workbench_initialization():
    """Test UIWorkbench initialization and scene attachment."""
    scene = window.Scene()
    btn = ui.TextButton2D(label="Test", position=(10, 10))
    scene.add(btn)

    workbench = UIWorkbench(scene=scene)
    assert len(workbench.items) >= 1
    assert workbench.find_item_by_ui(btn) is not None
    assert workbench.selected_item is None


def test_workbench_register_unregister():
    """Test manual register and unregister of UI elements."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    rect = ui.Rectangle2D(size=(100, 50), position=(20, 20))
    item = workbench.register(rect, name="test_rect")

    assert item.id in workbench.items
    assert item.name == "test_rect"
    assert item in workbench.root_items
    assert rect in scene.ui_elements

    workbench.unregister(item)
    assert item.id not in workbench.items
    assert item not in workbench.root_items
    assert rect not in scene.ui_elements


def test_workbench_parenting_and_unparenting():
    """Test parenting a child to a Panel2D and unparenting to root."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    panel = ui.Panel2D(size=(300, 200), position=(50, 50))
    panel_item = workbench.register(panel, name="parent_panel")

    button = ui.TextButton2D(label="ChildBtn", position=(80, 80))
    btn_item = workbench.register(button, name="child_btn")

    # Initial state: both are root items
    assert btn_item in workbench.root_items
    assert btn_item.parent is None

    # Parent button to panel
    success = workbench.parent_element(btn_item, panel_item, offset=(30, 30))
    assert success is True
    assert btn_item.parent is panel_item
    assert btn_item in panel_item.children
    assert btn_item not in workbench.root_items
    assert button in panel._elements

    # Check button position was updated relative to panel
    npt.assert_array_almost_equal(btn_item.get_position(), (80, 80))

    # Unparent back to root scene
    success_unparent = workbench.unparent_element(btn_item)
    assert success_unparent is True
    assert btn_item.parent is None
    assert btn_item in workbench.root_items
    assert button not in panel._elements
    assert button in scene.ui_elements
    npt.assert_array_almost_equal(btn_item.get_position(), (80, 80))


def test_workbench_reparenting():
    """Test reparenting a child between two different panels."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    panel_a = ui.Panel2D(size=(200, 200), position=(0, 0))
    panel_b = ui.Panel2D(size=(200, 200), position=(300, 0))
    item_a = workbench.register(panel_a, name="panel_a")
    item_b = workbench.register(panel_b, name="panel_b")

    slider = ui.LineSlider2D(position=(10, 20), length=150)
    slider_item = workbench.register(slider, name="slider")

    # Parent to panel A
    workbench.parent_element(slider_item, item_a, offset=(10, 20))
    assert slider_item.parent is item_a
    assert slider in panel_a._elements

    # Reparent to panel B
    success = workbench.reparent_element(slider_item, item_b, offset=(15, 25))
    assert success is True
    assert slider_item.parent is item_b
    assert slider not in panel_a._elements
    assert slider in panel_b._elements
    npt.assert_array_almost_equal(slider_item.get_position(), (315, 25))


def test_workbench_merge_elements():
    """Test merging multiple UI components into a new Panel2D."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    rect = ui.Rectangle2D(size=(80, 40), position=(100, 100))
    disk = ui.Disk2D(outer_radius=30, center=(200, 120))
    item_rect = workbench.register(rect, name="rect")
    item_disk = workbench.register(disk, name="disk")

    merged = workbench.merge_elements(
        [item_rect, item_disk], padding=10.0, panel_name="my_merged_panel"
    )

    assert merged is not None
    assert isinstance(merged.ui, ui.Panel2D)
    assert item_rect.parent is merged
    assert item_disk.parent is merged
    assert item_rect in merged.children
    assert item_disk in merged.children


def test_workbench_duplicate():
    """Test duplicating an existing element."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    btn = ui.TextButton2D(label="Original", position=(50, 50), size=(100, 40))
    btn_item = workbench.register(btn, name="btn_orig")

    dup_item = workbench.duplicate_element(btn_item, offset=(20, 20))
    assert dup_item is not None
    assert dup_item is not btn_item
    npt.assert_array_almost_equal(dup_item.get_position(), (70, 70))


def test_workbench_property_helpers():
    """Test UIWorkbenchItem property reading and writing."""
    rect = ui.Rectangle2D(size=(100, 60), position=(30, 40), color=(0.1, 0.2, 0.3))
    item = UIWorkbenchItem(id=1, name="rect_1", ui=rect)

    npt.assert_array_almost_equal(item.get_position(), (30, 40))
    item.set_position((50, 60))
    npt.assert_array_almost_equal(item.get_position(), (50, 60))

    w, h = item.get_size()
    assert (w, h) == (100.0, 60.0)
    item.set_size((120, 80))
    assert item.get_size() == (120.0, 80.0)

    item.set_color((0.8, 0.5, 0.2))
    npt.assert_array_almost_equal(item.get_color(), (0.8, 0.5, 0.2))

    item.set_opacity(0.65)
    npt.assert_almost_equal(item.get_opacity(), 0.65)

    item.set_z_order(5)
    assert item.get_z_order() == 5


def test_workbench_fuzzy_operations():
    """Test fuzzy randomize operations and boundary tests do not crash."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    p = ui.Panel2D(size=(200, 200), position=(50, 50))
    b = ui.TextButton2D(label="FuzzyBtn", position=(60, 60))
    workbench.register(p)
    workbench.register(b)

    # Fuzzy operations across all items
    workbench.fuzzy_randomize_colors("all")
    workbench.fuzzy_randomize_positions("all", max_delta=20.0)
    workbench.fuzzy_randomize_sizes("all", scale_min=0.8, scale_max=1.2)

    # Extreme bounds stress test
    item_b = workbench.find_item_by_ui(b)
    assert item_b is not None
    workbench.fuzzy_extreme_stress_test(item_b)


def test_workbench_reparent_stress_test():
    """Test reparenting stress test maintains actor counts."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    panel = ui.Panel2D(size=(300, 300), position=(0, 0))
    btn = ui.TextButton2D(label="StressBtn", position=(50, 50))
    panel_item = workbench.register(panel)
    btn_item = workbench.register(btn)

    success = workbench.run_reparent_stress_test(btn_item, panel_item, cycles=4)
    assert success is True


def test_workbench_create_elements_and_presets():
    """Test element creation wizard and presets."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    types_to_test = [
        "Panel2D",
        "TextButton2D",
        "LineSlider2D",
        "LineDoubleSlider2D",
        "RingSlider2D",
        "TextBlock2D",
        "TextBox2D",
        "Checkbox",
        "RadioButton",
        "ComboBox2D",
        "ListBox2D",
        "Rectangle2D",
        "RoundedRectangle2D",
        "Disk2D",
        "TabUI",
    ]

    for elem_type in types_to_test:
        item = workbench.create_element(elem_type, position=(100, 100), size=(120, 60))
        assert item is not None, f"Failed creating {elem_type}"
        assert item.type_name == elem_type

    # Presets
    p1 = workbench.create_preset("Settings Dialog")
    assert p1 is not None
    p2 = workbench.create_preset("Form Card")
    assert p2 is not None
    p3 = workbench.create_preset("Audio Mixer")
    assert p3 is not None


class _FakeEvent:
    """Minimal stand-in for a pygfx pointer event."""

    def __init__(self, x, y, button=1):
        """Store the pointer coordinates and the button index."""
        self.x = x
        self.y = y
        self.button = button


def _build_scene():
    """Build a small nested scene and return it with its components."""
    scene = window.Scene()
    panel = ui.Panel2D(
        size=(340, 250), position=(50, 60), has_border=True, border_width=2
    )
    title = ui.TextBlock2D(text="Title", size=(300, 30), bg_color=(0.2, 0.2, 0.3))
    panel.add_element(title, (15, 15))
    slider = ui.LineSlider2D(length=280, initial_value=60)
    panel.add_element(slider, (25, 120))
    scene.add(panel)
    disk = ui.Disk2D(outer_radius=30, center=(600, 200))
    scene.add(disk)
    return scene, panel, title, slider, disk


def test_workbench_auto_hierarchy_roles():
    """Test the workbench mirrors the scene with content/part roles."""
    scene, panel, title, slider, disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    panel_item = workbench.find_item_by_ui(panel)
    title_item = workbench.find_item_by_ui(title)
    slider_item = workbench.find_item_by_ui(slider)
    disk_item = workbench.find_item_by_ui(disk)

    assert panel_item.role == "root"
    assert disk_item.role == "root"
    assert title_item.role == "content"
    assert slider_item.role == "content"
    assert title_item.parent is panel_item
    assert not title_item.is_internal

    # Internal parts are discovered, labelled and flagged.
    background_item = workbench.find_item_by_ui(panel.background)
    assert background_item.role == "part"
    assert background_item.is_internal
    assert background_item.attr_label == "background"

    border_item = workbench.find_item_by_ui(panel.borders["top"])
    assert border_item.attr_label == "borders[top]"

    handle_item = workbench.find_item_by_ui(slider.handle)
    assert handle_item.attr_label == "handle"
    assert handle_item.is_internal
    assert handle_item.owner is slider_item

    # Content children come before internal parts.
    roles = [child.role for child in panel_item.children]
    assert roles == ["content", "content", "part", "part", "part", "part", "part"]


def test_workbench_sync_is_stable_and_prunes():
    """Test repeated syncs keep node identity and drop removed components."""
    scene, panel, title, _slider, disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    disk_item = workbench.find_item_by_ui(disk)
    disk_item.name = "my_disk"
    node_count = len(workbench.items)

    for _ in range(3):
        workbench.sync()

    assert len(workbench.items) == node_count
    assert workbench.find_item_by_ui(disk) is disk_item
    assert disk_item.name == "my_disk"

    scene.remove(disk)
    workbench.sync()
    assert workbench.find_item_by_ui(disk) is None
    assert len(workbench.items) == node_count - 1

    # Components added to a panel later are picked up on the next sync.
    late = ui.TextBlock2D(text="late", size=(60, 20))
    panel.add_element(late, (10, 200))
    workbench.sync()
    late_item = workbench.find_item_by_ui(late)
    assert late_item is not None
    assert late_item.parent is workbench.find_item_by_ui(panel)
    assert all(actor in scene.ui_scene.children for actor in late.actors)
    assert workbench.find_item_by_ui(title) is not None


def test_workbench_sync_restores_callbacks_on_removed_subtree():
    """Test removing and re-adding a subtree binds events to fresh items."""
    from fury.ui.workbench import HANDLER_NAMES, walk_ui

    scene, panel, _title, _slider, _disk = _build_scene()
    components = list(walk_ui(panel))
    originals = [
        {name: getattr(component, name) for name in HANDLER_NAMES}
        for component in components
    ]
    workbench = UIWorkbench(scene=scene)
    old_item = workbench.find_item_by_ui(panel)
    panel.background.on_left_mouse_button_pressed(_FakeEvent(100, 100))
    assert workbench.selected_item is old_item

    scene.remove(panel)
    workbench.sync()

    assert workbench.selected_item is None
    assert workbench.hit_item is None
    for component, handlers in zip(components, originals, strict=True):
        assert workbench.find_item_by_ui(component) is None
        for name, handler in handlers.items():
            assert getattr(component, name) is handler

    scene.add(panel)
    workbench.sync()
    new_item = workbench.find_item_by_ui(panel)
    assert new_item is not old_item
    old_event_count = old_item.event_counts["on_hover"]
    panel.on_hover(_FakeEvent(100, 100))
    assert new_item.event_counts["on_hover"] == 1
    assert old_item.event_counts["on_hover"] == old_event_count
    panel.background.on_left_mouse_button_pressed(_FakeEvent(100, 100))
    assert workbench.selected_item is new_item


def test_workbench_sync_preserves_replacement_callback_on_removal():
    """Test pruning leaves a user replacement callback intact."""
    scene = window.Scene()
    rect = ui.Rectangle2D()
    scene.add(rect)
    workbench = UIWorkbench(scene=scene)
    events = []
    replacement = events.append
    rect.on_hover = replacement

    scene.remove(rect)
    workbench.sync()

    assert rect.on_hover is replacement
    event = _FakeEvent(10, 10)
    rect.on_hover(event)
    assert events == [event]


def test_workbench_universal_drag_targets():
    """Test only components without a drag of their own become draggable."""
    scene, panel, title, slider, disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    assert workbench.find_item_by_ui(title).draggable
    assert workbench.find_item_by_ui(disk).draggable
    assert not workbench.find_item_by_ui(panel).draggable
    assert not workbench.find_item_by_ui(slider).draggable


def test_workbench_drag_moves_owner_and_updates_offset():
    """Test dragging an internal part moves the widget that owns it."""
    scene, panel, title, _slider, _disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    panel_item = workbench.find_item_by_ui(panel)
    title_item = workbench.find_item_by_ui(title)
    start = title_item.get_position()

    title.background.on_left_mouse_button_pressed(_FakeEvent(70, 80))
    assert workbench.selected_item is title_item
    assert workbench.hit_item is workbench.find_item_by_ui(title.background)

    title.background.on_left_mouse_button_dragged(_FakeEvent(120, 130))
    npt.assert_array_almost_equal(title_item.get_position(), start + (50, 50))

    offsets = {id(element): offset for element, offset in panel.element_offsets}
    npt.assert_array_almost_equal(
        offsets[id(title)], title_item.get_position() - panel_item.get_position()
    )

    title.background.on_left_mouse_button_released(_FakeEvent(120, 130))
    assert workbench._dragging is None


def test_workbench_keeps_native_widget_behaviour():
    """Test hooked callbacks still run the behaviour each widget defines."""
    scene, panel, _title, slider, _disk = _build_scene()
    workbench = UIWorkbench(scene=scene)
    panel_item = workbench.find_item_by_ui(panel)

    panel.background.on_left_mouse_button_pressed(_FakeEvent(100, 100))
    panel.background.on_left_mouse_button_dragged(_FakeEvent(150, 140))
    npt.assert_array_almost_equal(panel_item.get_position(), (100, 100))

    track_x = int(slider.track.get_position()[0])
    before = slider.value
    slider.track.on_left_mouse_button_pressed(_FakeEvent(track_x + 250, 200))
    slider.track.on_left_mouse_button_dragged(_FakeEvent(track_x + 250, 200))
    assert slider.value != before
    assert workbench.selected_item is workbench.find_item_by_ui(slider)


def test_workbench_event_counters_and_trace():
    """Test the workbench counts events and can log them."""
    scene, _panel, title, _slider, disk = _build_scene()
    workbench = UIWorkbench(scene=scene)
    disk_item = workbench.find_item_by_ui(disk)

    disk.on_hover(_FakeEvent(600, 200))
    disk.on_dishover(_FakeEvent(600, 200))

    assert disk_item.event_counts["on_hover"] == 1
    assert disk_item.last_event == "on_dishover"
    assert len(workbench.event_log) == 0

    disk_item.trace_events = True
    disk.on_hover(_FakeEvent(600, 200))
    assert disk_item.event_counts["on_hover"] == 2
    assert workbench.event_log[0]["event"] == "on_hover"
    assert workbench.event_log[0]["target"] == disk_item.display_name

    # A TextBlock2D forwards the events of its background to itself, so both
    # nodes see them.
    background_item = workbench.find_item_by_ui(title.background)
    title.background.on_hover(_FakeEvent(70, 80))
    assert background_item.event_counts["on_hover"] == 1
    assert workbench.find_item_by_ui(title).event_counts["on_hover"] == 1


def test_workbench_selection_outline():
    """Test the selection outline follows the selected component."""
    scene, _panel, title, _slider, _disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    workbench.update_outline()
    assert workbench._outline == []

    item = workbench.find_item_by_ui(title)
    workbench.select(item)
    workbench.update_outline()

    assert len(workbench._outline) == 4
    x, y, width, height = item.get_bounds()
    npt.assert_array_almost_equal(workbench._outline[0].get_position(), (x, y))
    npt.assert_almost_equal(workbench._outline[0].size[0], round(width))
    assert all(bar.actors[0].visible for bar in workbench._outline)

    workbench.select(None)
    workbench.update_outline()
    assert not any(bar.actors[0].visible for bar in workbench._outline)


def test_workbench_hierarchy_filtering():
    """Test the tree filter matches a node or one of its descendants."""
    scene, panel, title, _slider, disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    panel_item = workbench.find_item_by_ui(panel)
    disk_item = workbench.find_item_by_ui(disk)

    workbench.hierarchy_filter = "textblock"
    assert workbench._matches_filter(panel_item)
    assert not workbench._matches_filter(disk_item)

    workbench.hierarchy_filter = ""
    assert workbench._matches_filter(disk_item)

    visible = workbench._visible_children(panel_item)
    assert workbench.find_item_by_ui(panel.background) not in visible
    workbench.show_internals = True
    assert workbench.find_item_by_ui(panel.background) in (
        workbench._visible_children(panel_item)
    )


def test_workbench_property_introspection():
    """Test the reflective helpers used by the property inspector."""
    from fury.ui.workbench import iter_ui_properties, named_sub_ui

    panel = ui.Panel2D(size=(120, 80), has_border=True, border_width=2)
    names = {name for name, _prop, _cls in iter_ui_properties(panel)}
    assert {"color", "opacity", "z_order", "size", "border_color"} <= names
    assert "actors" not in names

    editable = {
        name for name, prop, _cls in iter_ui_properties(panel) if prop.fset is not None
    }
    assert {"color", "opacity", "z_order"} <= editable
    assert "size" not in editable

    labels = {label for _sub, label in named_sub_ui(panel)}
    assert "background" in labels
    assert "borders[left]" in labels

    # Reading border properties is skipped when the panel has no border.
    plain = ui.Panel2D(size=(120, 80))
    assert not UIWorkbench._property_applies(plain, "border_color")
    assert UIWorkbench._property_applies(panel, "border_color")


def test_workbench_value_editor_dispatch():
    """Test the generic editor recognises the value shapes it supports."""
    import numpy as np

    assert UIWorkbench._as_number_list((0.1, 0.2, 0.3)) == [0.1, 0.2, 0.3]
    assert UIWorkbench._as_number_list(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert UIWorkbench._as_number_list(["a", "b"]) is None
    assert UIWorkbench._as_number_list([True, False]) is None
    assert UIWorkbench._as_number_list([1, 2, 3, 4, 5]) is None

    rebuilt = UIWorkbench._rebuild_sequence((0.0, 0.0), [1.0, 2.0])
    assert rebuilt == (1.0, 2.0)
    assert isinstance(
        UIWorkbench._rebuild_sequence(np.zeros(2), [1.0, 2.0]), np.ndarray
    )

    assert UIWorkbench._short_repr("x" * 200).endswith("...")


def test_workbench_apply_value_reports_failure():
    """Test a failing property write lands in the status bar and the log."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)
    rect = ui.Rectangle2D(size=(50, 50))
    item = workbench.register(rect, name="rect")

    workbench._apply_value(item, "opacity", 0.25, setattr)
    npt.assert_almost_equal(rect.opacity, 0.25)

    workbench._apply_value(item, "z_order", "not-an-int", setattr)
    assert "z_order" in workbench.status_message
    assert workbench.event_log[0]["event"] == "set failed"


def test_workbench_parenting_rejects_cycles():
    """Test a container cannot be parented into one of its own children."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    outer = ui.Panel2D(size=(300, 300), position=(0, 0))
    inner = ui.Panel2D(size=(120, 120), position=(20, 20))
    outer_item = workbench.register(outer, name="outer")
    inner_item = workbench.register(inner, name="inner")

    assert workbench.parent_element(inner_item, outer_item, offset=(20, 20))
    assert not workbench.parent_element(outer_item, inner_item)
    assert outer_item.parent is None


def test_workbench_unregister_removes_subtree():
    """Test deleting a container also forgets everything it contains."""
    from fury.ui.workbench import walk_ui

    scene, panel, title, slider, _disk = _build_scene()
    workbench = UIWorkbench(scene=scene)

    panel_item = workbench.find_item_by_ui(panel)
    subtree_actors = {
        actor for component in walk_ui(panel) for actor in component.actors
    }
    remaining_actors = set(scene.ui_scene.children) - subtree_actors
    workbench.unregister(panel_item)

    assert workbench.find_item_by_ui(panel) is None
    assert workbench.find_item_by_ui(title) is None
    assert workbench.find_item_by_ui(slider) is None
    assert panel not in scene.ui_elements
    assert workbench.selected_item is None
    assert set(scene.ui_scene.children) == remaining_actors
    assert title in panel._children
    assert slider in panel._children

    # Deleting from the scene must not dismantle the reusable widget.
    scene.add(panel)
    workbench.sync()
    assert workbench.find_item_by_ui(title).parent.ui is panel
    assert workbench.find_item_by_ui(panel.background).is_internal


@pytest.mark.parametrize("same_scene", [False, True])
def test_workbench_set_scene_releases_old_hooks_and_outline(same_scene):
    """Test scene changes release callbacks, drag state and outline actors."""
    old_scene, panel, _title, _slider, disk = _build_scene()
    original = disk.on_hover
    workbench = UIWorkbench(scene=old_scene)
    disk.on_left_mouse_button_pressed(_FakeEvent(600, 200))
    workbench.update_outline()
    outline_actors = [bar.actors[0] for bar in workbench._outline]
    old_item = workbench.find_item_by_ui(disk)
    assert workbench._dragging is old_item

    new_scene = old_scene if same_scene else window.Scene()
    workbench.set_scene(new_scene)

    assert workbench._dragging is None
    assert workbench.hit_item is None
    assert workbench.selected_item is None
    assert not any(actor in old_scene.ui_scene.children for actor in outline_actors)
    if not same_scene:
        assert disk.on_hover is original
        new_scene.add(disk)
        workbench.sync()
    item = workbench.find_item_by_ui(disk)
    disk.on_hover(_FakeEvent(600, 200))
    assert item.event_counts["on_hover"] == 1
    assert old_item.event_counts["on_hover"] == 0
    workbench.select(item)
    workbench.update_outline()
    assert all(
        bar.actors[0] in new_scene.ui_scene.children for bar in workbench._outline
    )
    assert panel in old_scene.ui_elements


@pytest.mark.parametrize("destination", ["self", "descendant", "non_container"])
def test_workbench_invalid_reparent_preserves_hierarchy(destination):
    """Test rejected reparenting leaves the original parent and actors intact."""
    scene = window.Scene()
    outer = ui.Panel2D(size=(400, 400))
    inner = ui.Panel2D(size=(200, 200))
    child = ui.Panel2D(size=(50, 50))
    rect = ui.Rectangle2D()
    inner.add_element(child, (10, 10))
    outer.add_element(inner, (20, 20))
    scene.add(outer, rect)
    workbench = UIWorkbench(scene=scene)
    item = workbench.find_item_by_ui(inner)
    parent = item.parent
    target_ui = {"self": inner, "descendant": child, "non_container": rect}[destination]
    before = set(scene.ui_scene.children)
    position = item.get_position().copy()

    assert not workbench.reparent_element(item, workbench.find_item_by_ui(target_ui))

    assert item.parent is parent
    assert inner in outer._children
    assert inner not in scene.ui_elements
    assert set(scene.ui_scene.children) == before
    npt.assert_array_equal(item.get_position(), position)


@pytest.mark.parametrize("operation", ["parent", "unparent", "delete"])
def test_workbench_internal_parts_cannot_be_detached(operation):
    """Test hierarchy actions preserve a widget's internal structure."""
    scene, panel, _title, slider, _disk = _build_scene()
    workbench = UIWorkbench(scene=scene)
    item = workbench.find_item_by_ui(slider.handle)
    parent = item.parent
    before = set(scene.ui_scene.children)
    if operation == "parent":
        assert not workbench.parent_element(item, workbench.find_item_by_ui(panel))
    elif operation == "unparent":
        assert not workbench.unparent_element(item)
    else:
        workbench.unregister(item)
    workbench.sync()

    assert workbench.find_item_by_ui(slider.handle) is item
    assert item.parent is parent
    assert slider.handle not in scene.ui_elements
    assert slider.handle not in panel._children
    assert set(scene.ui_scene.children) == before


def test_workbench_create_playback_and_range_widgets():
    """Test the widgets whose constructors do not take a size."""
    scene = window.Scene()
    workbench = UIWorkbench(scene=scene)

    for element_type in ("PlaybackPanel", "RangeSlider"):
        item = workbench.create_element(
            element_type, position=(60.0, 60.0), size=(240.0, 60.0)
        )
        assert item is not None, f"Failed creating {element_type}"
        assert item.type_name == element_type

    assert workbench.create_element("NotAWidget") is None
    assert "Unknown element type" in workbench.status_message


def test_workbench_tabs_render_headlessly(monkeypatch, tmp_path):
    """Test all inspector widget paths and tabs in a real ImGui context."""
    imgui = pytest.importorskip("imgui_bundle").imgui
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(str(tmp_path / "imgui.ini"))
        io.display_size = imgui.ImVec2(1400, 820)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        monkeypatch.setattr(imgui, "collapsing_header", lambda *args, **kwargs: True)
        workbench = UIWorkbench(scene=window.Scene())
        for kind in workbench.create_types:
            assert workbench.create_element(kind) is not None, kind

        for item in list(workbench.items.values()):
            workbench.select(item)
            imgui.new_frame()
            imgui.begin("Workbench test")
            workbench._render_inspector_tab()
            workbench._render_hierarchy_operations()
            imgui.end()
            imgui.render()

        for tab in ("hierarchy", "builder", "fuzzy", "monitor"):
            imgui.new_frame()
            imgui.begin("Workbench test")
            getattr(workbench, f"_render_{tab}_tab")()
            imgui.end()
            imgui.render()
    finally:
        imgui.destroy_context(context)
