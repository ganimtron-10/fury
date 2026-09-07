"""Test UI Workbench and Playground module."""

import numpy.testing as npt

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
