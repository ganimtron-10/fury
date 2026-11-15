"""Test UI themes."""

import numpy as np
import numpy.testing as npt
import pytest

from fury import io, ui, window


@pytest.fixture(autouse=True)
def classic_theme():
    """Start and end every test with the classic global theme."""
    ui.set_theme("classic")
    yield
    ui.set_theme("classic")


def red_panel_theme():
    """
    Create a theme that draws opaque red panels.

    Returns
    -------
    Theme
        The theme.
    """
    return ui.builtin_theme("classic").derive(
        name="red", items={"Panel2D": {"color": (1, 0, 0), "opacity": 1.0}}
    )


def render_center_pixel(element):
    """
    Render one UI element and return the pixel at the canvas center.

    Parameters
    ----------
    element : UI
        Element to render.

    Returns
    -------
    ndarray
        RGB values of the center pixel.
    """
    scene = window.Scene()
    scene.add(element)
    image = window.snapshot(scene=scene, fname=None, return_array=True)
    height, width = image.shape[:2]
    return image[height // 2, width // 2, :3].astype(float)


def test_theme_lookup_follows_mro_and_base():
    """Check that a lookup tries the class, its bases, then the base theme."""
    base = ui.Theme(name="base", items={"UI": {"font_family": "Noto Sans"}})
    theme = ui.Theme(
        name="child",
        items={"ButtonGroup": {"font_size": 12}, "Checkbox": {"font_size": 14}},
        base=base,
    )

    assert theme.get(ui.Checkbox, "font_size") == 14
    assert theme.get(ui.RadioButton, "font_size") == 12
    assert theme.get(ui.RadioButton, "font_family") == "Noto Sans"
    assert theme.get("ButtonGroup", "font_size") == 12
    assert theme.base is base


def test_theme_missing_item_raises():
    """Check that an undefined item raises a KeyError naming it."""
    theme = ui.Theme(name="empty")

    with pytest.raises(KeyError, match="no item 'color' for 'Panel2D'"):
        theme.get(ui.Panel2D, "color")
    assert not theme.has(ui.Panel2D, "color")
    assert ui.builtin_theme("classic").has(ui.Panel2D, "color")


def test_theme_derive_overrides():
    """Check that a derived theme overrides only the given items."""
    classic = ui.builtin_theme("classic")
    theme = classic.derive(name="dark", items={"Panel2D": {"color": (0, 0, 0)}})

    assert theme.name == "dark"
    assert theme.get(ui.Panel2D, "color") == (0, 0, 0)
    assert theme.get(ui.Panel2D, "opacity") == classic.get(ui.Panel2D, "opacity")
    assert classic.get(ui.Panel2D, "color") == (0.1, 0.1, 0.1)


def test_theme_set_accepts_class():
    """Check that items can be set with a class or its name."""
    theme = ui.Theme()
    theme.set(ui.Panel2D, "opacity", 0.5)
    theme.set("TextBlock2D", "font_size", 11)

    assert theme.get(ui.Panel2D, "opacity") == 0.5
    assert theme.get(ui.TextBlock2D, "font_size") == 11


def test_theme_save_load_roundtrip(tmp_path):
    """Check that saving and loading keeps inherited and own items."""
    theme = red_panel_theme()
    path = tmp_path / "red.json"
    theme.save(path)
    loaded = ui.Theme.load(path)

    assert loaded.name == "red"
    assert loaded.base is None
    npt.assert_array_equal(loaded.get(ui.Panel2D, "color"), (1, 0, 0))
    npt.assert_array_equal(loaded.get(ui.LineSlider2D, "track_color"), (1, 0, 0))
    assert loaded.get(ui.TextButton2D, "states")["hover"] == [0.9, 0.9, 0.9]


def test_theme_save_rejects_unknown_values(tmp_path):
    """Check that a value JSON cannot store raises a TypeError."""
    theme = ui.Theme(items={"UI": {"font_family": object()}})

    with pytest.raises(TypeError, match="cannot be saved"):
        theme.save(tmp_path / "bad.json")


def test_set_theme_by_name():
    """Check that set_theme accepts a name or a Theme and rejects others."""
    theme = red_panel_theme()
    ui.set_theme(theme)
    assert ui.get_theme() is theme

    ui.set_theme("classic")
    assert ui.get_theme().name == "classic"

    with pytest.raises(ValueError, match="Unknown theme 'missing'"):
        ui.set_theme("missing")


def test_theme_scope_restores():
    """Check that theme_scope applies only inside its block, when nested too."""
    outer = red_panel_theme()
    inner = ui.Theme(name="inner", base=outer)
    global_theme = ui.get_theme()

    with ui.theme_scope(outer) as scoped:
        assert scoped is outer
        assert ui.get_theme() is outer
        with ui.theme_scope(inner):
            assert ui.get_theme() is inner
        assert ui.get_theme() is outer
    assert ui.get_theme() is global_theme

    with pytest.raises(RuntimeError), ui.theme_scope(outer):
        raise RuntimeError
    assert ui.get_theme() is global_theme


def test_widgets_use_theme_defaults():
    """Check that widgets created in a scope take their defaults from it."""
    theme = ui.builtin_theme("classic").derive(
        items={
            "Panel2D": {"color": (0.2, 0.3, 0.4), "opacity": 0.9},
            "TextBlock2D": {"font_size": 11, "color": (0, 1, 0)},
            "Slider2D": {"track_color": (0, 0, 1), "handle_color": (0, 1, 0)},
        }
    )

    with ui.theme_scope(theme):
        panel = ui.Panel2D(size=(50, 50))
        text = ui.TextBlock2D(text="a", size=(20, 20))
        slider = ui.LineSlider2D()

    npt.assert_array_almost_equal(panel.color, (0.2, 0.3, 0.4))
    npt.assert_almost_equal(panel.opacity, 0.9)
    assert text.font_size == 11
    npt.assert_array_almost_equal(text.color[:3], (0, 1, 0))
    npt.assert_array_almost_equal(slider.track.color, (0, 0, 1))
    npt.assert_array_almost_equal(slider.handle.color, (0, 1, 0))
    assert panel.theme is theme
    assert slider.theme is theme


def test_explicit_argument_overrides_theme():
    """Check that a constructor argument beats the theme item."""
    with ui.theme_scope(red_panel_theme()):
        panel = ui.Panel2D(size=(50, 50), color=(0, 0, 1), opacity=0.4)

    npt.assert_array_almost_equal(panel.color, (0, 0, 1))
    npt.assert_almost_equal(panel.opacity, 0.4)


def test_composite_children_inherit_theme():
    """Check that sub-widgets built by a composite widget use its theme."""
    theme = red_panel_theme()
    with ui.theme_scope(theme):
        combo = ui.ComboBox2D(items=["a", "b"])
    created_after = ui.Panel2D(size=(10, 10))

    assert combo.drop_down_menu.theme is theme
    assert combo.drop_down_menu.panel.theme is theme
    assert created_after.theme is not theme
    npt.assert_array_almost_equal(created_after.color, (0.1, 0.1, 0.1))


def test_global_theme_applies_to_new_widgets():
    """Check that set_theme changes widgets created afterwards only."""
    before = ui.Panel2D(size=(10, 10))
    ui.set_theme(red_panel_theme())
    after = ui.Panel2D(size=(10, 10))

    npt.assert_array_almost_equal(before.color, (0.1, 0.1, 0.1))
    npt.assert_array_almost_equal(after.color, (1, 0, 0))


def test_theme_colors_render():
    """Check that a theme color reaches the rendered pixels."""
    classic_pixel = render_center_pixel(ui.Panel2D(size=(4000, 4000)))
    with ui.theme_scope(red_panel_theme()):
        red_pixel = render_center_pixel(ui.Panel2D(size=(4000, 4000)))

    assert red_pixel[0] > 200
    assert red_pixel[1] < 50 and red_pixel[2] < 50
    assert abs(classic_pixel[0] - classic_pixel[1]) < 5


def test_text_button_uses_theme_hex_states():
    """Check that a TextButton2D draws hex state colors from the theme."""
    theme = ui.builtin_theme("classic").derive(
        items={"TextButton2D": {"states": {"default": "#FF0000", "hover": "#00FF00"}}}
    )
    with ui.theme_scope(theme):
        button = ui.TextButton2D("Go", size=(4000, 4000))

    npt.assert_array_almost_equal(button.child.background.color, (1, 0, 0))
    button.states["default"] = "#0000FF"
    assert theme.get(ui.TextButton2D, "states")["default"] == "#FF0000"
    pixel = render_center_pixel(button)
    assert pixel[0] > 200 and pixel[1] < 50


def test_theme_lookup_prefers_specific_class():
    """Check that a class item in the base beats a general item in a derived theme."""
    theme = ui.builtin_theme("classic").derive(items={"UI": {"font_size": 30}})

    assert theme.get(ui.TextBlock2D, "font_size") == 18
    assert theme.get(ui.Panel2D, "font_size") == 30


def test_theme_save_load_keeps_lookups(tmp_path):
    """Check that a reloaded theme returns the same value for every lookup."""
    theme = ui.builtin_theme("classic").derive(
        items={
            "UI": {"font_size": 30, "font_family": "Noto Sans"},
            "Slider2D": {"track_color": (0, 0, 1)},
            "LineSlider2D": {"handle_color": (0, 1, 0)},
        }
    )
    path = tmp_path / "theme.json"
    theme.save(path)
    loaded = ui.Theme.load(path)

    widget_types = [
        ui.TextBlock2D,
        ui.Panel2D,
        ui.LineSlider2D,
        ui.RingSlider2D,
        ui.TextButton2D,
        ui.Checkbox,
        ui.ListBox2D,
    ]
    item_names = {item for items in theme.to_dict()["items"].values() for item in items}
    for widget_type in widget_types:
        for item in item_names:
            assert loaded.has(widget_type, item) == theme.has(widget_type, item)
            if theme.has(widget_type, item):
                npt.assert_equal(
                    loaded.get(widget_type, item), theme.get(widget_type, item)
                )


def test_theme_copies_values():
    """Check that changing a given or exported value does not change the theme."""
    color = [1, 0, 0]
    theme = ui.Theme(items={"Panel2D": {"color": color}})
    color[0] = 0
    exported = theme.to_dict()
    exported["items"]["Panel2D"]["color"][1] = 1

    assert theme.get(ui.Panel2D, "color") == [1, 0, 0]
    theme.get(ui.Panel2D, "color")[2] = 1
    assert theme.get(ui.Panel2D, "color") == [1, 0, 0]
    assert ui.builtin_theme("classic").derive().name == "custom"


@pytest.mark.parametrize("slider_type", [ui.LineSlider2D, ui.RingSlider2D])
def test_slider_font_size_from_theme(slider_type):
    """Check that concrete sliders take their label font size from the theme."""
    theme = ui.builtin_theme("classic").derive(items={"Slider2D": {"font_size": 13}})
    with ui.theme_scope(theme):
        slider = slider_type()
    explicit = slider_type(font_size=21)

    assert slider.text.font_size == 13
    assert explicit.text.font_size == 21


def test_lazily_created_children_keep_parent_theme():
    """Check that rows added by a later resize use the list box's theme."""
    theme = ui.builtin_theme("classic").derive(items={"UI": {"font_family": "Mono"}})
    with ui.theme_scope(theme):
        listbox = ui.ListBox2D(values=["a", "b", "c", "d"], size=(200, 60))
    slot_count = len(listbox.slots)
    listbox.resize((200, 300))

    assert len(listbox.slots) > slot_count
    assert all(slot.theme is theme for slot in listbox.slots)


def test_tree_nodes_keep_parent_theme():
    """Check that nodes added to a Tree2D later still use the tree's theme."""
    theme = ui.builtin_theme("classic").derive(name="tree")
    with ui.theme_scope(theme):
        tree = ui.elements.Tree2D(structure=[{"Root": ["Child1", "Child2"]}])
    tree.generate_tree([{"Root": ["Late"]}])

    assert len(tree.nodes_dict) == 4
    assert all(node.theme is theme for node in tree.nodes_dict.values())


def test_image_container_ignores_rectangle_color(tmp_path):
    """Check that a themed Rectangle2D color does not tint images."""
    image_path = tmp_path / "white.png"
    io.save_image(np.full((8, 8, 3), 255, dtype=np.uint8), str(image_path))
    theme = ui.builtin_theme("classic").derive(
        items={"Rectangle2D": {"color": (1, 0, 0)}}
    )
    with ui.theme_scope(theme):
        container = ui.ImageContainer2D(str(image_path), size=(4000, 4000))

    npt.assert_array_almost_equal(container.color, (1, 1, 1))
    pixel = render_center_pixel(container)
    assert pixel.min() > 200
