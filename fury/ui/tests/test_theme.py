"""Test UI themes."""

import gc

import numpy as np
import numpy.testing as npt
import pytest

from fury import io, ui, window
from fury.data import fetch_viz_icons, read_viz_icons
from fury.ui.helpers import Anchor
from fury.ui.tests._helpers import click, send_pointer


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


def test_global_theme_applies_to_new_and_existing_widgets():
    """Check that set_theme styles new widgets and restyles existing ones."""
    before = ui.Panel2D(size=(10, 10))
    ui.set_theme(red_panel_theme())
    after = ui.Panel2D(size=(10, 10))

    npt.assert_array_almost_equal(before.color, (1, 0, 0))
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


def _hex(color):
    """
    Convert a hex color to RGB values in [0, 1].

    Parameters
    ----------
    color : str
        Color such as ``"#1C1D20"``.

    Returns
    -------
    ndarray
        RGB values in [0, 1].
    """
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)]) / 255


def render_pixels(element, points):
    """
    Render one UI element in a 200x200 window and read some pixels.

    Parameters
    ----------
    element : UI
        Element to render.
    points : list of (int, int)
        Pixel coordinates (x, y), with y pointing down.

    Returns
    -------
    list of ndarray
        RGB values in [0, 255] of each point.
    """
    scene = window.Scene(background=(0, 0, 0))
    scene.add(element)
    show_m = window.ShowManager(
        scene=scene, size=(200, 200), window_type="offscreen", pixel_ratio=1
    )
    show_m.render()
    show_m.window.draw()
    image = np.asarray(show_m.snapshot())[..., :3].astype(int)
    show_m.close()
    return [image[y, x] for x, y in points]


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_graphite_themes_match_classic_items(name):
    """Graphite themes define every classic item, so restyling back works."""
    classic = ui.builtin_theme("classic").to_dict()["items"]
    graphite = ui.builtin_theme(name).to_dict()["items"]

    for type_name, items in classic.items():
        assert set(items) <= set(graphite[type_name]), type_name
    for type_name, items in graphite.items():
        assert set(items) <= set(classic.get(type_name, {})) | set(classic["UI"]), (
            type_name
        )


def test_graphite_theme_save_load_roundtrip(tmp_path):
    """The Graphite theme survives a JSON round trip."""
    path = tmp_path / "graphite.json"
    ui.builtin_theme("graphite").save(path)
    loaded = ui.Theme.load(path)

    assert loaded.get("Panel2D", "corner_radius") == 10
    npt.assert_allclose(loaded.get("Panel2D", "shadow_color"), (0, 0, 0, 0.35))


def test_set_theme_restyles_widget_but_keeps_explicit_color():
    """Live restyle changes an existing widget but keeps an explicit color."""
    themed = ui.Panel2D(size=(120, 100), position=(40, 40))
    explicit = ui.Panel2D(size=(120, 100), position=(40, 40), color=(0, 1, 0))
    classic_pixel = render_pixels(themed, [(100, 90)])[0]

    ui.set_theme("graphite")

    npt.assert_array_almost_equal(themed.color, _hex("#1C1D20"))
    assert themed.opacity == 1.0
    assert themed.background.corner_radius == 10
    npt.assert_array_almost_equal(explicit.color, (0, 1, 0))
    assert explicit.background.corner_radius == 10
    themed_pixel, corner = render_pixels(themed, [(100, 90), (40, 40)])
    npt.assert_allclose(themed_pixel, _hex("#1C1D20") * 255, atol=2)
    assert not np.allclose(themed_pixel, classic_pixel, atol=2)
    npt.assert_array_equal(corner, (0, 0, 0))
    npt.assert_allclose(render_pixels(explicit, [(100, 90)])[0], (0, 255, 0), atol=2)

    ui.set_theme("classic")
    npt.assert_array_almost_equal(themed.color, (0.1, 0.1, 0.1))
    assert themed.background.corner_radius == 0
    npt.assert_allclose(render_pixels(themed, [(100, 90)])[0], classic_pixel, atol=0)


def test_set_theme_skips_widgets_from_a_scope():
    """Widgets created in a theme_scope keep their theme when set_theme runs."""
    with ui.theme_scope(red_panel_theme()):
        scoped = ui.Panel2D(size=(10, 10))
    ui.set_theme("graphite")

    npt.assert_array_almost_equal(scoped.color, (1, 0, 0))
    assert scoped.theme.name == "red"


def test_apply_theme_restyles_children_and_stops_following_global():
    """apply_theme restyles a panel and its children, which then keep it."""
    panel = ui.Panel2D(size=(200, 200))
    title = ui.TextBlock2D(text="Title", dynamic_bbox=True)
    button = ui.TextButton2D("Go")
    panel.add(title)
    panel.add(button)

    panel.apply_theme("graphite_light")
    npt.assert_array_almost_equal(panel.color, (1, 1, 1))
    npt.assert_array_almost_equal(title.color, _hex("#18181B"))
    npt.assert_array_almost_equal(button.child.background.color, _hex("#18181B"))
    npt.assert_array_almost_equal(button.child.color, (1, 1, 1))
    assert button.child.font_weight == 600

    ui.set_theme("graphite")
    npt.assert_array_almost_equal(panel.color, (1, 1, 1))
    npt.assert_array_almost_equal(title.color, _hex("#18181B"))


def test_restyle_updates_composite_parts():
    """Restyling a composite widget updates the colors of its parts."""
    slider = ui.LineSlider2D()
    combo = ui.ComboBox2D(items=["a", "b"])
    text_box = ui.TextBox2D(10, 1)
    checkbox = ui.Checkbox(["a"])

    ui.set_theme("graphite")

    npt.assert_array_almost_equal(slider.track.color, _hex("#2C2E33"))
    npt.assert_array_almost_equal(slider.handle.color, _hex("#ECEDEF"))
    npt.assert_array_almost_equal(combo.selection_box.color, _hex("#ECEDEF"))
    npt.assert_array_almost_equal(combo.selection_box.background.color, _hex("#26282C"))
    assert combo.selection_box.background.corner_radius == 6
    slot = combo.drop_down_menu.slots[0]
    npt.assert_array_almost_equal(slot.textblock.color, _hex("#ECEDEF"))
    npt.assert_array_almost_equal(text_box.text.background.color, _hex("#26282C"))
    assert text_box.text.background.border_width == 1
    _, label = checkbox.options["a"]
    npt.assert_array_almost_equal(label.color, _hex("#ECEDEF"))


def test_restyle_keeps_explicit_composite_arguments():
    """Colors passed to a composite constructor survive a restyle."""
    combo = ui.ComboBox2D(items=["a"], selection_text_color=(1, 0, 0))
    list_box = ui.ListBox2D(values=["a"], text_color=(0, 1, 0))

    ui.set_theme("graphite")

    npt.assert_array_almost_equal(combo.selection_box.color, (1, 0, 0))
    npt.assert_array_almost_equal(list_box.slots[0].textblock.color, (0, 1, 0))


def _composite_factories(directory):
    """
    Get factories for every composite widget, built with default styles.

    Parameters
    ----------
    directory : pathlib.Path
        Folder with a few entries, listed by the file menu.

    Returns
    -------
    dict
        Mapping of a name to a function creating the widget.
    """
    fetch_viz_icons()
    icon = read_viz_icons(fname="play3.png")
    return {
        "panel": lambda: ui.Panel2D(size=(200, 150), position=(20, 20)),
        "panel_border": lambda: ui.Panel2D(
            size=(200, 150), position=(20, 20), has_border=True, border_width=3
        ),
        "textblock": lambda: ui.TextBlock2D(
            text="Hello", position=(50, 50), bg_color=(0.5, 0, 0), dynamic_bbox=True
        ),
        "textbutton": lambda: ui.TextButton2D("Go", position=(50, 50)),
        "line_slider": lambda: ui.LineSlider2D(position=(50, 100)),
        "double_slider": lambda: ui.LineDoubleSlider2D(position=(50, 100)),
        "ring_slider": lambda: ui.RingSlider2D(center=(150, 150)),
        "range_slider": lambda: ui.RangeSlider(
            range_slider_center=(200, 100), value_slider_center=(200, 200)
        ),
        "textbox": lambda: ui.TextBox2D(20, 2, text="Type", position=(50, 50)),
        "checkbox": lambda: ui.Checkbox(["a", "b"], checked_labels=["a"]),
        "radio": lambda: ui.RadioButton(["a", "b"], checked_labels=["a"]),
        "combo": lambda: ui.ComboBox2D(items=["x", "y"], position=(50, 50)),
        "listbox": lambda: ui.ListBox2D(
            values=["x", "y", "z"], position=(50, 20), size=(200, 200), font_size=24
        ),
        "playback": lambda: ui.PlaybackPanel(position=(20, 50), width=380),
        "spinbox": lambda: ui.SpinBox(position=(50, 50)),
        "tabui": lambda: ui.TabUI(
            position=(20, 20), size=(300, 200), tab_titles=["A", "B"], font_size=30
        ),
        "card": lambda: ui.Card2D(
            icon, title_text="T", body_text="B", position=(20, 20), size=(200, 220)
        ),
        "filemenu": lambda: ui.FileMenu2D(
            directory_path=str(directory), position=(20, 20), size=(300, 200)
        ),
        "tree": lambda: ui.elements.Tree2D(
            structure=[{"Root": ["Child1", "Child2"]}], tree_name="Tree"
        ),
        "loader": lambda: ui.Loader2D(size=(400, 300), message="Loading"),
        "switch": lambda: ui.Switch2D(label="Shading", checked=True, position=(50, 50)),
        "switch_off": lambda: ui.Switch2D(
            label="Axes", label_side="left", position=(50, 50)
        ),
        "segmented": lambda: ui.SegmentedControl2D(
            ["Axial", "Coronal", "Sagittal"], selected_index=1, position=(50, 50)
        ),
    }


def _render_scene(element):
    """
    Render one element over a colored background.

    Parameters
    ----------
    element : UI
        Element to render.

    Returns
    -------
    ndarray
        RGB image with values in [0, 255].
    """
    scene = window.Scene(background=(0.2, 0.3, 0.4))
    scene.add(element)
    show_m = window.ShowManager(
        scene=scene, size=(400, 300), window_type="offscreen", pixel_ratio=1
    )
    show_m.render()
    show_m.window.draw()
    image = np.asarray(show_m.snapshot())[..., :3].astype(int)
    show_m.close()
    return image


COMPOSITES = [
    "panel",
    "panel_border",
    "textblock",
    "textbutton",
    "line_slider",
    "double_slider",
    "ring_slider",
    "range_slider",
    "textbox",
    "checkbox",
    "radio",
    "combo",
    "listbox",
    "playback",
    "spinbox",
    "tabui",
    "card",
    "filemenu",
    "tree",
    "loader",
    "switch",
    "switch_off",
    "segmented",
]


@pytest.mark.parametrize("name", COMPOSITES)
def test_classic_graphite_classic_roundtrip_matches_fresh_render(name, tmp_path):
    """Restyling to Graphite and back draws the same pixels as a fresh widget."""
    (tmp_path / "folder").mkdir()
    (tmp_path / "notes.txt").write_text("notes")
    factory = _composite_factories(tmp_path)[name]
    reference = _render_scene(factory())

    widget = factory()
    ui.set_theme("graphite")
    ui.set_theme("classic")

    npt.assert_array_equal(_render_scene(widget), reference)


def test_property_values_survive_restyle():
    """Values set through public properties are kept by a restyle."""
    text = ui.TextBlock2D(text="Hi", dynamic_bbox=True)
    text.color = (1, 0, 0)
    text.font_size = 30
    list_box = ui.ListBox2D(values=["a"], font_size=24)
    panel = ui.Panel2D(size=(50, 50))
    panel.color = (0, 0, 1)

    ui.set_theme("graphite")

    npt.assert_array_almost_equal(text.color, (1, 0, 0))
    assert text.font_size == 30
    assert list_box.slots[0].textblock.font_size == 24
    npt.assert_array_almost_equal(panel.color, (0, 0, 1))
    assert panel.background.corner_radius == 10


def test_failed_construction_is_not_restyled():
    """A widget whose constructor raised never follows the global theme."""
    before = len(ui.UIContext.global_theme_followers)
    with pytest.raises(ValueError, match="shape"):
        ui.LineSlider2D(shape="triangle")

    assert len(ui.UIContext.global_theme_followers) == before
    panel = ui.Panel2D(size=(10, 10))
    ui.set_theme("graphite")
    assert panel.background.corner_radius == 10


def test_theme_registry_does_not_keep_widgets_alive():
    """The follower registry only holds weak references."""
    gc.collect()
    before = len(ui.UIContext.global_theme_followers)
    widgets = [ui.Panel2D(size=(10, 10)) for _ in range(20)]
    widgets += [ui.LineSlider2D() for _ in range(5)]
    assert len(ui.UIContext.global_theme_followers) == before + 25

    del widgets
    gc.collect()
    assert len(ui.UIContext.global_theme_followers) == before


def test_parts_of_widgets_do_not_follow_the_global_theme():
    """Only the widget a user creates is registered, not its parts."""
    gc.collect()
    before = len(ui.UIContext.global_theme_followers)
    combo = ui.ComboBox2D(items=["a", "b"])

    followers = ui.UIContext.global_theme_followers
    assert len(followers) == before + 1
    assert combo in followers


@pytest.mark.parametrize(
    "element, name, value",
    [
        (lambda: ui.TextBlock2D(text="a", dynamic_bbox=True), "font_weight", 1000),
        (lambda: ui.TextBlock2D(text="a", dynamic_bbox=True), "color", "notacolor"),
        (lambda: ui.Rectangle2D(size=(5, 5)), "color", "notacolor"),
        (lambda: ui.RoundedRectangle2D(size=(5, 5)), "border_width", -1),
        (lambda: ui.Panel2D(size=(5, 5)), "color", "notacolor"),
    ],
)
def test_invalid_setter_value_keeps_restyle_working(element, name, value):
    """A rejected value is not remembered, so later restyles still work."""
    widget = element()
    with pytest.raises(ValueError):
        setattr(widget, name, value)
    other = ui.Panel2D(size=(5, 5))

    ui.set_theme("graphite")

    assert other.background.corner_radius == 10
    assert widget._style_state()[0].get(name) is not value


@pytest.mark.parametrize("final", ["classic", "graphite"])
def test_user_customizations_survive_set_theme(final):
    """Values set through public attributes, on widgets or parts, are kept."""
    slider = ui.LineSlider2D()
    slider.default_color = (0, 1, 0)
    slider.track.color = (0, 0, 1)
    list_box = ui.ListBox2D(values=["a"])
    list_box.selected_color = (0, 1, 0)
    list_box.text_color = (1, 0, 1)
    button = ui.TextButton2D("Go")
    states = dict.fromkeys(("default", "hover", "pressed", "disabled"), (0, 1, 0))
    button.states = states
    panel = ui.Panel2D(size=(20, 20))
    panel.background.color = (0, 1, 0)
    knob = ui.LineSlider2D()
    knob.handle.color = (1, 0, 1)

    if final == "classic":
        ui.set_theme("classic")
    else:
        ui.set_theme("graphite")
        ui.set_theme("classic")
        ui.set_theme("graphite")

    npt.assert_array_almost_equal(slider.default_color, (0, 1, 0))
    npt.assert_array_almost_equal(slider.handle.color, (0, 1, 0))
    npt.assert_array_almost_equal(slider.track.color, (0, 0, 1))
    npt.assert_array_almost_equal(knob.handle.color, (1, 0, 1))
    npt.assert_array_almost_equal(list_box.selected_color, (0, 1, 0))
    npt.assert_array_almost_equal(list_box.slots[0].textblock.color, (1, 0, 1))
    assert button.states is states
    npt.assert_array_almost_equal(button.child.background.color, (0, 1, 0))
    npt.assert_array_almost_equal(panel.background.color, (0, 1, 0))


def test_parent_setter_replaces_value_set_on_part():
    """Setting the panel color after its background color uses the panel color."""
    panel = ui.Panel2D(size=(20, 20))
    panel.background.color = (0, 1, 0)
    panel.color = (1, 0, 0)

    ui.set_theme("graphite")
    npt.assert_array_almost_equal(panel.background.color, (1, 0, 0))


def test_set_theme_with_same_items_does_not_restyle():
    """Setting an equal theme leaves widgets untouched."""
    panel = ui.Panel2D(size=(20, 20))
    panel.background.actor.material.color = (0, 1, 0, 1)

    ui.set_theme("classic")

    npt.assert_array_almost_equal(panel.background.actor.material.color[:3], (0, 1, 0))


def test_text_color_none_follows_the_theme_again():
    """Setting a text color to None makes it follow the theme."""
    text = ui.TextBlock2D(text="a", dynamic_bbox=True)
    text.color = (1, 0, 0)
    text.color = None
    npt.assert_array_almost_equal(text.color, (1, 1, 1))

    ui.set_theme("graphite")
    npt.assert_array_almost_equal(text.color, _hex("#ECEDEF"))


def test_muted_text_follows_the_theme():
    """A muted text block uses the theme's muted color, also after restyle."""
    text = ui.TextBlock2D(text="a", dynamic_bbox=True, muted=True)
    npt.assert_array_almost_equal(text.color, (0.6, 0.6, 0.6))

    ui.set_theme("graphite_light")
    npt.assert_array_almost_equal(text.color, _hex("#71717A"))


def test_apply_theme_on_a_part_sticks():
    """A part given its own theme keeps it when its parent is restyled."""
    combo = ui.ComboBox2D(items=["a"])
    combo.drop_down_menu.apply_theme("graphite")

    ui.set_theme("graphite_light")
    ui.set_theme("classic")

    assert combo.drop_down_menu.theme.name == "graphite"
    assert combo.drop_down_menu.panel.background.corner_radius == 10


RESTYLE_MATCHES_FRESH = [
    "panel",
    "panel_border",
    "textblock",
    "textbutton",
    "double_slider",
    "ring_slider",
    "range_slider",
    "spinbox",
    "tabui",
    "card",
    "tree",
    "loader",
    "switch",
    "switch_off",
    "segmented",
]


@pytest.mark.parametrize("theme", ["graphite", "graphite_light"])
@pytest.mark.parametrize("name", RESTYLE_MATCHES_FRESH)
def test_live_restyle_matches_fresh_widget(name, theme, tmp_path):
    """
    A classic widget restyled to Graphite matches one created in Graphite.

    Widgets whose structure depends on the theme, such as inline sliders,
    box indicators or slot heights derived from the font size, are left out.
    """
    factory = _composite_factories(tmp_path)[name]
    widget = factory()
    ui.set_theme(theme)

    npt.assert_array_equal(_render_scene(widget), _render_scene(factory()))


def _render_with(element, interact):
    """
    Render an element after interacting with it through real events.

    Parameters
    ----------
    element : UI
        Element to render.
    interact : callable
        Called with the show manager after the first frame.

    Returns
    -------
    ndarray
        RGB image with values in [0, 255].
    """
    scene = window.Scene(background=(0.2, 0.3, 0.4))
    scene.add(element)
    show_m = window.ShowManager(
        scene=scene, size=(400, 300), window_type="offscreen", pixel_ratio=1
    )
    show_m.render()
    show_m.window.draw()
    interact(show_m)
    show_m.render()
    show_m.window.draw()
    image = np.asarray(show_m.snapshot())[..., :3].astype(int)
    show_m.close()
    return image


def _center(element):
    """
    Get the center of an element.

    Parameters
    ----------
    element : UI
        The element.

    Returns
    -------
    ndarray
        Center (x, y).
    """
    return element.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)


@pytest.mark.parametrize(
    "state",
    [
        "hovered_button",
        "focused_textbox",
        "open_combo",
        "hovered_switch",
        "hovered_segment",
    ],
)
def test_roundtrip_keeps_interaction_states(state):
    """A restyle round trip during hover, focus or an open menu changes nothing."""
    factories = {
        "hovered_button": lambda: ui.TextButton2D("Go", position=(50, 50)),
        "focused_textbox": lambda: ui.TextBox2D(20, 2, text="Type", position=(50, 50)),
        "open_combo": lambda: ui.ComboBox2D(items=["x", "y"], position=(50, 50)),
        "hovered_switch": lambda: ui.Switch2D(label="Axes", position=(50, 50)),
        "hovered_segment": lambda: ui.SegmentedControl2D(["a", "b"], position=(50, 50)),
    }

    def interact(show_m, widget, restyle):
        if state == "hovered_button":
            send_pointer(show_m, "pointer_move", _center(widget))
        elif state == "hovered_switch":
            send_pointer(show_m, "pointer_move", _center(widget.track))
        elif state == "hovered_segment":
            send_pointer(show_m, "pointer_move", _center(widget.segments[1]))
        elif state == "focused_textbox":
            click(show_m, _center(widget))
        else:
            click(show_m, _center(widget.drop_down_button))
        if restyle:
            ui.set_theme("graphite")
            ui.set_theme("classic")

    reference_widget = factories[state]()
    reference = _render_with(
        reference_widget, lambda sm: interact(sm, reference_widget, False)
    )
    widget = factories[state]()
    image = _render_with(widget, lambda sm: interact(sm, widget, True))

    npt.assert_array_equal(image, reference)


RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)


def _tab_ui():
    """
    Build a tab UI whose first tab is open.

    Returns
    -------
    TabUI
        The tab UI.
    """
    tab_ui = ui.TabUI(size=(300, 200), tab_titles=["A", "B"])
    tab_ui.select_tab(0)
    return tab_ui


def _combo_slot(combo):
    """
    Get the first menu item of a combo box.

    Parameters
    ----------
    combo : ComboBox2D
        The combo box.

    Returns
    -------
    ListBoxItem2D
        The first item.
    """
    return combo.drop_down_menu.slots[0]


def _set(name, value):
    """
    Create a function that sets an attribute.

    Parameters
    ----------
    name : str
        Dotted path of the attribute, relative to the widget.
    value : object
        Value to set.

    Returns
    -------
    callable
        Function taking the widget.
    """

    def apply(widget):
        """
        Set the attribute on a widget.

        Parameters
        ----------
        widget : UI
            The widget.
        """
        *path, attribute = name.split(".")
        target = widget
        for part in path:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        setattr(target, attribute, value)

    return apply


USER_VALUES = [
    (
        "rectangle.color",
        lambda: ui.Rectangle2D(),
        _set("color", RED),
        lambda w: w.actor.material.color[:3],
        RED,
    ),
    (
        "rectangle.opacity",
        lambda: ui.Rectangle2D(),
        _set("opacity", 0.4),
        lambda w: w.actor.material.opacity,
        0.4,
    ),
    (
        "box.color",
        lambda: ui.RoundedRectangle2D(),
        _set("color", RED),
        lambda w: w.actor.material.color[:3],
        RED,
    ),
    (
        "box.opacity",
        lambda: ui.RoundedRectangle2D(),
        _set("opacity", 0.4),
        lambda w: w.actor.material.opacity,
        0.4,
    ),
    (
        "box.corner_radius",
        lambda: ui.RoundedRectangle2D(),
        _set("corner_radius", 4),
        lambda w: w.corner_radius,
        4,
    ),
    (
        "box.border_color",
        lambda: ui.RoundedRectangle2D(),
        _set("border_color", RED),
        lambda w: w.border_color[:3],
        RED,
    ),
    (
        "box.border_width",
        lambda: ui.RoundedRectangle2D(),
        _set("border_width", 3),
        lambda w: w.border_width,
        3,
    ),
    (
        "box.shadow_color",
        lambda: ui.RoundedRectangle2D(),
        _set("shadow_color", RED),
        lambda w: w.shadow_color[:3],
        RED,
    ),
    (
        "box.shadow_offset",
        lambda: ui.RoundedRectangle2D(),
        _set("shadow_offset", (2, 3)),
        lambda w: w.shadow_offset,
        (2, 3),
    ),
    (
        "box.shadow_blur",
        lambda: ui.RoundedRectangle2D(),
        _set("shadow_blur", 5),
        lambda w: w.shadow_blur,
        5,
    ),
    (
        "box.shadow_spread",
        lambda: ui.RoundedRectangle2D(),
        _set("shadow_spread", 2),
        lambda w: w.shadow_spread,
        2,
    ),
    (
        "disk.color",
        lambda: ui.Disk2D(outer_radius=5),
        _set("color", RED),
        lambda w: w.actor.material.color[:3],
        RED,
    ),
    (
        "disk.opacity",
        lambda: ui.Disk2D(outer_radius=5),
        _set("opacity", 0.4),
        lambda w: w.actor.material.opacity,
        0.4,
    ),
    (
        "text.color",
        lambda: ui.TextBlock2D(dynamic_bbox=True),
        _set("color", RED),
        lambda w: w.actor.material.color[:3],
        RED,
    ),
    (
        "text.font_size",
        lambda: ui.TextBlock2D(dynamic_bbox=True),
        _set("font_size", 31),
        lambda w: w.actor.font_size,
        31,
    ),
    (
        "text.font_weight",
        lambda: ui.TextBlock2D(dynamic_bbox=True),
        _set("font_weight", 700),
        lambda w: w.actor.material.weight_offset,
        300,
    ),
    (
        "text.background_color",
        lambda: ui.TextBlock2D(dynamic_bbox=True),
        _set("background_color", RED),
        lambda w: w.background.color,
        RED,
    ),
    (
        "slider.default_color",
        lambda: ui.LineSlider2D(),
        _set("default_color", RED),
        lambda w: w.handle.color,
        RED,
    ),
    (
        "slider.active_color",
        lambda: ui.LineSlider2D(),
        _set("active_color", RED),
        lambda w: w.active_color,
        RED,
    ),
    (
        "slider.hover_color",
        lambda: ui.LineSlider2D(),
        _set("hover_color", RED),
        lambda w: w.hover_color,
        RED,
    ),
    (
        "slider.track.color",
        lambda: ui.LineSlider2D(),
        _set("track.color", RED),
        lambda w: w.track.color,
        RED,
    ),
    (
        "slider.handle.color",
        lambda: ui.LineSlider2D(),
        _set("handle.color", RED),
        lambda w: w.handle.color,
        RED,
    ),
    (
        "double_slider.default_color",
        lambda: ui.LineDoubleSlider2D(),
        _set("default_color", RED),
        lambda w: w.handles[0].color,
        RED,
    ),
    (
        "panel.color",
        lambda: ui.Panel2D(size=(50, 50)),
        _set("color", RED),
        lambda w: w.background.color,
        RED,
    ),
    (
        "panel.opacity",
        lambda: ui.Panel2D(size=(50, 50)),
        _set("opacity", 0.4),
        lambda w: w.background.opacity,
        0.4,
    ),
    (
        "panel.background.color",
        lambda: ui.Panel2D(size=(50, 50)),
        _set("background.color", RED),
        lambda w: w.background.color,
        RED,
    ),
    (
        "panel.border_color",
        lambda: ui.Panel2D(size=(50, 50), has_border=True),
        _set("border_color", ("left", RED)),
        lambda w: w.borders["left"].color,
        RED,
    ),
    (
        "panel.border_width",
        lambda: ui.Panel2D(size=(50, 50), has_border=True),
        _set("border_width", ("left", 5)),
        lambda w: w.borders["left"].width,
        5,
    ),
    (
        "tab_panel.color",
        lambda: ui.TabPanel2D(title="x"),
        _set("color", RED),
        lambda w: w.panel.background.color,
        RED,
    ),
    (
        "tab_panel.title_color",
        lambda: ui.TabPanel2D(title="x"),
        _set("title_color", GREEN),
        lambda w: w.text_block.color,
        GREEN,
    ),
    (
        "tab_panel.title_font_size",
        lambda: ui.TabPanel2D(title="x"),
        _set("title_font_size", 27),
        lambda w: w.text_block.font_size,
        27,
    ),
    (
        "tab_ui.active_color",
        _tab_ui,
        _set("active_color", RED),
        lambda w: w.tabs[0].panel.background.color,
        RED,
    ),
    (
        "tab_ui.inactive_color",
        _tab_ui,
        _set("inactive_color", RED),
        lambda w: w.tabs[1].panel.background.color,
        RED,
    ),
    (
        "tab_ui.tab.color",
        _tab_ui,
        _set("tabs.1.color", RED),
        lambda w: w.tabs[1].panel.background.color,
        RED,
    ),
    (
        "button.states",
        lambda: ui.TextButton2D("Go"),
        _set("states", {"default": RED, "hover": RED, "pressed": RED, "disabled": RED}),
        lambda w: w.child.background.color,
        RED,
    ),
    (
        "button.child.color",
        lambda: ui.TextButton2D("Go"),
        _set("child.color", RED),
        lambda w: w.child.color,
        RED,
    ),
    (
        "checkbox.text_color",
        lambda: ui.Checkbox(["a"]),
        _set("text_color", RED),
        lambda w: w.options["a"][1].color,
        RED,
    ),
    (
        "combo.sel_text_color",
        lambda: ui.ComboBox2D(items=["x"]),
        _set("sel_text_color", RED),
        lambda w: w.selection_box.color,
        RED,
    ),
    (
        "combo.sel_bg_color",
        lambda: ui.ComboBox2D(items=["x"]),
        _set("sel_bg_color", RED),
        lambda w: w.selection_box.background.color,
        RED,
    ),
    (
        "combo.menu_txt_color",
        lambda: ui.ComboBox2D(items=["x"]),
        _set("menu_txt_color", RED),
        lambda w: _combo_slot(w).textblock.color,
        RED,
    ),
    (
        "combo.unselected_color",
        lambda: ui.ComboBox2D(items=["x"]),
        _set("unselected_color", RED),
        lambda w: _combo_slot(w).background.color,
        RED,
    ),
    (
        "listbox.text_color",
        lambda: ui.ListBox2D(values=["a"]),
        _set("text_color", RED),
        lambda w: w.slots[0].textblock.color,
        RED,
    ),
    (
        "listbox.unselected_color",
        lambda: ui.ListBox2D(values=["a"]),
        _set("unselected_color", RED),
        lambda w: w.slots[0].background.color,
        RED,
    ),
    (
        "listbox.selected_color",
        lambda: ui.ListBox2D(values=["a"]),
        _set("selected_color", RED),
        lambda w: w.slots[0].selected_color,
        RED,
    ),
    (
        "listbox.scroll_bar_inactive_color",
        lambda: ui.ListBox2D(values=["a"]),
        _set("scroll_bar_inactive_color", RED),
        lambda w: w.scroll_bar.color,
        RED,
    ),
    (
        "list_item.text_color",
        lambda: ui.ListBoxItem2D(on_select=lambda **kwargs: None, size=(50, 20)),
        _set("text_color", RED),
        lambda w: w.textblock.color,
        RED,
    ),
    (
        "list_item.unselected_color",
        lambda: ui.ListBoxItem2D(on_select=lambda **kwargs: None, size=(50, 20)),
        _set("unselected_color", RED),
        lambda w: w.background.color,
        RED,
    ),
    (
        "list_item.hover_color",
        lambda: ui.ListBoxItem2D(on_select=lambda **kwargs: None, size=(50, 20)),
        _set("hover_color", RED),
        lambda w: w.hover_color,
        RED,
    ),
    (
        "tree_node.color",
        lambda: ui.elements.TreeNode2D(label="n"),
        _set("color", RED),
        lambda w: w.title_panel.background.color,
        RED,
    ),
    (
        "tree_node.opacity",
        lambda: ui.elements.TreeNode2D(label="n"),
        _set("opacity", 0.4),
        lambda w: w.title_panel.background.opacity,
        0.4,
    ),
    (
        "tree_node.content_color",
        lambda: ui.elements.TreeNode2D(label="n"),
        _set("content_color", RED),
        lambda w: w.content_panel.background.color,
        RED,
    ),
    (
        "tree.scroll_bar_inactive_color",
        lambda: ui.elements.Tree2D(structure=[{"R": ["a"]}], tree_name="T"),
        _set("scroll_bar_inactive_color", RED),
        lambda w: w.scroll_bar.color,
        RED,
    ),
    (
        "card.color",
        lambda: _card(),
        _set("color", RED),
        lambda w: w.panel.background.color,
        RED,
    ),
    (
        "textbox.text.background_color",
        lambda: ui.TextBox2D(10, 1, text="a"),
        _set("text.background_color", RED),
        lambda w: w.text.background.color,
        RED,
    ),
    (
        "combo.selection_box.background_color",
        lambda: ui.ComboBox2D(items=["x"]),
        _set("selection_box.background_color", RED),
        lambda w: w.selection_box.background.color,
        RED,
    ),
    (
        "card.bg_color",
        lambda: _card(),
        _set("bg_color", RED),
        lambda w: w.panel.background.color,
        RED,
    ),
    (
        "card.opacity",
        lambda: _card(),
        _set("opacity", 0.4),
        lambda w: w.panel.background.opacity,
        0.4,
    ),
    (
        "spinbox.panel_color",
        lambda: ui.SpinBox(),
        _set("panel_color", RED),
        lambda w: w.panel.background.color,
        RED,
    ),
    (
        "switch.track_on_color",
        lambda: ui.Switch2D(checked=True),
        _set("track_on_color", RED),
        lambda w: w.track.color,
        RED,
    ),
    (
        "switch.track_color",
        lambda: ui.Switch2D(),
        _set("track_color", RED),
        lambda w: w.track.color,
        RED,
    ),
    (
        "switch.knob_color",
        lambda: ui.Switch2D(),
        _set("knob_color", RED),
        lambda w: w.knob.color,
        RED,
    ),
    (
        "switch.label_color",
        lambda: ui.Switch2D(label="a"),
        _set("label_color", RED),
        lambda w: w.label.color,
        RED,
    ),
    (
        "switch.track.color",
        lambda: ui.Switch2D(),
        _set("track.color", RED),
        lambda w: w.track.color,
        RED,
    ),
    (
        "segmented.color",
        lambda: ui.SegmentedControl2D(["a", "b"]),
        _set("color", RED),
        lambda w: w.container.color,
        RED,
    ),
    (
        "segmented.selected_color",
        lambda: ui.SegmentedControl2D(["a", "b"]),
        _set("selected_color", RED),
        lambda w: w.segments[0].color,
        RED,
    ),
    (
        "segmented.text_color",
        lambda: ui.SegmentedControl2D(["a", "b"]),
        _set("text_color", RED),
        lambda w: w.labels[1].color,
        RED,
    ),
    (
        "segmented.selected_text_color",
        lambda: ui.SegmentedControl2D(["a", "b"]),
        _set("selected_text_color", RED),
        lambda w: w.labels[0].color,
        RED,
    ),
]


def _card():
    """
    Build a small card.

    Returns
    -------
    Card2D
        The card.
    """
    fetch_viz_icons()
    return ui.Card2D(read_viz_icons(fname="play3.png"), title_text="T", size=(200, 220))


@pytest.mark.parametrize(
    "factory, apply, read, expected",
    [row[1:] for row in USER_VALUES],
    ids=[row[0] for row in USER_VALUES],
)
def test_user_values_survive_theme_round_trip(factory, apply, read, expected):
    """A value the user sets is applied at once and kept by restyles."""
    widget = factory()
    apply(widget)
    npt.assert_allclose(np.asarray(read(widget), dtype=float), expected, atol=1e-3)

    ui.set_theme("graphite")
    ui.set_theme("classic")

    npt.assert_allclose(np.asarray(read(widget), dtype=float), expected, atol=1e-3)
    _render_scene(widget)


def test_set_theme_accepts_numpy_items_and_equal_loaded_theme(tmp_path):
    """Themes with arrays compare without error, and an equal theme is a no-op."""
    first = ui.builtin_theme("classic").derive(
        name="a", items={"Panel2D": {"color": np.array([1.0, 0, 0])}}
    )
    second = ui.builtin_theme("classic").derive(
        name="b", items={"Panel2D": {"color": np.array([1.0, 0, 0])}}
    )
    ui.set_theme(first)
    panel = ui.Panel2D(size=(10, 10))
    panel.background.actor.material.color = (0, 1, 0, 1)
    ui.set_theme(second)
    npt.assert_array_almost_equal(panel.background.actor.material.color[:3], (0, 1, 0))

    path = tmp_path / "theme.json"
    first.save(path)
    ui.set_theme(ui.Theme.load(path))
    npt.assert_array_almost_equal(panel.background.actor.material.color[:3], (0, 1, 0))


def test_overflowing_tree_restyle_matches_fresh_render():
    """A restyled overflowing tree keeps its active scroll bar like a fresh one."""

    def tree():
        structure = [{f"Node{i}": [f"Leaf{i}{j}" for j in range(4)]} for i in range(6)]
        return ui.elements.Tree2D(
            structure=structure, tree_name="T", size=(200, 120), position=(10, 10)
        )

    widget = tree()
    ui.set_theme("graphite")
    npt.assert_array_equal(_render_scene(widget), _render_scene(tree()))
    npt.assert_array_almost_equal(widget.scroll_bar.color, _hex("#ECEDEF"))


def test_apply_theme_on_parent_skips_parts_with_their_own_theme():
    """A part given its own theme keeps it when the parent gets another one."""
    combo = ui.ComboBox2D(items=["a"])
    combo.drop_down_menu.apply_theme("graphite")
    combo.apply_theme("graphite_light")

    assert combo.drop_down_menu.theme.name == "graphite"
    assert combo.selection_box.theme.name == "graphite_light"


def test_combo_resize_font_scale_survives_restyle():
    """A combo box keeps the font scale of a resize through a round trip."""
    combo = ui.ComboBox2D(items=["x", "y"], size=(300, 200))
    combo.resize((300, 400))
    scaled = combo.font_size

    ui.set_theme("graphite")
    ui.set_theme("classic")

    assert combo.font_size == scaled
    assert combo.selection_box.font_size == scaled


def _all_parts(element):
    """
    Collect a widget and every UI component reachable from it.

    Parameters
    ----------
    element : UI
        The widget.

    Returns
    -------
    dict
        Mapping of ``id`` to component.
    """
    found = {}
    stack = [element]
    while stack:
        current = stack.pop()
        if id(current) in found:
            continue
        found[id(current)] = current
        for value in vars(current).values():
            values = value if isinstance(value, (list, tuple)) else [value]
            if isinstance(value, dict):
                values = list(value.values())
            for item in values:
                if isinstance(item, (list, tuple)):
                    stack.extend(x for x in item if isinstance(x, ui.UI))
                elif isinstance(item, ui.UI):
                    stack.append(item)
    return found


def test_set_theme_refreshes_each_widget_once():
    """A theme change refreshes every widget and part at most once."""
    widgets = [
        ui.ListBox2D(values=[str(i) for i in range(30)], size=(200, 300)),
        ui.ComboBox2D(items=["a", "b", "c"]),
        ui.LineSlider2D(),
        ui.TabUI(tab_titles=["A", "B"]),
    ]
    panel = ui.Panel2D(size=(200, 200))
    for _ in range(3):
        panel.add(ui.TextBlock2D(text="t", dynamic_bbox=True))
    widgets.append(panel)
    card = _card()
    widgets.append(card)

    ui.set_theme("graphite")

    refreshed = ui.UIContext.last_refreshes
    assert refreshed.index(id(card)) < refreshed.index(id(card.panel))
    combo = widgets[1]
    assert refreshed.index(id(combo)) < refreshed.index(id(combo.drop_down_menu))
    assert len(refreshed) == len(set(refreshed))
    for widget in widgets:
        assert id(widget) in refreshed
    for widget in widgets:
        for part_id in _all_parts(widget):
            assert refreshed.count(part_id) <= 1
    assert id(widgets[0].slots[0]) in refreshed


def test_restyled_text_button_uses_theme_font_family():
    """A button restyled to Graphite uses the Graphite font like a fresh one."""
    button = ui.TextButton2D("Go")
    ui.set_theme("graphite")
    fresh = ui.TextButton2D("Go")

    assert button.child.font_family == fresh.child.font_family
    npt.assert_array_almost_equal(button.child.background.color, _hex("#F2F3F5"))


def test_restyled_playback_panel_hides_square_border():
    """With rounded corners, the panel border is drawn by the rounded box."""
    playback = ui.PlaybackPanel()
    assert playback.panel.borders["left"].actor.visible

    ui.set_theme("graphite")

    assert not playback.panel.borders["left"].actor.visible
    assert playback.panel.background.border_width == 2
    playback.panel.set_visibility(True)
    assert not playback.panel.borders["left"].actor.visible

    ui.set_theme("classic")
    assert playback.panel.borders["left"].actor.visible
    assert playback.panel.background.border_width == 0


def _rects_visible(panel):
    """
    Get the visibility of the four border rectangles of a panel.

    Parameters
    ----------
    panel : Panel2D
        Panel with ``has_border``.

    Returns
    -------
    list of bool
        Visibility of the left, right, top and bottom rectangles.
    """
    return [panel.borders[side].actor.visible for side in panel.border_sides]


def test_rounded_panel_border_setters_reach_the_background(caplog):
    """With rounded corners, per-side setters change the drawn border."""
    ui.set_theme("graphite")
    panel = ui.Panel2D(
        size=(200, 150), has_border=True, border_width=3, border_color=(1, 0, 0)
    )
    assert _rects_visible(panel) == [False] * 4
    assert panel.background.border_width == 3

    with caplog.at_level("WARNING"):
        panel.border_color = ("left", (0, 1, 0))
        panel.border_width = ("top", 8)
    assert "all sides" in caplog.text
    npt.assert_array_almost_equal(panel.background.border_color[:3], (0, 1, 0))
    assert panel.background.border_width == 8
    assert panel.border_width == [8] * 4
    npt.assert_array_almost_equal(panel.border_color[0], (0, 1, 0))

    ui.set_theme("graphite_light")
    assert panel.background.border_width == 8

    ui.set_theme("classic")
    assert _rects_visible(panel) == [True] * 4
    assert panel.background.border_width == 0
    npt.assert_array_almost_equal(panel.borders["left"].color, (0, 1, 0))
    npt.assert_array_almost_equal(panel.borders["right"].color, (1, 0, 0))
    assert panel.borders["top"].height == 8


def test_hidden_panel_stays_hidden_through_restyles():
    """A hidden bordered panel stays hidden until it is shown again."""
    ui.set_theme("graphite")
    panel = ui.Panel2D(size=(100, 80), has_border=True, border_width=2)
    panel.set_visibility(False)

    ui.set_theme("classic")
    assert _rects_visible(panel) == [False] * 4
    assert not panel.background.actor.visible

    panel.set_visibility(True)
    assert _rects_visible(panel) == [True] * 4


def test_restyle_keeps_a_border_side_the_user_hid():
    """Only rectangles hidden for rounding are shown again."""
    panel = ui.Panel2D(size=(100, 80), has_border=True, border_width=2)
    panel.borders["left"].set_visibility(False)

    ui.set_theme("graphite")
    ui.set_theme("classic")

    assert not panel.borders["left"].actor.visible
    assert panel.borders["right"].actor.visible


def test_panel_rounding_follows_the_background_radius():
    """A radius set on the background counts as rounded corners."""
    panel = ui.Panel2D(size=(100, 80), has_border=True, border_width=2)
    panel.background.corner_radius = 8

    assert _rects_visible(panel) == [False] * 4
    assert panel.background.border_width == 2

    panel.background.corner_radius = 0
    assert _rects_visible(panel) == [True] * 4
    assert panel.background.border_width == 0


def test_color_attributes_are_rgb_tuples_in_every_theme():
    """Themed color attributes read back as RGB float tuples."""
    classic = ui.LineSlider2D()
    ui.set_theme("graphite")
    graphite = ui.LineSlider2D()

    for slider in (classic, graphite):
        assert isinstance(slider.active_color, tuple)
        assert len(slider.active_color) == 3
        assert all(isinstance(c, float) for c in slider.active_color)
    npt.assert_allclose(graphite.active_color, _hex("#F2F3F5"))


@pytest.mark.parametrize(
    "start, end", [("classic", "graphite"), ("graphite", "classic")]
)
@pytest.mark.parametrize("kind", ["panel", "playback"])
def test_hide_switch_show_matches_fresh_widget(kind, start, end):
    """Hiding, switching theme and showing again draws like a shown widget."""

    def make():
        if kind == "panel":
            return ui.Panel2D(
                size=(200, 150), position=(20, 20), has_border=True, border_width=3
            )
        return ui.PlaybackPanel(position=(20, 50), width=380)

    ui.set_theme(start)
    widget = make()
    reference = make()
    widget.set_visibility(False)
    ui.set_theme(end)
    widget.set_visibility(True)

    npt.assert_array_equal(_render_scene(widget), _render_scene(reference))
    if kind == "panel":
        npt.assert_array_equal(_render_scene(widget), _render_scene(make()))


def test_rounded_border_uses_the_latest_side_color():
    """The rounded border shows the color set last, even if set while square."""
    ui.set_theme("graphite")
    panel = ui.Panel2D(size=(200, 150), has_border=True, border_width=2)
    panel.border_color = ("left", (1, 0, 0))
    ui.set_theme("classic")
    panel.border_color = ("left", (0, 0, 1))
    ui.set_theme("graphite")

    npt.assert_array_almost_equal(panel.background.border_color[:3], (0, 0, 1))
    assert panel.border_color == [(0.0, 0.0, 1.0)] * 4


def test_border_color_getter_returns_float_tuples():
    """The border colors read back as float tuples, square or rounded."""
    panel = ui.Panel2D(size=(100, 100), has_border=True, border_color="#ff0000")
    assert panel.border_color == [(1.0, 0.0, 0.0)] * 4
    ui.set_theme("graphite")
    assert all(isinstance(color, tuple) for color in panel.border_color)
    assert all(isinstance(c, float) for c in panel.border_color[0])


def test_showing_a_side_follows_the_border_rule():
    """Showing one side directly cannot bypass rounding or a hidden panel."""
    ui.set_theme("graphite")
    rounded = ui.Panel2D(size=(100, 80), has_border=True, border_width=2)
    rounded.borders["left"].set_visibility(False)
    rounded.borders["left"].set_visibility(True)
    assert _rects_visible(rounded) == [False] * 4

    ui.set_theme("classic")
    assert _rects_visible(rounded) == [True] * 4
    hidden = ui.Panel2D(size=(100, 80), has_border=True, border_width=2)
    hidden.set_visibility(False)
    hidden.borders["left"].set_visibility(True)
    assert _rects_visible(hidden) == [False] * 4
    hidden.set_visibility(True)
    assert _rects_visible(hidden) == [True] * 4


def test_per_side_values_read_back_after_leaving_rounded_mode():
    """After rounding ends, the getters report the values set per side."""
    ui.set_theme("graphite")
    panel = ui.Panel2D(
        size=(100, 80), has_border=True, border_width=3, border_color=(1, 0, 0)
    )
    panel.border_color = ("left", (0, 1, 0))
    panel.border_width = ("top", 6)

    ui.set_theme("classic")

    assert panel.border_color == [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    assert panel.border_width[2] == 6


def _box(element):
    """
    Get the pixel box an element covers.

    Parameters
    ----------
    element : UI
        The element.

    Returns
    -------
    tuple of int
        Left, top, right and bottom edges, with y pointing down.
    """
    x, y = (int(round(v)) for v in element.get_position())
    width, height = (int(round(v)) for v in element.size)
    return x, y, x + width, y + height


def _region(image, element, within):
    """
    Get the pixels an element covers inside the straight part of another.

    The clipping box is inset horizontally by its corner radius, so rounded
    corners and the scene behind them are left out.

    Parameters
    ----------
    image : ndarray
        RGB image with y pointing down, as rendered by ``_render_scene``.
    element : UI
        Element whose box is cut out.
    within : Panel2D
        Panel whose box clips the region.

    Returns
    -------
    ndarray
        Pixels of the clipped box, of shape (height, width, 3).
    """
    left, top, right, bottom = _box(element)
    clip = _box(within)
    inset = int(np.ceil(np.max(within.background.corner_radius)))
    left, top = max(left, clip[0] + inset), max(top, clip[1])
    right, bottom = min(right, clip[2] - inset), min(bottom, clip[3])
    return image[top:bottom, left:right]


def _contrast_spread(pixels, background):
    """
    Get how far the most different pixel is from a background color.

    Parameters
    ----------
    pixels : ndarray
        RGB pixels in [0, 255].
    background : array_like
        Rendered RGB background color in [0, 255].

    Returns
    -------
    float
        Largest per-pixel color distance to the background, in [0, 255].
    """
    difference = pixels.reshape(-1, 3) - np.asarray(background, dtype=float)[:3]
    return float(np.abs(difference).max())


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_graphite_tree_labels_contrast_with_title_background(name):
    """Tree node labels and expand icons stand out from their title bars."""
    with ui.theme_scope(name):
        tree = ui.elements.Tree2D(
            structure=[{"Root": ["Child1", "Child2"]}],
            tree_name="Tree",
            position=(20, 20),
            size=(300, 200),
        )

    image = _render_scene(tree)
    for node in [tree.base_node, *tree.base_node.child_nodes]:
        title = node.title_panel
        left, top, right, bottom = _box(title)
        inset = int(np.ceil(np.max(title.background.corner_radius)))
        background = image[(top + bottom) // 2, right - inset - 2]
        label = _region(image, node.label_text, title)
        assert _contrast_spread(label, background) > 180
        if node is not tree.base_node:
            icon = _region(image, node.button, title)
            assert _contrast_spread(icon, background) > 150


def _playback_buttons(playback):
    """
    Get the textured buttons of a playback panel.

    Parameters
    ----------
    playback : PlaybackPanel
        The panel.

    Returns
    -------
    list of TexturedButton2D
        Play or pause, stop, loop, slower and faster buttons.
    """
    return [
        playback._play_pause_btn,
        playback._stop_btn,
        playback._loop_btn,
        playback._slow_down_btn,
        playback._speed_up_btn,
    ]


def _playback_icon_spreads(playback):
    """
    Render a playback panel and measure how its icons stand out.

    Parameters
    ----------
    playback : PlaybackPanel
        The panel.

    Returns
    -------
    list of float
        Contrast spread of each play, stop and loop icon against the panel.
    """
    image = _render_scene(playback)
    panel = playback.panel
    left, top, _, bottom = _box(panel)
    background = image[(top + bottom) // 2, left + 4]
    return [
        _contrast_spread(_region(image, button, panel), background)
        for button in _playback_buttons(playback)[:3]
    ]


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_graphite_playback_icons_contrast_with_panel(name):
    """Playback button icons are drawn in a color that contrasts the panel."""
    with ui.theme_scope(name):
        playback = ui.PlaybackPanel(position=(10, 50), width=380)

    for spread in _playback_icon_spreads(playback):
        assert spread > 150


def test_playback_icons_follow_live_theme_change():
    """set_theme tints the icons of an existing classic playback panel."""
    playback = ui.PlaybackPanel(position=(10, 50), width=380)
    assert playback._stop_btn.icon_color is None

    ui.set_theme("graphite")

    expected = tuple(_hex("#ECEDEF"))
    for button in _playback_buttons(playback):
        npt.assert_allclose(button.icon_color, expected)
    for spread in _playback_icon_spreads(playback):
        assert spread > 150

    ui.set_theme("classic")
    for button in _playback_buttons(playback):
        assert button.icon_color is None


def test_icon_color_is_an_rgb_tuple_and_restores_image_colors():
    """icon_color reads back as RGB floats; None shows the images again."""
    fetch_viz_icons()
    button = ui.TexturedButton2D(states={"default": read_viz_icons(fname="stop2.png")})
    original = button.child.actor.material.map.texture
    assert button.icon_color is None

    button.icon_color = "#FF0000"
    assert button.icon_color == (1.0, 0.0, 0.0)
    tinted = button.child.actor.material.map.texture
    assert tinted is not original
    tinted_data = np.asarray(tinted.data)
    npt.assert_array_equal(tinted_data[..., 3], np.asarray(original.data)[..., 3])
    visible = tinted_data[..., 3] > 0
    npt.assert_array_equal(tinted_data[visible][:, :3], [[255, 0, 0]] * visible.sum())

    button.icon_color = (255, 0, 0)
    assert button.child.actor.material.map.texture is tinted

    button.icon_color = None
    assert button.child.actor.material.map.texture is original
    assert button.icon_color is None


def test_icon_tint_is_kept_when_the_button_changes_state():
    """A toggled or pressed button still shows tinted images."""
    with ui.theme_scope("graphite"):
        playback = ui.PlaybackPanel(width=380)
    button = playback._play_pause_btn
    tinted = set(map(id, button._tinted_map.values()))

    button.do_click()
    assert button.toggled
    assert id(button.child.actor.material.map.texture) in tinted
    button.is_pressed = True
    button.update_visual_state()
    assert id(button.child.actor.material.map.texture) in tinted


@pytest.mark.parametrize("theme_name", ["graphite", "classic"])
def test_user_icon_color_survives_set_theme(theme_name):
    """An icon color set by the user is kept when the global theme changes."""
    playback = ui.PlaybackPanel(width=380)
    with ui.theme_scope("graphite"):
        tree = ui.elements.Tree2D(
            structure=[{"Root": ["Child"]}], tree_name="Tree", size=(200, 100)
        )
    node = tree.base_node.child_nodes[0]
    playback._stop_btn.icon_color = (0, 1, 0)
    node.button.icon_color = (0, 0, 1)

    ui.set_theme("graphite" if theme_name == "classic" else "classic")
    ui.set_theme(theme_name)
    node.selected_color = (0.5, 0.1, 0.1)

    assert playback._stop_btn.icon_color == (0.0, 1.0, 0.0)
    assert node.button.icon_color == (0.0, 0.0, 1.0)


def test_icon_color_defaults_to_none_without_theme_item():
    """A theme without icon_color draws images in their own colors."""
    fetch_viz_icons()
    bare = ui.Theme(name="bare", items={"UI": {"font_family": "Noto Sans"}})
    with ui.theme_scope(bare):
        button = ui.TexturedButton2D(
            states={"default": read_viz_icons(fname="stop2.png")}
        )

    assert button.icon_color is None


def test_icon_color_none_after_theme_change_matches_siblings():
    """Clearing a user icon color shows the color the parent last pushed."""
    playback = ui.PlaybackPanel(width=380)
    tree = ui.elements.Tree2D(
        structure=[{"A": ["a1"]}, {"B": ["b1"]}], tree_name="Tree", size=(200, 100)
    )
    first, second = tree.base_node.child_nodes[:2]
    playback._stop_btn.icon_color = (0, 1, 0)
    first.button.icon_color = (0, 0, 1)

    ui.set_theme("graphite")
    playback._stop_btn.icon_color = None
    first.button.icon_color = None

    assert playback._stop_btn.icon_color == playback._play_pause_btn.icon_color
    npt.assert_allclose(playback._stop_btn.icon_color, _hex("#ECEDEF"))
    assert first.button.icon_color == second.button.icon_color
    npt.assert_allclose(first.button.icon_color, _hex("#ECEDEF"))


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_graphite_spinbox_icons_contrast_with_panel(name):
    """SpinBox arrow icons are drawn in a color that contrasts the panel."""
    with ui.theme_scope(name):
        spinbox = ui.SpinBox(position=(50, 50))

    spinbox.panel.update_layout()
    image = _render_scene(spinbox)
    left, top, _, bottom = _box(spinbox.panel)
    background = image[(top + bottom) // 2, left + 2]
    for button in (spinbox.increment_button, spinbox.decrement_button):
        assert button.icon_color is not None
        assert _contrast_spread(_region(image, button, spinbox.panel), background) > 150


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_graphite_combobox_arrow_uses_text_color(name):
    """The ComboBox arrow is drawn in the theme text color."""
    with ui.theme_scope(name):
        combo = ui.ComboBox2D(items=["x", "y"], position=(50, 50))
    text = ui.builtin_theme(name).get(ui.TextBlock2D, "color")

    button = combo.drop_down_button
    npt.assert_allclose(button.icon_color, _hex(text))
    data = np.asarray(button.child.actor.material.map.texture.data)
    visible = data[..., 3] > 0
    npt.assert_allclose(
        data[visible][:, :3] / 255, [_hex(text)] * visible.sum(), atol=1 / 255
    )


def test_tinted_texture_handles_gray_alpha_images():
    """A gray and alpha image is tinted into an RGBA image with its alpha."""
    alpha = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    gray_alpha = np.stack([np.full((2, 2), 40, dtype=np.uint8), alpha], axis=-1)
    texture = ui.elements._tinted_texture(
        ui.elements.Texture(gray_alpha, dim=2), (1.0, 0.0, 0.0)
    )

    data = np.asarray(texture.data)
    assert data.shape == (2, 2, 4)
    npt.assert_array_equal(data[..., 3], alpha)
    npt.assert_array_equal(data[..., :3], [[[255, 0, 0]] * 2] * 2)


def test_text_color_none_after_theme_change_follows_button():
    """Clearing a user text color shows the color the button last pushed."""
    button = ui.TextButton2D("Go")
    button.child.color = (1, 0, 0)

    ui.set_theme("graphite")
    button.child.color = None

    expected = ui.TextButton2D("Go").child.color
    npt.assert_allclose(button.child.color, expected)
    npt.assert_allclose(button.child.color, _hex("#141518"))


@pytest.mark.parametrize("name", ["graphite", "graphite_light"])
def test_switch_and_segmented_control_follow_the_mockup(name):
    """Fresh Graphite widgets use the mockup sizes and palette colors."""
    palette = ui.theme.GRAPHITE_PALETTE
    if name == "graphite_light":
        palette = ui.theme.GRAPHITE_LIGHT_PALETTE
    with ui.theme_scope(name):
        switch = ui.Switch2D(checked=True)
        control = ui.SegmentedControl2D(["Axial", "Cor", "Sag"])

    npt.assert_array_equal(switch.track.size, (30, 18))
    npt.assert_array_equal(switch.knob.size, (12, 12))
    assert switch.knob.corner_radius == 6
    npt.assert_allclose(switch.track.color, _hex(palette["accent"]))
    npt.assert_allclose(switch.knob.color, _hex(palette["on_accent"]))
    assert control.container.corner_radius == 8
    assert control.container.border_width == 1
    assert control.segments[0].corner_radius == 6
    npt.assert_allclose(control.container.color, _hex(palette["fill"]))
    npt.assert_allclose(control.segments[0].color, _hex(palette["accent"]))
    npt.assert_allclose(control.labels[0].color, _hex(palette["on_accent"]))
    npt.assert_allclose(control.labels[1].color, _hex(palette["muted"]))
    assert control.labels[0].font_weight == 600


def test_user_switch_and_segment_colors_survive_graphite():
    """Colors set by the user are drawn after a switch to Graphite."""
    switch = ui.Switch2D(checked=True, track_on_color=GREEN, position=(20, 20))
    control = ui.SegmentedControl2D(["a", "b"], position=(20, 80))
    control.selected_color = RED

    ui.set_theme("graphite")

    npt.assert_array_almost_equal(switch.track.color, GREEN)
    npt.assert_array_almost_equal(control.segments[0].color, RED)
    npt.assert_array_almost_equal(control.container.color, _hex("#26282C"))
    track_x, track_y = switch.track.get_position() + (5, 9)
    segment_x, segment_y = control.segments[0].get_position() + (4, 4)
    scene = window.Scene(background=(0, 0, 0))
    scene.add(switch, control)
    show_m = window.ShowManager(
        scene=scene, size=(200, 200), window_type="offscreen", pixel_ratio=1
    )
    show_m.render()
    show_m.window.draw()
    image = np.asarray(show_m.snapshot())[..., :3].astype(int)
    show_m.close()
    npt.assert_allclose(image[int(track_y), int(track_x)], (0, 255, 0), atol=3)
    npt.assert_allclose(image[int(segment_y), int(segment_x)], (255, 0, 0), atol=3)
