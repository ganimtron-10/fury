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


@pytest.mark.parametrize("state", ["hovered_button", "focused_textbox", "open_combo"])
def test_roundtrip_keeps_interaction_states(state):
    """A restyle round trip during hover, focus or an open menu changes nothing."""
    factories = {
        "hovered_button": lambda: ui.TextButton2D("Go", position=(50, 50)),
        "focused_textbox": lambda: ui.TextBox2D(20, 2, text="Type", position=(50, 50)),
        "open_combo": lambda: ui.ComboBox2D(items=["x", "y"], position=(50, 50)),
    }

    def interact(show_m, widget, restyle):
        if state == "hovered_button":
            send_pointer(show_m, "pointer_move", _center(widget))
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


NEW_THEMES = ["hairline", "soft_pill", "frost", "ember"]

NEW_THEME_COLORS = {
    "hairline": {"accent": "#FFFFFF", "fill": "#FFFFFF", "round": False},
    "soft_pill": {"accent": "#8B7CFF", "fill": "#8B7CFF", "round": True},
    "frost": {"accent": "#2563EB", "fill": "#2563EB", "round": True},
    "ember": {"accent": "#FF7A45", "fill": "#FF7A45", "round": True},
}


def _render_on(element, background, points):
    """
    Render one UI element over a background color and read some pixels.

    Parameters
    ----------
    element : UI
        Element to render.
    background : tuple of float
        RGB background of the scene, in [0, 1].
    points : list of (int, int)
        Pixel coordinates (x, y), with y pointing down.

    Returns
    -------
    list of ndarray
        RGB values in [0, 255] of each point.
    """
    scene = window.Scene(background=background)
    scene.add(element)
    show_m = window.ShowManager(
        scene=scene, size=(200, 200), window_type="offscreen", pixel_ratio=1
    )
    show_m.render()
    show_m.window.draw()
    image = np.asarray(show_m.snapshot())[..., :3].astype(int)
    show_m.close()
    return [image[y, x] for x, y in points]


def _over_black(color, opacity):
    """
    Get the color of a translucent surface drawn over black.

    Blending happens in linear light, like the renderer.

    Parameters
    ----------
    color : str
        Hex color of the surface.
    opacity : float
        Opacity of the surface.

    Returns
    -------
    ndarray
        RGB values in [0, 255].
    """
    srgb = _hex(color)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    linear = linear * opacity
    blended = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return blended * 255


def test_theme_names_lists_builtin_themes():
    """theme_names lists every built-in theme, and density is not one."""
    names = ui.theme_names()

    assert names == [
        "classic",
        "graphite",
        "graphite_light",
        "hairline",
        "soft_pill",
        "frost",
        "ember",
    ]
    assert "compact" not in names
    for name in names:
        assert ui.builtin_theme(name).name == name


@pytest.mark.parametrize("name", ["missing", "compact", "Graphite"])
def test_unknown_theme_names_raise(name):
    """Unknown names raise a ValueError listing the available themes."""
    with pytest.raises(ValueError, match=f"Unknown theme '{name}'.*hairline"):
        ui.builtin_theme(name)
    with pytest.raises(ValueError, match=f"Unknown theme '{name}'"):
        ui.set_theme(name)
    assert ui.get_theme().name == "classic"


@pytest.mark.parametrize("name", NEW_THEMES)
def test_new_themes_define_every_graphite_item(name):
    """New themes define the same items as Graphite, so restyling back works."""
    graphite = ui.builtin_theme("graphite").to_dict()["items"]
    theme = ui.builtin_theme(name).to_dict()["items"]

    assert set(theme) == set(graphite)
    for type_name, items in graphite.items():
        assert set(theme[type_name]) == set(items), type_name


@pytest.mark.parametrize("name", NEW_THEMES)
def test_new_theme_panel_corner_and_border(name):
    """Round themes leave the panel corner empty; Hairline fills it."""
    with ui.theme_scope(name):
        panel = ui.Panel2D(size=(120, 100), position=(40, 40))
    center, border, corner = render_pixels(panel, [(100, 90), (40, 90), (40, 40)])
    theme = ui.builtin_theme(name)
    background = _over_black(
        theme.get(ui.Panel2D, "color"), theme.get(ui.Panel2D, "opacity")
    )

    border_color = _over_black(
        theme.get(ui.Panel2D, "border_color"), theme.get(ui.Panel2D, "opacity")
    )

    npt.assert_allclose(center, background, atol=3)
    assert np.abs(border - border_color).sum() < np.abs(border - background).sum()
    if NEW_THEME_COLORS[name]["round"]:
        assert corner.max() <= 2
    else:
        assert corner.sum() > center.sum()


@pytest.mark.parametrize("name", NEW_THEMES)
def test_new_theme_button_colors_and_radius(name):
    """A primary button is drawn in the accent color, with the theme radius."""
    with ui.theme_scope(name):
        button = ui.TextButton2D("", size=(100, 40), position=(50, 80))
    inside, corner = render_pixels(button, [(60, 100), (50, 80)])

    npt.assert_allclose(inside, _hex(NEW_THEME_COLORS[name]["accent"]) * 255, atol=3)
    if NEW_THEME_COLORS[name]["round"]:
        assert corner.max() < 40
    else:
        assert corner.min() > 150


@pytest.mark.parametrize("name", NEW_THEMES)
def test_new_theme_slider_fill_and_track(name):
    """The slider fill left of the handle uses the accent, the track does not."""
    with ui.theme_scope(name):
        slider = ui.LineSlider2D(length=160, position=(20, 80), initial_value=75)
    left, top = slider.track.get_position()
    y = int(top + slider.track.size[1] / 2 - 0.5)
    fill, track = render_pixels(slider, [(int(left) + 30, y), (int(left) + 150, y)])

    npt.assert_allclose(fill, _hex(NEW_THEME_COLORS[name]["fill"]) * 255, atol=3)
    assert np.abs(track - fill).max() > 30


def test_frost_panel_shows_the_scene_without_shadow_inside():
    """Frost lets the scene show through and its shadow stays outside."""
    strong_shadow = ui.builtin_theme("frost").derive(
        items={"Panel2D": {"shadow_color": (0, 0, 0, 0.9)}}
    )
    no_shadow = ui.builtin_theme("frost").derive(
        items={"Panel2D": {"shadow_color": (0, 0, 0, 0), "shadow_blur": 0}}
    )
    points = [(100, 80), (100, 135)]
    with ui.theme_scope(strong_shadow):
        shadowed = ui.Panel2D(size=(120, 100), position=(40, 30))
    with ui.theme_scope(no_shadow):
        plain = ui.Panel2D(size=(120, 100), position=(40, 30))

    light = (0.85, 0.87, 0.9)
    inside, below = _render_on(shadowed, light, points)
    plain_inside, plain_below = _render_on(plain, light, points)
    on_red = _render_on(plain, (1, 0, 0), points)[0]

    npt.assert_allclose(inside, plain_inside, atol=2)
    assert below.sum() < plain_below.sum() - 60
    assert on_red[0] - on_red[2] > 20
    assert ui.builtin_theme("frost").get(ui.Panel2D, "opacity") == 0.78


@pytest.mark.parametrize("name", NEW_THEMES)
@pytest.mark.parametrize("widget", RESTYLE_MATCHES_FRESH)
def test_graphite_to_new_theme_and_back_matches_fresh(name, widget, tmp_path):
    """Graphite widgets restyled to a new theme and back match fresh ones."""
    factory = _composite_factories(tmp_path)[widget]
    ui.set_theme("graphite")
    element = factory()
    graphite_reference = _render_scene(factory())

    ui.set_theme(name)
    npt.assert_array_equal(_render_scene(element), _render_scene(factory()))

    ui.set_theme("graphite")
    npt.assert_array_equal(_render_scene(element), graphite_reference)


@pytest.mark.parametrize("name", ui.theme_names())
def test_compact_density_scales_metrics_down(name):
    """Compact density shrinks fonts, spacing, radii and slider sizes."""
    comfortable = ui.builtin_theme(name)
    compact = comfortable.derive(density="compact")

    assert compact.density == "compact"
    assert compact.name == "custom"
    for ui_type, item in [
        (ui.TextBlock2D, "font_size"),
        (ui.TextButton2D, "font_size"),
        (ui.LineSlider2D, "font_size"),
        (ui.LineSlider2D, "handle_outer_radius"),
    ]:
        assert compact.get(ui_type, item) < comfortable.get(ui_type, item), item
    line_width = comfortable.get(ui.LineSlider2D, "line_width")
    assert compact.get(ui.LineSlider2D, "line_width") == max(1, round(line_width * 0.8))
    for ui_type, item in [
        (ui.Panel2D, "padding"),
        (ui.Panel2D, "gap"),
        (ui.Panel2D, "corner_radius"),
        (ui.TextButton2D, "corner_radius"),
    ]:
        value = comfortable.get(ui_type, item)
        assert compact.get(ui_type, item) == (round(value * 0.8) if value else 0)
    assert compact.get(ui.Panel2D, "color") == comfortable.get(ui.Panel2D, "color")
    assert compact.get(ui.Panel2D, "border_width") == comfortable.get(
        ui.Panel2D, "border_width"
    )


def test_compact_graphite_matches_the_dense_mockup():
    """Compact Graphite uses 12 px text, 10 px padding and 6 px gaps."""
    compact = ui.builtin_theme("graphite").derive(density="compact")

    assert compact.get(ui.TextBlock2D, "font_size") == 12
    assert compact.get(ui.Panel2D, "padding") == 10
    assert compact.get(ui.Panel2D, "gap") == 6
    assert compact.get(ui.TextBox2D, "text_padding") == (0, 6)
    assert compact.get(ui.LineSlider2D, "line_width") == 3


def test_compact_panel_renders_smaller():
    """Widgets created with compact density are smaller than comfortable ones."""
    sizes = {}
    for density in ("comfortable", "compact"):
        theme = ui.builtin_theme("graphite").derive(density=density)
        with ui.theme_scope(theme):
            panel = ui.Panel2D(size=(200, 50))
            panel.set_layout(direction=ui.Direction.COLUMN, height=ui.fit())
            panel.add(ui.TextBlock2D(text="Rendering", dynamic_bbox=True))
            panel.add(ui.LineSlider2D(length=150, label="Opacity"))
            panel.add(ui.Checkbox(["Shading", "Show axes"]))
        ui.UIContext.flush_layouts()
        sizes[density] = panel.size[1]

    assert sizes["compact"] < sizes["comfortable"]


def test_comfortable_density_is_the_theme_itself():
    """Comfortable density leaves every item unchanged and restyles nothing."""
    for name in ui.theme_names():
        theme = ui.builtin_theme(name).to_dict()
        comfortable = ui.builtin_theme(name).derive(density="comfortable").to_dict()
        assert comfortable["density"] == theme["density"] == "comfortable"
        assert comfortable["items"] == theme["items"]

    ui.set_theme("graphite")
    graphite = ui.get_theme()
    ui.set_theme(graphite, density="comfortable")
    assert ui.get_theme() is graphite


def test_set_theme_density_restyles_live_and_back():
    """Changing the density restyles existing widgets like fresh ones."""
    ui.set_theme("graphite")
    text = ui.TextBlock2D(text="Hello", position=(50, 50), dynamic_bbox=True)
    button = ui.TextButton2D("Go", position=(50, 120))
    graphite = ui.get_theme()
    comfortable_text = _render_scene(text)
    comfortable_button = _render_scene(button)

    ui.set_theme(ui.get_theme(), density="compact")
    assert ui.get_theme().density == "compact"
    assert text.font_size == 12
    assert button.font_size == 12
    fresh_text = ui.TextBlock2D(text="Hello", position=(50, 50), dynamic_bbox=True)
    npt.assert_array_equal(_render_scene(text), _render_scene(fresh_text))
    fresh_button = ui.TextButton2D("Go", position=(50, 120))
    npt.assert_array_equal(_render_scene(button), _render_scene(fresh_button))

    ui.set_theme(ui.get_theme(), density="comfortable")
    assert ui.get_theme() is graphite
    assert text.font_size == 13
    npt.assert_array_equal(_render_scene(text), comfortable_text)
    npt.assert_array_equal(_render_scene(button), comfortable_button)


def test_density_change_restyles_panel_spacing():
    """A panel laid out with the theme spacing follows a density change."""
    ui.set_theme("soft_pill")
    panel = ui.Panel2D(size=(200, 50))
    panel.set_layout(direction=ui.Direction.COLUMN, height=ui.fit())
    panel.add(ui.TextBlock2D(text="A", dynamic_bbox=True))
    panel.add(ui.TextBlock2D(text="B", dynamic_bbox=True))
    ui.UIContext.flush_layouts()
    comfortable_height = panel.size[1]

    ui.set_theme("soft_pill", density="compact")
    ui.UIContext.flush_layouts()

    assert panel.background.corner_radius == round(18 * 0.8)
    assert panel.size[1] < comfortable_height


def test_unknown_density_raises():
    """An unknown density raises a ValueError naming the densities."""
    with pytest.raises(ValueError, match="Unknown density 'dense'.*compact"):
        ui.builtin_theme("graphite").derive(density="dense")
    with pytest.raises(ValueError, match="Unknown density 'dense'"):
        ui.set_theme("graphite", density="dense")


def test_compact_theme_save_load_roundtrip(tmp_path):
    """A saved compact theme loads with its density and scales once."""
    path = tmp_path / "compact.json"
    compact = ui.builtin_theme("frost").derive(name="frost_compact", density="compact")
    compact.save(path)
    loaded = ui.Theme.load(path)

    assert loaded.density == "compact"
    assert loaded.get(ui.TextBlock2D, "font_size") == 12
    assert loaded.get(ui.Panel2D, "corner_radius") == 10
    loaded.save(tmp_path / "again.json")
    assert (tmp_path / "again.json").read_text() == path.read_text()


@pytest.mark.parametrize("base_density", [None, "comfortable", "compact"])
def test_compact_theme_dict_roundtrip_with_a_base(base_density):
    """A compact theme rebuilt from its dict, on any base, scales once."""
    compact = ui.builtin_theme("graphite").derive(density="compact")
    base = None
    if base_density is not None:
        base = ui.builtin_theme("graphite").derive(density=base_density)
    rebuilt = ui.Theme.from_dict(compact.to_dict(), base=base)

    assert rebuilt.density == "compact"
    assert rebuilt.get(ui.TextBlock2D, "font_size") == 12
    assert rebuilt.get(ui.Panel2D, "padding") == 10


def test_loaded_compact_theme_switches_back_to_comfortable(tmp_path):
    """A loaded compact theme returns to the original values when comfortable."""
    path = tmp_path / "compact.json"
    ui.builtin_theme("graphite").derive(density="compact").save(path)
    ui.set_theme(ui.Theme.load(path))
    assert ui.get_theme().get(ui.TextBlock2D, "font_size") == 12

    ui.set_theme(density="comfortable")

    assert ui.get_theme().density == "comfortable"
    assert ui.get_theme().get(ui.TextBlock2D, "font_size") == 13
    assert ui.get_theme().get(ui.Panel2D, "padding") == 12


def test_derived_items_under_compact_are_scaled_on_lookup():
    """Items given to a compact theme are stored as given and scaled once."""
    compact = ui.builtin_theme("graphite").derive(density="compact")
    custom = compact.derive(items={"TextBlock2D": {"font_size": 26}})
    custom.set("Panel2D", "padding", 0.5)

    assert custom.get(ui.TextBlock2D, "font_size") == 24
    assert custom.get(ui.Panel2D, "padding") == pytest.approx(0.4)
    assert custom.to_dict()["items"]["TextBlock2D"]["font_size"] == 26
    assert custom.derive(density="comfortable").get(ui.TextBlock2D, "font_size") == 26


def test_set_theme_density_uses_the_global_theme_not_the_scope():
    """set_theme with only a density changes the global theme."""
    ui.set_theme("ember")
    with ui.theme_scope("frost"):
        ui.set_theme(density="compact")
        assert ui.get_theme().name == "frost"
    theme = ui.get_theme()

    assert theme.name == "ember"
    assert theme.density == "compact"
    assert theme.get(ui.Panel2D, "color") == "#1A1716"


def test_theme_scope_accepts_a_density():
    """theme_scope can build elements with a density without changing the theme."""
    with ui.theme_scope("soft_pill", density="compact") as theme:
        text = ui.TextBlock2D(text="A", dynamic_bbox=True)

    assert theme.name == "soft_pill"
    assert theme.density == "compact"
    assert text.font_size == 12
    assert ui.get_theme().density == "comfortable"


@pytest.mark.parametrize("name", ["graphite", "soft_pill", "hairline", "classic"])
def test_density_change_restyles_existing_slider(name):
    """A slider restyled to compact matches a fresh compact slider."""
    ui.set_theme(name)
    slider = ui.LineSlider2D(length=160, position=(20, 80), initial_value=75)
    comfortable = _render_scene(slider)

    ui.set_theme(density="compact")
    fresh = ui.LineSlider2D(length=160, position=(20, 80), initial_value=75)
    npt.assert_array_equal(slider.size, fresh.size)
    npt.assert_array_equal(slider.handle.size, fresh.handle.size)
    npt.assert_array_equal(slider.track.size, fresh.track.size)
    npt.assert_array_equal(_render_scene(slider), _render_scene(fresh))

    ui.set_theme(density="comfortable")
    npt.assert_array_equal(_render_scene(slider), comfortable)


@pytest.mark.parametrize("group", [ui.Checkbox, ui.RadioButton])
def test_density_change_restyles_existing_box_indicators(group):
    """Box indicators and their label gap follow a density change."""
    ui.set_theme("graphite")
    options = group(["Shading", "Show axes"], checked_labels=["Shading"])
    options.set_position((40, 40))
    comfortable = _render_scene(options)

    ui.set_theme(density="compact")
    fresh = group(["Shading", "Show axes"], checked_labels=["Shading"])
    fresh.set_position((40, 40))
    ui.UIContext.flush_layouts()
    assert options.button_size == fresh.button_size
    assert options.button_label_gap == fresh.button_label_gap == 6
    npt.assert_array_equal(_render_scene(options), _render_scene(fresh))

    ui.set_theme(density="comfortable")
    npt.assert_array_equal(_render_scene(options), comfortable)


def test_hairline_panel_shows_the_scene_through():
    """The Hairline panel is translucent, without a shadow around it."""
    with ui.theme_scope("hairline"):
        panel = ui.Panel2D(size=(120, 100), position=(40, 30))
    points = [(100, 80), (100, 140), (190, 190)]

    inside, below, far = _render_on(panel, (0.5, 0.5, 0.5), points)
    on_red = _render_on(panel, (1, 0, 0), points)[0]

    assert on_red[0] - on_red[2] > 20
    assert inside.max() > 40
    npt.assert_array_equal(below, far)


def test_density_keeps_a_ring_handle_around_its_inner_radius():
    """A ring handle stays larger than its inner radius when compact."""
    ui.set_theme("classic")
    slider = ui.LineSlider2D(inner_radius=9, outer_radius=None)
    assert slider.handle.outer_radius == 10

    ui.set_theme(density="compact")
    fresh = ui.LineSlider2D(inner_radius=9, outer_radius=None)

    assert slider.handle.outer_radius == fresh.handle.outer_radius == 10
    assert slider.handle.inner_radius == 9


def test_user_slider_sizes_survive_a_density_change():
    """line_width and outer_radius given by the user are kept."""
    ui.set_theme("graphite")
    slider = ui.LineSlider2D(line_width=7, outer_radius=11, length=150)
    handle_size = tuple(slider.handle.size)

    ui.set_theme(density="compact")
    ui.set_theme(density="comfortable")
    ui.set_theme(density="compact")

    assert slider.track.size[1] == 7
    assert tuple(slider.handle.size) == handle_size == (22, 22)


def test_scaled_float_items_stay_floats():
    """Density keeps float items floats and integer items integers."""
    theme = ui.Theme(
        items={"UI": {"padding": 12.0, "gap": 8, "corner_radius": 0.5}},
        density="compact",
    )

    assert theme.get(ui.UI, "padding") == 10.0
    assert isinstance(theme.get(ui.UI, "padding"), float)
    assert theme.get(ui.UI, "gap") == 6
    assert isinstance(theme.get(ui.UI, "gap"), int)
    assert theme.get(ui.UI, "corner_radius") == pytest.approx(0.4)


def test_panel_relays_out_slider_and_checkbox_after_density_change():
    """A panel stacks its slider and checkbox again for the new density."""
    ui.set_theme("graphite")

    def build():
        panel = ui.Panel2D(size=(220, 50), position=(10, 10))
        panel.set_layout(direction=ui.Direction.COLUMN, height=ui.fit())
        slider = ui.LineSlider2D(length=180, label="Opacity")
        checkbox = ui.Checkbox(["Shading", "Show axes"], checked_labels=["Shading"])
        panel.add(slider)
        panel.add(checkbox)
        return panel, slider, checkbox

    panel, slider, checkbox = build()
    ui.UIContext.flush_layouts()
    comfortable_height = panel.size[1]

    ui.set_theme(density="compact")
    fresh, fresh_slider, fresh_checkbox = build()
    ui.UIContext.flush_layouts()

    assert panel.size[1] < comfortable_height
    npt.assert_array_equal(panel.size, fresh.size)
    npt.assert_array_equal(slider.get_position(), fresh_slider.get_position())
    npt.assert_array_equal(checkbox.get_position(), fresh_checkbox.get_position())
    npt.assert_array_equal(_render_scene(panel), _render_scene(fresh))
