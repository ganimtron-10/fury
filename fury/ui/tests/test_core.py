"""Core module testing."""

from PIL import Image
import numpy as np
import numpy.testing as npt
import pytest

from fury import ui, window
from fury.actor import Mesh
from fury.testing import assert_visibility
from fury.ui import UIContext
from fury.ui.helpers import Anchor
from fury.ui.layout import LayoutNode, fit, grow, solve


def test_rectangle2d_initialization_default():
    """
    Test Rectangle2D initialization with default parameters.

    Checks default size, color, opacity, and position.
    """
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.size, (100, 100))
    npt.assert_array_equal(rect.color, [1, 1, 1])
    npt.assert_equal(rect.opacity, 1.0)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors

    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.BOTTOM), [0, 100])


def test_rectangle2d_initialization_custom():
    """
    Test Rectangle2D initialization with custom parameters.

    Checks custom size, position, color, and opacity.
    """
    custom_size = (200, 100)
    custom_position = (50, 80)
    custom_color = (0.5, 0.2, 0.8)
    custom_opacity = 0.7

    rect = ui.Rectangle2D(
        size=custom_size,
        position=custom_position,
        color=custom_color,
        opacity=custom_opacity,
    )

    npt.assert_equal(rect.size, custom_size)
    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.TOP), custom_position)
    npt.assert_array_almost_equal(rect.color, custom_color)
    npt.assert_almost_equal(rect.opacity, custom_opacity)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors


def test_rectangle2d_width_property():
    """Test width getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.width, 100)
    rect.width = 120
    npt.assert_equal(rect.width, 120)
    npt.assert_equal(rect.height, 50)
    npt.assert_equal(rect.size, (120, 50))


def test_rectangle2d_height_property():
    """Test height getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.height, 50)
    rect.height = 70
    npt.assert_equal(rect.height, 70)
    npt.assert_equal(rect.width, 100)
    npt.assert_equal(rect.size, (100, 70))


def test_rectangle2d_color_property():
    """Test color getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_array_equal(rect.color, [1, 1, 1])
    new_color = (0.1, 0.2, 0.3)
    rect.color = new_color
    npt.assert_array_almost_equal(rect.color, new_color)


def test_rectangle2d_color_normalization():
    """Test Rectangle2D accepts hex strings and [0, 255] colors."""
    rect = ui.Rectangle2D(color="#FF0000")
    npt.assert_array_almost_equal(rect.color, [1, 0, 0])

    rect.color = "#00FF00"
    npt.assert_array_almost_equal(rect.color, [0, 1, 0])

    rect.color = (0, 0, 255)
    npt.assert_array_almost_equal(rect.color, [0, 0, 1])


def test_rectangle2d_opacity_property():
    """Test opacity getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.opacity, 1.0)
    new_opacity = 0.5
    rect.opacity = new_opacity
    npt.assert_equal(rect.opacity, new_opacity)


def test_rectangle2d_resize():
    """Test resize method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    new_size = (250, 150)
    rect.resize(new_size)
    npt.assert_equal(rect.size, new_size)


def test_rectangle2d_set_visibility():
    """Test set_visibility method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(10, 10))
    rect.set_visibility(False)
    assert rect.actor.visible is False
    rect.set_visibility(True)
    assert rect.actor.visible is True


def test_rounded_rectangle2d_initialization_default():
    """
    Test RoundedRectangle2D initialization with default parameters.

    Checks default size, color, opacity, corner_radius and position.
    """
    rect = ui.RoundedRectangle2D()
    npt.assert_equal(rect.size, (100, 100))
    npt.assert_array_equal(rect.color, [1, 1, 1])
    npt.assert_equal(rect.opacity, 1.0)
    npt.assert_equal(rect.corner_radius, 10.0)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors

    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.BOTTOM), [0, 100])


def test_rounded_rectangle2d_initialization_custom():
    """
    Test RoundedRectangle2D initialization with custom parameters.

    Checks custom size, position, color, opacity, and corner_radius.
    """
    custom_size = (200, 100)
    custom_position = (50, 80)
    custom_color = (0.5, 0.2, 0.8)
    custom_opacity = 0.7
    custom_radius = 25.0

    rect = ui.RoundedRectangle2D(
        size=custom_size,
        position=custom_position,
        color=custom_color,
        opacity=custom_opacity,
        corner_radius=custom_radius,
    )

    npt.assert_equal(rect.size, custom_size)
    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.TOP), custom_position)
    npt.assert_array_almost_equal(rect.color, custom_color)
    npt.assert_almost_equal(rect.opacity, custom_opacity)
    npt.assert_almost_equal(rect.corner_radius, custom_radius)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors


def test_rounded_rectangle2d_width_property():
    """Test width getter and setter for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D(size=(100, 50))
    npt.assert_equal(rect.width, 100)
    rect.width = 120
    npt.assert_equal(rect.width, 120)
    npt.assert_equal(rect.height, 50)
    npt.assert_equal(rect.size, (120, 50))


def test_rounded_rectangle2d_height_property():
    """Test height getter and setter for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D(size=(100, 50))
    npt.assert_equal(rect.height, 50)
    rect.height = 70
    npt.assert_equal(rect.height, 70)
    npt.assert_equal(rect.width, 100)
    npt.assert_equal(rect.size, (100, 70))


def test_rounded_rectangle2d_color_property():
    """Test color getter and setter for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D()
    npt.assert_array_equal(rect.color, [1, 1, 1])
    new_color = (0.1, 0.2, 0.3)
    rect.color = new_color
    npt.assert_array_almost_equal(rect.color, new_color)


def test_rounded_rectangle2d_color_normalization():
    """Test RoundedRectangle2D accepts hex strings and [0, 255] colors."""
    rect = ui.RoundedRectangle2D(color="#FF0000")
    npt.assert_array_almost_equal(rect.color, [1, 0, 0])

    rect.color = "#00FF00"
    npt.assert_array_almost_equal(rect.color, [0, 1, 0])

    rect.color = (0, 0, 255)
    npt.assert_array_almost_equal(rect.color, [0, 0, 1])


def test_rounded_rectangle2d_opacity_property():
    """Test opacity getter and setter for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D()
    npt.assert_equal(rect.opacity, 1.0)
    new_opacity = 0.5
    rect.opacity = new_opacity
    npt.assert_equal(rect.opacity, new_opacity)


def test_rounded_rectangle2d_corner_radius_property():
    """Test corner_radius getter and setter for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D()
    npt.assert_equal(rect.corner_radius, 10.0)
    new_radius = 20.0
    rect.corner_radius = new_radius
    npt.assert_equal(rect.corner_radius, new_radius)


def test_rounded_rectangle2d_resize():
    """Test resize method for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D(size=(100, 50))
    new_size = (250, 150)
    rect.resize(new_size)
    npt.assert_equal(rect.size, new_size)


def test_rounded_rectangle2d_set_visibility():
    """Test set_visibility method for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D(size=(10, 10))
    rect.set_visibility(False)
    assert rect.actor.visible is False
    rect.set_visibility(True)
    assert rect.actor.visible is True


def test_disk2d_initialization_default():
    """
    Test Disk2D initialization with default parameters.

    Checks default center, color, opacity, and required outer_radius.
    """
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_array_equal(disk_ui.get_position(Anchor.CENTER, Anchor.CENTER), [0, 0])
    npt.assert_array_equal(disk_ui.color, [1, 1, 1])
    npt.assert_equal(disk_ui.opacity, 1.0)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (20, 20))


def test_disk2d_initialization_custom():
    """
    Test Disk2D initialization with custom parameters.

    Checks custom outer_radius, center, color, and opacity.
    """
    custom_radius = 25
    custom_center = (100, 150)
    custom_color = (0.9, 0.1, 0.4)
    custom_opacity = 0.6

    disk_ui = ui.Disk2D(
        outer_radius=custom_radius,
        center=custom_center,
        color=custom_color,
        opacity=custom_opacity,
    )

    npt.assert_equal(disk_ui.outer_radius, custom_radius)
    npt.assert_array_equal(
        disk_ui.get_position(Anchor.CENTER, Anchor.CENTER), custom_center
    )
    npt.assert_array_almost_equal(disk_ui.color, custom_color)
    npt.assert_almost_equal(disk_ui.opacity, custom_opacity)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (50, 50))


def test_disk2d_outer_radius_property():
    """Test outer_radius getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_equal(disk_ui.size, (20, 20))

    disk_ui.outer_radius = 15
    npt.assert_equal(disk_ui.outer_radius, 15)
    npt.assert_equal(disk_ui.size, (30, 30))


def test_disk2d_inner_radius_property():
    """Test inner_radius getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=20, inner_radius=10)
    npt.assert_equal(disk_ui.inner_radius, 10)

    disk_ui.inner_radius = 15
    npt.assert_equal(disk_ui.inner_radius, 15)

    with npt.assert_raises(ValueError):
        disk_ui.inner_radius = 25


def test_disk2d_color_property():
    """Test color getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_array_almost_equal(disk_ui.color, [1, 1, 1])
    new_color = (0.6, 0.7, 0.8)
    disk_ui.color = new_color
    npt.assert_array_almost_equal(disk_ui.color, new_color)


def test_disk2d_color_normalization():
    """Test Disk2D accepts hex strings and [0, 255] colors."""
    disk_ui = ui.Disk2D(outer_radius=10, color="#FF0000")
    npt.assert_array_almost_equal(disk_ui.color, [1, 0, 0])

    disk_ui.color = (0, 255, 0)
    npt.assert_array_almost_equal(disk_ui.color, [0, 1, 0])


def test_disk2d_opacity_property():
    """Test opacity getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_almost_equal(disk_ui.opacity, 1.0)
    new_opacity = 0.3
    disk_ui.opacity = new_opacity
    npt.assert_almost_equal(disk_ui.opacity, new_opacity)


def test_disk2d_set_visibility():
    """Test set_visibility method for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    disk_ui.set_visibility(False)
    assert disk_ui.actor.visible is False
    disk_ui.set_visibility(True)
    assert disk_ui.actor.visible is True


def test_rectangle2d_visual_snapshot():
    """Visibility snapshot test for Rectangle2D."""
    rect = ui.Rectangle2D(
        size=(50, 50),
        position=(75, 75),
        color=(1.0, 0.0, 0.0),
        opacity=1.0,
    )

    assert_visibility(rect, toggle=rect.set_visibility, colors=[(1.0, 0.0, 0.0)])


def test_rounded_rectangle2d_visual_snapshot():
    """Visibility snapshot test for RoundedRectangle2D."""
    rect = ui.RoundedRectangle2D(
        size=(50, 50),
        position=(75, 75),
        color=(1.0, 0.0, 0.0),
        corner_radius=15.0,
        opacity=1.0,
    )

    assert_visibility(rect, toggle=rect.set_visibility, colors=[(1.0, 0.0, 0.0)])


def test_disk2d_visual_snapshot():
    """Visibility snapshot test for Disk2D."""
    disk = ui.Disk2D(
        outer_radius=25,
        center=(100, 100),
        color=(0.0, 1.0, 0.0),
    )

    assert_visibility(disk, toggle=disk.set_visibility, colors=[(0.0, 1.0, 0.0)])


@pytest.mark.usefixtures("classic_theme")
def test_textblock2d_initialization_default():
    """
    Test TextBlock2D initialization with minimal valid parameters.

    Checks default font, color, alignment, and flags.
    """
    tb = ui.TextBlock2D(size=(200, 100))

    npt.assert_equal(tb.message, "Text Block")
    npt.assert_equal(tb.font_size, 18)
    npt.assert_equal(tb.font_family[0], "Arial")
    npt.assert_array_equal(tb.color, [1, 1, 1])

    assert tb.bold is False
    assert tb.italic is False
    assert tb.have_bg is False
    assert tb.background_color is None

    assert tb.justification == "left"
    assert tb.vertical_justification == "top"


def test_textblock2d_initialization_custom():
    """Test TextBlock2D initialization with full custom parameters."""
    custom_color = (1.0, 1.0, 0.0)
    custom_bg = (0.0, 0.0, 1.0)

    tb = ui.TextBlock2D(
        text="Custom",
        font_size=24,
        font_family="Times New Roman",
        justification="center",
        vertical_justification="top",
        bold=True,
        italic=False,
        size=(300, 150),
        color=custom_color,
        bg_color=custom_bg,
        position=(10, 10),
    )

    npt.assert_equal(tb.message, "Custom")
    npt.assert_equal(tb.font_size, 24)
    npt.assert_equal(tb.font_family[0], "Times New Roman")
    npt.assert_array_almost_equal(tb.color, custom_color)

    assert tb.have_bg is True
    npt.assert_array_almost_equal(tb.background_color, custom_bg)
    assert isinstance(tb.background, ui.RoundedRectangle2D)

    assert "**Custom**" in tb.get_formatted_text("Custom")


def test_textblock2d_message_property():
    """Test message getter and setter."""
    tb = ui.TextBlock2D(size=(100, 50))
    npt.assert_equal(tb.message, "Text Block")

    new_msg = "New Message"
    tb.message = new_msg
    npt.assert_equal(tb.message, new_msg)


def test_textblock2d_font_properties():
    """Test font_size and font_family properties."""
    tb = ui.TextBlock2D(size=(100, 50))

    tb.font_size = 30
    npt.assert_equal(tb.font_size, 30)

    tb.font_family = "Courier"
    npt.assert_equal(tb.font_family[0], "Courier")


def test_textblock2d_style_properties():
    """Test bold and italic properties."""
    tb = ui.TextBlock2D(text="Style", size=(100, 50))

    assert tb.get_formatted_text("Style") == "Style"

    tb.bold = True
    assert tb.bold is True
    assert tb.get_formatted_text("Style") == "**Style**"

    tb.bold = False
    tb.italic = True
    assert tb.italic is True
    assert tb.get_formatted_text("Style") == "*Style*"


def test_textblock2d_alignment_valid():
    """Test valid inputs for justification properties."""
    tb = ui.TextBlock2D(size=(100, 100))

    valid_h = ["left", "center", "right"]
    valid_v = ["top", "middle", "bottom"]

    for h in valid_h:
        tb.justification = h
        npt.assert_equal(tb.justification, h)

    for v in valid_v:
        tb.vertical_justification = v
        npt.assert_equal(tb.vertical_justification, v)


def test_textblock2d_alignment_invalid():
    """Test invalid inputs for justification properties."""
    tb = ui.TextBlock2D(size=(100, 100))

    with npt.assert_raises(ValueError):
        tb.justification = "diagonal"

    with npt.assert_raises(ValueError):
        tb.vertical_justification = "sideways"


@pytest.mark.usefixtures("classic_theme")
def test_textblock2d_color_property():
    """Test text color getter and setter."""
    tb = ui.TextBlock2D(size=(100, 50))
    npt.assert_array_equal(tb.color, [1, 1, 1])

    tb.color = (0.5, 0.5, 0.5)
    npt.assert_array_almost_equal(tb.color, [0.5, 0.5, 0.5])


@pytest.mark.usefixtures("classic_theme")
def test_textblock2d_color_normalization():
    """Test TextBlock2D accepts hex strings and [0, 255] colors."""
    tb = ui.TextBlock2D(size=(100, 50), color="#0000FF")
    npt.assert_array_almost_equal(tb.color, [0, 0, 1])

    tb.color = (255, 0, 0)
    npt.assert_array_almost_equal(tb.color, [1, 0, 0])

    # None falls back to white.
    tb.color = None
    npt.assert_array_almost_equal(tb.color, [1, 1, 1])

    # Background color is normalized via the same path.
    tb.background_color = "#00FF00"
    npt.assert_array_almost_equal(tb.background_color, [0, 1, 0])


def test_textblock2d_background_color_property():
    """Test background color logic."""
    tb = ui.TextBlock2D(size=(100, 50))

    assert tb.background_color is None
    assert tb.have_bg is False

    tb.background_color = (1, 0, 0)
    assert tb.have_bg is True
    npt.assert_array_equal(tb.background_color, (1, 0, 0))
    assert tb.background.actor.visible is True

    tb.background_color = None
    assert tb.have_bg is False
    assert tb.background_color is None
    assert tb.background.actor.visible is False


def test_textblock2d_visibility_preserves_disabled_background():
    """Test visibility does not show background when bg_color is None."""
    tb = ui.TextBlock2D(size=(100, 50), bg_color=None)

    assert tb.have_bg is False
    assert tb.background.actor.visible is False

    tb.set_visibility(False)
    assert tb.actor.visible is False
    assert tb.background.actor.visible is False

    tb.set_visibility(True)
    assert tb.actor.visible is True
    assert tb.background.actor.visible is False


def test_textblock2d_dynamic_bbox_property():
    """Test dynamic_bbox property toggling."""
    tb = ui.TextBlock2D(size=(100, 50))
    assert tb.dynamic_bbox is False

    tb.dynamic_bbox = True
    assert tb.dynamic_bbox is True

    size = tb._get_size()
    assert len(size) == 2


def test_textblock2d_resize():
    """Test resize method."""
    tb = ui.TextBlock2D(size=(100, 50))
    new_size = (200, 150)

    tb.resize(new_size)

    npt.assert_equal(tb.actor.max_width, new_size[0])
    npt.assert_equal(tb.background.size, new_size)

    npt.assert_equal(tb.boundingbox[2] - tb.boundingbox[0], new_size[0])
    npt.assert_equal(tb.boundingbox[3] - tb.boundingbox[1], new_size[1])


@pytest.mark.parametrize("dynamic_bbox", [False, True])
@pytest.mark.parametrize(
    "justification, vertical_justification",
    [("left", "top"), ("center", "middle"), ("right", "bottom")],
)
def test_textblock2d_first_frame_layout(
    dynamic_bbox, justification, vertical_justification
):
    """Text and background must be aligned on the first frame after a change."""
    block = ui.TextBlock2D(
        text="HUD",
        position=(24, 24),
        size=None if dynamic_bbox else (560, 160),
        dynamic_bbox=dynamic_bbox,
        font_size=18,
        justification=justification,
        vertical_justification=vertical_justification,
        color=(1, 1, 1),
        bg_color=(1, 0, 0),
    )
    scene = window.Scene(background=(0, 0, 0, 1))
    scene.add(block)
    show_m = window.ShowManager(
        scene=scene, size=(640, 240), pixel_ratio=1, window_type="offscreen"
    )

    def capture_masks():
        window.render_screens(show_m.renderer, show_m.screens)
        pixels = show_m.snapshot(fname=None)[:, :, :3]
        foreground = np.all(pixels > 180, axis=2)
        background = (
            (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80) & (pixels[:, :, 2] < 80)
        )
        return foreground, background

    def check_first_frame():
        first_foreground, first_background = capture_masks()
        next_foreground, next_background = capture_masks()
        npt.assert_array_equal(first_foreground, next_foreground)
        npt.assert_array_equal(first_background, next_background)

        foreground_y, foreground_x = np.where(first_foreground)
        background_y, background_x = np.where(first_background)
        x, y = block.get_position(Anchor.LEFT, Anchor.TOP)
        expected_size = block.get_text_actor_size() if dynamic_bbox else (560, 160)
        npt.assert_allclose(
            (
                background_x.max() - background_x.min() + 1,
                background_y.max() - background_y.min() + 1,
            ),
            expected_size,
            atol=1,
        )
        npt.assert_allclose((background_x.min(), background_y.min()), (x, y), atol=1)
        assert 0 < foreground_x.min() <= foreground_x.max() < 639
        assert 0 < foreground_y.min() <= foreground_y.max() < 239
        assert foreground_x.min() >= background_x.min() - 1
        assert foreground_x.max() <= background_x.max() + 1
        assert foreground_y.min() >= background_y.min() - 1
        assert foreground_y.max() <= background_y.max() + 1
        if justification == "left" and vertical_justification == "top":
            assert x - 1 <= foreground_x.min() <= x + 10

    try:
        check_first_frame()
        block.message = "HUD: a longer status"
        check_first_frame()
        block.message = "HUD"
        check_first_frame()
        block.message = "HUD: a longer status"
        check_first_frame()
        block.font_size = 28
        check_first_frame()
        block.message = "HUD: moved status"
        block.set_position((40, 35))
        check_first_frame()
        block.set_position((24, 24))
        block.message = "HUD"
        check_first_frame()
        block.set_position((40, 35))
        block.message = "HUD: a longer status"
        check_first_frame()
    finally:
        show_m.window.close()


def test_textblock2d_visual_snapshot():
    """
    Visibility snapshot test for TextBlock2D.

    Checks that the red background renders when visible and nothing renders
    once the block is hidden.
    """
    tb = ui.TextBlock2D(
        text="Visual Test",
        size=(200, 100),
        position=(50, 50),
        bg_color=(1.0, 0.0, 0.0),
        color=(1.0, 1.0, 1.0),
    )

    assert_visibility(tb, toggle=tb.set_visibility, colors=[(1.0, 0.0, 0.0)])


def test_slider2d_base_logic():
    """Test the core logic of the Slider2D base class using RingSlider2D."""
    slider = ui.RingSlider2D(initial_value=50, min_value=0, max_value=100)

    npt.assert_equal(slider.value, 50)
    npt.assert_almost_equal(slider.ratio, 0.5)

    slider.value = 150
    npt.assert_equal(slider.value, 100)
    npt.assert_equal(slider.ratio, 1.0)

    slider.value = -50
    npt.assert_equal(slider.value, 0)
    npt.assert_equal(slider.ratio, 0.0)

    slider.ratio = 1.5
    npt.assert_equal(slider.ratio, 1.0)
    npt.assert_equal(slider.value, 100)

    slider.value = 75
    slider.text_template = "V: {value:.0f}"
    assert slider.format_text() == "V: 75"

    slider.text_template = lambda s: f"R: {s.ratio:.2f}"
    assert slider.format_text() == "R: 0.75"

    hook_data = {"called": False}

    def on_change(s):
        hook_data["called"] = True

    slider.on_change = on_change
    slider.value = 25
    assert hook_data["called"] is True


def test_textblock2d_text_renders_above_background():
    """
    Text must draw on top of its own background.

    The text actor and its background share the component's z_order, so they
    are coplanar. The text is given a higher render_order so its anti-aliased
    edges blend against the background instead of whatever lies behind the UI
    (which produced ragged, clipped-looking glyphs). Anti-aliasing is also
    enabled for smooth glyph edges.
    """
    tb = ui.TextBlock2D(
        text="Above background",
        size=(200, 100),
        position=(50, 50),
        bg_color=(1.0, 1.0, 1.0),
        color=(0.0, 0.0, 0.0),
    )

    background_actor = tb.background.actors[0]
    assert tb.actor.render_order > background_actor.render_order
    assert tb.actor.material.aa is True


def test_textblock2d_render_order_tracks_z_order():
    """
    render_order increases with z_order while text stays above its background.

    Draw order is derived from z_order so overlapping components stack
    predictably, with a fixed sub-order keeping a component's text above its
    own background.
    """
    tb = ui.TextBlock2D(
        text="Layered",
        size=(200, 100),
        position=(50, 50),
        bg_color=(1.0, 1.0, 1.0),
    )
    background_actor = tb.background.actors[0]

    low_text = tb.actor.render_order
    low_bg = background_actor.render_order
    assert low_text > low_bg

    tb.z_order = 5
    tb.update_bounding_box()

    assert tb.actor.render_order > low_text
    assert background_actor.render_order > low_bg
    assert tb.actor.render_order > background_actor.render_order


def test_textblock2d_solid_text_over_background_snapshot():
    """
    Black text on a white background renders as solid dark glyphs.

    End-to-end guard for the coplanar draw order: over a red scene, the white
    background fills the box and the black text on top is the only source of
    near-black pixels, so its presence confirms the text composites over its
    background rather than bleeding into the scene behind the UI.
    """
    tb = ui.TextBlock2D(
        text="Solid",
        font_size=40,
        size=(200, 60),
        position=(50, 50),
        bg_color=(1.0, 1.0, 1.0),
        color=(0.0, 0.0, 0.0),
    )

    scene = window.Scene(background=(1.0, 0.0, 0.0, 1.0))
    scene.add(tb)

    fname = "textblock_solid_over_bg.png"
    window.snapshot(scene=scene, fname=fname)

    img_array = np.array(Image.open(fname))[:, :, :3]

    white_box = np.all(img_array > 200, axis=2).sum()
    dark_text = np.all(img_array < 40, axis=2).sum()

    assert white_box > 0
    assert dark_text > 0


def _text_top_left(block):
    """Return the on-screen top-left corner of a text block's glyphs."""
    block.actor._update_object()
    aabb = block.actor._aabb
    return (
        float(block.actor.local.x + aabb[0][0]),
        float(UIContext.canvas_size[1] - (block.actor.local.y + aabb[1][1])),
    )


def test_layout_natural_size_survives_layout_resizes():
    """A size applied by the layout must not become the natural size."""
    parent = LayoutNode(width=200, height=20)
    rect = ui.Rectangle2D(size=(50, 20))
    rect.set_layout(width=grow())
    parent.add(rect._layout_node)
    solve(parent)
    npt.assert_array_equal(rect.size, (200, 20))

    parent.style.update(width=fit())
    solve(parent)
    npt.assert_array_equal(rect.size, (50, 20))


def test_layout_outside_resize_replaces_natural_size():
    """A resize from user code changes what the layout starts from."""
    parent = LayoutNode()
    rect = ui.Rectangle2D(size=(50, 20))
    parent.add(rect._layout_node)
    solve(parent)
    rect.resize((80, 30))
    rect.mark_layout_dirty()
    assert parent.dirty
    assert solve(parent).size == (80, 30)


def test_text_measured_before_first_frame_matches_rendered_size():
    """Text can be sized without waiting for a rendered frame."""
    block = ui.TextBlock2D(text="Measured", font_size=24, dynamic_bbox=True)
    before = block.get_text_actor_size()
    scene = window.Scene()
    scene.add(block)
    show_m = window.ShowManager(scene=scene, size=(200, 100), window_type="offscreen")
    show_m.render()
    show_m.window.draw()
    npt.assert_allclose(before, block.get_text_actor_size())
    show_m.close()


@pytest.mark.parametrize("dynamic_bbox", [True, False])
def test_font_size_change_realigns_immediately(dynamic_bbox):
    """Changing the font size must not wait for a later frame (#1194)."""
    block = ui.TextBlock2D(
        text="Hello",
        font_size=12,
        size=None if dynamic_bbox else (200, 100),
        dynamic_bbox=dynamic_bbox,
        bg_color=(1, 0, 0),
        position=(10, 10),
        vertical_justification="top",
    )
    block.font_size = 40
    npt.assert_allclose(_text_top_left(block)[1], 10, atol=1)
    if dynamic_bbox:
        npt.assert_allclose(block.background.size, block.size)


def test_text_padding_grows_dynamic_box_and_insets_text():
    """Padding belongs to the box and moves the text inside it (#838)."""
    block = ui.TextBlock2D(
        text="Padded", font_size=20, dynamic_bbox=True, bg_color=(1, 0, 0)
    )
    text_w, text_h = block.get_text_actor_size()
    block.set_position((50, 50))
    block.set_layout(padding=(6, 10))
    npt.assert_array_equal(block.size, (int(text_w) + 20, int(text_h) + 12))
    npt.assert_array_equal(block.background.size, block.size)
    npt.assert_allclose(_text_top_left(block), (60, 56), atol=1)


def test_text_fit_in_layout_measures_without_a_frame():
    block = ui.TextBlock2D(text="Measure me", font_size=18, dynamic_bbox=True)
    parent = LayoutNode(padding=4)
    parent.add(block._layout_node)
    rect = solve(parent)
    text_w, text_h = block.get_text_actor_size()
    assert rect.size == (int(np.ceil(text_w)) + 8, int(np.ceil(text_h)) + 8)


def test_text_grows_its_box_in_layout_and_keeps_it_when_moved():
    block = ui.TextBlock2D(
        text="Hi", dynamic_bbox=True, bg_color=(1, 0, 0), justification="center"
    )
    block.set_layout(width=grow())
    parent = LayoutNode(width=200)
    parent.add(block._layout_node)
    solve(parent)
    npt.assert_array_equal(block.background.size[0], 200)
    block.set_position((5, 5))
    npt.assert_array_equal(block.background.size[0], 200)
    npt.assert_allclose(block.actor.local.x, 105)


def test_text_wraps_to_narrow_layout_width():
    text = "several words that need to wrap in a narrow column"
    block = ui.TextBlock2D(text=text, dynamic_bbox=True)
    single_line = block.get_text_actor_size()[1]
    parent = LayoutNode(width=100, direction="column", align_cross="stretch")
    parent.add(block._layout_node)
    rect = solve(parent)
    assert rect.width == 100
    assert rect.height > 2 * single_line
    npt.assert_allclose(block.get_text_actor_size()[1], rect.height, atol=1)


def test_text_change_marks_layout_dirty():
    block = ui.TextBlock2D(text="a", dynamic_bbox=True)
    parent = LayoutNode()
    parent.add(block._layout_node)
    first = solve(parent).width
    block.message = "a much longer message"
    assert parent.dirty
    assert solve(parent).width > first


@pytest.mark.parametrize(
    "text,font_size",
    [
        ("Quality: 50", 26),
        ("Two words", 18),
        ("a b c d e f", 31),
        ("Render settings", 22),
    ],
)
def test_text_at_its_natural_width_stays_on_one_line(text, font_size):
    """Text sized to fit exactly must not wrap on its own width."""
    block = ui.TextBlock2D(text=text, font_size=font_size, dynamic_bbox=True)
    one_line = block.get_text_actor_size()[1]
    parent = LayoutNode()
    parent.add(block._layout_node)
    assert solve(parent).height == int(np.ceil(one_line))
    npt.assert_allclose(block.get_text_actor_size()[1], one_line)


def test_text_survives_half_pixel_centering():
    """Rounding a centered box must not make its text wrap."""
    block = ui.TextBlock2D(text="Quality: 50", font_size=26, dynamic_bbox=True)
    one_line = block.get_text_actor_size()[1]
    natural = int(np.ceil(block.get_text_actor_size()[0]))
    parent = LayoutNode(
        width=natural + 101, height=100, align_main="center", align_cross="center"
    )
    parent.add(block._layout_node)
    solve(parent)
    npt.assert_allclose(block.get_text_actor_size()[1], one_line)


# ---------------------------------------------------------------------------
# RoundedRectangle2D box style: border, corner radii and drop shadow.
# ---------------------------------------------------------------------------

BACKGROUND = (1.0, 1.0, 1.0)
FILL = (0.0, 0.0, 1.0)


@pytest.fixture
def show_window():
    """
    Provide a factory for offscreen windows that are closed after the test.

    Yields
    ------
    callable
        ``show_window(*elements, size=(200, 200), pixel_ratio=1,
        background=BACKGROUND)`` adds the elements to a new scene and returns
        its offscreen ShowManager. Nothing is drawn until :func:`_snapshot`.
    """
    managers = []

    def factory(*elements, size=(200, 200), pixel_ratio=1, background=BACKGROUND):
        """
        Create an offscreen window showing the elements.

        Parameters
        ----------
        *elements : UI
            Elements to add to the scene.
        size : (int, int), optional
            Window size in pixels.
        pixel_ratio : float, optional
            Ratio of physical to logical pixels.
        background : tuple, optional
            RGB background color of the scene.

        Returns
        -------
        ShowManager
            The show manager. No frame has been drawn yet.
        """
        scene = window.Scene()
        scene.background = background
        scene.add(*elements)
        show_m = window.ShowManager(
            scene=scene, size=size, pixel_ratio=pixel_ratio, window_type="offscreen"
        )
        managers.append(show_m)
        return show_m

    yield factory
    for show_m in managers:
        show_m.close()


def _snapshot(show_m):
    """
    Draw a frame and return its RGB pixels.

    Parameters
    ----------
    show_m : ShowManager
        The show manager to draw.

    Returns
    -------
    ndarray
        Image of shape (height, width, 3), indexed as ``image[y, x]`` with
        ``y`` pointing down.
    """
    show_m.render()
    show_m.window.draw()
    return np.asarray(show_m.snapshot())[..., :3].astype(int)


def _assert_color(pixel, color, *, atol=3):
    """
    Assert that an 8-bit pixel matches a color given in [0, 1].

    Parameters
    ----------
    pixel : ndarray
        RGB pixel in [0, 255].
    color : tuple
        Expected RGB color in [0, 1].
    atol : int, optional
        Allowed difference per channel.
    """
    npt.assert_allclose(pixel, np.asarray(color) * 255, atol=atol)


def _box(**kwargs):
    """
    Build a blue 100x100 box with its top-left corner at (50, 50).

    Parameters
    ----------
    **kwargs : dict
        Style arguments forwarded to ``RoundedRectangle2D``.

    Returns
    -------
    RoundedRectangle2D
        The box.
    """
    kwargs.setdefault("corner_radius", 0)
    return ui.RoundedRectangle2D(
        size=(100, 100), position=(50, 50), color=FILL, **kwargs
    )


def test_default_box_render_is_pinned(show_window):
    """The default box renders exactly as before the style options existed."""
    box = ui.RoundedRectangle2D()
    image = _snapshot(show_window(box, background=(0.1, 0.1, 0.1)))

    expected_corner = [
        [26, 26, 26, 52, 103, 154],
        [26, 26, 64, 124, 176, 214],
        [26, 64, 132, 192, 231, 254],
        [52, 124, 192, 234, 255, 255],
        [103, 176, 231, 255, 255, 255],
        [154, 214, 254, 255, 255, 255],
    ]
    npt.assert_array_equal(image[0:6, 0:6, 0], expected_corner)
    npt.assert_array_equal(image[50, 0:3, 0], [237, 255, 255])
    npt.assert_array_equal(image[50, 97:101, 0], [255, 255, 237, 26])
    npt.assert_array_equal(image[50, 50], [255, 255, 255])
    npt.assert_array_equal(image[150, 150], [26, 26, 26])


def test_default_style_has_no_border_and_no_shadow():
    """The defaults draw a plain fill and hide the shadow actor."""
    box = ui.RoundedRectangle2D()
    assert box.border_width == 0
    assert box.shadow_blur == 0
    assert box.shadow_spread == 0
    assert box.shadow_offset == (0.0, 0.0)
    assert box.shadow_actor.visible is False
    assert box.actors == [box.actor]
    assert box.shadow_actor.parent is box.actor


def test_border_is_drawn_inside_the_edge(show_window):
    """Border pixels sit inside the box edge, the fill stays in the middle."""
    box = _box(border_color="#FF0000", border_width=6, corner_radius=10)
    image = _snapshot(show_window(box))

    _assert_color(image[100, 53], (1, 0, 0))
    _assert_color(image[53, 100], (1, 0, 0))
    _assert_color(image[146, 100], (1, 0, 0))
    _assert_color(image[100, 146], (1, 0, 0))
    _assert_color(image[100, 100], FILL)
    _assert_color(image[100, 58], FILL)
    _assert_color(image[100, 47], BACKGROUND)


def test_border_follows_rounded_corner(show_window):
    """Along the diagonal of a rounded corner the border keeps its width."""
    box = _box(border_color=(1, 0, 0), border_width=4, corner_radius=30)
    image = _snapshot(show_window(box))

    def diagonal_pixel(distance_from_corner_center):
        """
        Return the pixel on the top-left corner diagonal.

        Parameters
        ----------
        distance_from_corner_center : float
            Distance from the center of the corner arc, in pixels.

        Returns
        -------
        ndarray
            RGB pixel in [0, 255].
        """
        offset = distance_from_corner_center / np.sqrt(2)
        index = int(50 + 30 - offset)
        return image[index, index]

    _assert_color(diagonal_pixel(28), (1, 0, 0))
    _assert_color(diagonal_pixel(23), FILL)
    _assert_color(diagonal_pixel(33), BACKGROUND)


def test_border_with_alpha_blends_over_the_fill(show_window):
    """A translucent border shows the fill through it."""
    box = _box(border_color=(1, 0, 0, 0.5), border_width=6)
    image = _snapshot(show_window(box))

    half_in_srgb = 255 * (1.055 * 0.5 ** (1 / 2.4) - 0.055)
    npt.assert_allclose(image[100, 53], (half_in_srgb, 0, half_in_srgb), atol=2)


@pytest.mark.parametrize(
    "corner, pixel",
    [(0, (52, 52)), (1, (52, 147)), (2, (147, 147)), (3, (147, 52))],
    ids=["top_left", "top_right", "bottom_right", "bottom_left"],
)
def test_each_corner_radius_is_independent(corner, pixel, show_window):
    """Only the corner given a large radius is cut away."""
    radii = [0, 0, 0, 0]
    radii[corner] = 40
    box = _box(corner_radius=tuple(radii))
    image = _snapshot(show_window(box))

    corner_pixels = [(52, 52), (52, 147), (147, 147), (147, 52)]
    for y, x in corner_pixels:
        expected = BACKGROUND if (y, x) == pixel else FILL
        _assert_color(image[y, x], expected)
    assert box.corner_radius == tuple(float(r) for r in radii)


def test_corner_radius_is_clamped_to_half_the_smaller_side(show_window):
    """A radius larger than the box draws a pill, not a broken shape."""
    box = ui.RoundedRectangle2D(
        size=(120, 40), position=(40, 80), color=FILL, corner_radius=500
    )
    image = _snapshot(show_window(box))

    _assert_color(image[100, 100], FILL)
    _assert_color(image[100, 42], FILL)
    _assert_color(image[82, 42], BACKGROUND)
    assert box.corner_radius == 500


def test_small_box_with_default_radius_is_a_proper_rounded_shape(show_window):
    """A box smaller than twice the radius is drawn with clamped corners."""
    box = ui.RoundedRectangle2D(size=(10, 10), position=(95, 95), color=(0, 0, 0))
    image = _snapshot(show_window(box))

    _assert_color(image[100, 100], (0, 0, 0))
    _assert_color(image[100, 97], (0, 0, 0))
    _assert_color(image[97, 100], (0, 0, 0))
    _assert_color(image[102, 100], (0, 0, 0))
    _assert_color(image[100, 102], (0, 0, 0))
    for y, x in [(95, 95), (95, 104), (104, 104), (104, 95)]:
        assert image[y, x, 0] > 150
    npt.assert_array_equal(image[95:105, 95:105], image[95:105, 95:105][::-1])
    npt.assert_array_equal(image[95:105, 95:105], image[95:105, 95:105][:, ::-1])


def test_border_and_shadow_scale_with_pixel_ratio(show_window):
    """At pixel ratio 2 widths in pixels cover twice as many device pixels."""
    box = _box(
        border_color=(1, 0, 0),
        border_width=4,
        shadow_color=(0, 0, 0, 1),
        shadow_spread=6,
    )
    image = _snapshot(show_window(box, pixel_ratio=2))
    assert image.shape[:2] == (400, 400)

    _assert_color(image[200, 101], (1, 0, 0))
    _assert_color(image[200, 106], (1, 0, 0))
    _assert_color(image[200, 110], FILL)
    _assert_color(image[200, 98], (0, 0, 0))
    _assert_color(image[200, 89], (0, 0, 0))
    _assert_color(image[200, 86], BACKGROUND)


def test_style_color_getters_round_trip():
    """Border and shadow colors read back as RGBA in [0, 1]."""
    box = ui.RoundedRectangle2D(border_color="#FF0000", shadow_color=(0, 0, 0, 0.35))
    npt.assert_allclose(box.border_color, (1, 0, 0, 1))
    npt.assert_allclose(box.shadow_color, (0, 0, 0, 0.35), atol=1e-6)

    box.border_color = (0, 128, 255)
    box.shadow_color = "#00FF00"
    npt.assert_allclose(box.border_color, (0, 128 / 255, 1, 1), atol=1e-6)
    npt.assert_allclose(box.shadow_color, (0, 1, 0, 1))


def test_shadow_does_not_show_through_a_translucent_box(show_window):
    """Like a CSS box shadow, the shadow is cut out under the box."""
    plain = ui.RoundedRectangle2D(
        size=(100, 100), position=(50, 50), color=(1, 1, 1), opacity=0.5
    )
    plain_image = _snapshot(show_window(plain))
    shadowed = ui.RoundedRectangle2D(
        size=(100, 100),
        position=(50, 50),
        color=(1, 1, 1),
        opacity=0.5,
        shadow_color=(0, 0, 0, 1),
        shadow_blur=10,
        shadow_offset=(10, 10),
    )
    image = _snapshot(show_window(shadowed))

    npt.assert_array_equal(image[100, 100], plain_image[100, 100])
    npt.assert_array_equal(image[140, 140], plain_image[140, 140])
    assert image[100, 155].sum() < plain_image[100, 155].sum()


def test_large_blur_is_not_cut_off_at_the_quad_edge(show_window):
    """A wide blur fades out before the edge of the shadow geometry."""
    box = ui.RoundedRectangle2D(
        size=(40, 40),
        position=(130, 130),
        color=FILL,
        corner_radius=0,
        shadow_color=(0, 0, 0, 1),
        shadow_blur=40,
    )
    image = _snapshot(show_window(box, size=(300, 300)))
    quad_edge = 170 + int(1.5 * 40 + 1)
    row = image[150, 170 : quad_edge + 10, 0]

    assert row[0] < 200
    assert np.all(np.diff(row) >= 0)
    assert np.abs(np.diff(row)).max() <= 12
    assert row[quad_edge - 170 - 1] >= 253
    assert row[-1] == 255


def test_shadow_is_drawn_on_the_offset_side_only(show_window):
    """The shadow shows below and right of the box, not above or left."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_blur=10, shadow_offset=(15, 15))
    image = _snapshot(show_window(box))

    assert image[100, 155].sum() < 3 * 200
    assert image[155, 100].sum() < 3 * 200
    _assert_color(image[100, 44], BACKGROUND)
    _assert_color(image[44, 100], BACKGROUND)
    _assert_color(image[100, 100], FILL)


def test_shadow_is_soft(show_window):
    """The blurred shadow fades out gradually away from the box."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_blur=20, shadow_offset=(0, 0))
    image = _snapshot(show_window(box))
    row = image[100, 151:180, 0]

    assert np.all(np.diff(row) >= 0)
    assert row[0] < 200
    assert row[-1] > 245
    assert len(np.unique(row)) > 10


def test_shadow_spread_without_blur_draws_a_hard_ring(show_window):
    """Spread alone grows the shadow shape with a hard edge."""
    box = _box(shadow_color=(1, 0, 0, 1), shadow_spread=8)
    image = _snapshot(show_window(box))

    _assert_color(image[100, 45], (1, 0, 0))
    _assert_color(image[100, 40], BACKGROUND)


def test_shadow_is_drawn_behind_the_box(show_window):
    """The box covers its shadow at the anti-aliased edge too."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_spread=10)
    assert box.shadow_actor.render_order < box.actor.render_order
    image = _snapshot(show_window(box))

    _assert_color(image[100, 100], FILL)
    assert image[100, 149, 2] > 200
    assert image[100, 149, 0] < 80


def test_shadow_spread_setter_after_creation(show_window):
    """Setting a spread on a box without shadow makes the shadow appear."""
    box = _box(shadow_color=(1, 0, 0, 1))
    show_m = show_window(box)
    _assert_color(_snapshot(show_m)[100, 45], BACKGROUND)

    box.shadow_spread = 8
    assert box.shadow_actor.visible is True
    image = _snapshot(show_m)
    _assert_color(image[100, 45], (1, 0, 0))
    _assert_color(image[100, 40], BACKGROUND)


def test_shadow_follows_set_position(show_window):
    """Moving the box moves its shadow with it."""
    box = _box(shadow_color=(1, 0, 0, 1), shadow_offset=(0, 10))
    show_m = show_window(box)
    _assert_color(_snapshot(show_m)[155, 100], (1, 0, 0))

    box.set_position((20, 20))
    image = _snapshot(show_m)
    _assert_color(image[125, 70], (1, 0, 0))
    _assert_color(image[155, 100], BACKGROUND)


def test_shadow_offset_scales_with_pixel_ratio(show_window):
    """At pixel ratio 2 the offset covers twice as many device pixels."""
    box = _box(shadow_color=(1, 0, 0, 1), shadow_offset=(10, 0))
    image = _snapshot(show_window(box, pixel_ratio=2))

    _assert_color(image[200, 296], FILL)
    _assert_color(image[200, 304], (1, 0, 0))
    _assert_color(image[200, 316], (1, 0, 0))
    _assert_color(image[200, 324], BACKGROUND)


def test_sub_pixel_blur_keeps_the_edge_anti_aliased(show_window):
    """A tiny blur still gives an anti-aliased, not a jagged, shadow edge."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_spread=10, shadow_blur=0.1)
    image = _snapshot(show_window(box))
    row = image[100, 36:45, 0]

    assert row[0] == 255
    assert row[-1] == 0
    assert np.any((row > 20) & (row < 235))


def test_shadow_does_not_change_size_position_or_picking(show_window):
    """The shadow is not part of the box for size, position and hit tests."""
    plain = _box()
    box = _box(shadow_color=(0, 0, 0, 1), shadow_blur=10, shadow_offset=(20, 20))
    npt.assert_array_equal(box.size, plain.size)
    npt.assert_array_equal(box.get_position(), plain.get_position())

    show_m = show_window(box)
    _snapshot(show_m)
    assert show_m.renderer.get_pick_info((100, 100))["world_object"] is box.actor
    assert show_m.renderer.get_pick_info((160, 160))["world_object"] is None


def test_shadow_does_not_change_layout():
    """Neighbors in a row are placed by the box size, not the shadow."""
    first = ui.RoundedRectangle2D(
        size=(40, 30), shadow_blur=20, shadow_offset=(10, 10), shadow_spread=4
    )
    second = ui.RoundedRectangle2D(size=(40, 30))
    row = ui.Row(first, second, gap=0, position=(10, 10))
    row.update_layout()

    npt.assert_array_equal(first.size, (40, 30))
    assert second.get_position()[0] == first.get_position()[0] + 40


def test_shadow_does_not_hide_picks_of_elements_below(show_window):
    """An element under the shadow can still be picked."""
    below = ui.Rectangle2D(size=(40, 40), position=(150, 150), color=(0, 1, 0))
    box = _box(shadow_color=(0, 0, 0, 1), shadow_blur=10, shadow_offset=(20, 20))
    box.z_order = 1
    show_m = show_window(below, box)
    image = _snapshot(show_m)

    assert image[165, 165, 1] < 200
    assert show_m.renderer.get_pick_info((165, 165))["world_object"] is below.actor


def test_setters_update_the_render(show_window):
    """Changing the style after creation redraws the box with it."""
    box = _box()
    show_m = show_window(box)
    image = _snapshot(show_m)
    _assert_color(image[100, 52], FILL)
    _assert_color(image[52, 52], FILL)
    _assert_color(image[100, 160], BACKGROUND)

    box.border_color = (0, 1, 0)
    box.border_width = 5
    box.corner_radius = (40, 0, 0, 0)
    box.shadow_color = (0, 0, 0, 1)
    box.shadow_offset = (15, 0)
    box.shadow_blur = 4
    image = _snapshot(show_m)
    _assert_color(image[100, 52], (0, 1, 0))
    _assert_color(image[52, 52], BACKGROUND)
    assert image[100, 155].sum() < 3 * 100

    box.border_width = 0
    box.corner_radius = 0
    box.shadow_blur = 0
    box.shadow_offset = (0, 0)
    image = _snapshot(show_m)
    _assert_color(image[100, 52], FILL)
    _assert_color(image[52, 52], FILL)
    _assert_color(image[100, 160], BACKGROUND)
    assert box.shadow_actor.visible is False


def test_resize_keeps_the_shadow_around_the_box(show_window):
    """After a resize the shadow follows the new box size."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_spread=6)
    show_m = show_window(box)
    box.resize((60, 60))
    image = _snapshot(show_m)

    _assert_color(image[80, 113], (0, 0, 0))
    _assert_color(image[80, 120], BACKGROUND)
    npt.assert_array_equal(box.size, (60, 60))


def test_hiding_the_box_hides_its_shadow():
    """The shadow follows the box visibility and only returns when enabled."""
    box = _box(shadow_blur=10)
    box.set_visibility(False)
    assert box.actor.visible is False
    assert box.shadow_actor.visible is False

    box.set_visibility(True)
    assert box.shadow_actor.visible is True

    box.shadow_blur = 0
    box.set_visibility(False)
    box.set_visibility(True)
    assert box.shadow_actor.visible is False


def test_opacity_applies_to_the_shadow(show_window):
    """The box opacity also fades its shadow."""
    box = _box(shadow_color=(0, 0, 0, 1), shadow_spread=10)
    box.opacity = 0.5
    image = _snapshot(show_window(box))

    half_in_srgb = 255 * (1.055 * 0.5 ** (1 / 2.4) - 0.055)
    npt.assert_allclose(image[100, 45], [half_in_srgb] * 3, atol=5)


def test_opaque_border_hides_the_fill_when_faded(show_window):
    """Opacity fades the composited box, so the fill stays under the border."""
    box = _box(border_color=(1, 0, 0), border_width=6, opacity=0.5)
    image = _snapshot(show_window(box))

    half_in_srgb = 255 * (1.055 * 0.5 ** (1 / 2.4) - 0.055)
    npt.assert_allclose(image[100, 53], (255, half_in_srgb, half_in_srgb), atol=2)
    npt.assert_allclose(image[100, 100], (half_in_srgb, half_in_srgb, 255), atol=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"border_width": -1},
        {"shadow_blur": -2},
        {"corner_radius": (1, 2, 3)},
        {"corner_radius": (1, 2, 3, 4, 5)},
        {"corner_radius": -1},
        {"corner_radius": (1, -2, 3, 4)},
        {"shadow_offset": (1, 2, 3)},
    ],
)
def test_invalid_style_arguments_raise(kwargs):
    """Negative sizes and wrongly shaped tuples are rejected."""
    with pytest.raises(ValueError):
        ui.RoundedRectangle2D(**kwargs)

    box = ui.RoundedRectangle2D()
    name, value = next(iter(kwargs.items()))
    with pytest.raises(ValueError):
        setattr(box, name, value)


@pytest.mark.parametrize("opacity", [1.0, 0.7])
def test_square_borderless_box_matches_rectangle_pixels(show_window, opacity):
    """A box with square corners and no border draws exactly like Rectangle2D."""
    box = _box(opacity=opacity)
    rectangle = ui.Rectangle2D(
        size=(100, 100), position=(50, 50), color=FILL, opacity=opacity
    )

    box_image = _snapshot(show_window(box))
    rectangle_image = _snapshot(show_window(rectangle))

    npt.assert_array_equal(box_image, rectangle_image)


def test_box_switches_material_and_keeps_events(show_window):
    """Rounding and squaring a box changes its look but keeps its events."""
    box = _box()
    presses = []
    box.on_left_mouse_button_pressed = presses.append
    show_m = show_window(box)

    box.corner_radius = 30
    _assert_color(_snapshot(show_m)[52, 52], BACKGROUND)
    box.corner_radius = 0
    _assert_color(_snapshot(show_m)[52, 52], FILL)
    box.border_width = 4
    box.border_color = (0, 1, 0)
    _assert_color(_snapshot(show_m)[100, 51], (0, 1, 0))

    box.border_width = 0
    _assert_color(_snapshot(show_m)[100, 51], FILL)

    pick_info = show_m.renderer.get_pick_info((100, 100))
    assert pick_info["world_object"] is box.actor
    show_m.renderer.dispatch_event(
        window.PointerEvent(
            x=100,
            y=100,
            type=window.EventType.POINTER_DOWN,
            target=pick_info["world_object"],
            button=1,
        )
    )
    assert len(presses) == 1


def test_box_material_switch_keeps_picking_settings(show_window):
    """A box made non-pickable stays non-pickable after it is rounded."""
    box = _box()
    box.actor.material.pick_write = False
    show_m = show_window(box)
    box.corner_radius = 20
    _snapshot(show_m)

    assert show_m.renderer.get_pick_info((100, 100))["world_object"] is None


def test_text_font_weight_thickens_glyphs(show_window):
    """A semibold text block covers more pixels than a regular one."""
    images = []
    for weight in (400, 600):
        text = ui.TextBlock2D(
            text="Weight",
            font_size=24,
            color=(0, 0, 0),
            position=(20, 80),
            dynamic_bbox=True,
            font_weight=weight,
        )
        assert text.font_weight == weight
        images.append(_snapshot(show_window(text)))

    regular, semibold = ((image.sum(axis=-1) < 3 * 128).sum() for image in images)
    assert semibold > regular * 1.1

    with pytest.raises(ValueError, match="font_weight"):
        text.font_weight = 1000
