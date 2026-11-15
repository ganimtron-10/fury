"""Shared helpers for UI module tests."""

from fury import ui
from fury.data import fetch_viz_icons, read_viz_icons


def icon(fname="play3.png"):
    """Return a real icon image path, fetching the asset set on first use."""
    fetch_viz_icons()
    return read_viz_icons(fname=fname)


def send_pointer(show_m, kind, position, *, button=0, pointer_id=1):
    """
    Send a pointer event through the renderer, as the canvas does.

    The renderer picks the object under the pointer and dispatches the event,
    generating enter and leave events when the object under the pointer
    changes.

    Parameters
    ----------
    show_m : ShowManager
        The show manager whose renderer receives the event.
    kind : str
        ``"pointer_move"``, ``"pointer_down"`` or ``"pointer_up"``.
    position : (float, float)
        Logical canvas coordinates, with y pointing down.
    button : int, optional
        Mouse button, 1 for the left button.
    pointer_id : int, optional
        Identifier of the pointer.
    """
    show_m.renderer.convert_event(
        {
            "event_type": kind,
            "x": float(position[0]),
            "y": float(position[1]),
            "button": button,
            "buttons": (button,) if kind == "pointer_down" else (),
            "modifiers": (),
            "ntouches": 0,
            "touches": {},
            "pointer_id": pointer_id,
        }
    )


def click(show_m, position):
    """
    Press and release the left button at a position.

    Parameters
    ----------
    show_m : ShowManager
        The show manager whose renderer receives the events.
    position : (float, float)
        Logical canvas coordinates, with y pointing down.
    """
    send_pointer(show_m, "pointer_move", position)
    send_pointer(show_m, "pointer_down", position, button=1)
    send_pointer(show_m, "pointer_up", position, button=1)


def list_box_item():
    """Build a standalone ListBoxItem2D with visible text."""
    item = ui.ListBoxItem2D(on_select=lambda **kwargs: None, size=(100, 30))
    item.element = "Hello"
    return item


# The full set of UI elements keyed by id. Each value is a factory that builds
# a fresh element (with real inputs) on every call.
ELEMENT_FACTORIES = {
    "textured_button_2d": lambda: ui.TexturedButton2D(
        states={"default": icon("play3.png")},
        position=(100, 100),
        size=(50, 50),
    ),
    "text_button_2d": lambda: ui.TextButton2D(
        label="BlueBtn", size=(120, 50), position=(50, 50)
    ),
    "line_slider_2d": lambda: ui.LineSlider2D(
        position=(100, 100), initial_value=25, min_value=0, max_value=100, length=250
    ),
    "playback_panel": lambda: ui.PlaybackPanel(position=(0, 0), width=900),
    "text_box_2d": lambda: ui.TextBox2D(
        width=10, height=3, text="Hello", position=(100, 100)
    ),
    "line_double_slider_2d": lambda: ui.LineDoubleSlider2D(
        position=(100, 100),
        initial_values=(20, 80),
        min_value=0,
        max_value=100,
        length=200,
    ),
    "ring_slider_2d": lambda: ui.RingSlider2D(
        center=(150, 150), initial_value=50, min_value=0, max_value=100
    ),
    "range_slider": lambda: ui.RangeSlider(
        range_slider_center=(450, 400),
        value_slider_center=(450, 300),
        length=200,
        min_value=0,
        max_value=100,
    ),
    "combo_box_2d": lambda: ui.ComboBox2D(
        items=["Item0", "Item1", "Item2"],
        position=(100, 200),
        size=(300, 200),
        placeholder="Pick one...",
    ),
    "list_box_2d": lambda: ui.ListBox2D(
        values=["Item 1", "Item 2", "Item 3"],
        position=(0, 0),
        size=(200, 200),
        multiselection=True,
    ),
    "list_box_item_2d": list_box_item,
    "card_2d": lambda: ui.Card2D(
        image_path=icon("play3.png"),
        title_text="Test Title",
        body_text="Test Body",
        size=(400, 400),
        image_scale=0.5,
        padding=10,
        border_width=2,
    ),
    "panel_2d": lambda: ui.Panel2D(size=(400, 400), position=(400, 400)),
    "tab_panel_2d": lambda: ui.TabPanel2D(size=(120, 30), title="Tab 1"),
    "tab_ui": lambda: ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        startup_tab_id=1,
    ),
    "image_container_2d": lambda: ui.ImageContainer2D(
        img_path=icon("home3.png"), size=(100, 100)
    ),
    "switch_2d": lambda: ui.Switch2D(label="Shading", position=(50, 50)),
    "segmented_control_2d": lambda: ui.SegmentedControl2D(
        ["Axial", "Coronal", "Sagittal"], position=(50, 50)
    ),
}
