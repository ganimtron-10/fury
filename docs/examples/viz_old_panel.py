"""
========
Panel UI
========

This example shows how old Panel UI works with new UI System.

First, some imports.
"""

import numpy as np

from fury.ui import UI, Rectangle2D, Anchor, Disk2D, UIContext
from fury.window import Scene, ShowManager
from fury.decorators import warn_on_args_to_kwargs
from fury.lib import PointerEvent

###############################################################################
# First, we reuse the old Panel UI Code.


class Panel2D(UI):
    """A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.
    """

    @warn_on_args_to_kwargs()
    def __init__(
        self,
        size,
        *,
        position=(0, 0),
        color=(0.1, 0.1, 0.1),
        opacity=0.7,
        align="left",
        border_color=(1, 1, 1),
        border_width=0,
        has_border=False,
    ):
        """Init class instance.

        Parameters
        ----------
        size : (int, int)
            Size (width, height) in pixels of the panel.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of the panel.
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].
        align : [left, right]
            Alignment of the panel with respect to the overall screen.
        border_color: (float, float, float), optional
            Must take values in [0, 1].
        border_width: float, optional
            width of the border
        has_border: bool, optional
            If the panel should have borders.

        """
        self.has_border = has_border
        self._border_color = border_color
        self._border_width = border_width
        super(Panel2D, self).__init__(position=position)
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self.position = position
        self._drag_offset = None

    def _setup(self):
        """Setup this UI component.

        Create the background (Rectangle2D) of the panel.
        Create the borders (Rectangle2D) of the panel.
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D()

        if self.has_border:
            self.borders = {
                "left": Rectangle2D(),
                "right": Rectangle2D(),
                "top": Rectangle2D(),
                "bottom": Rectangle2D(),
            }

            self.border_coords = {
                "left": (0.0, 0.0),
                "right": (1.0, 0.0),
                "top": (0.0, 1.0),
                "bottom": (0.0, 0.0),
            }

            for key in self.borders.keys():
                self.borders[key].color = self._border_color
                self.add_element(self.borders[key], self.border_coords[key])

            for key in self.borders.keys():
                self.borders[
                    key
                ].on_left_mouse_button_pressed = self.left_button_pressed

                self.borders[
                    key
                ].on_left_mouse_button_dragged = self.left_button_dragged

        self.add_element(self.background, (0, 0))

        # Add default events listener for this UI component.
        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    # v2: Added this function
    def _update_actors_position(self):
        self._set_position(
            self.get_position(x_anchor=Anchor.LEFT, y_anchor=Anchor.BOTTOM)
        )

    def _get_actors(self):
        """Get the actors composing this UI component."""
        actors = []
        for element in self._elements:
            actors += element.actors

        return actors

    def _add_to_scene(self, scene):
        """Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        scene : scene

        """
        for element in self._elements:
            element.add_to_scene(scene)

    def _get_size(self):
        return self.background.size

    def resize(self, size):
        """Set the panel size.

        Parameters
        ----------
        size : (float, float)
            Panel size (width, height) in pixels.

        """
        self.background.resize(size)

        if self.has_border:
            self.borders["left"].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders["right"].resize(
                (self._border_width, size[1] + self._border_width)
            )

            self.borders["top"].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.borders["bottom"].resize(
                (self.size[0] + self._border_width, self._border_width)
            )

            self.update_border_coords()

    def _set_position(self, coords):
        """Set the lower-left corner position of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).

        """
        coords = np.array(coords)
        for element, offset in self.element_offsets:
            element.position = coords + offset

    def set_visibility(self, visibility):
        for element in self._elements:
            element.set_visibility(visibility)

    @property
    def color(self):
        return self.background.color

    @color.setter
    def color(self, color):
        self.background.color = color

    @property
    def opacity(self):
        return self.background.opacity

    @opacity.setter
    def opacity(self, opacity):
        self.background.opacity = opacity

    @warn_on_args_to_kwargs()
    def add_element(self, element, coords, *, anchor="position"):
        """Add a UI component to the panel.

        The coordinates represent an offset from the lower left corner of the
        panel.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        coords : (float, float) or (int, int)
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.

        """
        coords = np.array(coords)

        if np.issubdtype(coords.dtype, np.floating):
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError("Normalized coordinates must be in [0,1].")

            coords = coords * self.size

        if anchor == "center":
            element.center = self.position + coords
        elif anchor == "position":
            element.position = self.position + coords
        else:
            msg = "Unknown anchor {}. Supported anchors are 'position' and 'center'."
            raise ValueError(msg)

        self._elements.append(element)
        offset = element.position - self.position
        self.element_offsets.append((element, offset))
        self._children.append(element)

    def remove_element(self, element):
        """Remove a UI component from the panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.

        """
        idx = self._elements.index(element)
        del self._elements[idx]
        del self.element_offsets[idx]

    @warn_on_args_to_kwargs()
    def update_element(self, element, coords, *, anchor="position"):
        """Update the position of a UI component in the panel.

        Parameters
        ----------
        element : UI
            The UI item to be updated.
        coords : (float, float) or (int, int)
            New coordinates.
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.

        """
        self.remove_element(element)
        self.add_element(element, coords, anchor=anchor)

    def left_button_pressed(self, event: PointerEvent):
        click_pos = np.array([event.x, UIContext.get_canvas_size()[1] - event.y])
        self._drag_offset = click_pos - self.position
        event.cancel()

    def left_button_dragged(self, event: PointerEvent):
        if self._drag_offset is not None:
            click_position = np.array(
                [event.x, UIContext.get_canvas_size()[1] - event.y]
            )
            new_position = click_position - self._drag_offset
            self.position = new_position

    def re_align(self, window_size_change):
        """Re-organise the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.

        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.position += np.array(window_size_change)
        else:
            msg = "You can only left-align or right-align objects in a panel."
            raise ValueError(msg)

    def update_border_coords(self):
        """Update the coordinates of the borders"""
        self.border_coords = {
            "left": (0.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, 1.0),
            "bottom": (0.0, 0.0),
        }

        for key in self.borders.keys():
            self.update_element(self.borders[key], self.border_coords[key])

    @property
    def border_color(self):
        sides = ["left", "right", "top", "bottom"]
        return [self.borders[side].color for side in sides]

    @border_color.setter
    def border_color(self, side_color):
        """Set the color of a specific border

        Parameters
        ----------
        side_color: Iterable
            Iterable to pack side, color values

        """
        side, color = side_color

        if side.lower() not in ["left", "right", "top", "bottom"]:
            raise ValueError(f"{side} not a valid border side")

        self.borders[side].color = color

    @property
    def border_width(self):
        sides = ["left", "right", "top", "bottom"]
        widths = []

        for side in sides:
            if side in ["left", "right"]:
                widths.append(self.borders[side].width)
            elif side in ["top", "bottom"]:
                widths.append(self.borders[side].height)
            else:
                raise ValueError(f"{side} not a valid border side")
        return widths

    @border_width.setter
    def border_width(self, side_width):
        """Set the border width of a specific border

        Parameters
        ----------
        side_width: Iterable
            Iterable to pack side, width values

        """
        side, border_width = side_width

        if side.lower() in ["left", "right"]:
            self.borders[side].width = border_width
        elif side.lower() in ["top", "bottom"]:
            self.borders[side].height = border_width
        else:
            raise ValueError(f"{side} not a valid border side")


###############################################################################
# Intialize Panel UI.

panel_ui = Panel2D(
    size=(300, 150),
    color=(1, 1, 1),
    align="right",
    position=(200, 200),
    has_border=True,
    border_width=5,
    border_color=(1, 0, 1),
)

###############################################################################
# Adding elements to Panel.

rect = Rectangle2D(size=(50, 50), color=(0, 0, 1))
panel_ui.add_element(rect, coords=(0, 0.5))

disk = Disk2D(outer_radius=25, color=(1, 1, 0))
panel_ui.add_element(disk, coords=(0, 0))

###############################################################################
# Next we prepare the scene and render it with the help of show manager.

scene = Scene()
scene.add(panel_ui)

if __name__ == "__main__":
    current_size = (800, 800)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Old Panel UI",
        use_old_ui=True,
    )
    show_manager.start()
