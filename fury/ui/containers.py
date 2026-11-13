"""UI container module."""

import logging

import numpy as np

from fury.io import load_image
from fury.lib import Texture
from fury.ui.core import UI, Anchor, Rectangle2D, TextBlock2D
from fury.ui.layout import Align, Direction, Edges, Position, place


class Panel2D(UI):
    """
    A 2D UI Panel.

    Can contain one or more UI elements.

    Parameters
    ----------
    size : (int, int)
        Size (width, height) in pixels of the panel.
    position : (float, float)
        Absolute coordinates (x, y) of the top-left corner of the panel.
    color : str, tuple, list or ndarray
        A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
    opacity : float
        Must take values in [0, 1].
    align : [left, right]
        Alignment of the panel with respect to the overall screen.
    border_color : str, tuple, list or ndarray, optional
        Border color, same formats as ``color``.
    border_width : float, optional
        Width of the border.
    has_border : bool, optional
        If the panel should have borders.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.
    """

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
        """Initialize class instance."""
        self.border_sides = ["left", "right", "top", "bottom"]
        self.border_coords = {
            "left": (0.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, 0.0),
            "bottom": (0.0, 1.0),
        }
        self.has_border = has_border
        self._border_color = border_color
        side_width = border_width if has_border else 0
        self._border_widths = dict.fromkeys(self.border_sides, side_width)
        self._content_padding = Edges()
        super(Panel2D, self).__init__(position=position)
        self.set_layout(width=size[0], height=size[1])
        self._update_border_inset()
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self._drag_offset = None

    def _setup(self):
        """
        Set up this UI component.

        Create the background (Rectangle2D) of the panel and initialize
        the border elements (Rectangle2D).
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D(size=(1, 1))

        if self.has_border:
            self.borders = {
                "left": Rectangle2D(size=(1, 1)),
                "right": Rectangle2D(size=(1, 1)),
                "top": Rectangle2D(size=(1, 1)),
                "bottom": Rectangle2D(size=(1, 1)),
            }

            for key in self.borders.keys():
                self.borders[key].color = self._border_color
                self.borders[
                    key
                ].on_left_mouse_button_pressed = self.left_button_pressed

                self.borders[
                    key
                ].on_left_mouse_button_dragged = self.left_button_dragged
                self.add_element(
                    self.borders[key], self.border_coords[key], _is_internal=True
                )

        self.add_element(self.background, (0, 0), _is_internal=True)

        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _get_actors(self):
        """
        Get actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        actors = []

        actors.extend(self.background.actors)
        if self.has_border:
            for border in self.borders.values():
                actors.extend(border.actors)

        return actors

    def _get_size(self):
        """
        Get the actual size of the panel.

        Returns
        -------
        (float, float)
            The (width, height) size of the panel.
        """
        return self.background.size

    def resize(self, size):
        """
        Set the panel size.

        Parameters
        ----------
        size : (float, float)
            Panel size (width, height) in pixels.
        """
        style = self.layout_style
        if style.width.is_fixed:
            style.update(width=size[0])
        if style.height.is_fixed:
            style.update(height=size[1])
        self.mark_layout_dirty()

        self.background.resize(size)

        if self.has_border:
            widths = self._border_widths
            for side in ("left", "right"):
                self.borders[side].resize((widths[side], size[1] + widths["bottom"]))
            for side in ("top", "bottom"):
                self.borders[side].resize(
                    (self.size[0] + widths["right"], widths[side])
                )

            self.update_border_coords()

    def set_layout(self, **properties):
        """
        Change the layout properties of this panel.

        The panel's borders are added to ``padding``: the layout ``border``
        holds the width shared by all sides, and ``padding`` holds the given
        padding plus any extra width of a thicker side.

        Parameters
        ----------
        **properties : dict
            Properties of :class:`fury.ui.layout.LayoutStyle`, such as
            ``width``, ``padding`` or ``gap``.

        Returns
        -------
        Panel2D
            This panel, so calls can be chained.
        """
        if "padding" in properties:
            self._content_padding = Edges.of(properties.pop("padding"))
            properties.update(self._border_inset())
        return super().set_layout(**properties)

    def _border_inset(self):
        """
        Get the layout border and padding that keep content inside the borders.

        Returns
        -------
        dict
            ``border`` and ``padding`` layout properties.
        """
        widths = self._border_widths
        shared = min(widths.values())
        padding = self._content_padding
        return {
            "border": shared,
            "padding": Edges(
                padding.top + widths["top"] - shared,
                padding.right + widths["right"] - shared,
                padding.bottom + widths["bottom"] - shared,
                padding.left + widths["left"] - shared,
            ),
        }

    def _update_border_inset(self):
        """Apply the current border widths to the layout."""
        super().set_layout(**self._border_inset())

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        coords = self.get_position()

        for element, offset in self.element_offsets:
            element.z_order = self._element_z_order(element)
            element.set_position(coords + offset)

    def _element_z_order(self, element):
        """
        Get the Z-order that draws an element above the panel's background.

        Parameters
        ----------
        element : UI
            An element of the panel.

        Returns
        -------
        int
            The panel's Z-order for the background, one more for the
            borders, and two more for the other elements.
        """
        if element is self.background:
            return self.z_order
        if self.has_border and element in self.borders.values():
            return self.z_order + 1
        return self.z_order + 2

    def set_visibility(self, visibility):
        """
        Set visibility of this UI component.

        Parameters
        ----------
        visibility : bool
            If True, the panel and its elements will be visible. If False, it will
            be hidden.
        """
        for element in self._elements:
            element.set_visibility(visibility)

    @property
    def color(self):
        """
        Get the background color of the panel.

        Returns
        -------
        (float, float, float)
            RGB color of the panel background.
        """
        return self.background.color

    @color.setter
    def color(self, color):
        """
        Set the background color of the panel.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            New background color. A hex string ("#FF0000"), RGB(A) in [0, 1],
            or RGB(A) in [0, 255].
        """
        self.background.color = color

    @property
    def opacity(self):
        """
        Get the opacity of the panel.

        Returns
        -------
        float
            Opacity value.
        """
        return self.background.opacity

    @opacity.setter
    def opacity(self, opacity):
        """
        Set the opacity of the panel.

        Parameters
        ----------
        opacity : float
            New opacity value.
        """
        self.background.opacity = opacity

    def add_element(self, element, coords, *, anchor="position", _is_internal=False):
        """
        Add a UI component to the panel.

        The coordinates represent an offset from the top-left corner of the
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
        anchor : str, optional
            Supported anchors are 'position' (top-left) and 'center'.
        _is_internal : bool, optional
            Flag used to distinguish between user-added elements
            and internal elements added by the Panel itself.

        Raises
        ------
        ValueError
            If coordinates are normalized but outside the [0,1] range, or if
            an unknown anchor is provided.
        """
        self._place_element(element, coords, anchor=anchor)
        if not _is_internal:
            self._add_child(element)

    def _place_element(self, element, coords, *, anchor="position"):
        """
        Position a UI component in the panel and record its offset.

        Parameters
        ----------
        element : UI
            The UI item to be placed.
        coords : (float, float) or (int, int)
            Normalized coordinates in [0, 1] if float, pixels if int.
        anchor : str, optional
            Supported anchors are 'position' (top-left) and 'center'.

        Raises
        ------
        ValueError
            If coordinates are normalized but outside the [0,1] range, or if
            an unknown anchor is provided.
        """
        coords = np.array(coords)

        if np.issubdtype(coords.dtype, np.floating):
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError("Normalized coordinates must be in [0,1].")

            coords = coords * self.size

        element.z_order = self._element_z_order(element)
        if anchor == "center":
            element.set_position(
                self.get_position() + coords,
                x_anchor=Anchor.CENTER,
                y_anchor=Anchor.CENTER,
            )
        elif anchor == "position":
            element.set_position(
                self.get_position() + coords,
            )
        else:
            msg = f"Unknown anchor {anchor}. Supported anchors are 'position' and \
                'center'."
            raise ValueError(msg)

        self._elements.append(element)
        offset = element.get_position() - self.get_position()
        self.element_offsets.append((element, offset))

    def add(self, element, **properties):
        """
        Add a UI component to be positioned by the panel's layout.

        Use :meth:`set_layout` on the panel to choose the direction, padding
        and alignment. Use :meth:`add_element` for manual placement instead.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        **properties : dict
            Layout properties applied to ``element``.

        Returns
        -------
        UI
            The added element.
        """
        if properties:
            element.set_layout(**properties)
        element.z_order = self._element_z_order(element)
        self._layout_node.add(element._layout_node)
        self._elements.append(element)
        self.element_offsets.append((element, np.zeros(2)))
        self._add_child(element)
        return element

    def _apply_layout(self, rect):
        """
        Record where the layout placed the children, then move the panel.

        The offsets let the panel be dragged without solving the layout again.

        Parameters
        ----------
        rect : LayoutRect
            The computed position and size.
        """
        origin = np.array(rect.position, dtype=float)
        for index, (element, _) in enumerate(self.element_offsets):
            if element._layout_node.parent is self._layout_node:
                offset = np.array(element.layout_rect.position, dtype=float) - origin
                self.element_offsets[index] = (element, offset)
        super()._apply_layout(rect)

    def remove_element(self, element):
        """
        Remove a UI component from the panel.

        Parameters
        ----------
        element : UI
            The UI item to be removed.

        Raises
        ------
        ValueError
            If the element is not found in the panel's elements list.
        """
        self._forget_element(element)
        if element in self._children:
            self._remove_child(element)

    def _forget_element(self, element):
        """
        Stop positioning a UI component, leaving it drawn.

        Parameters
        ----------
        element : UI
            The UI item to forget.

        Raises
        ------
        ValueError
            If the element is not found in the panel's elements list.
        """
        idx = self._elements.index(element)
        node = self._layout_node
        if element._layout_node.parent is node:
            node.remove(element._layout_node)
            if not node.children:
                self._layout_natural_size = (node.style.inset(0), node.style.inset(1))
        del self._elements[idx]
        del self.element_offsets[idx]

    def update_element(self, element, coords, *, anchor="position"):
        """
        Update the position of a UI component in the panel.

        Parameters
        ----------
        element : UI
            The UI item to be updated.
        coords : (float, float) or (int, int)
            New coordinates.
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixel coordinates are assumed and it must fit within the
            panel's size.
        anchor : str, optional
            Supported anchors are 'position' (top-left) and 'center'.

        Raises
        ------
        ValueError
            If coordinates are normalized but outside the [0,1] range, or if
            an unknown anchor is provided.
        """
        self._forget_element(element)
        self._place_element(element, coords, anchor=anchor)

    def left_button_pressed(self, event):
        """
        Handle left mouse button press event for panel.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        click_pos = np.array([event.x, event.y])
        self._drag_offset = click_pos - self.get_position()

    def left_button_dragged(self, event):
        """
        Handle left mouse button drag event for panel movement.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self._drag_offset is not None:
            click_position = np.array([event.x, event.y])
            new_position = click_position - self._drag_offset
            self.set_position(new_position)

    def re_align(self, window_size_change):
        """
        Re-organise the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.

        Raises
        ------
        ValueError
            If alignment is not 'left' or 'right'.
        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.set_position(window_size_change)
        else:
            msg = "You can only left-align or right-align objects in a panel."
            raise ValueError(msg)

    def update_border_coords(self):
        """Update the coordinates of the borders."""
        if not self.has_border:
            return

        for key, border in self.borders.items():
            offset = np.array(self.border_coords[key]) * self.size
            for index, (element, _) in enumerate(self.element_offsets):
                if element is border:
                    self.element_offsets[index] = (border, offset)
            border.set_position(self.get_position() + offset)

    @property
    def border_color(self):
        """
        Get the current color of all four borders.

        Returns
        -------
        list
            A list containing the color (RGB tuple) of the left, right, top, and bottom
            borders, respectively.
        """
        if not self.has_border:
            logging.warning("Border is not present, border color is not available.")
            return []
        return [self.borders[side].color for side in self.border_sides]

    @border_color.setter
    def border_color(self, side_color):
        """
        Set the color of a specific border.

        Parameters
        ----------
        side_color : Iterable
            Iterable `[side, color]` containing the side (str) and the color.
            The color accepts a hex string ("#FF0000"), RGB(A) in [0, 1], or
            RGB(A) in [0, 255].
        """
        side, color = side_color

        if side.lower() not in ["left", "right", "top", "bottom"]:
            raise ValueError(f"{side} not a valid border side")

        if not self.has_border:
            logging.warning(
                "Border is not present, setting border color will be ignored."
            )
            return

        self.borders[side].color = color

    @property
    def border_width(self):
        """
        Get the current width/height of the borders.

        Returns
        -------
        list
            A list containing the width (for left/right) and height (for top/bottom)
            of the borders.
        """
        if not self.has_border:
            logging.warning("Border is not present, border width is not available.")
            return []

        widths = []

        for side in self.border_sides:
            if side in ["left", "right"]:
                widths.append(self.borders[side].width)
            elif side in ["top", "bottom"]:
                widths.append(self.borders[side].height)
            else:
                raise ValueError(f"{side} not a valid border side")
        return widths

    @border_width.setter
    def border_width(self, side_width):
        """
        Set the width of a specific border.

        Parameters
        ----------
        side_width : Iterable
            Iterable `[side, width]` containing the side (str) and the width (float).
            Only that side's inset of the layout changes.

        Raises
        ------
        ValueError
            If the side is not one of left, right, top or bottom, or the width
            is negative.
        """
        side, border_width = side_width
        side = side.lower()

        if side not in self.border_sides:
            raise ValueError(f"{side} not a valid border side")
        if border_width < 0:
            raise ValueError(f"Border width must not be negative, got {border_width}.")

        if not self.has_border:
            logging.warning(
                "Border is not present, setting border width will be ignored."
            )
            return

        self._border_widths[side] = border_width
        if side in ["left", "right"]:
            self.borders[side].width = border_width
        else:
            self.borders[side].height = border_width
        self._update_border_inset()


class LayoutBox(UI):
    """
    An invisible container that positions its children with the layout engine.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) of the box.
    children : iterable of UI, optional
        Initial children.
    **properties : dict
        Layout properties of the box, such as ``padding`` or ``gap``.
    """

    def __init__(self, *, position=(0, 0), children=(), **properties):
        """Initialize the box."""
        self._box_children = []
        super(LayoutBox, self).__init__(position=position)
        self.set_layout(**properties)
        for child in children:
            self.add(child)

    def _setup(self):
        """Set up this UI component, which has nothing to create."""

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            An empty list, as a box draws nothing itself.
        """
        return []

    def _measure_layout(self, max_width):
        """
        Report the content size of the box when it has no children.

        Parameters
        ----------
        max_width : float or None
            Width available to the content, or None when unconstrained.

        Returns
        -------
        (float, float)
            Zero, as a box has no content of its own.
        """
        return (0.0, 0.0)

    def _get_size(self):
        """
        Get the computed size of the box.

        Returns
        -------
        (int, int)
            Width and height in pixels.
        """
        self.update_layout()
        return self._layout_node.rect.size

    def resize(self, size):
        """
        Give the box a fixed size.

        Parameters
        ----------
        size : (int, int)
            Width and height in pixels.
        """
        self.set_layout(width=size[0], height=size[1])

    def _update_actors_position(self):
        """Move the children together with the box."""
        position = self.get_position()
        node = self._layout_node
        if node.parent is None:
            if tuple(position) != node.rect.position:
                place(node, position)
            return

        origin = np.array(node.rect.position)
        for child in self._box_children:
            offset = np.array(child.layout_rect.position) - origin
            child.set_position(position + offset)

    def add(self, element, **properties):
        """
        Add a UI component to the box.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        **properties : dict
            Layout properties applied to ``element``.

        Returns
        -------
        UI
            The added element.
        """
        if properties:
            element.set_layout(**properties)
        self._layout_node.add(element._layout_node)
        self._box_children.append(element)
        element.z_order = self.z_order
        self._add_child(element)
        return element

    def remove(self, element):
        """
        Remove a UI component from the box.

        Parameters
        ----------
        element : UI
            The UI item to be removed.
        """
        self._layout_node.remove(element._layout_node)
        self._box_children.remove(element)
        self._remove_child(element)


class Row(LayoutBox):
    """
    An invisible container placing its children from left to right.

    Parameters
    ----------
    *children : UI
        Initial children.
    position : (float, float), optional
        Absolute coordinates (x, y) of the row.
    **properties : dict
        Layout properties of the row, such as ``padding`` or ``gap``.
    """

    def __init__(self, *children, position=(0, 0), **properties):
        """Initialize the row."""
        properties.setdefault("direction", Direction.ROW)
        super().__init__(position=position, children=children, **properties)


class Column(LayoutBox):
    """
    An invisible container placing its children from top to bottom.

    Parameters
    ----------
    *children : UI
        Initial children.
    position : (float, float), optional
        Absolute coordinates (x, y) of the column.
    **properties : dict
        Layout properties of the column, such as ``padding`` or ``gap``.
    """

    def __init__(self, *children, position=(0, 0), **properties):
        """Initialize the column."""
        properties.setdefault("direction", Direction.COLUMN)
        super().__init__(position=position, children=children, **properties)


class TabPanel2D(UI):
    """
    A 2D tab header with an associated content panel.

    This component represents a single tab inside :class:`TabUI`. It owns the
    clickable tab header and delegates user-added UI elements to its
    ``content_panel``. The content panel visibility is controlled by
    :class:`TabUI`.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates `(x, y)` of the upper-left corner of the tab
        header.
    size : (int, int), optional
        Width and height in pixels of the tab header.
    title : str, optional
        Initial text displayed in the tab header.
    color : str, tuple, list or ndarray, optional
        Tab header background color. A hex string ("#FF0000"), RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    content_size : (int, int), optional
        Width and height in pixels of the content panel. If None, ``size`` is
        used.
    content_visible : bool, optional
        Initial visibility of the content panel.
    content_panel : :class:`Panel2D`, optional
        Panel used to store content for this tab. If None, a new panel with
        ``content_size`` is created.

    Attributes
    ----------
    panel : :class:`Panel2D`
        Header panel used as the clickable tab background.
    content_panel : :class:`Panel2D`
        Panel that stores the UI elements displayed when this tab is active.
    text_block : :class:`TextBlock2D`
        Text component used to render the tab title.
    """

    def __init__(
        self,
        *,
        position=(0, 0),
        size=(100, 100),
        title="New Tab",
        color=(0.5, 0.5, 0.5),
        content_size=None,
        content_visible=False,
        content_panel=None,
    ):
        """Initialize the tab panel."""
        self.content_panel = content_panel or Panel2D(size=content_size or size)
        self.content_panel.set_visibility(content_visible)
        self.panel_size = size
        self._text_size = size

        super(TabPanel2D, self).__init__(position=position)
        self.title = title
        self.color = color

    def _setup(self):
        """Set up the tab header panel and title text."""
        self.panel = Panel2D(size=self.panel_size)
        self.text_block = TextBlock2D(
            text="",
            size=self._text_size,
            color=(0, 0, 0),
            bg_color=None,
            justification="center",
            vertical_justification="middle",
        )
        self.panel.set_layout(align_main=Align.CENTER, align_cross=Align.CENTER)
        self.panel.add(self.text_block)
        self._children.extend([self.panel, self.content_panel])

    def _get_actors(self):
        """
        Get actors composing this component.

        Returns
        -------
        list
            Empty list because child UI elements own the actors.
        """
        return []

    def _get_size(self):
        """
        Get the tab header size.

        Returns
        -------
        (int, int)
            Width and height of the tab header in pixels.
        """
        return self.panel.size

    def _update_actors_position(self):
        """Update the tab header position."""
        self.panel.set_position(self.get_position())

    def resize(self, size):
        """
        Resize the tab header.

        Parameters
        ----------
        size : (int, int)
            New width and height in pixels of the tab header.
        """
        self.panel_size = size
        self._text_size = size
        self.panel.resize(size)
        self.text_block.resize(size)
        self.panel.update_layout()

    @property
    def color(self):
        """
        Get the tab header background color.

        Returns
        -------
        (float, float, float)
            RGB color of the tab header background.
        """
        return self.panel.color

    @color.setter
    def color(self, color):
        """
        Set the tab header background color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            New tab header background color. A hex string ("#FF0000"), RGB(A)
            in [0, 1], or RGB(A) in [0, 255].
        """
        self.panel.color = color

    @property
    def title(self):
        """
        Get the tab title.

        Returns
        -------
        str
            Text displayed in the tab header.
        """
        return self.text_block.message

    @title.setter
    def title(self, text):
        """
        Set the tab title.

        Parameters
        ----------
        text : str
            New text displayed in the tab header.
        """
        self.text_block.message = text

    @property
    def title_bold(self):
        """
        Get whether the tab title is bold.

        Returns
        -------
        bool
            True when the tab title is rendered in bold.
        """
        return self.text_block.bold

    @title_bold.setter
    def title_bold(self, bold):
        """
        Set whether the tab title is bold.

        Parameters
        ----------
        bold : bool
            If True, render the tab title in bold.
        """
        self.text_block.bold = bold
        self.text_block.message = self.text_block.message

    @property
    def title_color(self):
        """
        Get the tab title color.

        Returns
        -------
        (float, float, float)
            RGB color of the tab title.
        """
        return self.text_block.color

    @title_color.setter
    def title_color(self, color):
        """
        Set the tab title color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            New tab title color. A hex string ("#FF0000"), RGB(A) in [0, 1],
            or RGB(A) in [0, 255].
        """
        self.text_block.color = color

    @property
    def title_font_size(self):
        """
        Get the tab title font size.

        Returns
        -------
        int
            Font size of the tab title.
        """
        return self.text_block.font_size

    @title_font_size.setter
    def title_font_size(self, font_size):
        """
        Set the tab title font size.

        Parameters
        ----------
        font_size : int
            New font size of the tab title.
        """
        self.text_block.font_size = font_size

    @property
    def title_italic(self):
        """
        Get whether the tab title is italic.

        Returns
        -------
        bool
            True when the tab title is rendered in italic.
        """
        return self.text_block.italic

    @title_italic.setter
    def title_italic(self, italic):
        """
        Set whether the tab title is italic.

        Parameters
        ----------
        italic : bool
            If True, render the tab title in italic.
        """
        self.text_block.italic = italic
        self.text_block.message = self.text_block.message

    def add_element(self, element, coords, *, anchor="position"):
        """
        Add an element to this tab's content panel.

        Parameters
        ----------
        element : UI
            UI component to add to the tab content area.
        coords : (float, float) or (int, int)
            Coordinates relative to the content panel. If floats are supplied,
            normalized coordinates in [0, 1] are assumed. If integers are
            supplied, pixel coordinates are assumed.
        anchor : str, optional
            Anchor used to position ``element``. Supported values are the same
            as :meth:`Panel2D.add_element`.
        """
        element.set_visibility(False)
        self.content_panel.add_element(element, coords, anchor=anchor)

    def remove_element(self, element):
        """
        Remove an element from this tab's content panel.

        Parameters
        ----------
        element : UI
            UI component to remove from the tab content area.
        """
        self.content_panel.remove_element(element)

    def update_element(self, element, coords, *, anchor="position"):
        """
        Update an element in this tab's content panel.

        Parameters
        ----------
        element : UI
            UI component already present in the tab content area.
        coords : (float, float) or (int, int)
            New coordinates relative to the content panel.
        anchor : str, optional
            Anchor used to position ``element``. Supported values are the same
            as :meth:`Panel2D.add_element`.
        """
        self.content_panel.update_element(element, coords, anchor=anchor)


class TabUI(UI):
    """
    A 2D container that switches between multiple content panels.

    ``TabUI`` creates tab headers and one content panel per tab. The tab bar can
    be placed horizontally at the top or bottom, or vertically on the left or
    right. It can also use an accordion layout where each tab title spans the
    width of the widget and the selected tab expands below its title. A left
    click on a tab header selects that tab and hides the other tab content
    panels. A second left click on the active tab hides its content while
    keeping the tab selected. A right click collapses the tab UI and clears the
    active tab.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates `(x, y)` of the upper-left corner of the tab UI.
    size : (int, int), optional
        Width and height in pixels of the full tab UI.
    tab_titles : list of str, optional
        Titles used to create tabs during initialization. If None, one tab with
        the default title is created.
    active_color : str, tuple, list or ndarray, optional
        Color of the active tab header. A hex string ("#FF0000"), RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    inactive_color : str, tuple, list or ndarray, optional
        Color of inactive tab headers, same formats as ``active_color``.
    font_size : int, optional
        Font size used by tab titles.
    draggable : bool, optional
        If True, the tab UI can be dragged from tab headers or content panel
        backgrounds.
    startup_tab_id : int, optional
        Index of the tab to show initially. If None, all content panels are
        hidden on startup.
    tab_bar_pos : {'top', 'bottom', 'left', 'right', 'accordion'}, optional
        Position of the tab bar relative to the content panel. ``"accordion"``
        creates an accordion-style layout.

    Attributes
    ----------
    tabs : list of :class:`TabPanel2D`
        Tab panels managed by this widget.
    parent_panel : :class:`Panel2D`
        Transparent panel that anchors tab headers and content panels.
    active_tab_idx : int or None
        Index of the currently selected tab. None when the tab UI is collapsed.
    collapsed : bool
        True when no tab content panel is visible due to right-click collapse.
    on_change : callable
        Callback invoked after a tab is selected. The callback receives this
        ``TabUI`` instance.
    on_collapse : callable
        Callback invoked after the tab UI is collapsed. The callback receives
        this ``TabUI`` instance.

    Raises
    ------
    ValueError
        If ``tab_titles`` is empty or is not a list of strings.
    """

    _VALID_TAB_BAR_POSITIONS = ["top", "bottom", "left", "right", "accordion"]

    def __init__(
        self,
        *,
        position=(0, 0),
        size=(100, 100),
        tab_titles=None,
        active_color=(1, 1, 1),
        inactive_color=(0.5, 0.5, 0.5),
        font_size=18,
        draggable=False,
        startup_tab_id=None,
        tab_bar_pos="top",
    ):
        """Initialize the tab UI."""
        if tab_titles is not None:
            if not isinstance(tab_titles, list) or not all(
                isinstance(title, str) for title in tab_titles
            ):
                raise ValueError("tab_titles must be a list of strings.")
            if len(tab_titles) < 1:
                raise ValueError("TabUI requires at least one tab title.")
        else:
            tab_titles = ["New Tab"]

        self.tabs = []
        self.tab_titles = tab_titles
        self.parent_size = size
        self.draggable = draggable
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.font_size = font_size
        self.active_tab_idx = startup_tab_id
        self.collapsed = startup_tab_id is None
        self.tab_bar_pos = tab_bar_pos.lower()
        self._validate_tab_bar_pos()
        self._drag_offset = None
        self._drag_start_position = None
        self._drag_moved = False
        self._drag_threshold = 3

        super(TabUI, self).__init__(position=position)

    def _setup(self):
        """Set up the parent panel and tab panels."""
        self.parent_panel = Panel2D(size=self.parent_size, opacity=0.0)
        self._children.append(self.parent_panel)
        self.on_change = lambda ui: None
        self.on_collapse = lambda ui: None

        self._update_sizes()
        for title in self.tab_titles:
            tab_panel = TabPanel2D(
                size=self.tab_panel_size,
                title=title,
                color=self.inactive_color,
                content_size=self.content_size,
                content_visible=False,
            )
            tab_panel.title_font_size = self.font_size
            self.tabs.append(tab_panel)
            self._children.append(tab_panel)

        self.update_tabs()
        if self.active_tab_idx is not None:
            self._validate_tab_idx(self.active_tab_idx)
            self._show_tab(self.active_tab_idx)

    def _get_actors(self):
        """
        Get actors composing this component.

        Returns
        -------
        list
            Empty list because child UI elements own the actors.
        """
        return []

    def _get_size(self):
        """
        Get the full tab UI size.

        Returns
        -------
        (int, int)
            Width and height of the tab UI in pixels.
        """
        return self.parent_panel.size

    def _update_actors_position(self):
        """Update internal component positions."""
        self.parent_panel.set_position(self.get_position())
        if hasattr(self, "tabs"):
            self.update_tabs()

    def _validate_tab_bar_pos(self):
        """Fallback to the default layout when tab bar position is invalid."""
        if self.tab_bar_pos in self._VALID_TAB_BAR_POSITIONS:
            return

        logging.warning(
            "tab_bar_pos can only have value top/bottom/left/right/accordion"
        )
        self.tab_bar_pos = "top"

    def _update_sizes(self):
        """Update cached tab header and content panel sizes."""
        if self.tab_bar_pos == "accordion":
            tab_height = int(0.1 * self.parent_size[1])
            content_height = self.parent_size[1] - tab_height * self.nb_tabs
            self.content_size = (self.parent_size[0], content_height)
            self.tab_panel_size = (self.parent_size[0], tab_height)
        elif self.tab_bar_pos in ["left", "right"]:
            tab_width = int(0.1 * self.parent_size[0])
            self.content_size = (
                self.parent_size[0] - tab_width,
                self.parent_size[1],
            )
            self.tab_panel_size = (tab_width, self.parent_size[1] // self.nb_tabs)
        else:
            tab_height = int(0.1 * self.parent_size[1])
            self.content_size = (self.parent_size[0], self.parent_size[1] - tab_height)
            self.tab_panel_size = (self.parent_size[0] // self.nb_tabs, tab_height)

    def resize(self, size):
        """
        Resize the full tab UI.

        Parameters
        ----------
        size : (int, int)
            New width and height in pixels of the full tab UI.
        """
        self.parent_size = size
        self.parent_panel.resize(size)
        self._update_sizes()
        self.update_tabs()

    def update_tabs(self):
        """
        Update tab layout and callbacks.

        This recomputes each tab header position, content panel position, tab
        size, and event callbacks. If ``tab_bar_pos`` is invalid, it falls back
        to ``"top"`` and emits a warning.
        """
        previous_tab_bar_pos = self.tab_bar_pos
        self._validate_tab_bar_pos()
        if self.tab_bar_pos != previous_tab_bar_pos:
            self._update_sizes()

        bar = self.tab_bar_pos
        tab_width, tab_height = self.tab_panel_size
        content_offset = {"top": (0, tab_height), "left": (tab_width, 0)}.get(
            bar, (0, 0)
        )
        panel = self.parent_panel
        panel.set_layout(
            direction=(
                Direction.COLUMN
                if bar in ("left", "right", "accordion")
                else Direction.ROW
            ),
            align_cross=Align.END if bar in ("bottom", "right") else Align.START,
        )

        for idx, tab_panel in enumerate(self.tabs):
            tab_panel.resize(self.tab_panel_size)
            tab_panel.content_panel.resize(self.content_size)
            tab_panel.set_layout(width=tab_width, height=tab_height)

            # In an accordion only the open tab pushes the next header down.
            content = tab_panel.content_panel
            in_flow = (
                bar == "accordion"
                and self.active_tab_idx == idx
                and self._is_content_visible(tab_panel)
            )
            if in_flow:
                content.set_layout(position=Position.FLOW)
            else:
                content.set_layout(position=Position.ABSOLUTE, offset=content_offset)

            if tab_panel not in panel._elements:
                panel.add(tab_panel)
                panel.add(content)

            self._setup_tab_callbacks(idx, tab_panel)

        panel.update_layout()

    def _setup_tab_callbacks(self, idx, tab_panel):
        """
        Attach event callbacks to a tab header and content panel.

        Parameters
        ----------
        idx : int
            Index of ``tab_panel`` in :attr:`tabs`.
        tab_panel : :class:`TabPanel2D`
            Tab panel receiving selection, collapse, and optional drag
            callbacks.
        """
        tab_panel.text_block.on_right_mouse_button_clicked = lambda event: (
            self.collapse_tab_ui()
        )
        tab_panel.panel.background.on_right_mouse_button_clicked = lambda event: (
            self.collapse_tab_ui()
        )

        if self.draggable:
            for element in [tab_panel.panel.background, tab_panel.text_block]:
                element.on_left_mouse_button_pressed = lambda event, tab_idx=idx: (
                    self.left_button_pressed(event, tab_idx)
                )
                element.on_left_mouse_button_dragged = self.left_button_dragged
                element.on_left_mouse_button_released = lambda event: None

            for element in [tab_panel.content_panel.background]:
                element.on_left_mouse_button_pressed = self.left_button_pressed
                element.on_left_mouse_button_dragged = self.left_button_dragged
                element.on_left_mouse_button_released = lambda event: None
        else:
            tab_panel.text_block.on_left_mouse_button_clicked = (
                lambda event, tab_idx=idx: self.select_tab(tab_idx)
            )
            tab_panel.panel.background.on_left_mouse_button_clicked = (
                lambda event, tab_idx=idx: self.select_tab(tab_idx)
            )

    def _validate_tab_idx(self, tab_idx):
        """
        Validate a tab index.

        Parameters
        ----------
        tab_idx : int
            Index of the tab to validate.

        Raises
        ------
        IndexError
            If ``tab_idx`` is outside the range of available tabs.
        """
        if tab_idx < 0 or tab_idx >= self.nb_tabs:
            raise IndexError(f"Tab with index {tab_idx} does not exist")

    def _is_content_visible(self, tab_panel):
        """
        Return whether a tab content panel is visible.

        Parameters
        ----------
        tab_panel : :class:`TabPanel2D`
            Tab panel whose content visibility is queried.

        Returns
        -------
        bool
            True when the tab content panel background actor is visible.
        """
        return bool(tab_panel.content_panel.actors[0].visible)

    def _show_tab(self, tab_idx):
        """
        Show one tab and hide all others.

        Parameters
        ----------
        tab_idx : int
            Index of the tab to activate.
        """
        for idx, tab_panel in enumerate(self.tabs):
            is_active = idx == tab_idx
            tab_panel.color = self.active_color if is_active else self.inactive_color
            tab_panel.content_panel.set_visibility(is_active)
        self.active_tab_idx = tab_idx
        self.collapsed = False
        self.update_tabs()

    def select_tab(self, tab_idx):
        """
        Select a tab.

        Selecting a tab hides all other content panels. If the selected tab is
        already visible, its content panel is hidden and the tab remains the
        active tab. The :attr:`on_change` callback is invoked after selection.

        Parameters
        ----------
        tab_idx : int
            Index of the tab to select.

        Raises
        ------
        IndexError
            If ``tab_idx`` is outside the range of available tabs.
        """
        self._validate_tab_idx(tab_idx)

        for idx, tab_panel in enumerate(self.tabs):
            if idx != tab_idx:
                tab_panel.color = self.inactive_color
                tab_panel.content_panel.set_visibility(False)
                continue

            visible = self._is_content_visible(tab_panel)
            tab_panel.color = self.inactive_color if visible else self.active_color
            tab_panel.content_panel.set_visibility(not visible)

        self.active_tab_idx = tab_idx
        self.collapsed = False
        self.update_tabs()
        self.on_change(self)

    def collapse_tab_ui(self):
        """
        Collapse the active tab content.

        Hides the active tab content panel, resets :attr:`active_tab_idx` to
        None, marks the tab UI as collapsed, and invokes :attr:`on_collapse`.
        """
        if self.active_tab_idx is not None:
            active_tab_panel = self.tabs[self.active_tab_idx]
            active_tab_panel.color = self.inactive_color
            active_tab_panel.content_panel.set_visibility(False)

        self.active_tab_idx = None
        self.collapsed = True
        self.update_tabs()
        self.on_collapse(self)

    def add_element(self, tab_idx, element, coords, *, anchor="position"):
        """
        Add an element to a tab content panel.

        Parameters
        ----------
        tab_idx : int
            Index of the tab receiving the element.
        element : UI
            UI component to add to the tab content panel.
        coords : (float, float) or (int, int)
            Coordinates relative to the tab content panel. If floats are
            supplied, normalized coordinates in [0, 1] are assumed. If integers
            are supplied, pixel coordinates are assumed.
        anchor : str, optional
            Anchor used to position ``element``. Supported values are the same
            as :meth:`Panel2D.add_element`.

        Raises
        ------
        IndexError
            If ``tab_idx`` is outside the range of available tabs.
        """
        self._validate_tab_idx(tab_idx)
        self.tabs[tab_idx].add_element(element, coords, anchor=anchor)
        if tab_idx == self.active_tab_idx and not self.collapsed:
            element.set_visibility(True)

    def remove_element(self, tab_idx, element):
        """
        Remove an element from a tab content panel.

        Parameters
        ----------
        tab_idx : int
            Index of the tab containing the element.
        element : UI
            UI component to remove from the tab content panel.

        Raises
        ------
        IndexError
            If ``tab_idx`` is outside the range of available tabs.
        """
        self._validate_tab_idx(tab_idx)
        self.tabs[tab_idx].remove_element(element)

    def update_element(self, tab_idx, element, coords, *, anchor="position"):
        """
        Update an element in a tab content panel.

        Parameters
        ----------
        tab_idx : int
            Index of the tab containing the element.
        element : UI
            UI component already present in the tab content panel.
        coords : (float, float) or (int, int)
            New coordinates relative to the tab content panel.
        anchor : str, optional
            Anchor used to position ``element``. Supported values are the same
            as :meth:`Panel2D.add_element`.

        Raises
        ------
        IndexError
            If ``tab_idx`` is outside the range of available tabs.
        """
        self._validate_tab_idx(tab_idx)
        self.tabs[tab_idx].update_element(element, coords, anchor=anchor)

    def left_button_pressed(self, event, tab_idx=None):
        """
        Start dragging the tab UI.

        Parameters
        ----------
        event : PointerEvent
            PyGfx pointer event containing the current pointer coordinates.
        tab_idx : int, optional
            Index of the tab to select before starting the drag operation. If
            None, only dragging is initialized.
        """
        if tab_idx is not None:
            self.select_tab(tab_idx)

        click_pos = np.array([event.x, event.y])
        self._drag_start_position = click_pos
        self._drag_offset = click_pos - self.get_position()
        self._drag_moved = False

    def left_button_dragged(self, event):
        """
        Drag the tab UI.

        Parameters
        ----------
        event : PointerEvent
            PyGfx pointer event containing the current pointer coordinates.
        """
        if self._drag_offset is None:
            return
        click_pos = np.array([event.x, event.y])
        if self._drag_start_position is not None:
            self._drag_moved = (
                np.linalg.norm(click_pos - self._drag_start_position)
                > self._drag_threshold
            )
        self.set_position(click_pos - self._drag_offset)

    @property
    def nb_tabs(self):
        """
        Get the number of tabs in this tab UI.

        Returns
        -------
        int
            Number of tabs in this tab UI.
        """
        return len(self.tab_titles)


# class TabPanel2D(UI):
#     """Render content within a Tab.

#     Attributes
#     ----------
#     content_panel: :class: 'Panel2D'
#         Hold all the content UI components.
#     text_block: :class: 'TextBlock2D'
#         Renders the title of the tab.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         *,
#         position=(0, 0),
#         size=(100, 100),
#         title="New Tab",
#         color=(0.5, 0.5, 0.5),
#         content_panel=None,
#     ):
#         """Init class instance.

#         Parameters
#         ----------
#         position : (float, float)
#             Absolute coordinates (x, y) of the lower-left corner of the
#             UI component
#         size : (int, int)
#             Width and height of the pixels of this UI component.
#         title : str
#             Renders the title for Tab panel.
#         color : list of 3 floats
#             Background color of tab panel.
#         content_panel : Panel2D
#             Panel consisting of the content UI elements.

#         """
#         self.content_panel = content_panel
#         self.panel_size = size
#         self._text_size = (int(1.0 * size[0]), size[1])

#         super(TabPanel2D, self).__init__()
#         self.title = title
#         self.panel.position = position
#         self.color = color

#     def _setup(self):
#         """Setup this UI component.

#         Create parent panel.
#         Create Text to hold tab information.
#         Create Button to close tab.

#         """
#         self.panel = Panel2D(size=self.panel_size)
#         self.text_block = TextBlock2D(size=self._text_size, color=(0, 0, 0))
#         self.panel.add_element(self.text_block, (0, 0))

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         return self.panel.actors + self.content_panel.actors

#     def _add_to_scene(self, _scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self.panel.add_to_scene(_scene)
#         self.content_panel.add_to_scene(_scene)

#     def _set_position(self, _coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.panel.position = _coords

#     def _get_size(self):
#         return self.panel.size

#     def resize(self, size):
#         """Resize Tab panel.

#         Parameters
#         ----------
#         size : (int, int)
#             New width and height in pixels.

#         """
#         self._text_size = (int(0.7 * size[0]), size[1])
#         self._button_size = (int(0.3 * size[0]), size[1])
#         self.panel.resize(size)
#         self.text_block.resize(self._text_size)

#     @property
#     def color(self):
#         """Return the background color of tab panel."""
#         return self.panel.color

#     @color.setter
#     def color(self, color):
#         """Set background color of tab panel.

#         Parameters
#         ----------
#         color : list of 3 floats.

#         """
#         self.panel.color = color

#     @property
#     def title(self):
#         """Return the title of tab panel."""
#         return self.text_block.message

#     @title.setter
#     def title(self, text):
#         """Set the title of tab panel.

#         Parameters
#         ----------
#         text : str
#             New title for tab panel.

#         """
#         self.text_block.message = text

#     @property
#     def title_bold(self):
#         """Is the title of a tab panel bold."""
#         return self.text_block.bold

#     @title_bold.setter
#     def title_bold(self, bold):
#         """Determine if the text title of a tab panel must be bold.

#         Parameters
#         ----------
#         bold : bool
#             Bold property for a text title in a tab panel.

#         """
#         self.text_block.bold = bold

#     @property
#     def title_color(self):
#         """Return the title color of tab panel."""
#         return self.text_block.color

#     @title_color.setter
#     def title_color(self, color):
#         """Set the title color of tab panel.

#         Parameters
#         ----------
#         color : tuple
#             New title color for tab panel.

#         """
#         self.text_block.color = color

#     @property
#     def title_font_size(self):
#         """Return the title font size of tab panel."""
#         return self.text_block.font_size

#     @title_font_size.setter
#     def title_font_size(self, font_size):
#         """Set the title font size of tab panel.

#         Parameters
#         ----------
#         font_size : int
#             New title font size for tab panel.

#         """
#         self.text_block.font_size = font_size

#     @property
#     def title_italic(self):
#         """Is the title of a tab panel italic."""
#         return self.text_block.italic

#     @title_italic.setter
#     def title_italic(self, italic):
#         """Determine if the text title of a tab panel must be italic.

#         Parameters
#         ----------
#         italic : bool
#             Italic property for a text title in a tab panel.

#         """
#         self.text_block.italic = italic

#     @warn_on_args_to_kwargs()
#     def add_element(self, element, coords, *, anchor="position"):
#         """Add a UI component to the content panel.

#         The coordinates represent an offset from the lower left corner of the
#         panel.

#         Parameters
#         ----------
#         element : UI
#             The UI item to be added.
#         coords : (float, float) or (int, int)
#             If float, normalized coordinates are assumed and they must be
#             between [0,1].
#             If int, pixels coordinates are assumed and it must fit within the
#             panel's size.

#         """
#         element.set_visibility(False)
#         self.content_panel.add_element(element, coords, anchor=anchor)

#     def remove_element(self, element):
#         """Remove a UI component from the content panel.

#         Parameters
#         ----------
#         element : UI
#             The UI item to be removed.

#         """
#         self.content_panel.remove_element(element)

#     @warn_on_args_to_kwargs()
#     def update_element(self, element, coords, *, anchor="position"):
#         """Update the position of a UI component in the content panel.

#         Parameters
#         ----------
#         element : UI
#             The UI item to be updated.
#         coords : (float, float) or (int, int)
#             New coordinates.
#             If float, normalized coordinates are assumed and they must be
#             between [0,1].
#             If int, pixels coordinates are assumed and it must fit within the
#             panel's size.

#         """
#         self.content_panel.update_element(element, coords, anchor="position")


# class TabUI(UI):
#     """UI element to add multiple panels within a single window.

#     Attributes
#     ----------
#     tabs: :class: List of 'TabPanel2D'
#         Stores all the instances of 'TabPanel2D' that renders the contents.

#     """

#     @warn_on_args_to_kwargs()
#     def __init__(
#         self,
#         *,
#         position=(0, 0),
#         size=(100, 100),
#         nb_tabs=1,
#         active_color=(1, 1, 1),
#         inactive_color=(0.5, 0.5, 0.5),
#         draggable=False,
#         startup_tab_id=None,
#         tab_bar_pos="top",
#     ):
#         """Init class instance.

#         Parameters
#         ----------
#         position : (float, float)
#             Absolute coordinates (x, y) of the lower-left corner of this
#             UI component.
#         size : (int, int)
#             Width and height in pixels of this UI component.
#         nb_tabs : int
#             Number of tabs to be renders.
#         active_color : tuple of 3 floats.
#             Background color of active tab panel.
#         inactive_color : tuple of 3 floats.
#             Background color of inactive tab panels.
#         draggable : bool
#             Whether the UI element is draggable or not.
#         startup_tab_id : int, optional
#             Tab to be activated and uncollapsed on startup.
#             by default None is activated/ all collapsed.
#         tab_bar_pos : str, optional
#             Position of the Tab Bar in the panel
#         """
#         self.tabs = []
#         self.nb_tabs = nb_tabs
#         self.parent_size = size
#         self.content_size = (size[0], int(0.9 * size[1]))
#         self.draggable = draggable
#         self.active_color = active_color
#         self.inactive_color = inactive_color
#         self.active_tab_idx = startup_tab_id
#         self.collapsed = True
#         self.tab_bar_pos = tab_bar_pos

#         super(TabUI, self).__init__()
#         self.position = position

#     def _setup(self):
#         """Setup this UI component.

#         Create parent panel.
#         Create tab panels.
#         """
#         self.parent_panel = Panel2D(self.parent_size, opacity=0.0)

#         # Offer some standard hooks to the user.
#         self.on_change = lambda ui: None
#         self.on_collapse = lambda ui: None

#         for _ in range(self.nb_tabs):
#             content_panel = Panel2D(size=self.content_size)
#             content_panel.set_visibility(False)
#             tab_panel = TabPanel2D(content_panel=content_panel)
#             self.tabs.append(tab_panel)
#         self.update_tabs()

#         if self.active_tab_idx is not None:
#             self.tabs[self.active_tab_idx].color = self.active_color
#             self.tabs[self.active_tab_idx].content_panel.set_visibility(True)

#     def _get_actors(self):
#         """Get the actors composing this UI component."""
#         actors = []
#         actors += self.parent_panel.actors
#         for tab_panel in self.tabs:
#             actors += tab_panel.actors

#         return actors

#     def _add_to_scene(self, _scene):
#         """Add all subcomponents or VTK props that compose this UI component.

#         Parameters
#         ----------
#         scene : scene

#         """
#         self.parent_panel.add_to_scene(_scene)
#         for tab_panel in self.tabs:
#             tab_panel.add_to_scene(_scene)

#     def _set_position(self, _coords):
#         """Set the lower-left corner position of this UI component.

#         Parameters
#         ----------
#         coords: (float, float)
#             Absolute pixel coordinates (x, y).

#         """
#         self.parent_panel.position = _coords

#     def _get_size(self):
#         return self.parent_panel.size

#     def update_tabs(self):
#         """Update position, size and callbacks for tab panels."""
#         self.tab_panel_size = (self.size[0] // self.nb_tabs, int(0.1 * self.size[1]))
#         if self.tab_bar_pos.lower() not in ["top", "bottom"]:
#             warn("tab_bar_pos can only have value top/bottom", stacklevel=2)
#             self.tab_bar_pos = "top"

#         if self.tab_bar_pos.lower() == "top":
#             tab_panel_pos = [0.0, 0.9]
#         elif self.tab_bar_pos.lower() == "bottom":
#             tab_panel_pos = [0.0, 0.0]

#         for tab_panel in self.tabs:
#             tab_panel.resize(self.tab_panel_size)
#             tab_panel.content_panel.position = self.position

#             content_panel = tab_panel.content_panel
#             if self.draggable:
#                 tab_panel.panel.background.on_left_mouse_button_pressed = (
#                     self.left_button_pressed
#                 )
#                 content_panel.background.on_left_mouse_button_pressed = (
#                     self.left_button_pressed
#                 )
#                 tab_panel.text_block.on_left_mouse_button_pressed = (
#                     self.left_button_pressed
#                 )

#                 tab_panel.panel.background.on_left_mouse_button_dragged = (
#                     self.left_button_dragged
#                 )
#                 content_panel.background.on_left_mouse_button_dragged = (
#                     self.left_button_dragged
#                 )
#                 tab_panel.text_block.on_left_mouse_button_dragged = (
#                     self.left_button_dragged
#                 )
#             else:
#                 tab_panel.panel.background.on_left_mouse_button_dragged = (
#                     lambda i_ren, _obj, _comp: i_ren.force_render
#                 )
#                 content_panel.background.on_left_mouse_button_dragged = (
#                     lambda i_ren, _obj, _comp: i_ren.force_render
#                 )

#             tab_panel.text_block.on_left_mouse_button_clicked =
# self.select_tab_callback
#             tab_panel.panel.background.on_left_mouse_button_clicked = (
#                 self.select_tab_callback
#             )

#             tab_panel.text_block.on_right_mouse_button_clicked = self.collapse_tab_ui
#             tab_panel.panel.background.on_right_mouse_button_clicked = (
#                 self.collapse_tab_ui
#             )

#             tab_panel.content_panel.resize(self.content_size)
#             self.parent_panel.add_element(tab_panel, tab_panel_pos)
#             if self.tab_bar_pos.lower() == "top":
#                 self.parent_panel.add_element(tab_panel.content_panel, (0.0, 0.0))
#             elif self.tab_bar_pos.lower() == "bottom":
#                 self.parent_panel.add_element(tab_panel.content_panel, (0.0, 0.1))
#             tab_panel_pos[0] += 1 / self.nb_tabs

#     def select_tab_callback(self, iren, _obj, _tab_comp):
#         """Handle events when a tab is selected."""
#         for idx, tab_panel in enumerate(self.tabs):
#             if (
#                 tab_panel.text_block is not _tab_comp
#                 and tab_panel.panel.background is not _tab_comp
#             ):
#                 tab_panel.color = self.inactive_color
#                 tab_panel.content_panel.set_visibility(False)
#             else:
#                 current_visibility = tab_panel.content_panel.actors[0].GetVisibility()
#                 if not current_visibility:
#                     tab_panel.color = self.active_color
#                 else:
#                     tab_panel.color = self.inactive_color
#                 tab_panel.content_panel.set_visibility(not current_visibility)
#                 self.active_tab_idx = idx

#         self.collapsed = False
#         self.on_change(self)
#         iren.force_render()
#         iren.event.abort()

#     def collapse_tab_ui(self, iren, _obj, _tab_comp):
#         """Handle events when Tab UI is collapsed."""
#         if self.active_tab_idx is not None:
#             active_tab_panel = self.tabs[self.active_tab_idx]
#             active_tab_panel.color = self.inactive_color
#             active_tab_panel.content_panel.set_visibility(False)
#         self.active_tab_idx = None
#         self.collapsed = True
#         self.on_collapse(self)
#         iren.force_render()
#         iren.event.abort()

#     @warn_on_args_to_kwargs()
#     def add_element(self, tab_idx, element, coords, *, anchor="position"):
#         """Add element to content panel after checking its existence."""
#         if tab_idx < self.nb_tabs and tab_idx >= 0:
#             self.tabs[tab_idx].add_element(element, coords, anchor=anchor)
#             if tab_idx == self.active_tab_idx:
#                 element.set_visibility(True)
#         else:
#             raise IndexError("Tab with index " "{} does not exist".format(tab_idx))

#     def remove_element(self, tab_idx, element):
#         """Remove element from content panel after checking its existence."""
#         if tab_idx < self.nb_tabs and tab_idx >= 0:
#             self.tabs[tab_idx].remove_element(element)
#         else:
#             raise IndexError("Tab with index " "{} does not exist".format(tab_idx))

#     @warn_on_args_to_kwargs()
#     def update_element(self, tab_idx, element, coords, *, anchor="position"):
#         """Update element on content panel after checking its existence."""
#         if tab_idx < self.nb_tabs and tab_idx >= 0:
#             self.tabs[tab_idx].update_element(element, coords, anchor=anchor)
#         else:
#             raise IndexError("Tab with index " "{} does not exist".format(tab_idx))

#     def left_button_pressed(self, i_ren, _obj, _sub_component):
#         click_pos = np.array(i_ren.event.position)
#         self._click_position = click_pos
#         i_ren.event.abort()  # Stop propagating the event.

#     def left_button_dragged(self, i_ren, _obj, _sub_component):
#         click_position = np.array(i_ren.event.position)
#         change = click_position - self._click_position
#         self.parent_panel.position += change
#         self._click_position = click_position
#         i_ren.force_render()


class ImageContainer2D(Rectangle2D):
    """
    A 2D container to hold an image.

    Currently Supports:
    - png and jpg/jpeg images

    Parameters
    ----------
    img_path : str
        URL or local path of the image.
    position : (float, float), optional
        Absolute coordinates (x, y) of the top-left corner of the image.
    size : (int, int), optional
        Width and height in pixels of the image.

    Attributes
    ----------
    size : (float, float)
        Image size (width, height) in pixels.
    img : ndarray
        The image loaded from the specified path as a NumPy array.
    """

    def __init__(self, img_path, *, position=(0, 0), size=(100, 100)):
        """
        Init class instance.

        Parameters
        ----------
        img_path : str or ndarray
            URL, local path of the image, or a NumPy array containing image data.
        position : (float, float), optional
            Absolute coordinates (x, y) of the top-left corner of the image.
        size : (int, int), optional
            Width and height in pixels of the image.

        """
        self._img_path = img_path

        super(ImageContainer2D, self).__init__(size=size, position=position)

        if isinstance(img_path, np.ndarray):
            self.img = img_path
        else:
            self.img = load_image(img_path)

        self.set_img(self.img)

    def scale(self, factor):
        """
        Scale the image.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.
        """
        self.resize(self.size * factor)

    def set_img(self, img):
        """
        Modify the image displayed by this container.

        Parameters
        ----------
        img : ndarray
            Image data as a NumPy array. Supports grayscale (H, W),
            RGB (H, W, 3), and RGBA (H, W, 4) formats.
        """
        self.img = img

        img_float = img.astype(np.float32)

        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        if img_float.ndim == 3 and img_float.shape[2] == 1:
            img_float = img_float[:, :, 0]
        elif img_float.ndim == 3 and img_float.shape[2] == 3:
            alpha = np.ones((*img_float.shape[:2], 1), dtype=np.float32)
            img_float = np.concatenate([img_float, alpha], axis=-1)

        self.actor.material.map = Texture(img_float, dim=2)
        self.actor.material.needs_update = True

        self.resize(self.size)


# # class GridUI(UI):
# #     """Add actors in a grid and interact with them individually."""

# #     @warn_on_args_to_kwargs()
# #     def __init__(
# #         self,
# #         actors,
# #         *,
# #         captions=None,
# #         caption_offset=(0, -100, 0),
# #         cell_padding=0,
# #         cell_shape="rect",
# #         aspect_ratio=16 / 9.0,
# #         dim=None,
# #         rotation_speed=1,
# #         rotation_axis=(0, 1, 0),
# #     ):
# #         # TODO: add rotation axis None by default

# #         self.container = grid(
# #             actors,
# #             captions=captions,
# #             caption_offset=caption_offset,
# #             cell_padding=cell_padding,
# #             cell_shape=cell_shape,
# #             aspect_ratio=aspect_ratio,
# #             dim=dim,
# #         )
# #         self._actors = []
# #         self._actors_dict = {}
# #         self.rotation_speed = rotation_speed
# #         self.rotation_axis = rotation_axis

# #         for item in self.container._items:
# #             actor = item if captions is None else item._items[0]
# #             self._actors.append(actor)
# #             self._actors_dict[actor] = {"x": -np.inf, "y": -np.inf}

# #         super(GridUI, self).__init__(position=(0, 0, 0))

# #     def _get_size(self):
# #         return

# #     @staticmethod
# #     def left_click_callback(istyle, _obj, _what):
# #         istyle.trackball_actor.OnLeftButtonDown()
# #         istyle.force_render()
# #         istyle.event.abort()

# #     @staticmethod
# #     def left_release_callback(istyle, _obj, _what):
# #         istyle.trackball_actor.OnLeftButtonUp()
# #         istyle.force_render()
# #         istyle.event.abort()

# #     @staticmethod
# #     def mouse_move_callback(istyle, _obj, _what):
# #         istyle.trackball_actor.OnMouseMove()
# #         istyle.force_render()
# #         istyle.event.abort()

# #     @staticmethod
# #     def left_click_callback2(istyle, obj, self):
# #         rx, ry, rz = self.rotation_axis
# #         clockwise_rotation = np.array([self.rotation_speed, rx, ry, rz])
# #         rotate(obj, clockwise_rotation)

# #         istyle.force_render()
# #         istyle.event.abort()

# #     @staticmethod
# #     def left_release_callback2(istyle, _obj, _what):
# #         istyle.force_render()
# #         istyle.event.abort()

# #     @staticmethod
# #     def mouse_move_callback2(istyle, obj, self):
# #         if self._actors_dict[obj]["y"] == -np.inf:
# #             iren = istyle.GetInteractor()
# #             event_pos = iren.GetEventPosition()
# #             self._actors_dict[obj]["y"] = event_pos[1]

# #         else:
# #             iren = istyle.GetInteractor()
# #             event_pos = iren.GetEventPosition()
# #             rx, ry, rz = self.rotation_axis

# #             if event_pos[1] >= self._actors_dict[obj]["y"]:
# #                 clockwise_rotation = np.array([-self.rotation_speed, rx, ry, rz])
# #                 rotate(obj, clockwise_rotation)
# #             else:
# #                 anti_clockwise_rotation = np.array(
# [self.rotation_speed, rx, ry, rz])
# #                 rotate(obj, anti_clockwise_rotation)

# #             self._actors_dict[obj]["y"] = event_pos[1]

# #             istyle.force_render()
# #             istyle.event.abort()

# #     ANTICLOCKWISE_ROTATION_Y = np.array([-10, 0, 1, 0])
# #     CLOCKWISE_ROTATION_Y = np.array([10, 0, 1, 0])
# #     ANTICLOCKWISE_ROTATION_X = np.array([-10, 1, 0, 0])
# #     CLOCKWISE_ROTATION_X = np.array([10, 1, 0, 0])

# #     def key_press_callback(self, istyle, obj, _what):
# #         has_changed = False
# #         if istyle.event.key == "Left":
# #             has_changed = True
# #             for a in self._actors:
# #                 rotate(a, self.ANTICLOCKWISE_ROTATION_Y)
# #         elif istyle.event.key == "Right":
# #             has_changed = True
# #             for a in self._actors:
# #                 rotate(a, self.CLOCKWISE_ROTATION_Y)
# #         elif istyle.event.key == "Up":
# #             has_changed = True
# #             for a in self._actors:
# #                 rotate(a, self.ANTICLOCKWISE_ROTATION_X)
# #         elif istyle.event.key == "Down":
# #             has_changed = True
# #             for a in self._actors:
# #                 rotate(a, self.CLOCKWISE_ROTATION_X)

# #         if has_changed:
# #             istyle.force_render()

# #     def _setup(self):
# #         """Set up this UI component and the events of its actor."""
# #         # Add default events listener to the VTK actor.
# #         for actor in self._actors:
# #             # self.handle_events(actor)

# #             if self.rotation_axis is None:
# #                 self.add_callback(
# #                     actor, "LeftButtonPressEvent", self.left_click_callback
# #                 )
# #                 self.add_callback(
# #                     actor, "LeftButtonReleaseEvent", self.left_release_callback
# #                 )
# #                 self.add_callback(actor, "MouseMoveEvent", self.mouse_move_callback)
# #             else:
# #                 self.add_callback(
# #                     actor, "LeftButtonPressEvent", self.left_click_callback2
# #                 )
# #                 # TODO: possibly add this too
# #                 self.add_callback(
# #                     actor, "LeftButtonReleaseEvent", self.left_release_callback2
# #                 )
# #                 self.add_callback(
# actor, "MouseMoveEvent", self.mouse_move_callback2)

# #             # TODO: this is currently not running
# #             self.add_callback(actor, "KeyPressEvent", self.key_press_callback)
# #         # self.on_key_press = self.key_press_callback2

# #     def _get_actors(self):
# #         """Get the actors composing this UI component."""
# #         return self._actors

# #     def _add_to_scene(self, scene):
# #         """Add all subcomponents or VTK props that compose this UI component.

# #         Parameters
# #         ----------
# #         scene : scene

# #         """
# #         self.container.add_to_scene(scene)

# #     def resize(self, size):
# #         """Resize the button.

# #         Parameters
# #         ----------
# #         size : (float, float)
# #             Button size (width, height) in pixels.

# #         """
# #         # Update actor.
# #         pass

# #     def _set_position(self, coords):
# #         """Set the lower-left corner position of this UI component.

# #         Parameters
# #         ----------
# #         coords: (float, float)
# #             Absolute pixel coordinates (x, y).

# #         """
# #         # coords = (0, 0, 0)
# #         pass
# #         # self.actor.SetPosition(*coords)
# #         # self.container.SetPosition(*coords)
