"""UI core module that describe UI abstract class."""

import abc
import math

import numpy as np

from fury.actor import Mesh, Text, create_mesh
from fury.colormap import normalize_colors
from fury.geometry import buffer_to_geometry
from fury.lib import (
    EventType,
    plane_geometry,
    register_wgpu_render_function,
)
from fury.material import RoundedRectangleMaterial, _create_mesh_material
from fury.primitive import prim_ring
from fury.shader import RoundedRectangleShader
from fury.ui import UIContext
from fury.ui.helpers import (
    UI_SUBLAYER_SCALE,
    UI_Z_RANGE,
    Anchor,
    get_anchor_to_multiplier,
)
from fury.ui.layout import LayoutNode, compute_sizes, place
from fury.ui.theme import _as_theme, _inherited_theme_scope, get_theme

_TEXT_FIT_TOLERANCE = 1

BOX_STYLE_ITEMS = (
    "corner_radius",
    "border_color",
    "border_width",
    "shadow_color",
    "shadow_offset",
    "shadow_blur",
    "shadow_spread",
)


def _to_rgba(color):
    """
    Convert a color in any supported format to RGBA.

    Parameters
    ----------
    color : str, tuple, list or ndarray
        A hex string, RGB(A) in [0, 1], or RGB(A) in [0, 255].

    Returns
    -------
    ndarray
        RGBA color in [0, 1]. Opaque when no alpha was given.
    """
    color = normalize_colors(color)[0]
    if color.shape[0] == 3:
        color = np.append(color, 1.0)
    return color.astype(float)


class themed:
    """
    Descriptor for a themed value of a UI component.

    The value is stored under its theme item name. A value assigned through
    the attribute is the user's choice and is kept when the theme changes; a
    parent widget writes with ``_push_style``, and the theme with
    ``_set_style``. Like a property, it takes an optional getter, and a
    function that displays a new value.

    Parameters
    ----------
    fget : callable, optional
        Function returning the current value. Without it, the last value set
        is returned, as an RGB tuple of floats for color items.
    fshow : callable, optional
        Function displaying a new value. It may raise to reject the value,
        which is then not stored.
    item : str, optional
        Theme item name. Defaults to the attribute name.
    doc : str, optional
        Docstring, taken from ``fget`` when not given.
    reset_on_none : bool, optional
        If True, assigning None makes the value follow the theme again.
    """

    def __init__(
        self, fget=None, fshow=None, *, item=None, doc=None, reset_on_none=False
    ):
        """
        Initialize the descriptor.

        Parameters
        ----------
        fget : callable, optional
            Function returning the current value.
        fshow : callable, optional
            Function displaying a new value.
        item : str, optional
            Theme item name.
        doc : str, optional
            Docstring.
        reset_on_none : bool, optional
            If True, assigning None makes the value follow the theme again.
        """
        self.fget = fget
        self.fshow = fshow
        self.item = item
        self.reset_on_none = reset_on_none
        self.__doc__ = doc if doc is not None else getattr(fget, "__doc__", None)

    def __set_name__(self, owner, name):
        """
        Use the attribute name as the item name unless one was given.

        Parameters
        ----------
        owner : type
            Class defining the attribute.
        name : str
            Attribute name.
        """
        self.item = self.item or name

    def __get__(self, element, owner=None):
        """
        Get the current value.

        Parameters
        ----------
        element : UI or None
            The component, or None for access through the class.
        owner : type, optional
            The class.

        Returns
        -------
        object
            The value, or the descriptor itself for class access.
        """
        if element is None:
            return self
        if self.fget is not None:
            return self.fget(element)
        value = element._style_state()[2].get(self.item)
        if value is not None and self.item.endswith("color"):
            return tuple(float(c) for c in normalize_colors(value)[0][:3])
        return value

    def __set__(self, element, value):
        """
        Set the value as the user's choice.

        Parameters
        ----------
        element : UI
            The component.
        value : object
            The new value, or None to follow the theme again when the
            descriptor allows it.
        """
        if value is None and self.reset_on_none:
            element._style_state()[0].pop(self.item, None)
            element._set_style(self.item, element._theme_value(self.item))
            element._request_refresh()
            return
        element._set_style(self.item, value, source="user")


def _same_value(first, second):
    """
    Check whether two themed values are equal.

    Parameters
    ----------
    first, second : object
        Values such as colors, sizes or names.

    Returns
    -------
    bool
        True if the values are equal.
    """
    try:
        return bool(np.array_equal(np.asarray(first), np.asarray(second)))
    except (TypeError, ValueError):
        return first == second


def _style_descriptors(cls):
    """
    Get the themed values of a UI class, keyed by theme item name.

    Parameters
    ----------
    cls : type
        UI class.

    Returns
    -------
    dict
        Mapping of item name to :class:`themed` descriptor.
    """
    cached = cls.__dict__.get("_themed_descriptors")
    if cached is None:
        cached = {}
        for klass in reversed(cls.__mro__):
            for value in vars(klass).values():
                if isinstance(value, themed):
                    cached[value.item] = value
        cls._themed_descriptors = cached
    return cached


class _UIMeta(abc.ABCMeta):
    """Metaclass that tells a UI element when its construction finished."""

    def __call__(cls, *args, **kwargs):
        """
        Create a UI element, then let it follow the global theme.

        Parameters
        ----------
        *args : tuple
            Positional arguments of the constructor.
        **kwargs : dict
            Keyword arguments of the constructor.

        Returns
        -------
        UI
            The fully constructed element.
        """
        element = super().__call__(*args, **kwargs)
        if getattr(element, "_theme_source", None) == "global":
            UIContext.follow_global_theme(element)
        return element


class UI(object, metaclass=_UIMeta):
    """
    An umbrella class for all UI elements.

    While adding UI elements to the scene, we go over all the sub-elements
    that come with it and add those to the scene automatically.

    Parameters
    ----------
    position : (float, float)
        Absolute pixel coordinates `(x, y)` which, in combination with
        `x_anchor` and `y_anchor`, define the initial placement of this
        UI component.
    x_anchor : str, optional
        Define the horizontal anchor point for `position`. Can be "LEFT",
        "CENTER", or "RIGHT".
    y_anchor : str, optional
        Define the vertical anchor point for `position`. Can be "BOTTOM",
        "CENTER", or "TOP".
    z_order : int, optional
        The initial Z-order of the UI component.

    Attributes
    ----------
    position : (float, float)
        Absolute coordinates (x, y) of the top-left corner of this
        UI component.
    center : (float, float)
        Absolute coordinates (x, y) of the center of this UI component.
    on_left_mouse_button_pressed: function
        Callback function for when the left mouse button is pressed.
    on_left_mouse_button_released: function
        Callback function for when the left mouse button is released.
    on_left_mouse_button_clicked: function
        Callback function for when clicked using the left mouse button
        (i.e. pressed -> released).
    on_left_mouse_double_clicked: function
        Callback function for when left mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_left_mouse_button_dragged: function
        Callback function for when dragging using the left mouse button.
    on_right_mouse_button_pressed: function
        Callback function for when the right mouse button is pressed.
    on_right_mouse_button_released: function
        Callback function for when the right mouse button is released.
    on_right_mouse_button_clicked: function
        Callback function for when clicking using the right mouse button
        (i.e. pressed -> released).
    on_right_mouse_double_clicked: function
        Callback function for when right mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_right_mouse_button_dragged: function
        Callback function for when dragging using the right mouse button.
    on_middle_mouse_button_pressed: function
        Callback function for when the middle mouse button is pressed.
    on_middle_mouse_button_released: function
        Callback function for when the middle mouse button is released.
    on_middle_mouse_button_clicked: function
        Callback function for when clicking using the middle mouse button
        (i.e. pressed -> released).
    on_middle_mouse_double_clicked: function
        Callback function for when middle mouse button is double clicked
        (i.e pressed -> released -> pressed -> released).
    on_middle_mouse_button_dragged: function
        Callback function for when dragging using the middle mouse button.
    on_key_press: function
        Callback function for when a keyboard key is pressed.
    """

    _styled_by_parent = False
    _on_style_change = None
    _on_visibility_change = None

    def __init__(
        self, *, position=(0, 0), x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP, z_order=0
    ):
        """Init scene."""
        self._position = np.array([0, 0])
        self._children = []
        self._anchors = [x_anchor, y_anchor]
        self.z_order = z_order
        self._layout_node = LayoutNode(
            measure=self._measure_layout,
            apply=self._apply_layout,
            resizable=callable(getattr(self, "resize", None)),
            on_dirty=lambda: UIContext.request_layout(self),
        )
        self._layout_natural_size = None
        self._layout_applied_size = None
        self._hover_needs_sync = False
        self._requested_visibility = True

        if getattr(self, "_theme", None) is None:
            self._theme = get_theme()
        self._theme_source = UIContext.theme_source()
        with _inherited_theme_scope(self._theme):
            self._setup()
        self.set_position(position, x_anchor, y_anchor)

        self.left_button_state = "released"
        self.right_button_state = "released"
        self.middle_button_state = "released"

        self.on_left_mouse_button_pressed = lambda event: None
        self.on_left_mouse_button_dragged = lambda event: None
        self.on_left_mouse_button_released = lambda event: None
        self.on_left_mouse_button_clicked = lambda event: None
        self.on_left_mouse_double_clicked = lambda event: None
        self.on_right_mouse_button_pressed = lambda event: None
        self.on_right_mouse_button_released = lambda event: None
        self.on_right_mouse_button_clicked = lambda event: None
        self.on_right_mouse_double_clicked = lambda event: None
        self.on_right_mouse_button_dragged = lambda event: None
        self.on_middle_mouse_button_pressed = lambda event: None
        self.on_middle_mouse_button_released = lambda event: None
        self.on_middle_mouse_button_clicked = lambda event: None
        self.on_middle_mouse_double_clicked = lambda event: None
        self.on_middle_mouse_button_dragged = lambda event: None
        self.on_key_press = lambda event: None
        self.on_key_release = lambda event: None
        self.on_hover = lambda event: None
        self.on_dishover = lambda event: None
        self.on_focus = lambda event: None
        self.on_blur = lambda event: None
        self.on_wheel = lambda event: None

    @property
    def theme(self):
        """
        Get the theme of this UI component.

        Returns
        -------
        Theme
            The theme from :func:`fury.ui.theme_scope` or
            :func:`fury.ui.set_theme` at creation time, or the theme given
            later through :meth:`apply_theme` or :func:`fury.ui.set_theme`.
        """
        return self._theme

    def _style_state(self):
        """
        Get the stored themed values.

        Returns
        -------
        tuple of dict
            Values chosen by the user, values passed by a parent widget, and
            the values currently shown, all keyed by theme item name.
        """
        if not hasattr(self, "_style_values"):
            self._style_values = ({}, {}, {})
        return self._style_values

    def _theme_item(self, item):
        """
        Get the name under which the theme stores an item.

        Parameters
        ----------
        item : str
            Item name used by this component.

        Returns
        -------
        str
            Item name in the theme. Subclasses map variants here.
        """
        return item

    def _theme_value(self, item, value=None):
        """
        Get a value, or the theme item for this component if it is None.

        A value that is not None is remembered, so a later
        :meth:`apply_theme` keeps it. For a component a user creates it is
        the user's choice; for a part created by another widget it is the
        parent's value, which the parent may replace when it is restyled.

        Parameters
        ----------
        item : str
            Name of the theme item.
        value : object, optional
            Value given by the caller. Used as is unless it is None.

        Returns
        -------
        object
            ``value``, a value given earlier for the item, or the theme item.
        """
        user, parent, _ = self._style_state()
        if value is not None:
            source = getattr(self, "_theme_source", None) or UIContext.theme_source()
            (parent if source == "inherited" else user)[item] = value
            return value
        if item in user:
            return user[item]
        if item in parent:
            return parent[item]
        if getattr(self, "_theme", None) is None:
            self._theme = get_theme()
        return self._theme.get(type(self), self._theme_item(item))

    def _set_style(self, item, value, *, source="theme", force=False):
        """
        Store and show a themed value.

        Parameters
        ----------
        item : str
            Theme item name.
        value : object
            The value.
        source : str, optional
            ``"user"``, ``"parent"`` or ``"theme"``. A parent's value never
            replaces the user's, unless ``force`` is True.
        force : bool, optional
            If True, a parent's value replaces the user's.
        """
        user, parent, shown = self._style_state()
        if source == "parent":
            parent[item] = value
            if item in user and not force:
                return
            user.pop(item, None)
        if source != "user" and item in shown and _same_value(shown[item], value):
            return
        descriptor = _style_descriptors(type(self)).get(item)
        previous = getattr(self, "_style_source", "theme")
        self._style_source = source
        try:
            if descriptor is not None and descriptor.fshow is not None:
                descriptor.fshow(self, value)
        finally:
            self._style_source = previous
        shown[item] = value
        if source == "user":
            user[item] = value
        if self._on_style_change is not None:
            self._on_style_change(item)
        if source != "theme":
            self._request_refresh()

    def _push_style(self, item, value, *, force=False):
        """
        Set a themed value of this part on behalf of its parent widget.

        Parameters
        ----------
        item : str
            Theme item name.
        value : object
            The value.
        force : bool, optional
            If True, replace a value the user set on the part.
        """
        self._set_style(item, value, source="parent", force=force)

    def _push_styles(self, **items):
        """
        Set several themed values of this part on behalf of its parent.

        Parameters
        ----------
        **items : dict
            Theme item names and values.
        """
        with UIContext.style_batch():
            for item, value in items.items():
                self._push_style(item, value)

    def _forward(self, part, item, value):
        """
        Pass a value from one of this component's setters to a part.

        When the user set the value, it also replaces a value the user had
        set on the part directly.

        Parameters
        ----------
        part : UI
            The part.
        item : str
            Theme item name of the part.
        value : object
            The value.
        """
        source = getattr(self, "_style_source", "theme")
        part._push_style(item, value, force=source == "user")

    def _request_refresh(self):
        """Refresh the look of this component now, or at the end of a batch."""
        if type(self)._refresh_style is UI._refresh_style:
            return
        if not UIContext.queue_refresh(self):
            self._refresh_style()

    def _refresh_style(self):
        """Show the current themed values on the parts of this component."""

    def _box_style(self, *, prefix=""):
        """
        Get the box style theme items of this component.

        Parameters
        ----------
        prefix : str, optional
            Prefix of the item names, for components with several boxes.

        Returns
        -------
        dict
            Corner radius, border and shadow keyword arguments of
            :class:`RoundedRectangle2D`.
        """
        return {name: self._theme_value(prefix + name) for name in BOX_STYLE_ITEMS}

    def _apply_box_style(self, box, *, prefix="", **overrides):
        """
        Style a box with the box style theme items of this component.

        Parameters
        ----------
        box : RoundedRectangle2D
            The box to style.
        prefix : str, optional
            Prefix of the item names, for components with several boxes.
        **overrides : dict
            Box properties that replace the theme items.
        """
        style = self._box_style(prefix=prefix)
        style.update(overrides)
        for name, value in style.items():
            box._push_style(name, value)

    def apply_theme(self, theme):
        """
        Restyle this component, its parts and its children with a theme.

        Values passed explicitly to a constructor, or set later through a
        public property such as ``color`` or ``font_size``, on the component
        or on one of its parts, are kept. Values a widget only accepts in its
        constructor, such as the text color of ``TextBox2D``, the title and
        body colors of ``Card2D``, ``ListBox2D.background_opacity``,
        ``ComboBox2D.menu_opacity``, and the colors of ``PlaybackPanel``,
        ``Loader2D`` and ``FileMenu2D``, follow the same rule. Every
        other themed value, such as colors, font sizes, fonts and box
        styles, is read again from ``theme``. Structural items keep the value
        read at creation, because they decide which parts a widget is built
        from or how large they are: the ``LineSlider2D`` item ``layout``,
        the ``ButtonGroup`` item ``indicator`` and the size of its icon
        buttons, the ``PlaybackPanel`` border kind, and sizes derived from
        font sizes, such as list slot heights. Slider track widths and
        handle radii, and the size and label gap of box indicators, are
        restyled. Children and parts that were given their
        own theme with this method keep it. The component stops following the
        global theme, so a later :func:`fury.ui.set_theme` leaves it alone. Called
        on a part of another widget, the part keeps the theme when its
        parent is restyled, but the parent still passes it the values it
        controls, such as a list's item colors.

        Parameters
        ----------
        theme : Theme or str
            Theme, or name of a built-in theme.
        """
        self._theme_source = "own"
        UIContext.unfollow_global_theme(self)
        self._apply_theme(_as_theme(theme), follow_global=False)

    def _apply_theme(self, theme, *, follow_global):
        """
        Restyle this component and its children.

        Each component reads its themed values again, then all of them are
        refreshed once, parents before their parts.

        Parameters
        ----------
        theme : Theme
            The new theme.
        follow_global : bool
            True when the global theme changed. Children created in their
            own ``theme_scope``, or given their own theme, are then skipped.
            False when the theme is applied explicitly: children then stop
            following the global theme.
        """
        with UIContext.style_batch():
            self._restyle(theme, follow_global=follow_global)

    def _restyle(self, theme, *, follow_global):
        """
        Read the themed values of this component and its children again.

        Parameters
        ----------
        theme : Theme
            The new theme.
        follow_global : bool
            See :meth:`_apply_theme`.
        """
        self._theme = theme
        if not follow_global and self._theme_source != "inherited":
            self._theme_source = "own"
            UIContext.unfollow_global_theme(self)
        self._request_refresh()
        user, parent, shown = self._style_state()
        for item in _style_descriptors(type(self)):
            if item in user or item in parent:
                continue
            if not theme.has(type(self), self._theme_item(item)):
                continue
            value = self._theme_value(item)
            if item not in shown or not _same_value(shown[item], value):
                self._set_style(item, value)
        for child in self._theme_children():
            inherited = child._theme_source == "inherited"
            if child._theme_source == "own" or (follow_global and not inherited):
                continue
            if inherited and child._styled_by_parent:
                continue
            child._restyle(theme, follow_global=follow_global)

    def _theme_children(self):
        """
        Get the components restyled together with this one.

        Returns
        -------
        list of UI
            The children and composite parts of this component.
        """
        unique = {}
        for child in self._children:
            unique.setdefault(id(child), child)
        return list(unique.values())

    def _reset_interaction_state(self) -> None:
        """
        Cancel held mouse input and invalidate hover without changing focus.

        Returns
        -------
        None
            State is updated in place without dispatching callbacks.
        """
        self.left_button_state = "released"
        self.right_button_state = "released"
        self.middle_button_state = "released"
        self._hover_needs_sync = True
        if UIContext.hot_ui is self:
            UIContext.hot_ui = None

    @abc.abstractmethod
    def _setup(self):
        """
        Set up this UI component.

        This is where you should create all your needed actors and sub
        UI components.
        """
        msg = "Subclasses of UI must implement `_setup(self)`."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def _get_actors(self):
        """Get the actors composing this UI component."""
        msg = "Subclasses of UI must implement `_get_actors(self)`."
        raise NotImplementedError(msg)

    @property
    def layout_style(self):
        """
        Get the layout properties of this UI component.

        Returns
        -------
        LayoutStyle
            The layout properties. Use :meth:`set_layout` to change them.
        """
        return self._layout_node.style

    @property
    def layout_rect(self):
        """
        Get the rectangle computed by the most recent layout.

        Returns
        -------
        LayoutRect
            The position and size in screen pixels.
        """
        return self._layout_node.rect

    def set_layout(self, **properties):
        """
        Change the layout properties of this UI component.

        Parameters
        ----------
        **properties : dict
            Properties of :class:`fury.ui.layout.LayoutStyle`, such as
            ``width``, ``padding`` or ``gap``.

        Returns
        -------
        UI
            This component, so calls can be chained.
        """
        self._layout_node.style.update(**properties)
        self.mark_layout_dirty()
        return self

    def mark_layout_dirty(self):
        """Request a new layout of the tree containing this component."""
        self._layout_node.mark_dirty()

    def update_layout(self):
        """
        Solve the layout rooted at this component if it is out of date.

        The render loop calls this before drawing. Call it directly to read
        up-to-date sizes and positions before the next frame.
        """
        node = self._layout_node
        if not node.dirty or not node.children or node.parent is not None:
            return

        node.dirty = False
        compute_sizes(node, available=UIContext.canvas_size)
        if node.resizable and tuple(self.size) != node.rect.size:
            self.resize(node.rect.size)
        place(node, origin=self.get_position())

    def _measure_layout(self, max_width):
        """
        Report the natural size of this component for the layout engine.

        A size set by the layout is not natural: if it were reported back,
        a component that had grown could never shrink again. Only a size
        changed from outside the layout replaces the natural size.

        Parameters
        ----------
        max_width : float or None
            Width available to the content, or None when unconstrained.

        Returns
        -------
        (float, float)
            The size of the content, excluding padding and border.
        """
        current = tuple(int(value) for value in self.size)
        if self._layout_natural_size is None or current != self._layout_applied_size:
            self._layout_natural_size = current
        style = self._layout_node.style
        return (
            max(self._layout_natural_size[0] - style.inset(0), 0),
            max(self._layout_natural_size[1] - style.inset(1), 0),
        )

    def _apply_layout(self, rect):
        """
        Move and resize this component to its computed rectangle.

        Parameters
        ----------
        rect : LayoutRect
            The computed position and size.
        """
        if self._layout_node.resizable and tuple(self.size) != rect.size:
            self.resize(rect.size)
        if self._layout_node.parent is not None:
            self.set_position(rect.position)
        self._layout_applied_size = tuple(int(value) for value in self.size)

    @property
    def actors(self):
        """
        Get actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return self._get_actors()

    def perform_position_validation(self, x_anchor, y_anchor):
        """
        Perform validation checks for anchor string and the 'size' property.

        Parameters
        ----------
        x_anchor : str
            Horizontal anchor string to validate (e.g., "LEFT", "CENTER", "RIGHT").
        y_anchor : str
            Vertical anchor string to validate (e.g., "TOP", "CENTER", "BOTTOM").
        """
        if not hasattr(self, "size"):
            msg = "Subclasses of UI must implement property `size`."
            raise NotImplementedError(msg)

        if x_anchor not in [Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT]:
            raise ValueError(
                f"x_anchor should be one of these {', '.join([Anchor.LEFT, Anchor.CENTER, Anchor.RIGHT])} but received {x_anchor}"  # noqa: E501
            )

        if y_anchor not in [Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM]:
            raise ValueError(
                f"y_anchor should be one of these {', '.join([Anchor.TOP, Anchor.CENTER, Anchor.BOTTOM])} but received {y_anchor}"  # noqa: E501
            )

    def set_actor_position(self, actor, center_position, z_order, *, sub_order=0):
        """
        Set the position of the PyGfx actor.

        Parameters
        ----------
        actor : Mesh
            The PyGfx mesh actor whose position needs to be set.
        center_position : tuple or ndarray
            A 2-element array `(x, y)` representing the desired center
            position of the actor.
        z_order : int
            The Z-order of the UI component.
        sub_order : float, optional
            Ordering of this actor relative to other actors of the *same*
            component (and `z_order`). Higher values are drawn on top. This
            disambiguates coplanar actors, such as a text actor and its
            background, whose draw order would otherwise be undefined.
            Negative values draw the actor below the other actors of the
            same `z_order`.
        """
        canvas_size = UIContext.canvas_size

        actor.local.x = center_position[0]
        actor.local.y = canvas_size[1] - center_position[1]
        actor.local.z = np.interp(z_order, UIContext.z_order_bounds, UI_Z_RANGE)
        actor.render_order = float(z_order * UI_SUBLAYER_SCALE + sub_order)

    def set_position(self, coords, x_anchor=Anchor.LEFT, y_anchor=Anchor.TOP):
        """
        Position this UI component according to the specified anchor.

        Parameters
        ----------
        coords : (float, float)
            Absolute pixel coordinates (x, y). These coordinates
            are interpreted based on `x_anchor` and `y_anchor`.
        x_anchor : str, optional
            Define the horizontal anchor point for `coords`. Can be "LEFT",
            "CENTER", or "RIGHT".
        y_anchor : str, optional
            Define the vertical anchor point for `coords`. Can be "TOP",
            "CENTER", or "BOTTOM".
        """
        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)

        self._position = np.array(coords)
        self._anchors = [x_anchor.upper(), y_anchor.upper()]
        self._update_actors_position()

    def get_position(
        self,
        x_anchor=Anchor.LEFT,
        y_anchor=Anchor.TOP,
    ):
        """
        Get the position of this UI component according to the specified anchor.

        Parameters
        ----------
        x_anchor : str, optional
            Define the horizontal anchor point for the returned coordinates.
            Can be "LEFT", "CENTER", or "RIGHT".
        y_anchor : str, optional
            Define the vertical anchor point for the returned coordinates.
            Can be "BOTTOM", "CENTER", or "TOP".

        Returns
        -------
        (float, float)
            The (x, y) pixel coordinates of the specified anchor point.
        """
        ANCHOR_TO_MULTIPLIER = get_anchor_to_multiplier()

        self.perform_position_validation(x_anchor=x_anchor, y_anchor=y_anchor)
        size = self.size

        return np.array(
            [
                self._position[0]
                + size[0]
                * (
                    ANCHOR_TO_MULTIPLIER[x_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[0].upper()]
                ),
                self._position[1]
                + size[1]
                * (
                    ANCHOR_TO_MULTIPLIER[y_anchor.upper()]
                    - ANCHOR_TO_MULTIPLIER[self._anchors[1].upper()]
                ),
            ]
        )

    @property
    def z_order(self):
        """
        Get the Z-order of this UI element.

        Returns
        -------
        int
            Z-order of the UI.
        """
        return self._z_order

    @z_order.setter
    def z_order(self, z_order):
        """
        Set the Z-order of this UI element.

        Parameters
        ----------
        z_order : int
            The new integer Z-order value.

        Raises
        ------
        ValueError
            If the provided `z_order` is not an integer.
        """
        if not isinstance(z_order, int):
            raise ValueError("Z-order must be an integer.")

        self._z_order = z_order
        for child in self._children:
            child.z_order = z_order
        UIContext.z_order_bounds = z_order

    @abc.abstractmethod
    def _update_actors_position(self):
        """Update the position of the internal actors."""
        msg = "Subclasses of UI must implement `_set_actors_position(self)`."
        raise NotImplementedError(msg)

    @property
    def size(self):
        """
        Get width and height of this UI component.

        Returns
        -------
        (int, int)
            Width and Height of UI component in pixels.
        """
        return np.asarray(self._get_size(), dtype=int)

    @abc.abstractmethod
    def _get_size(self):
        """
        Get the actual size of the UI component.

        Returns
        -------
        (int, int)
            Width and height of the UI component in pixels.
        """
        msg = "Subclasses of UI must implement property `size`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility):
        """
        Set visibility of this UI component.

        The requested visibility is kept in ``_requested_visibility``. At
        the end, the private single callback ``_on_visibility_change`` is
        called if a parent widget set one, for example to keep a part hidden
        that it must not show. Subclasses that override this method without
        calling it, such as :class:`fury.ui.Panel2D` and
        :class:`TextBlock2D`, do not call the callback.

        Parameters
        ----------
        visibility : bool
            If `True`, the UI component will be visible. If `False`, it will be hidden.
        """
        self._requested_visibility = bool(visibility)
        for actor in self.actors:
            actor.visible = visibility

        for child in self._children:
            child.set_visibility(visibility)
        if self._on_visibility_change is not None:
            self._on_visibility_change()

    def handle_events(self, actor):
        """
        Attach event handlers to the UI object.

        Parameters
        ----------
        actor : Mesh
            The PyGfx mesh to which event handlers should be attached.
        """
        actor.add_event_handler(self.mouse_button_down_callback, EventType.POINTER_DOWN)
        actor.add_event_handler(self.mouse_button_up_callback, EventType.POINTER_UP)
        actor.add_event_handler(self.mouse_move_callback, EventType.POINTER_DRAG)
        actor.add_event_handler(self._sync_hover_state, EventType.POINTER_MOVE)
        actor.add_event_handler(self.key_press_callback, EventType.KEY_DOWN)
        actor.add_event_handler(self.key_release_callback, EventType.KEY_UP)
        actor.add_event_handler(self.pointer_enter_callback, EventType.POINTER_ENTER)
        actor.add_event_handler(self.pointer_leave_callback, EventType.POINTER_LEAVE)
        actor.add_event_handler(self.wheel_callback, EventType.WHEEL)

    def add_callback(self, prop, event_type, callback, *, priority=0):
        """
        Add a callback to a specific event for this UI component.

        Parameters
        ----------
        prop : object
            The PyGfx actor on which this callback is to be added.
        event_type : str
            The event type (e.g., 'pointer_down', 'key_down').
        callback : function
            The callback function.
        priority : int
            Higher number is higher priority.
        """
        prop.add_event_handler(callback, event_type)

    def mouse_button_down_callback(self, event):
        """
        Handle mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if event.button == 1:
            self.left_button_click_callback(event)
        elif event.button == 2:
            self.right_button_click_callback(event)
        elif event.button == 3:
            self.middle_button_click_callback(event)

    def mouse_button_up_callback(self, event):
        """
        Handle mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if event.button == 1:
            self.left_button_release_callback(event)
        elif event.button == 2:
            self.right_button_release_callback(event)
        elif event.button == 3:
            self.middle_button_release_callback(event)

    def left_button_click_callback(self, event):
        """
        Handle left mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.left_button_state = "pressing"

        if UIContext.hot_ui is not None:
            if (
                UIContext.active_ui is not None
                and UIContext.active_ui is not UIContext.hot_ui
            ):
                UIContext.active_ui.on_blur(event)
                UIContext.active_ui = None
            UIContext.active_ui = UIContext.hot_ui
            UIContext.active_ui.on_focus(event)

        self.on_left_mouse_button_pressed(event)

    def left_button_release_callback(self, event):
        """
        Handle left mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.left_button_state == "pressing":
            self.on_left_mouse_button_clicked(event)
        self.left_button_state = "released"
        self.on_left_mouse_button_released(event)

    def right_button_click_callback(self, event):
        """
        Handle right mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.right_button_state = "pressing"
        self.on_right_mouse_button_pressed(event)

    def right_button_release_callback(self, event):
        """
        Handle right mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.right_button_state == "pressing":
            self.on_right_mouse_button_clicked(event)
        self.right_button_state = "released"
        self.on_right_mouse_button_released(event)

    def middle_button_click_callback(self, event):
        """
        Handle middle mouse button press event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self.middle_button_state = "pressing"
        self.on_middle_mouse_button_pressed(event)

    def middle_button_release_callback(self, event):
        """
        Handle middle mouse button release event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self.middle_button_state == "pressing":
            self.on_middle_mouse_button_clicked(event)
        self.middle_button_state = "released"
        self.on_middle_mouse_button_released(event)

    def mouse_move_callback(self, event):
        """
        Handle mouse move event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        left_pressing_or_dragging = (
            self.left_button_state == "pressing" or self.left_button_state == "dragging"
        )

        right_pressing_or_dragging = (
            self.right_button_state == "pressing"
            or self.right_button_state == "dragging"
        )

        middle_pressing_or_dragging = (
            self.middle_button_state == "pressing"
            or self.middle_button_state == "dragging"
        )

        if left_pressing_or_dragging:
            self.left_button_state = "dragging"
            self.on_left_mouse_button_dragged(event)
        elif right_pressing_or_dragging:
            self.right_button_state = "dragging"
            self.on_right_mouse_button_dragged(event)
        elif middle_pressing_or_dragging:
            self.middle_button_state = "dragging"
            self.on_middle_mouse_button_dragged(event)

    def key_press_callback(self, event):
        """
        Handle key press event.

        Parameters
        ----------
        event : KeyboardEvent
            The PyGfx keyboard event object.
        """
        if UIContext.active_ui is not None:
            UIContext.active_ui.on_key_press(event)
        else:
            self.on_key_press(event)

    def key_release_callback(self, event):
        """
        Handle key release event.

        Parameters
        ----------
        event : KeyboardEvent
            The PyGfx keyboard event object.
        """
        if UIContext.active_ui is not None:
            UIContext.active_ui.on_key_release(event)
        else:
            self.on_key_release(event)

    def wheel_callback(self, event):
        """
        Handle wheel event.

        Parameters
        ----------
        event : WheelEvent
            The PyGfx wheel event object.
        """
        if UIContext.hot_ui is not None:
            UIContext.hot_ui.on_wheel(event)
        else:
            self.on_wheel(event)
        event.stop_propagation()

    def _sync_hover_state(self, event) -> None:
        """
        Restore invalidated hover from a newly picked pointer movement.

        Parameters
        ----------
        event : PointerEvent
            The real pointer movement targeting this component.

        Returns
        -------
        None
            Hover entry is delivered at most once after each reset.
        """
        if self._hover_needs_sync:
            self.pointer_enter_callback(event.copy(type=EventType.POINTER_ENTER))

    def pointer_enter_callback(self, event):
        """
        Handle pointer enter event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self._hover_needs_sync = False
        UIContext.hot_ui = self
        self.on_hover(event)

    def pointer_leave_callback(self, event):
        """
        Handle pointer leave event.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        self._hover_needs_sync = False
        if UIContext.hot_ui is self:
            UIContext.hot_ui = None
        self.on_dishover(event)


class Rectangle2D(UI):
    """
    A 2D rectangle sub-classed from UI.

    Parameters
    ----------
    size : (int, int)
        The size of the rectangle (width, height) in pixels.
    position : (float, float)
        Coordinates (x, y) of the top-left corner of the rectangle.
    color : str, tuple, list or ndarray, optional
        A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
        Defaults to the theme item of the same name.
    opacity : float
        Must take values in [0, 1].
    """

    _styled_by_parent = True

    def __init__(self, *, size=(100, 100), position=(0, 0), color=None, opacity=1.0):
        """Initialize a rectangle."""
        super(Rectangle2D, self).__init__(position=position)
        self._set_style("color", self._theme_value("color", color))
        self._set_style("opacity", self._theme_value("opacity", opacity))
        self.resize(size)

    def _setup(self):
        """
        Set up this UI component.

        Create the plane actor used internally.
        """
        geo = plane_geometry(width=1, height=1)
        mat = _create_mesh_material(material="basic", alpha_mode="auto", mode="auto")
        self.actor = create_mesh(geometry=geo, material=mat)

        self.handle_events(self.actor)

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return [self.actor]

    def _get_size(self):
        """
        Get the current size of the rectangle actor.

        Returns
        -------
        (float, float)
            The current `(width, height)` of the rectangle in pixels.
        """
        if hasattr(self, "_size"):
            return self._size
        bounds = self.actor.get_bounding_box()
        minx, miny, minz = bounds[0]
        maxx, maxy, maxz = bounds[1]
        return [maxx - minx, maxy - miny]

    @property
    def width(self):
        """
        Get the current width of the rectangle.

        Returns
        -------
        float
            The width of the rectangle in pixels.
        """
        return self._get_size()[0]

    @width.setter
    def width(self, width):
        """
        Set the width of the rectangle.

        Parameters
        ----------
        width : float
            New width of the rectangle.
        """
        self.resize((width, self.height))

    @property
    def height(self):
        """
        Get the current height of the rectangle.

        Returns
        -------
        float
            The height of the rectangle in pixels.
        """
        return self._get_size()[1]

    @height.setter
    def height(self, height):
        """
        Set the height of the rectangle.

        Parameters
        ----------
        height : float
            New height of the rectangle.
        """
        self.resize((self.width, height))

    def resize(self, size):
        """
        Set the rectangle size.

        Parameters
        ----------
        size : (float, float)
            Rectangle size (width, height) in pixels.
        """
        w = max(size[0], 1)
        h = max(size[1], 1)
        size = (w, h)
        self.actor.geometry = plane_geometry(width=w, height=h)
        self._update_actors_position()

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.set_actor_position(self.actor, position, self.z_order)

    def _get_color(self):
        """
        Get the rectangle color.

        Returns
        -------
        (float, float, float)
            RGB color.
        """
        return self.actor.material.color[:3]

    def _show_color(self, color):
        """
        Set the rectangle color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string (``"#FF0000"``), RGB(A) values in [0, 1], or
            RGB(A) values in [0, 255]. See :func:`normalize_colors`.
        """
        color = normalize_colors(color)[0]
        self.actor.material.color = np.array([*color[:3], 1.0])

    color = themed(_get_color, _show_color)

    def _get_opacity(self):
        """
        Get the rectangle opacity.

        Returns
        -------
        float
            Opacity value.
        """
        return self.actor.material.opacity

    def _show_opacity(self, opacity):
        """
        Set the rectangle opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.material.opacity = opacity

    opacity = themed(_get_opacity, _show_opacity)


class RoundedRectangle2D(UI):
    """
    A 2D rounded rectangle sub-classed from UI.

    The box can have a border drawn inside its edge, a different radius for
    each corner and a soft drop shadow. The shadow is drawn outside the box
    but is not part of it: ``size``, ``position``, layout and picking only
    cover the box itself.

    Parameters
    ----------
    size : (int, int)
        The size of the rectangle (width, height) in pixels.
    position : (float, float)
        Coordinates (x, y) of the lower-left corner of the rectangle.
    color : str, tuple, list or ndarray, optional
        A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
        Defaults to the theme item of the same name.
        An alpha channel is ignored; use ``opacity`` to fade the box. The
        border and shadow colors, in contrast, keep their alpha.
    corner_radius : float or tuple of float
        Radius of the rounded corners in pixels. A single value rounds every
        corner. Four values give the top-left, top-right, bottom-right and
        bottom-left radii. Radii are limited to half of the smaller side when
        drawn.
    opacity : float
        Must take values in [0, 1].
    border_color : str, tuple, list or ndarray
        Border color as a hex string, RGB(A) in [0, 1], or RGB(A) in
        [0, 255].
    border_width : float
        Width of the border in pixels, drawn inside the box edge. 0 draws no
        border.
    shadow_color : str, tuple, list or ndarray
        Shadow color as a hex string, RGB(A) in [0, 1], or RGB(A) in
        [0, 255]. The alpha channel sets the shadow strength.
    shadow_offset : (float, float)
        Shadow displacement (x, y) in pixels. Positive y moves the shadow
        down.
    shadow_blur : float
        Blur radius of the shadow in pixels.
    shadow_spread : float
        Distance in pixels by which the shadow grows (positive) or shrinks
        (negative) relative to the box.

    Notes
    -----
    The shadow is drawn when ``shadow_blur``, ``shadow_spread`` or
    ``shadow_offset`` is non-zero. With the defaults no shadow is drawn. Like
    a CSS box shadow, it is not drawn under the box, so it does not show
    through a translucent box. Only within the 1-2 pixels of anti-aliasing
    at the box edge can a little shadow remain under a translucent box.

    The blur is an approximation of a gaussian blur. It matches CSS for boxes
    much larger than the blur, but gets darker than CSS when the blur is
    large relative to the box. Shadows are drawn beneath all boxes of the
    same ``z_order``, so a box never falls into the shadow of a neighbor at
    its own ``z_order``; raise the ``z_order`` of a box to cast its shadow
    over others.

    Corner radii larger than half of the smaller side are drawn as half of
    that side, so a small box keeps a proper rounded shape: a 10x10 box with
    the default radius of 10 is drawn as a circle. Earlier versions drew a
    distorted shape in that case.

    A box with square corners and no border is drawn with a plain mesh
    material, exactly like :class:`Rectangle2D`, so its edges are not
    anti-aliased. Widgets rely on this to draw a flat themed box with the
    same pixels as before themes had box styles.
    """

    _styled_by_parent = True

    def __init__(
        self,
        *,
        size=(100, 100),
        position=(0, 0),
        color=None,
        corner_radius=10.0,
        opacity=1.0,
        border_color=(0, 0, 0),
        border_width=0.0,
        shadow_color=(0, 0, 0, 0.5),
        shadow_offset=(0, 0),
        shadow_blur=0.0,
        shadow_spread=0.0,
    ):
        """Initialize a rounded rectangle."""
        super(RoundedRectangle2D, self).__init__(position=position)

        self._set_style("color", self._theme_value("color", color))
        self._set_style("opacity", self._theme_value("opacity", opacity))
        for name, value in (
            ("corner_radius", corner_radius),
            ("border_color", border_color),
            ("border_width", border_width),
            ("shadow_color", shadow_color),
            ("shadow_offset", shadow_offset),
            ("shadow_blur", shadow_blur),
            ("shadow_spread", shadow_spread),
        ):
            self._set_style(name, self._theme_value(name, value))
        self.resize(size)

    def _setup(self):
        """
        Set up this UI component.

        Create the box actor with the plain material. The rounded material
        and the shadow are created when a radius, border or shadow needs
        them.
        """
        self._size = [0.0, 0.0]
        self._color = np.array([1.0, 1.0, 1.0, 1.0])
        self._opacity = 1.0
        self._corner_radius = 0.0
        self._radii = np.zeros(4)
        self._border_color = np.array([0.0, 0.0, 0.0, 1.0])
        self._border_width = 0.0
        self._shadow_color = np.array([0.0, 0.0, 0.0, 0.5])
        self._shadow_offset = np.zeros(2)
        self._shadow_blur = 0.0
        self._shadow_spread = 0.0
        self._visible = True
        self._sdf_material = None
        self._shadow = None
        self._flat_material = _create_mesh_material(
            material="basic", alpha_mode="auto", mode="auto"
        )
        self.actor = create_mesh(
            geometry=plane_geometry(width=1, height=1), material=self._flat_material
        )
        self.handle_events(self.actor)

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            The box actor. The shadow, when there is one, is a child of it.
        """
        return [self.actor]

    @property
    def shadow_actor(self):
        """
        Get the shadow actor, a non-pickable child of the box actor.

        Returns
        -------
        Mesh
            The shadow mesh, hidden while the box has no shadow.
        """
        if self._shadow is None:
            self._build_shadow()
        return self._shadow

    def _get_size(self):
        """
        Get the current size of the rectangle actor.

        Returns
        -------
        list
            Width and height of the rectangle, without the shadow.
        """
        return self._size

    @property
    def width(self):
        """
        Get the current width of the rectangle.

        Returns
        -------
        float
            Width of the rectangle.
        """
        return self._get_size()[0]

    @width.setter
    def width(self, width):
        """
        Set the width of the rectangle.

        Parameters
        ----------
        width : float
            New width.
        """
        self.resize((width, self.height))

    @property
    def height(self):
        """
        Get the current height of the rectangle.

        Returns
        -------
        float
            Height of the rectangle.
        """
        return self._get_size()[1]

    @height.setter
    def height(self, height):
        """
        Set the height of the rectangle.

        Parameters
        ----------
        height : float
            New height.
        """
        self.resize((self.width, height))

    def resize(self, size):
        """
        Set the rectangle size.

        Parameters
        ----------
        size : (float, float)
            Rectangle size (width, height) in pixels.
        """
        w = max(size[0], 1)
        h = max(size[1], 1)
        if [w, h] == self._size:
            self._update_actors_position()
            return
        self._size = [w, h]
        self.actor.geometry = plane_geometry(width=w, height=h)
        if self._sdf_material is not None:
            self._sdf_material.size = (w, h)
        self._update_shadow()
        self._update_actors_position()

    def _update_actors_position(self):
        """Set the position of the box actor and of its shadow."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
        self.set_actor_position(self.actor, position, self.z_order)
        if self._shadow is not None:
            self._shadow.local.position = (
                self._shadow_offset[0],
                -self._shadow_offset[1],
                0,
            )
            self._shadow.render_order = self.actor.render_order - 0.5

    def _build_sdf_material(self):
        """
        Create the rounded material from the current style.

        Returns
        -------
        RoundedRectangleMaterial
            The material drawing corners and the border.
        """
        material = RoundedRectangleMaterial(
            size=tuple(self._size),
            corner_radius=self._radii,
            color=self._color,
            opacity=self._opacity,
            border_color=self._border_color,
            border_width=self._border_width,
            side="BOTH",
        )
        return material

    def _update_box_material(self):
        """
        Draw square, borderless boxes with the plain mesh material.

        The rounded material is created the first time a radius or border
        needs it. Picking and depth settings carry over to the other
        material. Each material keeps its own ``alpha_mode``: the plain
        material must keep the one of :class:`Rectangle2D` to draw the same
        pixels.
        """
        flat = not np.any(self._radii > 0) and self._border_width == 0
        if flat:
            target = self._flat_material
        else:
            if self._sdf_material is None:
                self._sdf_material = self._build_sdf_material()
            target = self._sdf_material
        current = self.actor.material
        if current is target:
            return
        target.pick_write = current.pick_write
        target.depth_write = current.depth_write
        target.depth_test = current.depth_test
        self.actor.material = target

    def _has_shadow(self):
        """
        Tell whether the box casts a shadow.

        Returns
        -------
        bool
            True if the blur, spread or offset is non-zero.
        """
        return (
            self._shadow_blur > 0
            or self._shadow_spread != 0
            or bool(np.any(self._shadow_offset != 0))
        )

    def _build_shadow(self):
        """Create the shadow mesh as a child of the box actor."""
        material = RoundedRectangleMaterial(
            size=tuple(self._size),
            corner_radius=self._radii,
            shadow=True,
            shadow_color=self._shadow_color,
            shadow_offset=(self._shadow_offset[0], -self._shadow_offset[1]),
            shadow_blur=self._shadow_blur,
            shadow_spread=self._shadow_spread,
            opacity=self._opacity,
            side="BOTH",
        )
        self._shadow = create_mesh(
            geometry=plane_geometry(width=1, height=1), material=material
        )
        self._shadow.visible = False
        self.actor.add(self._shadow)
        self._update_shadow()
        self._update_actors_position()

    def _update_shadow(self):
        """Size and show the shadow, creating it when it is first needed."""
        has_shadow = self._has_shadow()
        if self._shadow is None:
            if has_shadow:
                self._build_shadow()
            return
        material = self._shadow.material
        material.size = tuple(self._size)
        if has_shadow:
            extent = max(self._shadow_spread, 0) + 1.5 * self._shadow_blur + 1
            width, height = self._size
            self._shadow.geometry = plane_geometry(
                width=width + 2 * extent, height=height + 2 * extent
            )
        self._shadow.visible = self._visible and has_shadow

    def _set_shadow_uniform(self, name, value):
        """
        Set a value of the shadow material if the shadow exists.

        Parameters
        ----------
        name : str
            Name of the material property.
        value : object
            The value.
        """
        if self._shadow is not None:
            setattr(self._shadow.material, name, value)

    def set_visibility(self, visibility):
        """
        Set visibility of the rectangle and its shadow.

        Parameters
        ----------
        visibility : bool
            If `True`, the rectangle will be visible. If `False`, it will be
            hidden.
        """
        self._visible = bool(visibility)
        super(RoundedRectangle2D, self).set_visibility(visibility)
        if self._shadow is not None:
            self._shadow.visible = self._visible and self._has_shadow()

    def _get_color(self):
        """
        Get the rectangle color.

        Returns
        -------
        ndarray
            RGB color.
        """
        return self._color[:3]

    def _show_color(self, color):
        """
        Set the rectangle color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string, RGB(A) in [0, 1], or RGB(A) in [0, 255]. An alpha
            channel is ignored; use ``opacity`` instead.
        """
        color = normalize_colors(color)[0]
        self._color = np.array([*color[:3], 1.0])
        self._flat_material.color = self._color
        if self._sdf_material is not None:
            self._sdf_material.color = self._color

    color = themed(_get_color, _show_color)

    def _get_opacity(self):
        """
        Get the rectangle opacity.

        Returns
        -------
        float
            Opacity of the rectangle.
        """
        return self._opacity

    def _show_opacity(self, opacity):
        """
        Set the rectangle opacity.

        Parameters
        ----------
        opacity : float
            New opacity.
        """
        self._opacity = opacity
        self._flat_material.opacity = opacity
        if self._sdf_material is not None:
            self._sdf_material.opacity = opacity
        self._set_shadow_uniform("opacity", opacity)

    opacity = themed(_get_opacity, _show_opacity)

    def _get_corner_radius(self):
        """
        Get the corner radius.

        Returns
        -------
        float or tuple of float
            The radius as it was set: one value for every corner, or the
            top-left, top-right, bottom-right and bottom-left radii.
        """
        return self._corner_radius

    def _show_corner_radius(self, radius):
        """
        Set the corner radius.

        Parameters
        ----------
        radius : float or tuple of float
            One non-negative radius for every corner, or four radii for the
            top-left, top-right, bottom-right and bottom-left corners.
        """
        if np.ndim(radius) == 0:
            radii = np.full(4, float(radius))
            value = float(radius)
        else:
            radii = np.asarray(radius, dtype=float)
            if radii.shape != (4,):
                raise ValueError(
                    "corner_radius must be a number or a sequence of 4 numbers "
                    f"(top-left, top-right, bottom-right, bottom-left), got {radius}."
                )
            value = tuple(float(r) for r in radii)
        if np.any(radii < 0):
            raise ValueError(f"corner_radius must be non-negative, got {radius}.")
        self._corner_radius = value
        self._radii = radii
        if self._sdf_material is not None:
            self._sdf_material.corner_radius = radii
        self._set_shadow_uniform("corner_radius", radii)
        self._update_box_material()

    corner_radius = themed(_get_corner_radius, _show_corner_radius)

    def _get_border_color(self):
        """
        Get the border color.

        Returns
        -------
        ndarray
            RGBA border color in [0, 1].
        """
        return self._border_color

    def _show_border_color(self, color):
        """
        Set the border color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string, RGB(A) in [0, 1], or RGB(A) in [0, 255]. See
            :func:`normalize_colors`.
        """
        self._border_color = _to_rgba(color)
        if self._sdf_material is not None:
            self._sdf_material.border_color = self._border_color

    border_color = themed(_get_border_color, _show_border_color)

    def _get_border_width(self):
        """
        Get the border width.

        Returns
        -------
        float
            Border width in pixels.
        """
        return self._border_width

    def _show_border_width(self, width):
        """
        Set the border width.

        Parameters
        ----------
        width : float
            Non-negative border width in pixels. 0 draws no border.
        """
        if width < 0:
            raise ValueError(f"border_width must be non-negative, got {width}.")
        self._border_width = width
        if self._sdf_material is not None:
            self._sdf_material.border_width = width
        self._update_box_material()

    border_width = themed(_get_border_width, _show_border_width)

    def _get_shadow_color(self):
        """
        Get the shadow color.

        Returns
        -------
        ndarray
            RGBA shadow color in [0, 1].
        """
        return self._shadow_color

    def _show_shadow_color(self, color):
        """
        Set the shadow color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string, RGB(A) in [0, 1], or RGB(A) in [0, 255]. The alpha
            channel sets the shadow strength.
        """
        self._shadow_color = _to_rgba(color)
        self._set_shadow_uniform("shadow_color", self._shadow_color)

    shadow_color = themed(_get_shadow_color, _show_shadow_color)

    def _get_shadow_offset(self):
        """
        Get the shadow offset.

        Returns
        -------
        tuple of float
            Shadow displacement (x, y) in pixels, positive y pointing down.
        """
        return tuple(float(v) for v in self._shadow_offset)

    def _show_shadow_offset(self, offset):
        """
        Set the shadow offset.

        Parameters
        ----------
        offset : (float, float)
            Shadow displacement (x, y) in pixels, positive y pointing down.
        """
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (2,):
            raise ValueError(f"shadow_offset must be (x, y), got {offset.tolist()}.")
        self._shadow_offset = offset
        self._set_shadow_uniform("shadow_offset", (offset[0], -offset[1]))
        self._update_shadow()
        self._update_actors_position()

    shadow_offset = themed(_get_shadow_offset, _show_shadow_offset)

    def _get_shadow_blur(self):
        """
        Get the shadow blur radius.

        Returns
        -------
        float
            Blur radius in pixels.
        """
        return self._shadow_blur

    def _show_shadow_blur(self, blur):
        """
        Set the shadow blur radius.

        Parameters
        ----------
        blur : float
            Non-negative blur radius in pixels.
        """
        if blur < 0:
            raise ValueError(f"shadow_blur must be non-negative, got {blur}.")
        self._shadow_blur = blur
        self._set_shadow_uniform("shadow_blur", blur)
        self._update_shadow()

    shadow_blur = themed(_get_shadow_blur, _show_shadow_blur)

    def _get_shadow_spread(self):
        """
        Get the shadow spread.

        Returns
        -------
        float
            Spread in pixels.
        """
        return self._shadow_spread

    def _show_shadow_spread(self, spread):
        """
        Set the shadow spread.

        Parameters
        ----------
        spread : float
            Distance in pixels by which the shadow grows (positive) or
            shrinks (negative) relative to the box.
        """
        self._shadow_spread = spread
        self._set_shadow_uniform("shadow_spread", spread)
        self._update_shadow()

    shadow_spread = themed(_get_shadow_spread, _show_shadow_spread)


class Disk2D(UI):
    """
    A 2D disk UI component.

    Parameters
    ----------
    outer_radius : int
        Outer radius of the disk.
    inner_radius : int, optional
        Inner radius of the disk.
    center : (float, float), optional
        Coordinates (x, y) of the center of the disk.
    color : str, tuple, list or ndarray, optional
        A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
        Defaults to the theme item of the same name.
    opacity : float, optional
        Must take values in [0, 1].
    """

    _styled_by_parent = True

    def __init__(
        self,
        outer_radius,
        *,
        inner_radius=0,
        center=(0, 0),
        color=None,
        opacity=1.0,
    ):
        """Initialize a 2D Disk."""
        self.actor = None
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

        super(Disk2D, self).__init__(
            position=center, x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER
        )

        self._set_style("color", self._theme_value("color", color))
        self._set_style("opacity", self._theme_value("opacity", opacity))

    def _setup(self):
        """
        Set up this UI component.

        Create the disk actor used internally.
        """
        positions, indices = prim_ring(
            inner_radius=self.inner_radius, outer_radius=self.outer_radius
        )
        geo = buffer_to_geometry(positions=positions, indices=indices)
        mat = _create_mesh_material(material="basic", alpha_mode="auto")
        self.actor = create_mesh(geometry=geo, material=mat)

        self.handle_events(self.actor)

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List of actors composing this UI component.
        """
        return [self.actor]

    def _get_size(self):
        """
        Get the current size of the disk.

        Returns
        -------
        (float, float)
            The current `(diameter, diameter)` of the disk in pixels.
        """
        diameter = 2 * self.outer_radius
        size = (diameter, diameter)
        return size

    def _update_actors_position(self):
        """Set the position of the internal actor."""
        position = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)

        self.set_actor_position(self.actor, position, self.z_order)

    def _get_color(self):
        """
        Get the color of this UI component.

        Returns
        -------
        (float, float, float)
            RGB color.
        """
        return self.actor.material.color[:3]

    def _show_color(self, color):
        """
        Set the color of this UI component.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string (``"#FF0000"``), RGB(A) values in [0, 1], or
            RGB(A) values in [0, 255]. See :func:`normalize_colors`.
        """
        color = normalize_colors(color)[0]
        self.actor.material.color = np.array([*color[:3], 1.0])

    color = themed(_get_color, _show_color)

    def _get_opacity(self):
        """
        Get the opacity of this UI component.

        Returns
        -------
        float
            Opacity value.
        """
        return self.actor.material.opacity

    def _show_opacity(self, opacity):
        """
        Set the opacity of this UI component.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].
        """
        self.actor.material.opacity = opacity

    opacity = themed(_get_opacity, _show_opacity)

    @property
    def outer_radius(self):
        """
        Get the outer radius of the disk.

        Returns
        -------
        int
            Outer radius in pixels.
        """
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, radius):
        """
        Set the outer radius of the disk.

        Parameters
        ----------
        radius : int
            New outer radius.
        """
        if self.actor:
            positions, indices = prim_ring(
                inner_radius=self.inner_radius, outer_radius=radius
            )
            self.actor.geometry = buffer_to_geometry(
                positions=positions, indices=indices
            )
        self._outer_radius = radius

    @property
    def inner_radius(self):
        """
        Get the inner radius of the disk.

        Returns
        -------
        int
            Inner radius in pixels.
        """
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, radius):
        """
        Set the inner radius of the disk.

        Parameters
        ----------
        radius : int
            New inner radius.
        """
        if self.actor:
            positions, indices = prim_ring(
                inner_radius=radius, outer_radius=self.outer_radius
            )
            self.actor.geometry = buffer_to_geometry(
                positions=positions, indices=indices
            )
        self._inner_radius = radius


class TextBlock2D(UI):
    """
    A 2D text component with optional background.

    Parameters
    ----------
    text : str, optional
        The initial text message.
    font_size : int, optional
        Size of the text font.
        Defaults to the theme item of the same name.
    font_family : str, optional
        The font family name.
        Defaults to the theme item of the same name.
    justification : str, optional
        Horizontal alignment ("left", "center", "right").
    vertical_justification : str, optional
        Vertical alignment ("top", "middle", "bottom").
    bold : bool, optional
        If True, makes text bold.
    italic : bool, optional
        If True, makes text italicized.
    size : (int, int), optional
        The (width, height) in pixels for the text bounding box.
    color : str, tuple, list or ndarray, optional
        Text color. A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in
        [0, 255].
        Defaults to the theme item of the same name.
    bg_color : str, tuple, list or ndarray, optional
        Background color, same formats as ``color``. If None, no background
        is drawn.
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    dynamic_bbox : bool, optional
        If True, resizes the bounding box to fit the content.
    font_weight : int, optional
        Weight of the text, from 100 (thin) to 900 (black). 400 is regular
        and 600 semibold. The bundled font has one weight, so other weights
        are drawn by thickening its glyphs.
        Defaults to the theme item of the same name.
    muted : bool, optional
        If True, the default color is the theme item ``muted_color``, for
        secondary text such as subtitles, instead of ``color``.

    Notes
    -----
    ``background`` is a :class:`RoundedRectangle2D`, so it can take a
    themed corner radius and border. It was a :class:`Rectangle2D` before
    themes had box styles.
    """

    def __init__(
        self,
        *,
        text="Text Block",
        font_size=None,
        font_family=None,
        justification="left",
        vertical_justification="top",
        bold=False,
        italic=False,
        size=None,
        color=None,
        bg_color=None,
        position=(0, 0),
        dynamic_bbox=False,
        font_weight=None,
        muted=False,
    ):
        """Initialize the text block instance."""
        self._color_item = "muted_color" if muted else "color"
        self.boundingbox = [0, 0, 0, 0]
        self._message = text
        self._dynamic_bbox = dynamic_bbox
        self._bg_size = size
        self._shadow = False
        self._layout_managed_size = None
        self._measure_cache = {}

        if self._bg_size is None and not self.dynamic_bbox:
            raise ValueError("TextBlock size is required as it is not dynamic.")

        self._justification = justification
        self._vertical_justification = vertical_justification
        super(TextBlock2D, self).__init__(position=position)
        self.have_bg = bool(bg_color)
        self._set_style("color", self._theme_value("color", color))
        if bg_color is not None:
            self._theme_value("background_color", bg_color)
        self._set_style("background_color", bg_color)
        self._set_style("font_family", self._theme_value("font_family", font_family))
        self.bold = bold
        self.italic = italic
        self.message = text
        self._set_style("font_size", self._theme_value("font_size", font_size))
        self._set_style("font_weight", self._theme_value("font_weight", font_weight))

        self.update_bounding_box()

    def _theme_item(self, item):
        """
        Get the name under which the theme stores an item.

        Parameters
        ----------
        item : str
            Item name used by this text block.

        Returns
        -------
        str
            ``muted_color`` for the color of muted text, otherwise ``item``.
        """
        return self._color_item if item == "color" else item

    def _setup(self):
        """Set up this UI component."""
        self.actor = Text(
            markdown=self._message, screen_space=True, anchor="middle-center"
        )
        self.actor.material.aa = True
        self.background = RoundedRectangle2D(corner_radius=0)
        self._children.append(self.background)
        self.handle_events(self.actor)

        self.background.on_left_mouse_button_pressed = lambda event: (
            self.on_left_mouse_button_pressed(event)
        )
        self.background.on_left_mouse_button_dragged = lambda event: (
            self.on_left_mouse_button_dragged(event)
        )
        self.background.on_key_press = lambda event: self.on_key_press(event)
        self.background.on_key_release = lambda event: self.on_key_release(event)

        def _bg_hover(event):
            """
            Redirect hot_ui to this TextBlock2D on background hover.

            Parameters
            ----------
            event : PointerEvent
                The PyGfx pointer event object.
            """
            UIContext.hot_ui = self
            self.on_hover(event)

        def _bg_dishover(event):
            """
            Clear hot_ui on background pointer-leave.

            Parameters
            ----------
            event : PointerEvent
                The PyGfx pointer event object.
            """
            if UIContext.hot_ui is self:
                UIContext.hot_ui = None
            self.on_dishover(event)

        self.background.on_hover = _bg_hover
        self.background.on_dishover = _bg_dishover
        self.background.on_wheel = lambda event: self.on_wheel(event)

    def set_layout(self, **properties):
        """
        Change the layout properties of this text block.

        Parameters
        ----------
        **properties : dict
            Properties of :class:`fury.ui.layout.LayoutStyle`, such as
            ``width`` or ``padding``.

        Returns
        -------
        TextBlock2D
            This component, so calls can be chained.
        """
        super().set_layout(**properties)
        self._text_changed()
        return self

    def _measure_layout(self, max_width):
        """
        Report the size of the text for the layout engine.

        Parameters
        ----------
        max_width : float or None
            Width the text may wrap to, or None for no limit.

        Returns
        -------
        (float, float)
            Size of the text, excluding padding and border.
        """
        if not self.dynamic_bbox:
            return super()._measure_layout(max_width)

        natural = self._measure_text(None)
        if max_width is None or max_width + _TEXT_FIT_TOLERANCE >= natural[0]:
            return natural
        return self._measure_text(max_width)

    def _measure_text(self, max_width):
        """
        Measure the text at a given wrapping width, using a cache.

        Parameters
        ----------
        max_width : float or None
            Width the text may wrap to, or None for no limit.

        Returns
        -------
        (int, int)
            Size of the text, rounded up to whole pixels.
        """
        key = (
            self._message,
            float(self.actor.font_size),
            self.actor.family,
            self.bold,
            self.italic,
            max_width,
        )
        if key not in self._measure_cache:
            previous = self.actor.max_width
            self.actor.max_width = 0 if max_width is None else max(max_width, 1)
            width, height = self.get_text_actor_size()
            self.actor.max_width = previous
            self._measure_cache[key] = (math.ceil(width), math.ceil(height))
        return self._measure_cache[key]

    def _apply_layout(self, rect):
        """
        Remember the size given by the layout, then apply it.

        Parameters
        ----------
        rect : LayoutRect
            The computed position and size.
        """
        self._layout_managed_size = rect.size
        content_width = max(rect.width - self.layout_style.inset(0), 1)
        # Glyph bounds are slightly narrower than the width the text wraps at,
        # and rounding a box to whole pixels can take one pixel from it. Text
        # given its natural width must not be limited to that width.
        natural_width = self._measure_text(None)[0]
        fits = (
            self.dynamic_bbox and content_width + _TEXT_FIT_TOLERANCE >= natural_width
        )
        self.actor.max_width = 0 if fits else content_width
        super()._apply_layout(rect)

    def _content_inset(self):
        """
        Get the space between the box edges and the text.

        Returns
        -------
        (float, float, float, float)
            The left, top, right and bottom inset in pixels.
        """
        style = self.layout_style
        padding = style.padding
        return (
            padding.left + style.border,
            padding.top + style.border,
            padding.right + style.border,
            padding.bottom + style.border,
        )

    def resize(self, size):
        """
        Resize the TextBlock2D bounding box.

        Parameters
        ----------
        size : (int, int)
            The new (width, height) in pixels.
        """
        self.actor.max_width = max(size[0] - self.layout_style.inset(0), 1)
        self.update_bounding_box(size=size)

    def _text_changed(self):
        """Refresh the box and the layout after the text content changed."""
        if self.dynamic_bbox:
            self.update_bounding_box()
        else:
            self.update_alignment()
        self.mark_layout_dirty()

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List containing the text actor and background actors.
        """
        return [self.actor]

    def set_visibility(self, visibility):
        """
        Set visibility of this text block.

        Parameters
        ----------
        visibility : bool
            If True, the text will be visible. The background is shown only
            when this text block was created with a background color. If False,
            both text and background are hidden.
        """
        self.actor.visible = visibility
        self.background.set_visibility(visibility and self.have_bg)

    def get_formatted_text(self, text):
        """
        Format the given text with markdown syntax for bold/italic styles.

        Parameters
        ----------
        text : str
            The raw text to format.

        Returns
        -------
        str
            The formatted markdown string.
        """
        if not self.bold and not self.italic:
            return text

        def _wrap_word(word):
            """
            Wrap a single word with the appropriate markdown markers.

            Parameters
            ----------
            word : str
                The word to wrap with markdown markers.

            Returns
            -------
            str
                The word wrapped with bold/italic markdown markers.
            """
            if not word.strip():
                return word
            if self.bold and self.italic:
                zws = "\u200b"
                return f"*{zws}**{word}**{zws}*"
            if self.bold:
                return f"**{word}**"
            return f"*{word}*"

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        formatted_lines = []
        for line in lines:
            if not line.strip():
                formatted_lines.append(line)
                continue
            words = line.split(" ")
            formatted_words = [_wrap_word(w) if w else w for w in words]
            formatted_lines.append(" ".join(formatted_words))

        return "\n".join(formatted_lines)

    @property
    def message(self):
        """
        Get the current text message.

        Returns
        -------
        str
            The text message.
        """
        return self._message

    @message.setter
    def message(self, text):
        """
        Set the text message.

        Parameters
        ----------
        text : str
            The message to display.
        """
        self._message = text
        self.actor.set_markdown(self.get_formatted_text(text))
        self._text_changed()

    def _get_font_size(self):
        """
        Get text font size.

        Returns
        -------
        int
            Text font size.
        """
        return self.actor.font_size

    def _show_font_size(self, size):
        """
        Set text font size.

        Parameters
        ----------
        size : int
            Text font size.
        """
        self.actor.font_size = size
        self._text_changed()

    font_size = themed(_get_font_size, _show_font_size)

    def _get_font_family(self):
        """
        Get font family.

        Returns
        -------
        str
            Text font family.
        """
        return self.actor.family

    def _show_font_family(self, family):
        """
        Set font family.

        Parameters
        ----------
        family : str
            The font family.
        """
        self._family_name = family
        self.actor.family = family
        self._text_changed()

    font_family = themed(_get_font_family, _show_font_family)

    def _get_font_weight(self):
        """
        Get the font weight.

        Returns
        -------
        int
            Weight from 100 to 900, 400 being regular.
        """
        return self._font_weight

    def _show_font_weight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : int
            Weight from 100 (thin) to 900 (black). 400 is regular, 600
            semibold and 700 bold.
        """
        if not 100 <= weight <= 900:
            raise ValueError(f"font_weight must be in [100, 900], got {weight}.")
        self._font_weight = weight
        self.actor.material.weight_offset = weight - 400

    font_weight = themed(_get_font_weight, _show_font_weight)

    @property
    def justification(self):
        """
        Get text justification.

        Returns
        -------
        str
            Text justification.
        """
        return self._justification

    @justification.setter
    def justification(self, justification):
        """
        Justify text.

        Parameters
        ----------
        justification : str
            Possible values are left, center, right.
        """
        self._justification = justification
        self.update_alignment()

    @property
    def vertical_justification(self):
        """
        Get text vertical justification.

        Returns
        -------
        str
            Text vertical justification.
        """
        return self._vertical_justification

    @vertical_justification.setter
    def vertical_justification(self, vertical_justification):
        """
        Justify text vertically.

        Parameters
        ----------
        vertical_justification : str
            Possible values are top, middle, bottom.
        """
        self._vertical_justification = vertical_justification
        self.update_alignment()

    @property
    def bold(self):
        """
        Return whether the text is bold.

        Returns
        -------
        bool
            Text is bold if True.
        """
        return self._bold

    @bold.setter
    def bold(self, flag):
        """
        Bold/un-bold text.

        Parameters
        ----------
        flag : bool
            Sets text bold if True.
        """
        self._bold = flag

    @property
    def italic(self):
        """
        Return whether the text is italicised.

        Returns
        -------
        bool
            Text is italicised if True.
        """
        return self._italic

    @italic.setter
    def italic(self, flag):
        """
        Italicise/un-italicise text.

        Parameters
        ----------
        flag : bool
            Italicises text if True.
        """
        self._italic = flag

    @property
    def shadow(self):
        """
        Return whether the text has a shadow.

        Returns
        -------
        bool
            True if text has a shadow.
        """
        return self._shadow

    @shadow.setter
    def shadow(self, flag):
        """
        Set text shadow.

        Parameters
        ----------
        flag : bool
            True if text has a shadow.
        """
        self._shadow = flag

    def _get_color(self):
        """
        Get text color.

        Returns
        -------
        (float, float, float)
            Returns text color in RGB.
        """
        return self.actor.material.color[:3]

    def _show_color(self, color):
        """
        Set text color.

        Parameters
        ----------
        color : str, tuple, list or ndarray
            A hex string (``"#FF0000"``), RGB(A) values in [0, 1], or
            RGB(A) values in [0, 255]. ``None`` uses the theme color again.
            See :func:`normalize_colors`.
        """
        color = normalize_colors(color)[0]
        self.actor.material.color = np.array([*color[:3], 1.0])

    color = themed(_get_color, _show_color, reset_on_none=True)

    def _get_background_color(self):
        """
        Get the background color.

        Returns
        -------
        (float, float, float) or None
            The RGB color of the background, or None if no background exists.
        """
        if not self.have_bg:
            return None

        return self.background.color

    def _show_background_color(self, color):
        """
        Set the background color.

        Parameters
        ----------
        color : str, tuple, list, ndarray or None
            A hex string ("#FF0000"), RGB(A) in [0, 1], or RGB(A) in [0, 255].
            If None, the background is removed.
        """
        if color is None:
            # Remove background.
            self.have_bg = False
            self.background.set_visibility(False)

        else:
            self.have_bg = True
            self.background.set_visibility(True)
            self._forward(self.background, "color", color)

    background_color = themed(_get_background_color, _show_background_color)

    @property
    def dynamic_bbox(self):
        """
        Check if the bounding box is dynamic.

        Returns
        -------
        bool
            True if dynamic, False otherwise.
        """
        return self._dynamic_bbox

    @dynamic_bbox.setter
    def dynamic_bbox(self, flag):
        """
        Set the dynamic bounding box state.

        Parameters
        ----------
        flag : bool
            If True, the bounding box resizes to content.
        """
        self._dynamic_bbox = flag
        if flag:
            self.update_bounding_box()

    def update_alignment(self):
        """Update the text actor alignment within the bounding box."""
        updated_text_position = [0, 0]
        inset = self._content_inset()
        left = self.boundingbox[0] + inset[0]
        top = self.boundingbox[1] + inset[1]
        right = self.boundingbox[2] - inset[2]
        bottom = self.boundingbox[3] - inset[3]

        horiz_anchor = ""
        if self.justification.lower() == "left":
            self.actor.text_align = "left"
            horiz_anchor = "left"
            updated_text_position[0] = left
        elif self.justification.lower() == "center":
            self.actor.text_align = "center"
            horiz_anchor = "center"
            updated_text_position[0] = left + (right - left) // 2
        elif self.justification.lower() == "right":
            self.actor.text_align = "right"
            horiz_anchor = "right"
            updated_text_position[0] = right
        else:
            msg = "Text can only be justified left, center and right."
            raise ValueError(msg)

        vert_anchor = ""
        if self.vertical_justification.lower() == "top":
            vert_anchor = "top"
            updated_text_position[1] = top
        elif self.vertical_justification.lower() == "middle":
            vert_anchor = "middle"
            updated_text_position[1] = top + (bottom - top) // 2
        elif self.vertical_justification.lower() == "bottom":
            vert_anchor = "bottom"
            updated_text_position[1] = bottom
        else:
            msg = "Vertical justification must be: top, middle or bottom."
            raise ValueError(msg)

        self.actor.anchor = f"{vert_anchor}-{horiz_anchor}"

        self.set_actor_position(
            self.actor, updated_text_position, self.z_order, sub_order=1
        )

    def update_bounding_box(self, *, size=None):
        """
        Update the text bounding box and background.

        Parameters
        ----------
        size : (int, int), optional
            If provided, uses this size. Otherwise, uses the current size.
        """
        if size is None:
            size = self.size

        pos = self.get_position()
        self.boundingbox = [
            pos[0],
            pos[1],
            pos[0] + size[0],
            pos[1] + size[1],
        ]
        self.background.resize(size)
        self._bg_size = size
        self.background.set_position(pos)

        self.update_alignment()

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        self.update_bounding_box()

    def get_text_actor_size(self):
        """
        Get the rendered size of the text actor.

        Returns
        -------
        (float, float)
            The (width, height) of the rendered text.
        """
        self.actor._update_object()
        return (
            self.actor._aabb[1][0] - self.actor._aabb[0][0],
            self.actor._aabb[1][1] - self.actor._aabb[0][1],
        )

    def _get_size(self):
        """
        Get the size of the text block.

        Returns
        -------
        (float, float)
            The current size of the text block.
        """
        if self.dynamic_bbox:
            if self._layout_node.parent is not None and self._layout_managed_size:
                return self._layout_managed_size
            width, height = self.get_text_actor_size()
            style = self.layout_style
            return (width + style.inset(0), height + style.inset(1))
        else:
            return self._bg_size


class Button2D(UI):
    """
    Base class for interactive 2D Buttons.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    size : (int, int), optional
        Width and height in pixels.
    is_toggle : bool, optional
        If True, the button behaves as a toggle switch.
    """

    def __init__(self, position=(0, 0), size=(30, 30), is_toggle=False):
        """Initialize the button instance."""
        self._dims = size
        self.child = None
        self.is_toggle = is_toggle
        self.toggled = False

        super().__init__(position=position)

        self.is_hovered = False
        self.is_pressed = False
        self._enabled = True

        self.on_clicked = None

        self.on_hover = self._handle_hover
        self.on_dishover = self._handle_dishover
        self.on_left_mouse_button_pressed = self._handle_down
        self.on_left_mouse_button_released = self._handle_up

        self.resize(size)
        self.update_visual_state()

    def _reset_interaction_state(self) -> None:
        """
        Cancel transient button feedback while preserving toggle state.

        Returns
        -------
        None
            Press and hover feedback are cleared without invoking callbacks.
        """
        super()._reset_interaction_state()
        if self.is_pressed or self.is_hovered:
            self.is_pressed = False
            self.is_hovered = False
            self.update_visual_state()

    @property
    def enabled(self):
        """
        Check if the button is enabled.

        Returns
        -------
        bool
            True if interactive, False otherwise.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        """
        Set the button enabled state.

        Parameters
        ----------
        value : bool
            True to enable, False to disable.
        """
        self._enabled = bool(value)

        if not self._enabled:
            self.is_hovered = False
            self.is_pressed = False
        self.update_visual_state()

    @property
    def toggled(self):
        """
        Check if the button is toggled.

        Returns
        -------
        bool
            True if toggled, False otherwise.
        """
        return self._toggled

    @toggled.setter
    def toggled(self, value):
        """
        Set the button's toggled state.

        Parameters
        ----------
        value : bool
            True to toggle, False otherwise.
        """
        self._toggled = bool(value)
        self.update_visual_state()

    def _handle_hover(self, event):
        """
        Handle the hover on the button area.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        if not self._enabled:
            return
        self.is_hovered = True
        self.update_visual_state()

    def _handle_dishover(self, event):
        """
        Handle the dishover on the button area.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        if not self._enabled:
            return
        self.is_hovered = False
        self.is_pressed = False
        self.update_visual_state()

    def _handle_down(self, event):
        """
        Handle the pointer being pressed down.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        if not self._enabled:
            return
        self.is_pressed = True
        self.update_visual_state()

    def _handle_up(self, event):
        """
        Handle the pointer being released.

        Triggers a click action if the release occurs while the button
        is in a pressed state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        if not self._enabled:
            return
        if self.is_pressed:
            self.do_click()
        self.is_pressed = False
        self.update_visual_state()

    def do_click(self):
        """Trigger the assigned click callback and handle toggle state."""
        if self.is_toggle:
            self.toggled = not self.toggled

        if self.on_clicked:
            self.on_clicked(self)

        self.update_visual_state()

    def resolve_state_key(self, available_keys):
        """
        Determine the current visual state key based on priority.

        Two schemes are supported, chosen by the keys the subclass provides:

        * One-dimensional (default): ``'disabled'`` > ``'pressed'`` (when
          ``is_pressed`` or ``toggled``) > ``'hover'`` > ``'default'``.
        * Two-dimensional toggle icons: when a ``'selected'`` key is present,
          the selected dimension (``toggled``) is combined with interaction
          feedback, yielding ``'default'``, ``'selected'``, ``'pressed'`` or
          ``'pressed-selected'``. This suits checkbox/radio state icons.

        Parameters
        ----------
        available_keys : list or set
            The keys available in the subclass (e.g., in a color map).

        Returns
        -------
        str
            The key representing the highest priority active state.
        """
        if not self.enabled:
            return "disabled" if "disabled" in available_keys else "default"

        # Two-dimensional toggle icons: selected state x interaction feedback.
        if "selected" in available_keys:
            selected = self.is_toggle and self.toggled
            active = self.is_pressed or self.is_hovered
            if active and selected and "pressed-selected" in available_keys:
                return "pressed-selected"
            if active and not selected and "pressed" in available_keys:
                return "pressed"
            return "selected" if selected else "default"

        is_active = self.is_pressed or (self.is_toggle and self.toggled)

        if is_active and "pressed" in available_keys:
            return "pressed"

        if self.is_hovered and "hover" in available_keys:
            return "hover"

        return "default"

    def update_visual_state(self):
        """Update the visual appearance of the button."""
        pass

    def _get_size(self):
        """
        Get the current size of the button.

        Returns
        -------
        (float, float)
            The current (width, height) of the button in pixels.
        """
        if self.child and hasattr(self.child, "_get_size"):
            return self.child._get_size()
        return self._dims

    def resize(self, size):
        """
        Resize the button and its child components.

        Parameters
        ----------
        size : (float, float)
            New width and height in pixels.
        """
        self._dims = size
        if self.child:
            if isinstance(self.child, UI):
                self.child.resize(size)
            elif hasattr(self.child, "geometry"):
                self.child.geometry = plane_geometry(width=size[0], height=size[1])
        self._update_actors_position()

    def _update_actors_position(self):
        """Update the position of the internal actors."""
        if self.child:
            if isinstance(self.child, UI):
                self.child.set_position(self.get_position())
            else:
                pos = self.get_position(x_anchor=Anchor.CENTER, y_anchor=Anchor.CENTER)
                self.set_actor_position(self.child, pos, self.z_order)

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            List containing the child actor(s).
        """
        if self.child:
            if not isinstance(self.child, UI):
                return [self.child]
            else:
                self._children.append(self.child)
        return []


class Slider2D(UI):
    """
    Base class for interactive 2D Sliders.

    Parameters
    ----------
    position : (float, float), optional
        Absolute coordinates (x, y) for placement.
    initial_value : float, optional
        The starting value of the slider.
    min_value : float, optional
        The minimum value of the slider range.
    max_value : float, optional
        The maximum value of the slider range.
    handle_inner_radius : int, optional
        The inner radius for disk-shaped handles.
    handle_outer_radius : int, optional
        The outer radius for disk-shaped handles.
    handle_side : int, optional
        The side length for square-shaped handles.
    font_size : int, optional
        The font size for the value label.
        Defaults to the theme item of the same name.
    text_template : str or callable, optional
        A formatting string or callable for the label.
    shape : str, optional
        The handle shape: disk or square.
    z_order : int, optional
        The stacking priority.
    """

    default_color = themed(
        item="handle_color",
        doc="Color of the handle at rest (theme item ``handle_color``).",
    )
    active_color = themed(
        item="handle_active_color",
        doc="Color of the handle while dragged (theme item ``handle_active_color``).",
    )
    hover_color = themed(
        item="handle_hover_color",
        doc="Color of the handle under the pointer, or None to keep the rest color "
        "(theme item ``handle_hover_color``).",
    )

    def __init__(
        self,
        *,
        position=(0, 0),
        initial_value=50,
        min_value=0,
        max_value=100,
        handle_inner_radius=0,
        handle_outer_radius=10,
        handle_side=20,
        font_size=None,
        text_template="{value:.1f} ({ratio:.0%})",
        shape="disk",
        z_order=0,
    ):
        """
        Initialize the 2D slider.

        Parameters
        ----------
        position : (float, float), optional
            Absolute coordinates (x, y) for placement.
        initial_value : float, optional
            The starting value of the slider.
        min_value : float, optional
            The minimum value of the slider range.
        max_value : float, optional
            The maximum value of the slider range.
        handle_inner_radius : int, optional
            The inner radius for disk-shaped handles.
        handle_outer_radius : int, optional
            The outer radius for disk-shaped handles.
        handle_side : int, optional
            The side length for square-shaped handles.
        font_size : int, optional
            The font size for the value label.
            Defaults to the theme item of the same name.
        text_template : str or callable, optional
            A formatting string or callable for the label.
        shape : str, optional
            The handle shape: disk or square.
        z_order : int, optional
            The stacking priority.
        """
        font_size = self._theme_value("font_size", font_size)
        for item in ("handle_color", "handle_active_color", "handle_hover_color"):
            self._set_style(item, self._theme_value(item))
        self._handle_hovered = False
        self._dragging = False
        self._value = initial_value
        range_val = max_value - min_value
        self._ratio = (initial_value - min_value) / range_val if range_val != 0 else 0
        self._min_value = min_value
        self._max_value = max_value
        self.text_template = text_template

        self._handle_inner_radius = handle_inner_radius
        self._handle_outer_radius = handle_outer_radius
        self._handle_side = handle_side
        self._font_size = font_size
        self.shape = shape

        self.on_change = lambda ui: None
        self.on_value_changed = lambda ui: None
        self.on_moving_slider = lambda ui: None

        self.track = None
        self.handle = None
        self.text = None

        super().__init__(position=position, z_order=z_order)

    @property
    def value(self):
        """
        Get the current numeric value of the slider.

        Returns
        -------
        float
            The slider value.
        """
        return self._value

    @value.setter
    def value(self, val):
        """
        Set the slider numeric value.

        Parameters
        ----------
        val : float
            New numeric value. Will be clamped to [min_value, max_value].
        """
        val = np.clip(val, self.min_value, self.max_value)
        self._value = val
        range_val = self.max_value - self.min_value
        self._ratio = (val - self.min_value) / range_val if range_val != 0 else 0
        self._update_handle_position()
        self.on_value_changed(self)
        self.on_change(self)

    @property
    def ratio(self):
        """
        Get the current normalized ratio (0 to 1).

        Returns
        -------
        float
            The slider ratio.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, r):
        """
        Set the slider ratio.

        Parameters
        ----------
        r : float
            New ratio value. Will be clamped to [0, 1].
        """
        self._ratio = np.clip(r, 0, 1)
        self._value = self.min_value + self._ratio * (self.max_value - self.min_value)
        self._update_handle_position()
        self.on_change(self)

    @property
    def min_value(self):
        """
        Get the minimum value of the slider.

        Returns
        -------
        float
            The minimum value.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, val):
        """
        Set the minimum value of the slider.

        Parameters
        ----------
        val : float
            The minimum value.
        """
        self._min_value = val
        self.value = self._value

    @property
    def max_value(self):
        """
        Get the maximum value of the slider.

        Returns
        -------
        float
            The maximum value.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, val):
        """
        Set the maximum value of the slider.

        Parameters
        ----------
        val : float
            The maximum value.
        """
        self._max_value = val
        self.value = self._value

    def track_click_callback(self, event):
        """
        Handle mouse click events on the slider track.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self.handle_move_callback(event)

    @abc.abstractmethod
    def handle_move_callback(self, event):
        """
        Handle mouse drag events to update the slider state.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        pass

    def handle_release_callback(self, event):
        """
        Handle the release of the mouse button.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._dragging = False
        self._update_handle_color()

    def _handle_hover(self, event):
        """
        Show the hover color while the pointer is over the handle.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._handle_hovered = True
        if not getattr(event, "buttons", ()):
            self._dragging = False
        self._update_handle_color()

    def _handle_dishover(self, event):
        """
        Restore the handle color when the pointer leaves the handle.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event.
        """
        self._handle_hovered = False
        if not getattr(event, "buttons", ()):
            self._dragging = False
        self._update_handle_color()

    def _reset_interaction_state(self):
        """
        Cancel the drag and hover feedback of the handle.

        Returns
        -------
        None
            State is updated in place without dispatching callbacks.
        """
        super()._reset_interaction_state()
        self._dragging = False
        self._handle_hovered = False
        if self.handle is not None:
            self._update_handle_color()

    def _update_handle_color(self):
        """Color the handle for its dragged, hovered or default state."""
        if self._dragging:
            self.handle._push_style("color", self.active_color)
        elif self._handle_hovered and self.hover_color is not None:
            self.handle._push_style("color", self.hover_color)
        else:
            self.handle._push_style("color", self.default_color)

    def _refresh_style(self):
        """Show the handle colors and the font size of the label."""
        self._font_size = self._theme_value("font_size")
        if self.text.font_size != self._font_size:
            self.text._push_style("font_size", self._font_size)
        self._update_handle_color()

    @abc.abstractmethod
    def _update_handle_position(self):
        """Update the position of the track and handle actors."""
        pass

    def _get_actors(self):
        """
        Get the actors composing this UI component.

        Returns
        -------
        list
            Empty list, as child UI components are added directly when adding parent.
        """
        return []

    def format_text(self):
        """
        Return formatted text to display along the slider.

        Returns
        -------
        str
            The formatted text.
        """
        if callable(self.text_template):
            return self.text_template(self)

        context = {"value": self.value, "ratio": self.ratio}
        if hasattr(self, "angle"):
            context["angle"] = np.rad2deg(self.angle)
        return self.text_template.format(**context)

    def _make_handle(self):
        """
        Create the handle for the slider shape.

        Returns
        -------
        UI
            A :class:`Disk2D` for ``"disk"`` or a :class:`Rectangle2D` for
            ``"square"``.
        """
        if self.shape == "disk":
            return Disk2D(
                outer_radius=self._handle_outer_radius,
                inner_radius=self._handle_inner_radius,
            )
        if self.shape == "square":
            return Rectangle2D(size=(self._handle_side, self._handle_side))
        raise ValueError("shape must be 'disk' or 'square'")

    def _setup(self):
        """Set up the common slider components."""
        self.handle = self._make_handle()

        self.handle.z_order = self.z_order + 1
        self.handle.on_hover = self._handle_hover
        self.handle.on_dishover = self._handle_dishover

        self.text = TextBlock2D(
            justification="center",
            vertical_justification="middle",
            dynamic_bbox=True,
            font_size=self._font_size,
        )
        self.text.z_order = self.z_order + 2

        self._children.extend([self.handle, self.text])


@register_wgpu_render_function(Mesh, RoundedRectangleMaterial)
def register_rounded_rect_render_function(wobject):
    """
    Register rounded rectangle render function.

    Parameters
    ----------
    wobject : Mesh
        The mesh object to be rendered.

    Returns
    -------
    tuple
        A tuple containing the rounded rectangle shader.
    """
    return (RoundedRectangleShader(wobject),)
