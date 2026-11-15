"""Themes that provide the default look of UI elements."""

from contextlib import contextmanager
import copy
import json
import numbers

import numpy as np

from fury.ui.context import UIContext


class Theme:
    """
    Default colors and fonts for UI elements.

    Items are stored per UI class name. Looking up an item for a UI class
    checks that class in this theme and then in the ``base`` theme, then does
    the same for each base class in method resolution order. An item set for
    a specific class therefore wins over one set for a more general class,
    whichever theme defines it.

    Parameters
    ----------
    name : str, optional
        Name of the theme.
    items : dict, optional
        Mapping of UI class name to a mapping of item name to value, for
        example ``{"Panel2D": {"color": (0.1, 0.1, 0.1)}}``.
    base : Theme, optional
        Theme consulted for items this theme does not define.
    density : str, optional
        Spacing and size of the elements: ``"comfortable"`` uses the items as
        they are, and ``"compact"`` scales the metric items down. Items are
        always stored as given and scaled when they are looked up, so a
        compact theme scales the items of its base and its own items alike:
        font sizes by 12/13, and ``padding``, ``gap``, ``header_gap``,
        ``text_padding``, corner radii, the slider ``line_width`` and
        ``handle_outer_radius`` by 0.8. None uses the density of the
        ``base`` theme, or ``"comfortable"``.
    """

    def __init__(self, *, name="custom", items=None, base=None, density=None):
        """
        Initialize the theme.

        Parameters
        ----------
        name : str, optional
            Name of the theme.
        items : dict, optional
            Mapping of UI class name to a mapping of item name to value.
        base : Theme, optional
            Theme consulted for items this theme does not define.
        density : str, optional
            Spacing and size of the elements, ``"comfortable"`` or
            ``"compact"``, or None to use the density of ``base``.
        """
        if density is not None and density not in _DENSITIES:
            available = ", ".join(_DENSITIES)
            msg = f"Unknown density {density!r}. Available densities: {available}."
            raise ValueError(msg)
        self.name = name
        self._base = base
        self._density = density
        self._items = {}
        for type_name, type_items in (items or {}).items():
            for item, value in type_items.items():
                self.set(type_name, item, value)

    @property
    def base(self):
        """
        Get the theme consulted for items this theme does not define.

        Returns
        -------
        Theme or None
            The base theme.
        """
        return self._base

    @property
    def density(self):
        """
        Get the density of the theme.

        Returns
        -------
        str
            The density given to this theme, else the density of its base,
            else ``"comfortable"``.
        """
        theme = self
        while theme is not None:
            if theme._density is not None:
                return theme._density
            theme = theme._base
        return "comfortable"

    def set(self, type_name, item, value):
        """
        Set an item for a UI class.

        Parameters
        ----------
        type_name : str or type
            UI class, or its name, the item applies to. Subclasses inherit it.
        item : str
            Name of the item, such as ``"color"``.
        value : object
            Value of the item. A copy is stored, so later changes to the
            given object do not affect the theme. It is stored unscaled;
            :meth:`get` scales metric items for the theme density.
        """
        self._items.setdefault(_type_name(type_name), {})[item] = copy.deepcopy(value)

    def get(self, ui_type, item):
        """
        Get the value of an item for a UI class.

        Parameters
        ----------
        ui_type : type, UI or str
            UI class, an instance of it, or a class name. For a class or an
            instance, base classes are checked after the class itself, in
            this theme and its base.
        item : str
            Name of the item.

        Returns
        -------
        object
            A copy of the item's value, scaled for the theme density when it
            is a metric item.

        Raises
        ------
        KeyError
            If neither this theme nor its base defines the item.
        """
        names = _lookup_names(ui_type)
        for name in names:
            theme = self
            while theme is not None:
                type_items = theme._items.get(name)
                if type_items is not None and item in type_items:
                    return _scale_item(item, type_items[item], self.density)
                theme = theme._base
        msg = f"Theme {self.name!r} has no item {item!r} for {names[0]!r}."
        raise KeyError(msg)

    def has(self, ui_type, item):
        """
        Check whether an item is defined for a UI class.

        Parameters
        ----------
        ui_type : type, UI or str
            UI class, an instance of it, or a class name.
        item : str
            Name of the item.

        Returns
        -------
        bool
            True if :meth:`get` would find the item.
        """
        try:
            self.get(ui_type, item)
        except KeyError:
            return False
        return True

    def derive(self, *, name=None, items=None, density=None):
        """
        Create a theme that overrides some items of this one.

        Parameters
        ----------
        name : str, optional
            Name of the new theme.
        items : dict, optional
            Items to override, in the same form as the ``items`` parameter.
        density : str, optional
            Density of the new theme, ``"comfortable"`` or ``"compact"``.
            None keeps the density of this theme. The ``items`` are stored as
            given and, like the inherited ones, scaled for the density when
            they are looked up.

        Returns
        -------
        Theme
            A theme whose base is this theme.
        """
        return Theme(name=name or "custom", items=items, base=self, density=density)

    def to_dict(self):
        """
        Get a copy of all items, including those inherited from the base.

        A theme created from the result with :meth:`from_dict` returns the
        same value as this theme for every lookup. Items are stored unscaled,
        with the theme ``density``.

        Returns
        -------
        dict
            Mapping with the theme ``name``, its ``density`` and its
            ``items``.
        """
        chain = []
        theme = self
        while theme is not None:
            chain.append(theme)
            theme = theme._base

        items = {}
        for theme in reversed(chain):
            for type_name, type_items in theme._items.items():
                items.setdefault(type_name, {}).update(type_items)
        return {
            "name": self.name,
            "density": self.density,
            "items": copy.deepcopy(items),
        }

    @classmethod
    def from_dict(cls, data, *, base=None):
        """
        Create a theme from a mapping made by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Mapping with an optional ``name``, an optional ``density`` and the
            ``items``.
        base : Theme, optional
            Theme consulted for items the mapping does not define.

        Returns
        -------
        Theme
            The new theme. Without a ``density`` in ``data``, it uses the
            density of ``base``.
        """
        return cls(
            name=data.get("name", "custom"),
            items=data.get("items"),
            base=base,
            density=data.get("density"),
        )

    def save(self, path):
        """
        Save the theme, with inherited items, as a JSON file.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.
        """
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, default=_json_value)

    @classmethod
    def load(cls, path, *, base=None):
        """
        Load a theme from a JSON file.

        Parameters
        ----------
        path : str or os.PathLike
            File written by :meth:`save`.
        base : Theme, optional
            Theme consulted for items the file does not define.

        Returns
        -------
        Theme
            The loaded theme.
        """
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream), base=base)

    def __repr__(self):
        """
        Get a short description of the theme.

        Returns
        -------
        str
            Description with the theme name.
        """
        return f"Theme(name={self.name!r})"


def _type_name(type_name):
    """
    Get the name used to store items for a UI class.

    Parameters
    ----------
    type_name : str or type
        UI class or its name.

    Returns
    -------
    str
        The class name.
    """
    if isinstance(type_name, type):
        return type_name.__name__
    return str(type_name)


def _lookup_names(ui_type):
    """
    Get the class names checked, in order, when looking up an item.

    Parameters
    ----------
    ui_type : type, UI or str
        UI class, an instance of it, or a class name.

    Returns
    -------
    list of str
        The class name followed by its base class names.
    """
    if isinstance(ui_type, str):
        return [ui_type]
    if not isinstance(ui_type, type):
        ui_type = type(ui_type)
    return [cls.__name__ for cls in ui_type.__mro__ if cls is not object]


_DENSITIES = {"comfortable": (1.0, 1.0), "compact": (12 / 13, 0.8)}

_METRIC_ITEMS = frozenset(
    {
        "padding",
        "gap",
        "header_gap",
        "text_padding",
        "corner_radius",
        "handle_corner_radius",
        "line_width",
        "handle_outer_radius",
    }
)


def _scale_item(item, value, density):
    """
    Get a copy of a theme value, scaled for a density if it is a metric.

    Parameters
    ----------
    item : str
        Name of the item. Items whose name ends with ``font_size`` are
        scaled by the font factor of the density, the spacing, radius and
        slider size items by its metric factor, and others are not scaled.
    value : object
        Value of the item.
    density : str
        ``"comfortable"`` or ``"compact"``.

    Returns
    -------
    object
        The copied, possibly scaled, value.
    """
    font_scale, metric_scale = _DENSITIES[density]
    if item.endswith("font_size"):
        scale = font_scale
    elif item in _METRIC_ITEMS:
        scale = metric_scale
    else:
        scale = 1.0
    if scale == 1.0:
        return copy.deepcopy(value)
    return _scale_value(value, scale)


def _scale_value(value, scale):
    """
    Scale a number, or each number of a sequence or array, for a density.

    Parameters
    ----------
    value : object
        Number, or list, tuple or array of numbers. Other values are returned
        as they are.
    scale : float
        Scale factor.

    Returns
    -------
    object
        The scaled value. Zero stays zero. A number of at least 1 is rounded
        to the nearest whole pixel with :func:`round` (halves go to the even
        number) and stays at least 1; a smaller number is scaled without
        rounding. A float stays a float and an integer an integer.
    """
    if isinstance(value, np.ndarray):
        return np.asarray(_scale_value(value.tolist(), scale))
    if isinstance(value, (list, tuple)):
        return type(value)(_scale_value(item, scale) for item in value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value == 0:
        return copy.deepcopy(value)
    if abs(value) < 1:
        return float(value) * scale
    scaled = round(value * scale) or int(np.sign(value))
    return float(scaled) if isinstance(value, float) else scaled


def _plain_value(value):
    """
    Convert a theme value to plain Python values that compare with ``==``.

    Parameters
    ----------
    value : object
        Theme value, possibly containing arrays or tuples.

    Returns
    -------
    object
        The value with arrays and tuples as lists and NumPy scalars as Python
        numbers.
    """
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _plain_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _comparable_items(theme):
    """
    Get all items of a theme in a form that compares with ``==``.

    Parameters
    ----------
    theme : Theme
        The theme.

    Returns
    -------
    dict
        The density and the items with inherited ones, without the theme
        name.
    """
    data = theme.to_dict()
    return _plain_value({"density": data["density"], "items": data["items"]})


def _json_value(value):
    """
    Convert a value that JSON cannot store directly, such as an array.

    Parameters
    ----------
    value : object
        Value to convert.

    Returns
    -------
    list
        The value as a list.

    Raises
    ------
    TypeError
        If the value cannot be converted to a list.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    msg = f"Theme value {value!r} cannot be saved as JSON."
    raise TypeError(msg)


def _classic_items():
    """
    Get the items that reproduce the original look of FURY UI elements.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    return {
        "UI": {
            "font_family": "Arial",
            "corner_radius": 0,
            "border_color": (0, 0, 0),
            "border_width": 0,
            "shadow_color": (0, 0, 0, 0),
            "shadow_offset": (0, 0),
            "shadow_blur": 0,
            "shadow_spread": 0,
            "padding": 0,
            "gap": 0,
        },
        "Rectangle2D": {"color": (1, 1, 1)},
        "RoundedRectangle2D": {"color": (1, 1, 1)},
        "Disk2D": {"color": (1, 1, 1)},
        "TextBlock2D": {
            "color": (1, 1, 1),
            "muted_color": (0.6, 0.6, 0.6),
            "font_size": 18,
            "font_weight": 400,
        },
        "Panel2D": {
            "color": (0.1, 0.1, 0.1),
            "opacity": 0.7,
            "border_color": (1, 1, 1),
            "border_width": 0,
        },
        "TabPanel2D": {"color": (0.5, 0.5, 0.5), "text_color": (0, 0, 0)},
        "TabUI": {
            "active_color": (1, 1, 1),
            "inactive_color": (0.5, 0.5, 0.5),
            "font_size": 18,
        },
        "TexturedButton2D": {"icon_color": None},
        "Loader2D": {
            "overlay_color": (0, 0, 0),
            "overlay_opacity": 0.5,
            "text_color": (1, 1, 1),
            "font_size": 18,
        },
        "TextButton2D": {
            "font_size": 25,
            "font_weight": 400,
            "text_color": (0, 0, 0),
            "bg_color": (1, 1, 1),
            "states": {
                "default": (1, 1, 1),
                "hover": (0.9, 0.9, 0.9),
                "pressed": (0.5, 0.5, 0.5),
                "disabled": (0.2, 0.2, 0.2),
            },
            "secondary_text_color": (0, 0, 0),
            "secondary_border_color": (0, 0, 0),
            "secondary_states": {
                "default": (1, 1, 1),
                "hover": (0.9, 0.9, 0.9),
                "pressed": (0.5, 0.5, 0.5),
                "disabled": (0.2, 0.2, 0.2),
            },
        },
        "Slider2D": {
            "font_size": 16,
            "handle_color": (1, 1, 1),
            "handle_active_color": (0, 0, 1),
            "handle_hover_color": None,
            "track_color": (1, 0, 0),
            "layout": "classic",
            "line_width": 5,
            "handle_outer_radius": 10,
            "header_gap": 6,
            "fill_color": (1, 0, 0),
            "label_color": (1, 1, 1),
            "value_color": (1, 1, 1),
            "handle_corner_radius": 0,
            "handle_border_color": (0, 0, 0),
            "handle_border_width": 0,
            "handle_shadow_color": (0, 0, 0, 0),
            "handle_shadow_offset": (0, 0),
            "handle_shadow_blur": 0,
            "handle_shadow_spread": 0,
        },
        "LineDoubleSlider2D": {
            "font_size": 16,
            "handle_color": (1, 1, 1),
            "handle_active_color": (0, 0, 1),
            "handle_hover_color": None,
            "track_color": (1, 0, 0),
        },
        "RangeSlider": {"font_size": 16},
        "PlaybackPanel": {
            "panel_color": (1, 1, 1),
            "border_color": (0, 0.3, 0),
            "border_width": 2,
            "time_color": (1, 1, 1),
            "speed_color": (0.2, 0.2, 0.2),
            "progress_color": (1, 0, 0),
            "icon_color": None,
        },
        "TextBox2D": {
            "color": (0, 0, 0),
            "bg_color": (1, 1, 1),
            "font_size": 18,
            "focus_border_color": (0, 0, 0),
            "focus_border_width": 0,
            "text_padding": 0,
        },
        "ButtonGroup": {
            "font_size": 18,
            "text_color": (1, 1, 1),
            "indicator": "icon",
            "indicator_color": (1, 1, 1),
            "indicator_on_color": (1, 1, 1),
            "indicator_border_color": (0, 0, 0),
            "indicator_hover_border_color": (0, 0, 0),
            "indicator_disabled_opacity": 1.0,
            "mark_color": (0, 0, 0),
        },
        "ComboBox2D": {
            "selection_text_color": (0, 0, 0),
            "selection_bg_color": (1, 1, 1),
            "menu_text_color": (0.2, 0.2, 0.2),
            "selected_color": (0.9, 0.6, 0.6),
            "unselected_color": (0.6, 0.6, 0.6),
            "scroll_bar_active_color": (0.6, 0.2, 0.2),
            "scroll_bar_inactive_color": (0.9, 0.0, 0.0),
            "font_size": 20,
            "icon_color": None,
        },
        "ListBox2D": {
            "font_size": 20,
            "panel_color": (1, 1, 1),
            "text_color": (0.2, 0.2, 0.2),
            "selected_color": (0.9, 0.6, 0.6),
            "unselected_color": (0.6, 0.6, 0.6),
            "scroll_bar_active_color": (0.6, 0.2, 0.2),
            "scroll_bar_inactive_color": (0.9, 0.0, 0.0),
        },
        "ListBoxItem2D": {
            "text_color": (1.0, 0.0, 0.0),
            "selected_color": (0.4, 0.4, 0.4),
            "unselected_color": (0.9, 0.9, 0.9),
            "hover_color": None,
        },
        "TreeNode2D": {
            "color": (0, 0, 0),
            "selected_color": (0.8, 0.3, 0.3),
            "icon_color": None,
        },
        "Tree2D": {
            "color": (0, 0, 0),
            "scroll_bar_inactive_color": (0.6, 0.2, 0.2),
            "scroll_bar_active_color": (0.9, 0.0, 0.0),
        },
        "FileMenu2D": {
            "font_size": 20,
            "directory_color": (0, 0.6, 0),
            "file_color": (0, 0, 0.7),
        },
        "Card2D": {
            "bg_color": (0.5, 0.5, 0.5),
            "title_color": (0.0, 0.0, 0.0),
            "body_color": (0.0, 0.0, 0.0),
            "border_color": (1.0, 1.0, 1.0),
            "border_width": 0,
            "title_font_weight": 400,
        },
        "SpinBox": {"panel_color": (1, 1, 1), "icon_color": None},
    }


GRAPHITE_PALETTE = {
    "background": "#1C1D20",
    "border": "#2C2E33",
    "text": "#ECEDEF",
    "muted": "#8D9097",
    "accent": "#F2F3F5",
    "on_accent": "#141518",
    "fill": "#26282C",
    "hover": "#FFFFFF",
    "pressed": "#C9CBD0",
    "selected": "#2C2E33",
    "shadow_color": (0, 0, 0, 0.35),
    "shadow_offset": (0, 10),
    "shadow_blur": 30,
}

GRAPHITE_LIGHT_PALETTE = {
    "background": "#FFFFFF",
    "border": "#E4E4E7",
    "text": "#18181B",
    "muted": "#71717A",
    "accent": "#18181B",
    "on_accent": "#FFFFFF",
    "fill": "#F4F4F5",
    "hover": "#27272A",
    "pressed": "#3F3F46",
    "selected": "#E4E4E7",
    "shadow_color": (16 / 255, 24 / 255, 40 / 255, 0.08),
    "shadow_offset": (0, 8),
    "shadow_blur": 24,
}


def _merge_items(base, overrides):
    """
    Merge theme items, class by class.

    Parameters
    ----------
    base : dict
        Mapping of UI class name to items.
    overrides : dict
        Items that replace or extend those of ``base``.

    Returns
    -------
    dict
        The merged mapping.
    """
    items = copy.deepcopy(base)
    for type_name, type_items in overrides.items():
        items.setdefault(type_name, {}).update(type_items)
    return items


def _graphite_items(palette, *, panel_radius=10, control_radius=6, panel_opacity=1.0):
    """
    Get the items of a Graphite theme for a palette.

    Graphite is a neutral, monochrome design: flat panels with a hairline
    border, rounded corners and a soft shadow, a single light (or dark)
    accent and semibold titles. Items it does not change come from the
    classic theme. The other built-in themes share its structure.

    Parameters
    ----------
    palette : dict
        Colors of the theme, like ``GRAPHITE_PALETTE``.
    panel_radius : float, optional
        Corner radius of panels, in pixels.
    control_radius : float, optional
        Corner radius of buttons, text boxes and combo boxes, in pixels. A
        radius larger than half the control height draws a pill.
    panel_opacity : float, optional
        Opacity of panels, in [0, 1].

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    bg = palette["background"]
    border = palette["border"]
    text = palette["text"]
    muted = palette["muted"]
    accent = palette["accent"]
    on_accent = palette["on_accent"]
    fill = palette["fill"]
    selected = palette["selected"]
    control_box = {
        "corner_radius": control_radius,
        "border_color": border,
        "border_width": 1,
    }
    return _merge_items(
        _classic_items(),
        {
            "UI": {"font_family": "Noto Sans", "padding": 12, "gap": 8},
            "TextBlock2D": {
                "color": text,
                "muted_color": muted,
                "font_size": 13,
                "font_weight": 400,
            },
            "Panel2D": {
                "color": bg,
                "opacity": panel_opacity,
                "border_color": border,
                "border_width": 1,
                "corner_radius": panel_radius,
                "shadow_color": palette["shadow_color"],
                "shadow_offset": palette["shadow_offset"],
                "shadow_blur": palette["shadow_blur"],
            },
            "TabPanel2D": {"color": fill, "text_color": text},
            "TabUI": {
                "active_color": bg,
                "inactive_color": fill,
                "font_size": 13,
            },
            "Loader2D": {"text_color": text, "font_size": 13},
            "TextButton2D": {
                "font_size": 13,
                "font_weight": 600,
                "text_color": on_accent,
                "bg_color": accent,
                "corner_radius": control_radius,
                "border_color": accent,
                "border_width": 1,
                "states": {
                    "default": {"color": accent},
                    "hover": {"color": palette["hover"]},
                    "pressed": {"color": palette["pressed"]},
                    "disabled": {
                        "color": fill,
                        "text_color": muted,
                        "border_color": border,
                    },
                },
                "secondary_text_color": text,
                "secondary_border_color": border,
                "secondary_states": {
                    "default": {"color": bg},
                    "hover": {"color": fill},
                    "pressed": {"color": border},
                    "disabled": {"color": bg, "text_color": muted},
                },
            },
            "Slider2D": {
                "font_size": 13,
                "handle_color": text,
                "handle_active_color": accent,
                "handle_hover_color": palette["hover"],
                "track_color": border,
                "layout": "inline",
                "line_width": 4,
                "handle_outer_radius": 7,
                "header_gap": 8,
                "fill_color": accent,
                "label_color": text,
                "value_color": muted,
                "handle_shadow_color": (0, 0, 0, 0.3),
                "handle_shadow_offset": (0, 1),
                "handle_shadow_blur": 3,
            },
            "LineDoubleSlider2D": {
                "font_size": 13,
                "handle_color": text,
                "handle_active_color": accent,
                "handle_hover_color": palette["hover"],
                "track_color": border,
            },
            "RangeSlider": {"font_size": 13},
            "PlaybackPanel": {
                "panel_color": bg,
                "border_color": border,
                "border_width": 0,
                "time_color": text,
                "speed_color": muted,
                "progress_color": accent,
                "icon_color": text,
            },
            "TextBox2D": {
                "color": text,
                "bg_color": fill,
                "font_size": 13,
                "focus_border_color": muted,
                "focus_border_width": 2,
                "text_padding": (0, 8),
                **control_box,
            },
            "ButtonGroup": {
                "font_size": 13,
                "text_color": text,
                "indicator": "box",
                "indicator_color": fill,
                "indicator_on_color": accent,
                "indicator_border_color": muted,
                "indicator_hover_border_color": text,
                "indicator_disabled_opacity": 0.4,
                "mark_color": on_accent,
            },
            "ComboBox2D": {
                "selection_text_color": text,
                "selection_bg_color": fill,
                "menu_text_color": text,
                "selected_color": selected,
                "unselected_color": bg,
                "scroll_bar_active_color": text,
                "scroll_bar_inactive_color": muted,
                "font_size": 13,
                "icon_color": text,
                **control_box,
            },
            "ListBox2D": {
                "font_size": 13,
                "panel_color": bg,
                "text_color": text,
                "selected_color": selected,
                "unselected_color": bg,
                "scroll_bar_active_color": text,
                "scroll_bar_inactive_color": muted,
            },
            "ListBoxItem2D": {
                "text_color": text,
                "selected_color": selected,
                "unselected_color": bg,
                "hover_color": fill,
                "corner_radius": min(4, control_radius),
            },
            "TreeNode2D": {
                "color": fill,
                "selected_color": selected,
                "icon_color": text,
            },
            "Tree2D": {
                "color": bg,
                "scroll_bar_inactive_color": muted,
                "scroll_bar_active_color": text,
            },
            "FileMenu2D": {
                "font_size": 13,
                "directory_color": text,
                "file_color": muted,
            },
            "Card2D": {
                "bg_color": bg,
                "title_color": text,
                "body_color": muted,
                "border_color": border,
                "title_font_weight": 600,
            },
            "SpinBox": {"panel_color": bg, "icon_color": text},
        },
    )


def _graphite_dark_items():
    """
    Get the items of the dark Graphite theme.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    return _graphite_items(GRAPHITE_PALETTE)


def _graphite_light_items():
    """
    Get the items of the light Graphite theme.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    return _graphite_items(GRAPHITE_LIGHT_PALETTE)


NO_SHADOW = {"shadow_color": (0, 0, 0, 0), "shadow_offset": (0, 0), "shadow_blur": 0}

HAIRLINE_PALETTE = {
    "background": "#0C0D0F",
    "border": "#38393B",
    "text": "#EDEDED",
    "muted": "#9A9A9A",
    "accent": "#FFFFFF",
    "on_accent": "#0C0D0F",
    "fill": "#101215",
    "hover": "#D9D9D9",
    "pressed": "#B3B3B3",
    "selected": "#38393B",
    **NO_SHADOW,
}

SOFT_PILL_PALETTE = {
    "background": "#1E1F24",
    "border": "#2D2F36",
    "text": "#F1F1F4",
    "muted": "#9597A1",
    "accent": "#8B7CFF",
    "on_accent": "#FFFFFF",
    "fill": "#2A2C33",
    "hover": "#A396FF",
    "pressed": "#7464E6",
    "selected": "#2D2F36",
    "shadow_color": (0, 0, 0, 0.4),
    "shadow_offset": (0, 12),
    "shadow_blur": 32,
}

FROST_PALETTE = {
    "background": "#FFFFFF",
    "border": "#E3E5E9",
    "text": "#0F172A",
    "muted": "#64748B",
    "accent": "#2563EB",
    "on_accent": "#FFFFFF",
    "fill": "#F1F2F4",
    "hover": "#3B82F6",
    "pressed": "#1D4ED8",
    "selected": "#E3E5E9",
    "shadow_color": (15 / 255, 23 / 255, 42 / 255, 0.10),
    "shadow_offset": (0, 12),
    "shadow_blur": 32,
}

EMBER_PALETTE = {
    "background": "#1A1716",
    "border": "#2E2926",
    "text": "#F2ECE8",
    "muted": "#9C928B",
    "accent": "#FF7A45",
    "on_accent": "#1A0A03",
    "fill": "#241F1D",
    "hover": "#FF9466",
    "pressed": "#D9622F",
    "selected": "#2E2926",
    "shadow_color": (0, 0, 0, 0.35),
    "shadow_offset": (0, 10),
    "shadow_blur": 30,
}


def _hairline_items():
    """
    Get the items of the Hairline theme.

    Hairline is Graphite reduced to lines: square corners, 1 pixel borders,
    no shadow, a translucent panel and a 1 pixel slider track with a small
    flat handle, the lightest look over a scene. It differs from its design
    where theme items cannot express it: the slider handle is a small disk
    instead of a tick with tick marks, titles and button labels keep their
    case instead of letter-spaced capitals, check boxes keep their small
    corner radius, choices are radio buttons rather than an underlined
    segmented control, and fills the design leaves transparent use the
    panel color over a dark scene.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    palette = HAIRLINE_PALETTE
    return _merge_items(
        _graphite_items(palette, panel_radius=0, control_radius=0, panel_opacity=0.72),
        {
            "TextButton2D": {"font_size": 12},
            "Slider2D": {
                "track_color": palette["muted"],
                "line_width": 1,
                "handle_outer_radius": 5,
                **{f"handle_{name}": value for name, value in NO_SHADOW.items()},
            },
            "LineDoubleSlider2D": {"track_color": palette["muted"]},
        },
    )


def _soft_pill_items():
    """
    Get the items of the Soft Pill theme.

    Soft Pill rounds everything: panels with a large radius, pill shaped
    controls, a violet accent and a thick slider whose accent handle has a
    light ring.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    palette = SOFT_PILL_PALETTE
    return _merge_items(
        _graphite_items(palette, panel_radius=18, control_radius=999),
        {
            "Slider2D": {
                "handle_color": palette["accent"],
                "handle_active_color": palette["pressed"],
                "line_width": 6,
                "handle_outer_radius": 8,
                "handle_border_color": palette["text"],
                "handle_border_width": 2,
            },
        },
    )


def _frost_items():
    """
    Get the items of the Frost theme.

    Frost is a light, translucent design for light scenes and figures: a
    white panel at 78% opacity, slate text and a blue accent. Translucent
    fills and borders of the design use their color over the panel.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    return _graphite_items(
        FROST_PALETTE, panel_radius=12, control_radius=8, panel_opacity=0.78
    )


def _ember_items():
    """
    Get the items of the Ember theme.

    Ember is Graphite with warm neutrals and an orange accent.

    Returns
    -------
    dict
        Mapping of UI class name to items.
    """
    return _graphite_items(EMBER_PALETTE)


BUILTIN_THEMES = {
    "classic": _classic_items,
    "graphite": _graphite_dark_items,
    "graphite_light": _graphite_light_items,
    "hairline": _hairline_items,
    "soft_pill": _soft_pill_items,
    "frost": _frost_items,
    "ember": _ember_items,
}


def theme_names():
    """
    Get the names of the built-in themes.

    Returns
    -------
    list of str
        Names accepted by :func:`builtin_theme` and :func:`set_theme`.
    """
    return list(BUILTIN_THEMES)


def builtin_theme(name):
    """
    Create a new copy of a built-in theme.

    Parameters
    ----------
    name : str
        Name of the theme, one of :func:`theme_names`.

    Returns
    -------
    Theme
        The theme.

    Raises
    ------
    ValueError
        If no built-in theme has this name.
    """
    if name not in BUILTIN_THEMES:
        available = ", ".join(BUILTIN_THEMES)
        msg = f"Unknown theme {name!r}. Available themes: {available}."
        raise ValueError(msg)
    return Theme(name=name, items=BUILTIN_THEMES[name]())


def _as_theme(theme):
    """
    Get a theme from a theme or the name of a built-in theme.

    Parameters
    ----------
    theme : Theme or str
        Theme, or name of a built-in theme.

    Returns
    -------
    Theme
        The theme.
    """
    if isinstance(theme, Theme):
        return theme
    return builtin_theme(theme)


def set_theme(theme=None, *, density=None):
    """
    Set the global theme of UI elements and restyle the existing ones.

    Elements created from now on outside a :func:`theme_scope` use the
    theme. Nothing is restyled when the new theme has the same items as the
    current one. Existing elements that follow the global theme, those created
    outside any :func:`theme_scope` and never given their own theme with
    :meth:`fury.ui.core.UI.apply_theme`, are restyled. Values passed
    explicitly to their constructors or set through their properties are
    kept, and structural items keep the value read at creation; see
    :meth:`fury.ui.core.UI.apply_theme`.

    ``set_theme(density="compact")`` changes only the density of the global
    theme, and restyles fonts, spacing, corner radii and slider sizes of the
    existing elements.

    Parameters
    ----------
    theme : Theme or str, optional
        Theme, or name of a built-in theme such as ``"classic"`` or
        ``"graphite"``; see :func:`theme_names`. None keeps the global
        theme, ignoring any :func:`theme_scope`.
    density : str, optional
        Density to use with the theme: ``"compact"`` scales font sizes,
        padding, gaps, corner radii and slider sizes down, and
        ``"comfortable"`` uses the theme items as they are. None keeps the
        density of ``theme``.
    """
    if theme is None:
        theme = UIContext.theme or builtin_theme("classic")
    theme = _as_theme(theme)
    if density is not None:
        theme = _with_density(theme, density)
    previous = UIContext.theme
    UIContext.theme = theme
    if previous is not None and _comparable_items(previous) == _comparable_items(theme):
        return
    with UIContext.style_batch():
        for element in UIContext.global_theme_followers:
            element._apply_theme(theme, follow_global=True)


def _with_density(theme, density):
    """
    Get a theme with the items of another one and a given density.

    A theme that only sets a density, such as one made by an earlier call,
    is replaced rather than wrapped, so switching back and forth does not
    grow a chain of themes.

    Parameters
    ----------
    theme : Theme
        The theme.
    density : str
        ``"comfortable"`` or ``"compact"``.

    Returns
    -------
    Theme
        ``theme`` itself if it already has the density, else a theme deriving
        from it, or from its base when it only sets a density.
    """
    if density not in _DENSITIES:
        available = ", ".join(_DENSITIES)
        msg = f"Unknown density {density!r}. Available densities: {available}."
        raise ValueError(msg)
    if not theme._items and theme._density is not None and theme.base is not None:
        theme = theme.base
    if theme.density == density:
        return theme
    return theme.derive(name=theme.name, density=density)


def get_theme():
    """
    Get the theme a UI element created now would use.

    Returns
    -------
    Theme
        The innermost :func:`theme_scope` theme, or the global theme.
    """
    if UIContext.theme is None:
        UIContext.theme = builtin_theme("classic")
    return UIContext.current_theme()


@contextmanager
def theme_scope(theme, *, density=None):
    """
    Use a theme for UI elements created inside a ``with`` block.

    Parameters
    ----------
    theme : Theme or str
        Theme, or name of a built-in theme.
    density : str, optional
        Density to use with the theme, ``"comfortable"`` or ``"compact"``.
        None keeps the density of ``theme``.

    Yields
    ------
    Theme
        The theme in use inside the block.
    """
    theme = _as_theme(theme)
    if density is not None:
        theme = _with_density(theme, density)
    UIContext.push_theme(theme)
    try:
        yield theme
    finally:
        UIContext.pop_theme()


@contextmanager
def _inherited_theme_scope(theme):
    """
    Use the theme of a UI element for the parts it creates.

    Parts created inside are marked as inherited, so they are restyled with
    the element rather than on their own.

    Parameters
    ----------
    theme : Theme
        Theme of the element.

    Yields
    ------
    Theme
        The theme in use inside the block.
    """
    UIContext.push_theme(theme, inherited=True)
    try:
        yield theme
    finally:
        UIContext.pop_theme()
