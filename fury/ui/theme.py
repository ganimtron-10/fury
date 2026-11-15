"""Themes that provide the default look of UI elements."""

from contextlib import contextmanager
import copy
import json

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
    """

    def __init__(self, *, name="custom", items=None, base=None):
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
        """
        self.name = name
        self._base = base
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
            given object do not affect the theme.
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
            A copy of the item's value.

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
                    return copy.deepcopy(type_items[item])
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

    def derive(self, *, name=None, items=None):
        """
        Create a theme that overrides some items of this one.

        Parameters
        ----------
        name : str, optional
            Name of the new theme.
        items : dict, optional
            Items to override, in the same form as the ``items`` parameter.

        Returns
        -------
        Theme
            A theme whose base is this theme.
        """
        return Theme(name=name or "custom", items=items, base=self)

    def to_dict(self):
        """
        Get a copy of all items, including those inherited from the base.

        A theme created from the result with :meth:`from_dict` returns the
        same value as this theme for every lookup.

        Returns
        -------
        dict
            Mapping with the theme ``name`` and its ``items``.
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
        return {"name": self.name, "items": copy.deepcopy(items)}

    @classmethod
    def from_dict(cls, data, *, base=None):
        """
        Create a theme from a mapping made by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Mapping with an optional ``name`` and the ``items``.
        base : Theme, optional
            Theme consulted for items the mapping does not define.

        Returns
        -------
        Theme
            The new theme.
        """
        return cls(name=data.get("name", "custom"), items=data.get("items"), base=base)

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
        "UI": {"font_family": "Arial"},
        "Rectangle2D": {"color": (1, 1, 1)},
        "RoundedRectangle2D": {"color": (1, 1, 1)},
        "Disk2D": {"color": (1, 1, 1)},
        "TextBlock2D": {"color": (1, 1, 1), "font_size": 18},
        "Panel2D": {
            "color": (0.1, 0.1, 0.1),
            "opacity": 0.7,
            "border_color": (1, 1, 1),
        },
        "TabPanel2D": {"color": (0.5, 0.5, 0.5), "text_color": (0, 0, 0)},
        "TabUI": {
            "active_color": (1, 1, 1),
            "inactive_color": (0.5, 0.5, 0.5),
            "font_size": 18,
        },
        "Loader2D": {
            "overlay_color": (0, 0, 0),
            "overlay_opacity": 0.5,
            "text_color": (1, 1, 1),
            "font_size": 18,
        },
        "TextButton2D": {
            "font_size": 25,
            "text_color": (0, 0, 0),
            "bg_color": (1, 1, 1),
            "states": {
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
            "track_color": (1, 0, 0),
        },
        "LineDoubleSlider2D": {
            "font_size": 16,
            "handle_color": (1, 1, 1),
            "handle_active_color": (0, 0, 1),
            "track_color": (1, 0, 0),
        },
        "RangeSlider": {"font_size": 16},
        "PlaybackPanel": {
            "panel_color": (1, 1, 1),
            "border_color": (0, 0.3, 0),
            "time_color": (1, 1, 1),
            "speed_color": (0.2, 0.2, 0.2),
            "progress_color": (1, 0, 0),
        },
        "TextBox2D": {
            "color": (0, 0, 0),
            "bg_color": (1, 1, 1),
            "font_size": 18,
        },
        "ButtonGroup": {"font_size": 18, "text_color": (1, 1, 1)},
        "ComboBox2D": {
            "selection_text_color": (0, 0, 0),
            "selection_bg_color": (1, 1, 1),
            "menu_text_color": (0.2, 0.2, 0.2),
            "selected_color": (0.9, 0.6, 0.6),
            "unselected_color": (0.6, 0.6, 0.6),
            "scroll_bar_active_color": (0.6, 0.2, 0.2),
            "scroll_bar_inactive_color": (0.9, 0.0, 0.0),
            "font_size": 20,
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
        },
        "TreeNode2D": {"color": (0, 0, 0), "selected_color": (0.8, 0.3, 0.3)},
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
        },
        "SpinBox": {"panel_color": (1, 1, 1)},
    }


BUILTIN_THEMES = {"classic": _classic_items}


def builtin_theme(name):
    """
    Create a new copy of a built-in theme.

    Parameters
    ----------
    name : str
        Name of the theme. One of the keys of ``BUILTIN_THEMES``.

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
        available = ", ".join(sorted(BUILTIN_THEMES))
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


def set_theme(theme):
    """
    Set the theme used by UI elements created from now on.

    Parameters
    ----------
    theme : Theme or str
        Theme, or name of a built-in theme such as ``"classic"``.
    """
    UIContext.theme = _as_theme(theme)


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
def theme_scope(theme):
    """
    Use a theme for UI elements created inside a ``with`` block.

    Parameters
    ----------
    theme : Theme or str
        Theme, or name of a built-in theme.

    Yields
    ------
    Theme
        The theme in use inside the block.
    """
    theme = _as_theme(theme)
    UIContext.push_theme(theme)
    try:
        yield theme
    finally:
        UIContext.pop_theme()
