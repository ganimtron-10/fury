"""UI context module."""

from contextlib import contextmanager
from weakref import WeakSet

import numpy as np


class UIContextClass:
    """Manage global UI context."""

    _instance = None

    def __new__(cls):
        """
        Handle instance creation for the UI context singleton.

        Returns
        -------
        UIContextClass
            The single, shared instance of `UIContextClass`.
        """
        if cls._instance is None:
            cls._instance = super(UIContextClass, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the UIContext singleton."""
        if self._initialized:
            return
        self._hot_ui = None
        self._active_ui = None
        self._canvas_size = np.array([800, 800])
        self._z_max = 1
        self._z_min = -1
        self._dirty_layouts = WeakSet()
        self._theme = None
        self._theme_stack = []
        self._global_theme_followers = WeakSet()
        self._refresh_queue = None
        self._last_refreshes = []
        self._initialized = True

    @property
    def theme(self):
        """
        Get the global theme of UI elements.

        Returns
        -------
        Theme or None
            The global theme, or None before one is set or first needed.
        """
        return self._theme

    @theme.setter
    def theme(self, theme):
        """
        Set the global theme of UI elements.

        Parameters
        ----------
        theme : Theme
            The global theme.
        """
        self._theme = theme

    def push_theme(self, theme, *, inherited=False):
        """
        Use a theme for UI elements created until the matching pop.

        Parameters
        ----------
        theme : Theme
            The scoped theme.
        inherited : bool, optional
            True when a UI element pushes its own theme while it creates its
            parts, False for a user's ``theme_scope``.
        """
        self._theme_stack.append((theme, inherited))

    def pop_theme(self):
        """
        Stop using the most recently pushed theme.

        Returns
        -------
        Theme
            The theme that was removed.
        """
        return self._theme_stack.pop()[0]

    def current_theme(self):
        """
        Get the innermost scoped theme, or the global theme.

        Returns
        -------
        Theme or None
            The theme in use.
        """
        if self._theme_stack:
            return self._theme_stack[-1][0]
        return self._theme

    def theme_source(self):
        """
        Tell where the theme of a UI element created now comes from.

        Returns
        -------
        str
            ``"global"`` outside any scope, ``"inherited"`` while a UI
            element creates its parts, and ``"scoped"`` inside a user's
            ``theme_scope``.
        """
        if not self._theme_stack:
            return "global"
        return "inherited" if self._theme_stack[-1][1] else "scoped"

    def follow_global_theme(self, element):
        """
        Restyle a UI element whenever the global theme changes.

        Only a weak reference is kept, so the element can still be freed.

        Parameters
        ----------
        element : UI
            Element created with the global theme.
        """
        self._global_theme_followers.add(element)

    def unfollow_global_theme(self, element):
        """
        Stop restyling a UI element when the global theme changes.

        Parameters
        ----------
        element : UI
            Element that got its own theme.
        """
        self._global_theme_followers.discard(element)

    @contextmanager
    def style_batch(self):
        """
        Refresh each restyled UI element once, at the end of the block.

        Elements are refreshed in the order they asked, so a parent restyled
        before its parts is refreshed before them. Nested blocks join the
        outer one.

        Yields
        ------
        None
            Nothing; the refresh happens when the outermost block ends.
        """
        if self._refresh_queue is not None:
            yield
            return
        self._refresh_queue = {}
        self._last_refreshes = []
        try:
            yield
            while self._refresh_queue:
                key = next(iter(self._refresh_queue))
                element = self._refresh_queue.pop(key)
                self._last_refreshes.append(key)
                element._refresh_style()
        finally:
            self._refresh_queue = None

    @property
    def last_refreshes(self):
        """
        Get the elements refreshed by the last batch, in order.

        This is a debugging and testing aid, not part of the styling API.

        Returns
        -------
        list of int
            Identity (``id``) of each refreshed element, once per refresh.
        """
        return list(self._last_refreshes)

    def queue_refresh(self, element):
        """
        Ask for an element to be refreshed at the end of the current batch.

        Parameters
        ----------
        element : UI
            The element.

        Returns
        -------
        bool
            False when no batch is active, so the caller refreshes now.
        """
        if self._refresh_queue is None:
            return False
        self._refresh_queue[id(element)] = element
        return True

    @property
    def global_theme_followers(self):
        """
        Get the UI elements that follow the global theme.

        This is a debugging and testing aid, not part of the styling API.

        Returns
        -------
        list of UI
            Living elements created outside any theme scope.
        """
        return list(self._global_theme_followers)

    @property
    def hot_ui(self):
        """
        Get the currently hot UI element.

        Returns
        -------
        UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        return self._hot_ui

    @hot_ui.setter
    def hot_ui(self, element):
        """
        Set the currently hot UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently hot, or `None` if no element is hot.
        """
        self._hot_ui = element

    def request_layout(self, element):
        """
        Queue a UI element whose layout must be solved before the next frame.

        Parameters
        ----------
        element : UI
            The root of a layout tree that changed.
        """
        self._dirty_layouts.add(element)

    def flush_layouts(self):
        """Solve every queued layout."""
        while self._dirty_layouts:
            elements = list(self._dirty_layouts)
            self._dirty_layouts.clear()
            for element in elements:
                element.update_layout()

    @property
    def active_ui(self):
        """
        Get the currently active UI element.

        Returns
        -------
        UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        return self._active_ui

    @active_ui.setter
    def active_ui(self, element):
        """
        Set the currently active UI element.

        Parameters
        ----------
        element : UI or None
            UI element that is currently active, or `None` if no element is active.
        """
        self._active_ui = element

    @property
    def canvas_size(self):
        """
        Get the current size of the rendering canvas in pixels.

        Returns
        -------
        numpy.ndarray
            Canvas `(width, height)` dimensions.
        """
        return self._canvas_size

    @canvas_size.setter
    def canvas_size(self, size):
        """
        Set the current size of the rendering canvas in pixels.

        Parameters
        ----------
        size : (int, int)
            Canvas `(width, height)` dimensions.
        """
        size = np.array(size)
        if not np.array_equal(self._canvas_size, size):
            self._canvas_size = size

    @property
    def z_order_bounds(self):
        """
        Get the minimum and maximum Z-order values in the UI.

        Returns
        -------
        numpy.ndarray
            Z-order bounds `[z_min, z_max]`.
        """
        return np.array([self._z_min, self._z_max])

    @z_order_bounds.setter
    def z_order_bounds(self, z_order):
        """
        Update the minimum or maximum active Z-order values in the UI.

        Parameters
        ----------
        z_order : int
            Z-order value of a UI.
        """
        self._z_min = min(self._z_min, z_order)
        self._z_max = max(self._z_max, z_order)


UIContext = UIContextClass()
