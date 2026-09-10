"""
FURY UI Workbench & Playground module.

Provides an interactive ImGui-powered side panel that behaves like a browser
style inspector for FURY UI: it mirrors the live scene hierarchy (including the
internal parts every widget is built from), exposes every readable and writable
property of the selected component, lets components be created, parented,
reparented and merged at run time, and reports the events each component
receives.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import datetime
import random
from typing import Any

import numpy as np

from fury.lib import imgui_bundle
from fury.ui import (
    UI,
    Checkbox,
    ComboBox2D,
    Disk2D,
    LineDoubleSlider2D,
    LineSlider2D,
    ListBox2D,
    Panel2D,
    PlaybackPanel,
    RadioButton,
    RangeSlider,
    Rectangle2D,
    RingSlider2D,
    RoundedRectangle2D,
    TabUI,
    TextBlock2D,
    TextBox2D,
    TextButton2D,
    UIContext,
)
from fury.ui.core import Button2D, Slider2D
from fury.ui.elements import ButtonGroup
from fury.window import add_ui_to_scene, remove_ui_from_scene

if imgui_bundle is not None:
    from imgui_bundle import imgui
else:
    imgui = None  # type: ignore[assignment]

#: Role of a node that is directly attached to the scene.
ROLE_ROOT = "root"
#: Role of a node the user explicitly added to a container.
ROLE_CONTENT = "content"
#: Role of a node a widget built for itself (background, handle, border, ...).
ROLE_PART = "part"

#: Containers whose ``_children`` hold user content rather than internal parts.
CONTENT_CONTAINERS = (Panel2D,)

#: Every callback slot defined by :class:`fury.ui.UI`.
HANDLER_NAMES = (
    "on_left_mouse_button_pressed",
    "on_left_mouse_button_released",
    "on_left_mouse_button_clicked",
    "on_left_mouse_double_clicked",
    "on_left_mouse_button_dragged",
    "on_right_mouse_button_pressed",
    "on_right_mouse_button_released",
    "on_right_mouse_button_clicked",
    "on_right_mouse_double_clicked",
    "on_right_mouse_button_dragged",
    "on_middle_mouse_button_pressed",
    "on_middle_mouse_button_released",
    "on_middle_mouse_button_clicked",
    "on_middle_mouse_double_clicked",
    "on_middle_mouse_button_dragged",
    "on_key_press",
    "on_key_release",
    "on_hover",
    "on_dishover",
    "on_focus",
    "on_blur",
    "on_wheel",
)

#: Properties that must not be edited through the generic property editor
#: because their getter and setter use different shapes.
ASYMMETRIC_PROPERTIES = {
    "border_color",
    "border_width",
    "current_time_str",
    "element",
    "ratio",
}

#: Properties the generic inspector never reads (too large or side effecting).
SKIPPED_PROPERTIES = {"actors"}


def iter_ui_properties(ui_obj):
    """
    Iterate over every property declared by a UI component and its bases.

    Parameters
    ----------
    ui_obj : UI
        The UI component to introspect.

    Yields
    ------
    tuple
        A ``(name, property_object, declaring_class_name)`` triple, ordered
        from the most derived class to the least derived one.
    """
    seen = set()
    for cls in type(ui_obj).__mro__:
        if cls is object:
            continue
        for name, member in vars(cls).items():
            if name.startswith("_") or name in seen or name in SKIPPED_PROPERTIES:
                continue
            if isinstance(member, property):
                seen.add(name)
                yield name, member, cls.__name__


def named_sub_ui(ui_obj):
    """
    Collect the sub components a widget keeps in its public attributes.

    Attributes holding a :class:`fury.ui.UI`, or a list/tuple/dict of them,
    are reported together with a readable label such as ``borders[left]``.

    Parameters
    ----------
    ui_obj : UI
        The UI component to introspect.

    Returns
    -------
    list
        A list of ``(sub_component, label)`` tuples.
    """
    found = []
    for name, value in list(vars(ui_obj).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, UI):
            found.append((value, name))
        elif isinstance(value, dict):
            for key, sub in value.items():
                if isinstance(sub, UI):
                    found.append((sub, f"{name}[{key}]"))
        elif isinstance(value, (list, tuple)):
            for index, sub in enumerate(value):
                if isinstance(sub, UI):
                    found.append((sub, f"{name}[{index}]"))
    return found


def iter_child_specs(ui_obj):
    """
    Describe every direct sub component of a UI component.

    Children of a content container (a :class:`fury.ui.Panel2D`) are reported
    as content, everything else as an internal part. Sub components reachable
    only through a public attribute are picked up as well, so the hierarchy is
    complete even before the component has been rendered once.

    Parameters
    ----------
    ui_obj : UI
        The UI component to describe.

    Returns
    -------
    list
        A list of ``(child, role, label)`` tuples, content first.
    """
    named = named_sub_ui(ui_obj)
    label_of = {id(child): label for child, label in named}

    seen = set()
    content = []
    parts = []

    def _push(bucket, child):
        """
        Append a child to a bucket unless it was already classified.

        Parameters
        ----------
        bucket : list
            The list to append to.
        child : object
            The candidate child component.
        """
        if not isinstance(child, UI) or id(child) in seen:
            return
        seen.add(id(child))
        bucket.append(child)

    if isinstance(ui_obj, CONTENT_CONTAINERS):
        for child in list(ui_obj._children):
            _push(content, child)
        for element in list(getattr(ui_obj, "_elements", [])):
            _push(parts, element)
    else:
        for child in list(ui_obj._children):
            _push(parts, child)

    for child, _label in named:
        _push(parts, child)

    specs = [(child, ROLE_CONTENT, label_of.get(id(child), "")) for child in content]
    specs += [(child, ROLE_PART, label_of.get(id(child), "")) for child in parts]
    return specs


def walk_ui(ui_obj, _seen=None):
    """
    Walk a UI component and every sub component it owns.

    Parameters
    ----------
    ui_obj : UI
        The root of the walk.
    _seen : set, optional
        Identities already visited, used internally to stop on cycles.

    Yields
    ------
    UI
        Every component of the sub tree, ``ui_obj`` included.
    """
    if _seen is None:
        _seen = set()
    if id(ui_obj) in _seen:
        return
    _seen.add(id(ui_obj))
    yield ui_obj
    for child, _role, _label in iter_child_specs(ui_obj):
        yield from walk_ui(child, _seen)


def has_native_drag(ui_obj):
    """
    Check whether a component already implements its own drag behaviour.

    Parameters
    ----------
    ui_obj : UI
        The UI component to check.

    Returns
    -------
    bool
        True if the component, or one of its parts, binds a real callback to
        ``on_left_mouse_button_dragged``.
    """
    for node in walk_ui(ui_obj):
        handler = getattr(node, "on_left_mouse_button_dragged", None)
        handler = getattr(handler, "workbench_original", handler)
        if handler is None:
            continue
        if getattr(handler, "__name__", "<lambda>") != "<lambda>":
            return True
    return False


def describe_event(event):
    """
    Build a short readable description of a pygfx event.

    Parameters
    ----------
    event : object
        The event object handed to a UI callback.

    Returns
    -------
    str
        A compact description, empty when the event carries no useful field.
    """
    parts = []
    x, y = getattr(event, "x", None), getattr(event, "y", None)
    if x is not None and y is not None:
        parts.append(f"({x:.0f}, {y:.0f})")
    key = getattr(event, "key", None)
    if key:
        parts.append(f"key={key}")
    button = getattr(event, "button", None)
    if button:
        parts.append(f"button={button}")
    return " ".join(parts)


@dataclass
class UIWorkbenchItem:
    """
    A node of the workbench hierarchy wrapping a single UI component.

    Attributes
    ----------
    id : int
        Unique identifier of the node inside the workbench.
    name : str
        Editable display name of the node.
    ui : UI
        The wrapped UI component.
    parent : UIWorkbenchItem, optional
        Parent node, or None when the component sits directly in the scene.
    children : list, optional
        Child nodes, content first then internal parts.
    is_internal : bool, optional
        True when the node belongs to the internals of another widget.
    draggable : bool, optional
        True when the workbench moves this component on mouse drag.
    role : str, optional
        One of ``"root"``, ``"content"`` or ``"part"``.
    attr_label : str, optional
        Name of the attribute the parent keeps this component in.
    event_counts : collections.Counter, optional
        Number of times each callback slot fired.
    last_event : str, optional
        Name of the callback slot that fired last.
    trace_events : bool, optional
        True when every event of this component is written to the event log.
    """

    id: int
    name: str
    ui: UI
    parent: UIWorkbenchItem | None = None
    children: list[UIWorkbenchItem] = field(default_factory=list)
    is_internal: bool = False
    draggable: bool = False
    role: str = ROLE_ROOT
    attr_label: str = ""
    event_counts: collections.Counter = field(default_factory=collections.Counter)
    last_event: str = ""
    trace_events: bool = False
    _drag_start_click: np.ndarray | None = None
    _drag_start_pos: np.ndarray | None = None
    _original_handlers: dict[str, Any] = field(default_factory=dict)

    @property
    def type_name(self) -> str:
        """
        Get the class name of the wrapped UI component.

        Returns
        -------
        str
            Name of the component class.
        """
        return self.ui.__class__.__name__

    @property
    def display_name(self) -> str:
        """
        Get the label shown for this node in the hierarchy.

        Returns
        -------
        str
            The attribute label when the node is an internal part, the
            editable name otherwise.
        """
        return self.attr_label or self.name

    @property
    def owner(self) -> UIWorkbenchItem:
        """
        Get the closest ancestor that is not an internal part.

        Returns
        -------
        UIWorkbenchItem
            The widget a user would consider being clicked. Returns ``self``
            when this node is not an internal part.
        """
        node = self
        while node.role == ROLE_PART and node.parent is not None:
            node = node.parent
        return node

    @property
    def path(self) -> str:
        """
        Get the slash separated path of this node from its root.

        Returns
        -------
        str
            Readable hierarchy path, for example ``panel / title``.
        """
        names = []
        node: UIWorkbenchItem | None = self
        while node is not None:
            names.append(node.display_name)
            node = node.parent
        return " / ".join(reversed(names))

    def get_position(self) -> np.ndarray:
        """
        Get the absolute top-left position of the component.

        Returns
        -------
        numpy.ndarray
            The ``(x, y)`` position in canvas pixels.
        """
        return np.array(self.ui.get_position(), dtype=float)

    def set_position(self, pos) -> None:
        """
        Move the component and keep its parent panel offset in sync.

        Parameters
        ----------
        pos : (float, float) or ndarray
            The new absolute ``(x, y)`` position in canvas pixels.
        """
        pos_arr = np.array(pos, dtype=float)
        self.ui.set_position(pos_arr)
        if self.parent is not None and isinstance(self.parent.ui, Panel2D):
            rel_offset = pos_arr - self.parent.get_position()
            if hasattr(self.parent.ui, "update_element_offset"):
                try:
                    self.parent.ui.update_element_offset(self.ui, rel_offset)
                except ValueError:
                    pass

    def get_size(self) -> tuple[float, float]:
        """
        Get the size of the component.

        Returns
        -------
        tuple
            The ``(width, height)`` of the component in pixels.
        """
        try:
            size = self.ui.size
            if size is not None:
                return float(size[0]), float(size[1])
        except Exception:
            pass
        return (100.0, 100.0)

    @property
    def can_resize(self) -> bool:
        """
        Check whether the component exposes a way to change its size.

        Returns
        -------
        bool
            True when :meth:`set_size` can act on this component.
        """
        if isinstance(self.ui, Disk2D):
            return True
        if hasattr(self.ui, "resize"):
            return True
        return hasattr(self.ui, "width") and hasattr(self.ui, "height")

    def set_size(self, size) -> bool:
        """
        Resize the component using whichever knob the class exposes.

        Components such as the sliders keep their size in constructor only
        state and are left untouched.

        Parameters
        ----------
        size : (float, float)
            The requested ``(width, height)`` in pixels.

        Returns
        -------
        bool
            True when the component was resized.
        """
        width, height = max(1.0, float(size[0])), max(1.0, float(size[1]))

        if isinstance(self.ui, Disk2D):
            self.ui.outer_radius = max(1.0, width / 2.0)
            return True

        if hasattr(self.ui, "resize"):
            try:
                self.ui.resize((width, height))
                return True
            except Exception:
                pass
        if hasattr(self.ui, "width") and hasattr(self.ui, "height"):
            try:
                self.ui.width = width
                self.ui.height = height
                return True
            except Exception:
                pass
        return False

    def get_bounds(self) -> tuple[float, float, float, float]:
        """
        Get the axis aligned bounding box of the component.

        Returns
        -------
        tuple
            The ``(x, y, width, height)`` box in canvas pixels.
        """
        pos = self.get_position()
        width, height = self.get_size()
        return (float(pos[0]), float(pos[1]), width, height)

    def get_color(self) -> tuple[float, float, float]:
        """
        Get the main RGB color of the component.

        Returns
        -------
        tuple
            The ``(r, g, b)`` color, each channel in [0, 1].
        """
        for holder in (self.ui, getattr(self.ui, "child", None)):
            if holder is None:
                continue
            color = getattr(holder, "color", None)
            if color is not None and len(color) >= 3:
                return float(color[0]), float(color[1]), float(color[2])
            background = getattr(holder, "background", None)
            color = getattr(background, "color", None)
            if color is not None and len(color) >= 3:
                return float(color[0]), float(color[1]), float(color[2])
        return (1.0, 1.0, 1.0)

    def set_color(self, color) -> None:
        """
        Set the main RGB color of the component.

        Parameters
        ----------
        color : sequence of float
            The ``(r, g, b)`` color, each channel in [0, 1].
        """
        rgb = tuple(float(channel) for channel in color[:3])
        if hasattr(self.ui, "color"):
            try:
                self.ui.color = rgb
            except Exception:
                pass
        child = getattr(self.ui, "child", None)
        if child is not None and getattr(child, "background", None) is not None:
            try:
                child.background.color = rgb
            except Exception:
                pass

    def get_opacity(self) -> float:
        """
        Get the opacity of the component.

        Returns
        -------
        float
            The opacity in [0, 1].
        """
        if hasattr(self.ui, "opacity"):
            return float(self.ui.opacity)
        child = getattr(self.ui, "child", None)
        if child is not None and getattr(child, "background", None) is not None:
            return float(child.background.opacity)
        return 1.0

    def set_opacity(self, opacity) -> None:
        """
        Set the opacity of the component.

        Parameters
        ----------
        opacity : float
            The requested opacity, clamped to [0, 1].
        """
        value = max(0.0, min(1.0, float(opacity)))
        if hasattr(self.ui, "opacity"):
            try:
                self.ui.opacity = value
            except Exception:
                pass
        child = getattr(self.ui, "child", None)
        if child is not None and getattr(child, "background", None) is not None:
            try:
                child.background.opacity = value
            except Exception:
                pass

    def get_z_order(self) -> int:
        """
        Get the Z-order of the component.

        Returns
        -------
        int
            The current Z-order.
        """
        return int(getattr(self.ui, "z_order", 0))

    def set_z_order(self, z_order) -> None:
        """
        Set the Z-order of the component.

        Parameters
        ----------
        z_order : int
            The new Z-order.
        """
        try:
            self.ui.z_order = int(z_order)
        except Exception:
            pass

    def get_visible(self) -> bool:
        """
        Check whether the component is currently visible.

        Returns
        -------
        bool
            True when at least one actor of the component is visible.
        """
        try:
            return any(bool(actor.visible) for actor in self.ui.actors)
        except Exception:
            return True

    def set_visible(self, visible) -> None:
        """
        Show or hide the component and its sub components.

        Parameters
        ----------
        visible : bool
            True to show the component, False to hide it.
        """
        try:
            self.ui.set_visibility(bool(visible))
        except Exception:
            pass

    def get_text(self) -> str:
        """
        Get the text carried by the component, if any.

        Returns
        -------
        str
            The text message, or an empty string for components without text.
        """
        if hasattr(self.ui, "message"):
            return str(self.ui.message)
        if hasattr(self.ui, "default_label"):
            return str(self.ui.default_label)
        text = getattr(self.ui, "text", None)
        if isinstance(text, str):
            return text
        if text is not None and hasattr(text, "message"):
            return str(text.message)
        return ""

    def set_text(self, text) -> None:
        """
        Set the text carried by the component.

        Parameters
        ----------
        text : str
            The new text message.
        """
        if hasattr(self.ui, "message"):
            self.ui.message = text
        if hasattr(self.ui, "default_label"):
            self.ui.default_label = text
            child = getattr(self.ui, "child", None)
            if child is not None and hasattr(child, "message"):
                child.message = text
        sub_text = getattr(self.ui, "text", None)
        if sub_text is not None and hasattr(sub_text, "message"):
            sub_text.message = text

    def get_font_size(self) -> int:
        """
        Get the font size of the component.

        Returns
        -------
        int
            The current font size in points.
        """
        if hasattr(self.ui, "font_size"):
            return int(self.ui.font_size)
        return 18

    def set_font_size(self, size) -> None:
        """
        Set the font size of the component.

        Parameters
        ----------
        size : int
            The new font size in points.
        """
        if hasattr(self.ui, "font_size"):
            try:
                self.ui.font_size = int(size)
            except Exception:
                pass


class UIWorkbench:
    """
    Interactive UI workbench and debugger for FURY.

    The workbench mirrors the live scene as a hierarchy of
    :class:`UIWorkbenchItem` nodes, refreshed on every rendered frame, and
    draws an ImGui side panel to inspect and edit that hierarchy.

    Parameters
    ----------
    scene : Scene, optional
        The scene to mirror. Can also be set later with :meth:`set_scene`.
    show_manager : ShowManager, optional
        The show manager driving the window. Only used for diagnostics.
    title : str, optional
        Title of the ImGui window.
    panel_width : int, optional
        Width in pixels of the docked ImGui panel.
    enable_universal_drag : bool, optional
        When True, components that have no drag behaviour of their own become
        draggable as soon as the workbench discovers them.
    show_internals : bool, optional
        When True, the hierarchy also lists the internal parts of each widget.
    show_outline : bool, optional
        When True, an outline is drawn in the viewport around the selection.
    """

    def __init__(
        self,
        scene=None,
        show_manager=None,
        *,
        title="UI Workbench & Debugger",
        panel_width=440,
        enable_universal_drag=True,
        show_internals=False,
        show_outline=True,
    ):
        """Initialize the UI Workbench."""
        self.scene = scene
        self.show_manager = show_manager
        self.title = title
        self.panel_width = panel_width
        self.enable_universal_drag = enable_universal_drag

        self._next_id = 1
        self.items: dict[int, UIWorkbenchItem] = {}
        self.root_items: list[UIWorkbenchItem] = []
        self.selected_item: UIWorkbenchItem | None = None
        self.hit_item: UIWorkbenchItem | None = None
        self.multi_selected_ids: set[int] = set()

        self._by_ui: dict[int, UIWorkbenchItem] = {}
        self._dragging: UIWorkbenchItem | None = None
        self._outline: list[Rectangle2D] = []
        self._outline_attached = False

        self.event_log: collections.deque = collections.deque(maxlen=200)
        self.status_message = "Ready"

        # Hierarchy view options.
        self.auto_sync = True
        self.show_internals = show_internals
        self.show_outline = show_outline
        self.hierarchy_filter = ""
        self._force_tree_state: bool | None = None

        # Inspector view options.
        self.show_private_attrs = False

        # Creation wizard defaults.
        self.create_type_idx = 0
        self.create_types = [
            "Panel2D",
            "TextButton2D",
            "LineSlider2D",
            "LineDoubleSlider2D",
            "RingSlider2D",
            "TextBlock2D",
            "TextBox2D",
            "Checkbox",
            "RadioButton",
            "ComboBox2D",
            "ListBox2D",
            "Rectangle2D",
            "RoundedRectangle2D",
            "Disk2D",
            "TabUI",
            "PlaybackPanel",
        ]
        self.create_target_parent_id = 0
        self.create_pos = [100.0, 100.0]
        self.create_size = [200.0, 80.0]
        self.create_label = "My Element"
        self.create_color = [0.2, 0.4, 0.8]
        self.create_opacity = 0.85
        self.create_border = False
        self.create_border_width = 2.0
        self._create_dest_idx = 0
        self._parent_combo_idx = 0

        # Fuzzy controls.
        self.fuzzy_pos_delta = 30.0
        self.fuzzy_scale_min = 0.7
        self.fuzzy_scale_max = 1.4
        self.reparent_cycles = 5

        # Merge controls.
        self.merge_padding = 15.0
        self.merge_panel_name = "merged_panel"
        self.merge_color = [0.12, 0.14, 0.18]

        if self.scene is not None:
            self.scan_scene()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_scene(self, scene) -> None:
        """
        Attach a scene and mirror its UI hierarchy.

        Parameters
        ----------
        scene : Scene
            The scene the workbench should inspect.
        """
        self.scene = scene
        self._by_ui.clear()
        self.items.clear()
        self.root_items.clear()
        self.selected_item = None
        self.multi_selected_ids.clear()
        self.scan_scene()

    def set_show_manager(self, show_manager) -> None:
        """
        Attach the show manager driving the window.

        Parameters
        ----------
        show_manager : ShowManager
            The active show manager.
        """
        self.show_manager = show_manager

    def log_event(self, event_name, target_name, details="") -> None:
        """
        Append an entry to the workbench event log.

        Parameters
        ----------
        event_name : str
            Name of the event or operation.
        target_name : str
            Name of the component the event applies to.
        details : str, optional
            Extra information shown next to the entry.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.event_log.appendleft(
            {
                "time": timestamp,
                "event": event_name,
                "target": target_name,
                "details": details,
            }
        )

    # ------------------------------------------------------------------
    # Automatic hierarchy discovery
    # ------------------------------------------------------------------

    def scan_scene(self) -> None:
        """Rebuild the hierarchy from the scene and report the item count."""
        self.sync()
        content = sum(1 for item in self.items.values() if not item.is_internal)
        self.status_message = (
            f"Scanned scene: {content} components, {len(self.items)} nodes"
        )

    def sync(self) -> None:
        """
        Mirror the scene graph into the workbench hierarchy.

        Existing nodes are reused so that names, selection, drag flags and
        event counters survive a refresh. Nodes whose component has left the
        scene are dropped.
        """
        if self.scene is None:
            return

        alive: set[int] = set()
        roots = []
        for ui_obj in list(self.scene.ui_elements):
            if not isinstance(ui_obj, UI):
                continue
            roots.append(self._sync_node(ui_obj, None, ROLE_ROOT, "", alive))

        self.root_items = roots

        for item in list(self.items.values()):
            if id(item.ui) not in alive:
                self._forget(item)

    def _sync_node(self, ui_obj, parent_item, role, attr_label, alive):
        """
        Create or refresh the node wrapping a component and its sub tree.

        Parameters
        ----------
        ui_obj : UI
            The component to mirror.
        parent_item : UIWorkbenchItem or None
            The parent node.
        role : str
            Role of the node, see :data:`ROLE_ROOT`.
        attr_label : str
            Attribute label the parent stores this component under.
        alive : set
            Identities visited during this pass, updated in place.

        Returns
        -------
        UIWorkbenchItem
            The node mirroring ``ui_obj``.
        """
        alive.add(id(ui_obj))

        item = self._by_ui.get(id(ui_obj))
        if item is None:
            item = self._create_item(ui_obj)
            # A component added to a container after that container joined the
            # scene never got its actors registered. Re-adding them is a no-op
            # for actors already there and repairs the ones that are missing.
            if self.scene is not None:
                add_ui_to_scene(self.scene.ui_scene, ui_obj)

        item.parent = parent_item
        item.role = role
        item.is_internal = role == ROLE_PART or (
            parent_item is not None and parent_item.is_internal
        )
        if attr_label:
            item.attr_label = attr_label

        children = []
        for child_ui, child_role, child_label in iter_child_specs(ui_obj):
            if id(child_ui) in alive:
                continue
            children.append(
                self._sync_node(child_ui, item, child_role, child_label, alive)
            )
        item.children = children

        self._install_hooks(item)
        return item

    def _create_item(self, ui_obj, name=None):
        """
        Build a fresh node for a component and register it.

        Parameters
        ----------
        ui_obj : UI
            The component to wrap.
        name : str, optional
            Display name. Defaults to ``<classname>_<id>``.

        Returns
        -------
        UIWorkbenchItem
            The newly created node.
        """
        item_id = self._next_id
        self._next_id += 1

        item = UIWorkbenchItem(
            id=item_id,
            name=name or f"{ui_obj.__class__.__name__.lower()}_{item_id}",
            ui=ui_obj,
        )
        if self.enable_universal_drag and not has_native_drag(ui_obj):
            item.draggable = True

        self.items[item_id] = item
        self._by_ui[id(ui_obj)] = item
        return item

    def _forget(self, item) -> None:
        """
        Drop a node from the workbench bookkeeping.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node to forget.
        """
        key = id(item.ui)
        self.items.pop(item.id, None)
        if self._by_ui.get(key) is item:
            del self._by_ui[key]
        self.multi_selected_ids.discard(item.id)
        if self.selected_item is item:
            self.selected_item = None
        if self.hit_item is item:
            self.hit_item = None
        if self._dragging is item:
            self._dragging = None

    def find_item_by_ui(self, ui):
        """
        Look up the node wrapping a component.

        Parameters
        ----------
        ui : UI
            The component to look up.

        Returns
        -------
        UIWorkbenchItem or None
            The matching node, or None when the component is unknown.
        """
        return self._by_ui.get(id(ui))

    def iter_items(self, *, include_internal=False):
        """
        Iterate over every node of the hierarchy.

        Parameters
        ----------
        include_internal : bool, optional
            When True, internal widget parts are yielded as well.

        Yields
        ------
        UIWorkbenchItem
            The nodes of the hierarchy.
        """
        for item in self.items.values():
            if include_internal or not item.is_internal:
                yield item

    # ------------------------------------------------------------------
    # Event interception, selection and dragging
    # ------------------------------------------------------------------

    def _install_hooks(self, item) -> None:
        """
        Wrap every callback slot of a component with a workbench hook.

        The hook records the event, drives selection and dragging, then
        forwards to the callback the component had before, so the widget keeps
        behaving exactly as it did.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node whose component should be hooked.
        """
        for name in HANDLER_NAMES:
            handler = getattr(item.ui, name, None)
            if getattr(handler, "workbench_hook", False):
                continue
            item._original_handlers[name] = handler
            setattr(item.ui, name, self._make_hook(item, name, handler))

    def _remove_hooks(self, item) -> None:
        """
        Restore the callbacks a component had before it was hooked.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node whose component should be unhooked.
        """
        for name, handler in item._original_handlers.items():
            if getattr(getattr(item.ui, name, None), "workbench_hook", False):
                setattr(item.ui, name, handler)
        item._original_handlers.clear()

    def _make_hook(self, item, name, original):
        """
        Build the wrapper installed in place of a component callback.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node owning the callback.
        name : str
            Name of the callback slot.
        original : callable or None
            The callback the component had before hooking.

        Returns
        -------
        callable
            The wrapper to install on the component.
        """

        def hook(event):
            """
            Record an event, then let the workbench and the widget react.

            Parameters
            ----------
            event : object
                The pygfx event handed to the callback.
            """
            consumed = self._on_ui_event(item, name, event)
            if not consumed and original is not None:
                original(event)

        hook.workbench_hook = True
        hook.workbench_original = original
        return hook

    def _on_ui_event(self, item, name, event) -> bool:
        """
        React to an event received by a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node whose component received the event.
        name : str
            Name of the callback slot that fired.
        event : object
            The pygfx event object.

        Returns
        -------
        bool
            True when the workbench consumed the event and the component's own
            callback must be skipped.
        """
        item.event_counts[name] += 1
        item.last_event = name
        if item.trace_events:
            self.log_event(name, item.display_name, describe_event(event))

        if name == "on_left_mouse_button_pressed":
            self.hit_item = item
            target = item.owner
            self.select(target)
            if target.draggable:
                target._drag_start_click = np.array([event.x, event.y], dtype=float)
                target._drag_start_pos = target.get_position()
                self._dragging = target
                self.log_event("drag start", target.display_name, describe_event(event))
                return True
            return False

        if name == "on_left_mouse_button_dragged":
            target = self._dragging
            if target is None:
                return False
            start_click, start_pos = target._drag_start_click, target._drag_start_pos
            if start_click is None or start_pos is None:
                return False
            delta = np.array([event.x, event.y], dtype=float) - start_click
            target.set_position(start_pos + delta)
            return True

        if name == "on_left_mouse_button_released":
            target = self._dragging
            if target is not None:
                position = target.get_position()
                self.log_event(
                    "drag end",
                    target.display_name,
                    f"({position[0]:.0f}, {position[1]:.0f})",
                )
                target._drag_start_click = None
                target._drag_start_pos = None
                self._dragging = None
                return True
            return False

        return False

    def select(self, item, *, additive=False) -> None:
        """
        Make a node the current selection.

        Parameters
        ----------
        item : UIWorkbenchItem or None
            The node to select, or None to clear the selection.
        additive : bool, optional
            When True, the node is added to the multi-selection instead of
            replacing it.
        """
        self.selected_item = item
        if item is None:
            self.multi_selected_ids.clear()
            return
        if not additive:
            self.multi_selected_ids.clear()
        self.multi_selected_ids.add(item.id)

    # ------------------------------------------------------------------
    # Selection outline drawn in the viewport
    # ------------------------------------------------------------------

    def _ensure_outline(self) -> None:
        """Create the four bars of the selection outline once."""
        if self._outline or self.scene is None:
            return
        for _ in range(4):
            bar = Rectangle2D(size=(1, 1), color=(1.0, 0.75, 0.1), opacity=1.0)
            bar.z_order = 25
            bar.set_visibility(False)
            self._outline.append(bar)
        for bar in self._outline:
            add_ui_to_scene(self.scene.ui_scene, bar)
        self._outline_attached = True

    def update_outline(self) -> None:
        """Move the selection outline onto the currently selected component."""
        if self.scene is None:
            return
        if not self.show_outline or self.selected_item is None:
            for bar in self._outline:
                bar.set_visibility(False)
            return

        self._ensure_outline()
        x, y, width, height = self.selected_item.get_bounds()
        thickness = 2.0
        boxes = (
            ((x, y), (max(1.0, width), thickness)),
            ((x, y + height - thickness), (max(1.0, width), thickness)),
            ((x, y), (thickness, max(1.0, height))),
            ((x + width - thickness, y), (thickness, max(1.0, height))),
        )
        for bar, (position, size) in zip(self._outline, boxes, strict=False):
            bar.resize(size)
            bar.set_position(np.array([position[0], position[1]], dtype=float))
            bar.set_visibility(True)

    # ------------------------------------------------------------------
    # Registration helpers kept for direct scripting
    # ------------------------------------------------------------------

    def register(self, ui, parent=None, name=None, *, add_to_scene=True):
        """
        Register a component in the workbench, optionally adding it to a scene.

        Parameters
        ----------
        ui : UI
            The component to register.
        parent : UIWorkbenchItem, optional
            Node the component should be parented to. When given, the
            component is added to that container.
        name : str, optional
            Display name of the new node.
        add_to_scene : bool, optional
            When True and no parent is given, the component is added to the
            scene.

        Returns
        -------
        UIWorkbenchItem
            The node wrapping the component.
        """
        item = self._by_ui.get(id(ui)) or self._create_item(ui, name=name)
        if name:
            item.name = name

        if parent is not None:
            self.parent_element(item, parent)
            return item

        if add_to_scene and self.scene is not None and ui not in self.scene.ui_elements:
            self.scene.add(ui)

        alive: set[int] = set()
        self._sync_node(ui, None, ROLE_ROOT, "", alive)
        if item not in self.root_items:
            self.root_items.append(item)
        return item

    def unregister(self, item, *, remove_from_scene=True) -> None:
        """
        Remove a node, its sub tree and optionally its actors from the scene.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node to remove.
        remove_from_scene : bool, optional
            When True, the component is also detached from the scene.
        """
        for child in list(item.children):
            self.unregister(child, remove_from_scene=False)

        parent = item.parent
        if parent is not None:
            if item in parent.children:
                parent.children.remove(item)
            if isinstance(parent.ui, Panel2D) and item.ui in parent.ui._elements:
                try:
                    parent.ui.remove_element(item.ui)
                except ValueError:
                    pass
        elif item in self.root_items:
            self.root_items.remove(item)

        if remove_from_scene and self.scene is not None:
            if item.ui in self.scene.ui_elements:
                self.scene.ui_elements.remove(item.ui)
            try:
                remove_ui_from_scene(self.scene.ui_scene, item.ui)
            except Exception:
                pass

        self._remove_hooks(item)
        self._forget(item)

    # ------------------------------------------------------------------
    # Hierarchy operations
    # ------------------------------------------------------------------

    def parent_element(self, child_item, parent_item, offset=None, anchor="position"):
        """
        Move a component into a container, keeping its position on screen.

        Parameters
        ----------
        child_item : UIWorkbenchItem
            The node to parent.
        parent_item : UIWorkbenchItem
            The container node to parent into. Must wrap a Panel2D.
        offset : (float, float), optional
            Offset from the top-left corner of the container. Computed from
            the current positions when omitted.
        anchor : str, optional
            Panel anchor, either ``"position"`` or ``"center"``.

        Returns
        -------
        bool
            True when the component was parented.
        """
        if child_item is parent_item or child_item in parent_item.children:
            return False
        if not isinstance(parent_item.ui, Panel2D):
            self.status_message = f"{parent_item.name} is not a Panel2D container"
            return False
        if self._is_ancestor(child_item, parent_item):
            self.status_message = "Cannot parent a component into its own child"
            return False

        if child_item.parent is not None:
            self.unparent_element(child_item)

        if offset is None:
            offset = child_item.get_position() - parent_item.get_position()
        offset = (int(round(float(offset[0]))), int(round(float(offset[1]))))

        if child_item in self.root_items:
            self.root_items.remove(child_item)
        if self.scene is not None and child_item.ui in self.scene.ui_elements:
            self.scene.ui_elements.remove(child_item.ui)

        parent_item.ui.add_element(child_item.ui, offset, anchor=anchor)

        if self.scene is not None:
            add_ui_to_scene(self.scene.ui_scene, child_item.ui)

        child_item.parent = parent_item
        child_item.role = ROLE_CONTENT
        child_item.is_internal = parent_item.is_internal
        if child_item not in parent_item.children:
            parent_item.children.append(child_item)

        self.log_event("parented", child_item.name, f"into {parent_item.name}")
        self.status_message = f"Parented {child_item.name} into {parent_item.name}"
        return True

    def unparent_element(self, child_item):
        """
        Detach a component from its container back into the scene.

        Parameters
        ----------
        child_item : UIWorkbenchItem
            The node to detach.

        Returns
        -------
        bool
            True when the component was detached.
        """
        old_parent = child_item.parent
        if old_parent is None:
            return False

        abs_pos = child_item.get_position()

        if (
            isinstance(old_parent.ui, Panel2D)
            and child_item.ui in old_parent.ui._elements
        ):
            old_parent.ui.remove_element(child_item.ui)
        if child_item in old_parent.children:
            old_parent.children.remove(child_item)

        child_item.parent = None
        child_item.role = ROLE_ROOT
        child_item.is_internal = False

        if child_item not in self.root_items:
            self.root_items.append(child_item)
        if self.scene is not None and child_item.ui not in self.scene.ui_elements:
            self.scene.ui_elements.append(child_item.ui)
            add_ui_to_scene(self.scene.ui_scene, child_item.ui)

        child_item.set_position(abs_pos)

        self.log_event("unparented", child_item.name, f"from {old_parent.name}")
        self.status_message = f"Unparented {child_item.name} to the scene root"
        return True

    def reparent_element(
        self, child_item, new_parent_item, offset=None, anchor="position"
    ):
        """
        Move a component from its current container into another one.

        Parameters
        ----------
        child_item : UIWorkbenchItem
            The node to move.
        new_parent_item : UIWorkbenchItem
            The destination container node.
        offset : (float, float), optional
            Offset inside the destination container. Keeps the on screen
            position when omitted.
        anchor : str, optional
            Panel anchor, either ``"position"`` or ``"center"``.

        Returns
        -------
        bool
            True when the component was moved.
        """
        abs_pos = child_item.get_position()
        self.unparent_element(child_item)
        if offset is None:
            offset = abs_pos - new_parent_item.get_position()
        return self.parent_element(
            child_item, new_parent_item, offset=offset, anchor=anchor
        )

    def _is_ancestor(self, candidate, item):
        """
        Check whether a node is an ancestor of another one.

        Parameters
        ----------
        candidate : UIWorkbenchItem
            The possible ancestor.
        item : UIWorkbenchItem
            The node to test.

        Returns
        -------
        bool
            True when ``candidate`` is on the parent chain of ``item``.
        """
        node = item.parent
        while node is not None:
            if node is candidate:
                return True
            node = node.parent
        return False

    def merge_elements(
        self,
        items_to_merge,
        padding=15.0,
        panel_name=None,
        color=(0.15, 0.15, 0.18),
        opacity=0.85,
    ):
        """
        Wrap several components in a new panel sized to their bounding box.

        Parameters
        ----------
        items_to_merge : list of UIWorkbenchItem
            The nodes to gather.
        padding : float, optional
            Padding in pixels between the bounding box and the panel border.
        panel_name : str, optional
            Name of the created panel node.
        color : tuple, optional
            RGB color of the created panel.
        opacity : float, optional
            Opacity of the created panel.

        Returns
        -------
        UIWorkbenchItem or None
            The node of the created panel, or None when nothing was merged.
        """
        valid_items = [item for item in items_to_merge if not item.is_internal]
        if not valid_items:
            return None

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        for item in valid_items:
            x, y, width, height = item.get_bounds()
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + width), max(max_y, y + height)

        panel_x = max(0.0, min_x - padding)
        panel_y = max(0.0, min_y - padding)
        panel_w = max(50.0, (max_x - min_x) + 2 * padding)
        panel_h = max(50.0, (max_y - min_y) + 2 * padding)

        panel_ui = Panel2D(
            size=(panel_w, panel_h),
            position=(panel_x, panel_y),
            color=color,
            opacity=opacity,
            has_border=True,
            border_width=2,
            border_color=(0.4, 0.6, 0.9),
        )
        name = panel_name or f"merged_panel_{self._next_id}"
        new_panel_item = self.register(panel_ui, parent=None, name=name)

        for item in valid_items:
            offset = item.get_position() - np.array([panel_x, panel_y])
            self.parent_element(item, new_panel_item, offset=offset)

        self.select(new_panel_item)
        self.log_event("merged", new_panel_item.name, f"{len(valid_items)} components")
        self.status_message = (
            f"Merged {len(valid_items)} components into {new_panel_item.name}"
        )
        return new_panel_item

    def duplicate_element(self, item, offset=(25.0, 25.0)):
        """
        Create a copy of a component next to the original.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node to duplicate.
        offset : (float, float), optional
            Offset applied to the copy.

        Returns
        -------
        UIWorkbenchItem or None
            The node of the copy, or None when the type is not supported.
        """
        position = item.get_position() + np.array(offset, dtype=float)
        new_item = self.create_element(
            item.type_name,
            position=tuple(position),
            size=item.get_size(),
            color=item.get_color(),
            opacity=item.get_opacity(),
            label=item.get_text() or "Clone",
            parent=item.parent,
        )
        if new_item is not None:
            self.select(new_item)
            self.log_event("duplicated", item.name, f"as {new_item.name}")
        return new_item

    # ------------------------------------------------------------------
    # Component creation
    # ------------------------------------------------------------------

    def create_element(
        self,
        element_type,
        *,
        position=(100.0, 100.0),
        size=(150.0, 50.0),
        color=(0.2, 0.5, 0.8),
        opacity=0.9,
        label="New UI",
        parent=None,
    ):
        """
        Instantiate a FURY UI component and add it to the workbench.

        Parameters
        ----------
        element_type : str
            Class name of the component, see :attr:`create_types`.
        position : (float, float), optional
            Absolute position of the new component.
        size : (float, float), optional
            Requested size of the new component.
        color : tuple, optional
            RGB color of the new component.
        opacity : float, optional
            Opacity of the new component.
        label : str, optional
            Text used by components that display a label.
        parent : UIWorkbenchItem, optional
            Container node to add the component to.

        Returns
        -------
        UIWorkbenchItem or None
            The node of the new component, or None when creation failed.
        """
        pos = (float(position[0]), float(position[1]))
        sz = (int(max(10, size[0])), int(max(10, size[1])))

        builders = {
            "Panel2D": lambda: Panel2D(
                size=sz,
                position=pos,
                color=color,
                opacity=opacity,
                has_border=self.create_border,
                border_width=self.create_border_width,
            ),
            "TextButton2D": lambda: TextButton2D(label=label, position=pos, size=sz),
            "LineSlider2D": lambda: LineSlider2D(
                position=pos, initial_value=50, min_value=0, max_value=100, length=sz[0]
            ),
            "LineDoubleSlider2D": lambda: LineDoubleSlider2D(
                position=pos,
                initial_values=(25, 75),
                min_value=0,
                max_value=100,
                length=sz[0],
            ),
            "RingSlider2D": lambda: RingSlider2D(
                center=pos,
                initial_value=45,
                min_value=0,
                max_value=100,
                slider_inner_radius=max(10, int(sz[0] // 2) - 15),
                slider_outer_radius=max(20, int(sz[0] // 2)),
            ),
            "TextBlock2D": lambda: TextBlock2D(
                text=label,
                position=pos,
                size=sz,
                color=(1, 1, 1),
                bg_color=color,
                font_size=18,
            ),
            "TextBox2D": lambda: TextBox2D(
                width=max(10, sz[0] // 12),
                height=max(1, sz[1] // 25),
                text=label,
                position=pos,
            ),
            "Checkbox": lambda: Checkbox(
                labels=["Option A", "Option B", "Option C"],
                checked_labels=["Option A"],
                position=pos,
            ),
            "RadioButton": lambda: RadioButton(
                labels=["Choice 1", "Choice 2", "Choice 3"],
                checked_labels=["Choice 1"],
                position=pos,
            ),
            "ComboBox2D": lambda: ComboBox2D(
                items=["Item 1", "Item 2", "Item 3", "Item 4"], position=pos, size=sz
            ),
            "ListBox2D": lambda: ListBox2D(
                values=["Entry A", "Entry B", "Entry C", "Entry D"],
                position=pos,
                size=sz,
            ),
            "Rectangle2D": lambda: Rectangle2D(
                size=sz, position=pos, color=color, opacity=opacity
            ),
            "RoundedRectangle2D": lambda: RoundedRectangle2D(
                size=sz,
                position=pos,
                color=color,
                opacity=opacity,
                corner_radius=12.0,
            ),
            "Disk2D": lambda: Disk2D(
                outer_radius=max(10, int(sz[0] // 2)),
                inner_radius=0,
                center=pos,
                color=color,
                opacity=opacity,
            ),
            "TabUI": lambda: TabUI(
                position=pos, size=sz, tab_titles=["Tab 1", "Tab 2"], startup_tab_id=0
            ),
            "PlaybackPanel": lambda: PlaybackPanel(position=pos, width=max(200, sz[0])),
            "RangeSlider": lambda: RangeSlider(
                length=max(60, sz[0]),
                range_slider_center=pos,
                value_slider_center=(pos[0], pos[1] + max(60, sz[1])),
            ),
        }

        builder = builders.get(element_type)
        if builder is None:
            self.status_message = f"Unknown element type: {element_type}"
            return None

        try:
            ui_instance = builder()
        except Exception as err:
            self.status_message = f"Error creating {element_type}: {err}"
            self.log_event("create failed", element_type, str(err))
            return None

        if parent is not None:
            item = self._create_item(ui_instance)
            self.parent_element(item, parent, offset=pos)
        else:
            item = self.register(ui_instance, parent=None, add_to_scene=True)

        self.select(item)
        self.log_event("created", item.name, element_type)
        self.status_message = f"Created {item.name} ({element_type})"
        return item

    def create_preset(self, preset_name, position=(80.0, 80.0)):
        """
        Build one of the ready made composite layouts.

        Parameters
        ----------
        preset_name : str
            One of ``"Settings Dialog"``, ``"Form Card"`` or ``"Audio Mixer"``.
        position : (float, float), optional
            Absolute position of the created panel.

        Returns
        -------
        UIWorkbenchItem or None
            The node of the created panel, or None for an unknown preset.
        """
        px, py = float(position[0]), float(position[1])
        recipes = {
            "Settings Dialog": (
                "settings_dialog",
                Panel2D(
                    size=(320, 300),
                    position=(px, py),
                    color=(0.14, 0.16, 0.22),
                    opacity=0.92,
                    has_border=True,
                    border_width=2,
                    border_color=(0.3, 0.5, 0.8),
                ),
                [
                    (
                        lambda: TextBlock2D(
                            text="Settings",
                            font_size=20,
                            color=(1, 1, 1),
                            bg_color=(0.2, 0.25, 0.35),
                            size=(290, 34),
                        ),
                        (15, 15),
                    ),
                    (
                        lambda: Checkbox(
                            labels=["Enable Shadows", "Anti-Aliasing", "VSync"],
                            checked_labels=["Anti-Aliasing"],
                        ),
                        (15, 65),
                    ),
                    (
                        lambda: LineSlider2D(
                            initial_value=75,
                            min_value=0,
                            max_value=100,
                            length=270,
                            text_template="Volume: {value:.0f}%",
                        ),
                        (25, 200),
                    ),
                    (lambda: TextButton2D(label="Apply", size=(140, 32)), (15, 250)),
                ],
            ),
            "Form Card": (
                "form_card",
                Panel2D(
                    size=(300, 230),
                    position=(px, py),
                    color=(0.18, 0.18, 0.2),
                    opacity=0.9,
                    has_border=True,
                    border_width=1,
                ),
                [
                    (
                        lambda: TextBlock2D(
                            text="User Profile",
                            font_size=18,
                            color=(1, 1, 1),
                            bg_color=(0.26, 0.26, 0.3),
                            size=(270, 30),
                        ),
                        (15, 15),
                    ),
                    (
                        lambda: TextBox2D(width=20, height=1, text="Alice Doe"),
                        (15, 65),
                    ),
                    (lambda: TextButton2D(label="Save", size=(100, 32)), (15, 170)),
                ],
            ),
            "Audio Mixer": (
                "audio_mixer",
                Panel2D(
                    size=(380, 210),
                    position=(px, py),
                    color=(0.12, 0.12, 0.15),
                    opacity=0.9,
                    has_border=True,
                    border_width=2,
                ),
                [
                    (
                        lambda: LineSlider2D(
                            initial_value=80,
                            min_value=0,
                            max_value=100,
                            length=190,
                            text_template="Master: {value:.0f} dB",
                        ),
                        (25, 55),
                    ),
                    (
                        lambda: RingSlider2D(
                            initial_value=50,
                            min_value=0,
                            max_value=100,
                            slider_inner_radius=22,
                            slider_outer_radius=36,
                        ),
                        (250, 20),
                    ),
                    (lambda: TextButton2D(label="Mute", size=(90, 30)), (25, 150)),
                ],
            ),
        }

        recipe = recipes.get(preset_name)
        if recipe is None:
            self.status_message = f"Unknown preset: {preset_name}"
            return None

        name, panel_ui, children = recipe
        panel_item = self.register(panel_ui, name=name)
        for factory, offset in children:
            child_item = self._create_item(factory())
            self.parent_element(child_item, panel_item, offset=offset)

        self.select(panel_item)
        self.status_message = f"Created preset: {preset_name}"
        self.log_event("preset", panel_item.name, preset_name)
        return panel_item

    # ------------------------------------------------------------------
    # Fuzzy and stress experiments
    # ------------------------------------------------------------------

    def _get_target_items(self, targets):
        """
        Resolve a target selector into a list of nodes.

        Parameters
        ----------
        targets : str
            Either ``"selected"`` or ``"all"``.

        Returns
        -------
        list
            The nodes the operation should apply to.
        """
        if targets == "selected":
            selected = [
                self.items[item_id]
                for item_id in self.multi_selected_ids
                if item_id in self.items
            ]
            if selected:
                return selected
            if self.selected_item is not None:
                return [self.selected_item]
        return list(self.iter_items())

    def fuzzy_randomize_colors(self, targets="selected"):
        """
        Give random colors to the targeted components.

        Parameters
        ----------
        targets : str, optional
            Either ``"selected"`` or ``"all"``.
        """
        items = self._get_target_items(targets)
        for item in items:
            item.set_color(
                (
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0),
                    random.uniform(0.1, 1.0),
                )
            )
        self.status_message = f"Randomized colors on {len(items)} components"
        self.log_event("fuzzy colors", f"{len(items)} components")

    def fuzzy_randomize_positions(self, targets="selected", max_delta=35.0):
        """
        Jitter the position of the targeted components.

        Parameters
        ----------
        targets : str, optional
            Either ``"selected"`` or ``"all"``.
        max_delta : float, optional
            Maximum jitter applied on each axis, in pixels.
        """
        items = self._get_target_items(targets)
        for item in items:
            pos = item.get_position()
            item.set_position(
                (
                    max(0.0, pos[0] + random.uniform(-max_delta, max_delta)),
                    max(0.0, pos[1] + random.uniform(-max_delta, max_delta)),
                )
            )
        self.status_message = f"Jittered {len(items)} components"
        self.log_event("fuzzy positions", f"{len(items)} components", f"±{max_delta}px")

    def fuzzy_randomize_sizes(self, targets="selected", scale_min=0.7, scale_max=1.3):
        """
        Scale the targeted components by a random factor.

        Parameters
        ----------
        targets : str, optional
            Either ``"selected"`` or ``"all"``.
        scale_min : float, optional
            Lower bound of the random scale factor.
        scale_max : float, optional
            Upper bound of the random scale factor.
        """
        items = [item for item in self._get_target_items(targets) if item.can_resize]
        for item in items:
            width, height = item.get_size()
            item.set_size(
                (
                    max(20.0, width * random.uniform(scale_min, scale_max)),
                    max(20.0, height * random.uniform(scale_min, scale_max)),
                )
            )
        self.status_message = f"Scaled {len(items)} components"
        self.log_event(
            "fuzzy sizes", f"{len(items)} components", f"[{scale_min}, {scale_max}]"
        )

    def fuzzy_extreme_stress_test(self, target_item):
        """
        Push a component through extreme geometry values and restore it.

        Parameters
        ----------
        target_item : UIWorkbenchItem
            The node to stress.

        Returns
        -------
        int
            Number of stress steps that completed without raising.
        """
        original_size = target_item.get_size()
        original_pos = target_item.get_position()
        original_z = target_item.get_z_order()

        steps = [
            ("minimum size", lambda: target_item.set_size((1.0, 1.0))),
            ("oversize", lambda: target_item.set_size((1600.0, 1200.0))),
            ("negative position", lambda: target_item.set_position((-50.0, -50.0))),
            ("high z-order", lambda: target_item.set_z_order(999)),
            ("restore size", lambda: target_item.set_size(original_size)),
            ("restore position", lambda: target_item.set_position(original_pos)),
            ("restore z-order", lambda: target_item.set_z_order(original_z)),
        ]

        succeeded = 0
        for description, action in steps:
            try:
                action()
                succeeded += 1
            except Exception as err:
                self.log_event(
                    "stress failed", target_item.name, f"{description}: {err}"
                )

        self.status_message = (
            f"Stress test on {target_item.name}: {succeeded}/{len(steps)} steps passed"
        )
        self.log_event("stress test", target_item.name, f"{succeeded}/{len(steps)}")
        return succeeded

    def run_reparent_stress_test(self, child_item, target_panel, cycles=5):
        """
        Parent and unparent a component repeatedly to check for actor leaks.

        Parameters
        ----------
        child_item : UIWorkbenchItem
            The node to move back and forth.
        target_panel : UIWorkbenchItem
            The container node used as destination.
        cycles : int, optional
            Number of parent/unparent cycles to run.

        Returns
        -------
        bool
            True when the number of actors in the scene is unchanged.
        """
        start_actors = len(self.scene.ui_scene.children) if self.scene else 0
        for _ in range(cycles):
            self.parent_element(child_item, target_panel, offset=(20, 20))
            self.unparent_element(child_item)
        end_actors = len(self.scene.ui_scene.children) if self.scene else 0

        delta = end_actors - start_actors
        message = f"Reparent test ({cycles} cycles): actor delta = {delta}"
        self.status_message = message
        self.log_event("reparent stress", child_item.name, message)
        return delta == 0

    # ------------------------------------------------------------------
    # ImGui drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label(text):
        """
        Draw a line of text without letting ImGui interpret format markers.

        Parameters
        ----------
        text : str
            The text to draw.
        """
        imgui.text_unformatted(text)

    @staticmethod
    def _muted(text):
        """
        Draw a dimmed line of text.

        Parameters
        ----------
        text : str
            The text to draw.
        """
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.62, 1.0))
        imgui.text_unformatted(text)
        imgui.pop_style_color()

    @staticmethod
    def _heading(color, text):
        """
        Draw a colored heading.

        Parameters
        ----------
        color : tuple
            The RGBA color of the heading.
        text : str
            The text to draw.
        """
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
        imgui.text_unformatted(text)
        imgui.pop_style_color()

    @staticmethod
    def _short_repr(value):
        """
        Build a single line, length limited representation of a value.

        Parameters
        ----------
        value : object
            The value to describe.

        Returns
        -------
        str
            A compact representation, truncated when too long.
        """
        try:
            text = " ".join(repr(value).split())
        except Exception as err:
            text = f"<unrepresentable: {err}>"
        if len(text) > 96:
            text = text[:93] + "..."
        return text

    @staticmethod
    def _as_number_list(value):
        """
        Read a value as a short list of numbers when possible.

        Parameters
        ----------
        value : object
            The value to convert.

        Returns
        -------
        list or None
            The numbers as floats, or None when the value is not a short
            numeric sequence.
        """
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                return None
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 4:
            return None
        numbers = []
        for entry in value:
            if isinstance(entry, (bool, np.bool_)) or not isinstance(
                entry, (int, float, np.integer, np.floating)
            ):
                return None
            numbers.append(float(entry))
        return numbers

    @staticmethod
    def _rebuild_sequence(original, numbers):
        """
        Rebuild a numeric sequence using the container type of the original.

        Parameters
        ----------
        original : object
            The value the numbers were read from.
        numbers : list
            The edited numbers.

        Returns
        -------
        object
            The numbers wrapped in the same kind of container.
        """
        if isinstance(original, np.ndarray):
            return np.array(numbers, dtype=float)
        if isinstance(original, tuple):
            return tuple(numbers)
        return list(numbers)

    def _value_editor(self, label, value, *, editable=True):
        """
        Draw the widget that best fits a Python value.

        Parameters
        ----------
        label : str
            ImGui label, optionally carrying an ``##id`` suffix.
        value : object
            The current value.
        editable : bool, optional
            When False the value is only displayed.

        Returns
        -------
        tuple
            A ``(changed, new_value)`` pair.
        """
        name = label.split("##")[0].strip()

        if not editable or value is None:
            self._muted(f"{name} = {self._short_repr(value)}")
            return False, value

        if isinstance(value, (bool, np.bool_)):
            return imgui.checkbox(label, bool(value))

        if isinstance(value, (int, np.integer)):
            return imgui.drag_int(label, int(value), 1.0, -100000, 100000)

        if isinstance(value, (float, np.floating)):
            speed = max(0.01, abs(float(value)) * 0.01)
            return imgui.drag_float(label, float(value), speed, -100000.0, 100000.0)

        if isinstance(value, str):
            if "\n" in value:
                return imgui.input_text_multiline(label, value, imgui.ImVec2(0, 70))
            return imgui.input_text(label, value)

        numbers = self._as_number_list(value)
        if numbers is not None:
            is_color = (
                "color" in name.lower()
                and len(numbers) in (3, 4)
                and all(0.0 <= number <= 1.0 for number in numbers)
            )
            if is_color:
                editor = imgui.color_edit3 if len(numbers) == 3 else imgui.color_edit4
                changed, edited = editor(label, numbers)
                return changed, self._rebuild_sequence(value, edited)

            if len(numbers) == 1:
                changed, edited = imgui.drag_float(
                    label, numbers[0], 1.0, -10000.0, 10000.0
                )
                edited = [edited]
            else:
                editor = {
                    2: imgui.drag_float2,
                    3: imgui.drag_float3,
                    4: imgui.drag_float4,
                }[len(numbers)]
                changed, edited = editor(label, numbers, 1.0, -10000.0, 10000.0)
            return changed, self._rebuild_sequence(value, edited)

        self._muted(f"{name} = {self._short_repr(value)}")
        return False, value

    # ------------------------------------------------------------------
    # ImGui panel
    # ------------------------------------------------------------------

    def render(self) -> None:
        """
        Draw the workbench panel for the current ImGui frame.

        Pass this method as ``imgui_draw_function`` of the
        :class:`fury.window.ShowManager`.
        """
        if imgui is None:
            return

        if self.auto_sync:
            self.sync()
        self.update_outline()

        viewport = imgui.get_main_viewport()
        panel_w = max(380.0, float(self.panel_width))
        imgui.set_next_window_pos(
            imgui.ImVec2(
                viewport.pos.x + viewport.size.x - panel_w, float(viewport.pos.y)
            ),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(panel_w, float(viewport.size.y)), imgui.Cond_.always
        )

        window_flags = (
            imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
        )
        expanded, _ = imgui.begin(self.title, True, window_flags)
        if not expanded:
            imgui.end()
            return

        content = sum(1 for _ in self.iter_items())
        self._heading((0.35, 0.8, 1.0, 1.0), f"{content} components in scene")
        imgui.same_line()
        self._muted(f"| {len(self.items)} nodes")
        self._muted(self.status_message)
        imgui.separator()

        if imgui.begin_tab_bar("WorkbenchTabs"):
            if imgui.begin_tab_item("Hierarchy")[0]:
                self._render_hierarchy_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Inspector")[0]:
                self._render_inspector_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Create")[0]:
                self._render_builder_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Experiments")[0]:
                self._render_fuzzy_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Monitor")[0]:
                self._render_monitor_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.end()

    # ------------------------------------------------------------------
    # Hierarchy tab
    # ------------------------------------------------------------------

    def _visible_children(self, item):
        """
        List the child nodes the hierarchy view should draw.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node whose children are requested.

        Returns
        -------
        list
            The children, filtered by the "show internals" option.
        """
        if self.show_internals:
            return list(item.children)
        return [child for child in item.children if not child.is_internal]

    def _matches_filter(self, item):
        """
        Check whether a node or one of its descendants matches the filter.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node to test.

        Returns
        -------
        bool
            True when the node should stay visible.
        """
        needle = self.hierarchy_filter.strip().lower()
        if not needle:
            return True
        if needle in item.display_name.lower() or needle in item.type_name.lower():
            return True
        return any(self._matches_filter(child) for child in item.children)

    def _render_hierarchy_tab(self) -> None:
        """Draw the scene tree and the hierarchy operations."""
        if imgui.button("Rescan"):
            self.scan_scene()
        imgui.same_line()
        _, self.auto_sync = imgui.checkbox("Auto", self.auto_sync)
        imgui.same_line()
        _, self.show_internals = imgui.checkbox("Internals", self.show_internals)
        imgui.same_line()
        _, self.show_outline = imgui.checkbox("Outline", self.show_outline)

        imgui.set_next_item_width(-90)
        _, self.hierarchy_filter = imgui.input_text("Filter", self.hierarchy_filter)
        imgui.same_line()
        if imgui.small_button("Clear"):
            self.hierarchy_filter = ""

        if imgui.small_button("Expand all"):
            self._force_tree_state = True
        imgui.same_line()
        if imgui.small_button("Collapse all"):
            self._force_tree_state = False
        imgui.same_line()
        if imgui.small_button("Deselect"):
            self.select(None)

        imgui.begin_child(
            "HierarchyTree", imgui.ImVec2(0, 260), imgui.ChildFlags_.borders
        )
        if not self.root_items:
            self._muted("No UI component in the scene. Use the Create tab.")
        else:
            for root_item in list(self.root_items):
                self._render_tree_node(root_item)
        imgui.end_child()
        self._force_tree_state = None

        imgui.separator()
        self._render_hierarchy_operations()

    def _render_tree_node(self, item) -> None:
        """
        Draw one node of the hierarchy tree and recurse into its children.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node to draw.
        """
        if not self._matches_filter(item):
            return

        children = self._visible_children(item)
        imgui.push_id(item.id)

        changed, checked = imgui.checkbox("##pick", item.id in self.multi_selected_ids)
        if changed:
            if checked:
                self.multi_selected_ids.add(item.id)
            else:
                self.multi_selected_ids.discard(item.id)
        imgui.same_line()

        flags = (
            imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.span_avail_width
        )
        if self.selected_item is item:
            flags |= imgui.TreeNodeFlags_.selected
        if not children:
            flags |= imgui.TreeNodeFlags_.leaf
        if self._force_tree_state is not None:
            imgui.set_next_item_open(self._force_tree_state)

        if item.role == ROLE_PART:
            color = imgui.ImVec4(0.62, 0.62, 0.66, 1.0)
        elif isinstance(item.ui, Panel2D):
            color = imgui.ImVec4(0.45, 0.85, 1.0, 1.0)
        else:
            color = imgui.ImVec4(0.88, 0.9, 0.92, 1.0)

        marker = "o" if item.role == ROLE_PART else "*"
        label = f"{marker} {item.display_name}  [{item.type_name}]"

        imgui.push_style_color(imgui.Col_.text, color)
        opened = imgui.tree_node_ex(label, flags)
        imgui.pop_style_color()

        if imgui.is_item_clicked():
            self.select(item, additive=imgui.get_io().key_ctrl)
        self._render_node_tooltip(item)
        self._render_node_context_menu(item)

        if opened:
            for child in children:
                self._render_tree_node(child)
            imgui.tree_pop()

        imgui.pop_id()

    def _render_node_tooltip(self, item) -> None:
        """
        Show the geometry of a node when its tree row is hovered.

        Parameters
        ----------
        item : UIWorkbenchItem
            The hovered node.
        """
        if not imgui.is_item_hovered():
            return
        x, y, width, height = item.get_bounds()
        imgui.begin_tooltip()
        self._label(item.path)
        self._muted(f"role: {item.role}   id: {item.id}")
        self._muted(f"position: ({x:.0f}, {y:.0f})   size: {width:.0f} x {height:.0f}")
        self._muted(f"z-order: {item.get_z_order()}   visible: {item.get_visible()}")
        self._muted(f"draggable: {item.draggable}   children: {len(item.children)}")
        imgui.end_tooltip()

    def _render_node_context_menu(self, item) -> None:
        """
        Draw the right-click menu of a tree row.

        Parameters
        ----------
        item : UIWorkbenchItem
            The node the menu acts on.
        """
        if not imgui.begin_popup_context_item("##ctx"):
            return

        self._muted(item.display_name)
        imgui.separator()
        if imgui.menu_item("Select", "", False)[0]:
            self.select(item)
        if imgui.menu_item("Toggle visible", "", False)[0]:
            item.set_visible(not item.get_visible())
        if imgui.menu_item("Toggle draggable", "", False)[0]:
            item.draggable = not item.draggable
        if imgui.menu_item("Trace events", "", item.trace_events)[0]:
            item.trace_events = not item.trace_events
        imgui.separator()
        if imgui.menu_item("Duplicate", "", False, not item.is_internal)[0]:
            self.duplicate_element(item)
        if imgui.menu_item("Unparent", "", False, item.parent is not None)[0]:
            self.unparent_element(item)
        if imgui.menu_item("Delete", "", False, not item.is_internal)[0]:
            self.unregister(item)
        imgui.end_popup()

    def _panel_targets(self, exclude=None):
        """
        List the panels a component can be parented into.

        Parameters
        ----------
        exclude : UIWorkbenchItem, optional
            A node to leave out, together with its descendants.

        Returns
        -------
        list
            The candidate container nodes.
        """
        targets = []
        for item in self.items.values():
            if not isinstance(item.ui, Panel2D) or item.is_internal:
                continue
            if exclude is not None and (
                item is exclude or self._is_ancestor(exclude, item)
            ):
                continue
            targets.append(item)
        return targets

    def _render_hierarchy_operations(self) -> None:
        """Draw the parenting, duplication and merging controls."""
        self._heading((0.4, 0.9, 0.5, 1.0), "Hierarchy operations")

        item = self.selected_item
        if item is None:
            self._muted("Select a component in the tree above.")
        else:
            self._label(f"Selected: {item.path}")
            self._muted(f"{item.type_name}  |  role: {item.role}  |  id: {item.id}")

            targets = self._panel_targets(exclude=item)
            if targets:
                names = [target.name for target in targets]
                self._parent_combo_idx = min(self._parent_combo_idx, len(names) - 1)
                _, self._parent_combo_idx = imgui.combo(
                    "Target panel", self._parent_combo_idx, names
                )
                target = targets[self._parent_combo_idx]
                if imgui.button("Parent into panel"):
                    if item.parent is None:
                        self.parent_element(item, target)
                    else:
                        self.reparent_element(item, target)
                imgui.same_line()
            else:
                self._muted("No panel available as a parent.")

            if imgui.button("Unparent") and item.parent is not None:
                self.unparent_element(item)
            imgui.same_line()
            if imgui.button("Duplicate"):
                self.duplicate_element(item)
            imgui.same_line()
            if imgui.button("Delete"):
                self.unregister(item)

        imgui.separator()
        self._heading((0.95, 0.6, 0.25, 1.0), "Merge into a new panel")
        picked = [
            self.items[item_id]
            for item_id in sorted(self.multi_selected_ids)
            if item_id in self.items
        ]
        self._muted(f"{len(picked)} component(s) ticked in the tree")

        _, self.merge_padding = imgui.slider_float(
            "Padding", self.merge_padding, 0.0, 60.0
        )
        _, self.merge_color = imgui.color_edit3("Panel color", self.merge_color)

        can_merge = len(picked) >= 2
        if not can_merge:
            imgui.begin_disabled()
        if imgui.button("Merge ticked components"):
            self.merge_elements(
                picked, padding=self.merge_padding, color=tuple(self.merge_color)
            )
        if not can_merge:
            imgui.end_disabled()
            imgui.same_line()
            self._muted("tick at least two")

    # ------------------------------------------------------------------
    # Inspector tab
    # ------------------------------------------------------------------

    def _render_inspector_tab(self) -> None:
        """Draw every readable and writable property of the selection."""
        item = self.selected_item
        if item is None:
            self._muted("No component selected. Pick one in the Hierarchy tab")
            self._muted("or click a component directly in the viewport.")
            return

        imgui.begin_child("InspectorBody", imgui.ImVec2(0, 0))

        self._heading((0.35, 0.8, 1.0, 1.0), f"{item.type_name}  #{item.id}")
        self._muted(item.path)

        changed, new_name = imgui.input_text("Name", item.name)
        if changed:
            item.name = new_name

        changed, draggable = imgui.checkbox("Draggable", item.draggable)
        if changed:
            item.draggable = draggable
            self.log_event("draggable", item.name, str(draggable))
        imgui.same_line()
        changed, visible = imgui.checkbox("Visible", item.get_visible())
        if changed:
            item.set_visible(visible)
        imgui.same_line()
        _, item.trace_events = imgui.checkbox("Trace", item.trace_events)

        if item.parent is not None and imgui.small_button("Select parent"):
            self.select(item.parent)

        self._render_transform_section(item)
        self._render_appearance_section(item)
        self._render_text_section(item)
        self._render_widget_section(item)
        self._render_all_properties_section(item)
        self._render_attributes_section(item)
        self._render_actors_section(item)
        self._render_events_section(item)

        imgui.end_child()

    def _render_transform_section(self, item) -> None:
        """
        Draw the geometry controls of a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header("Transform", imgui.TreeNodeFlags_.default_open):
            return

        position = item.get_position()
        changed, new_position = imgui.drag_float2(
            "Position", [float(position[0]), float(position[1])], 1.0, -2000.0, 4000.0
        )
        if changed:
            item.set_position(new_position)

        width, height = item.get_size()
        if not item.can_resize:
            imgui.begin_disabled()
        changed, new_size = imgui.drag_float2("Size", [width, height], 1.0, 1.0, 4000.0)
        if changed:
            item.set_size(new_size)
        if not item.can_resize:
            imgui.end_disabled()
            self._muted(f"{item.type_name} keeps the size it was built with.")

        changed, new_z = imgui.slider_int("Z-order", item.get_z_order(), -10, 50)
        if changed:
            item.set_z_order(new_z)

        x, y, width, height = item.get_bounds()
        self._muted(f"bounds: x {x:.0f}  y {y:.0f}  w {width:.0f}  h {height:.0f}")
        anchors = getattr(item.ui, "_anchors", None)
        if anchors is not None:
            self._muted(f"anchors: {anchors[0]} / {anchors[1]}")

        if imgui.small_button("Snap to origin"):
            item.set_position((0.0, 0.0))
        imgui.same_line()
        if imgui.small_button("Center on canvas"):
            canvas = UIContext.canvas_size
            item.set_position(
                (
                    float(canvas[0]) / 2.0 - width / 2.0,
                    float(canvas[1]) / 2.0 - height / 2.0,
                )
            )

    def _render_appearance_section(self, item) -> None:
        """
        Draw the color, opacity and border controls of a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header("Appearance", imgui.TreeNodeFlags_.default_open):
            return

        changed, new_color = imgui.color_edit3("Color", list(item.get_color()))
        if changed:
            item.set_color(new_color)

        changed, new_opacity = imgui.slider_float(
            "Opacity", item.get_opacity(), 0.0, 1.0
        )
        if changed:
            item.set_opacity(new_opacity)

        if hasattr(item.ui, "corner_radius"):
            radius = float(getattr(item.ui, "corner_radius", 0.0))
            changed, new_radius = imgui.slider_float("Corner radius", radius, 0.0, 80.0)
            if changed:
                try:
                    item.ui.corner_radius = new_radius
                except Exception as err:
                    self.status_message = f"corner_radius: {err}"

        if isinstance(item.ui, Panel2D) and item.ui.has_border:
            self._muted("Borders")
            for side in item.ui.border_sides:
                border = item.ui.borders[side]
                imgui.push_id(f"border_{side}")
                changed, color = imgui.color_edit3(f"{side} color", list(border.color))
                if changed:
                    item.ui.border_color = [side, color]
                is_vertical = side in ("left", "right")
                current = float(border.width if is_vertical else border.height)
                changed, thickness = imgui.slider_float(
                    f"{side} width", current, 0.0, 20.0
                )
                if changed:
                    item.ui.border_width = [side, thickness]
                imgui.pop_id()

    def _render_text_section(self, item) -> None:
        """
        Draw the typography controls of a component that carries text.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        has_text = (
            hasattr(item.ui, "message")
            or hasattr(item.ui, "default_label")
            or hasattr(item.ui, "font_size")
        )
        if not has_text:
            return
        if not imgui.collapsing_header("Text", imgui.TreeNodeFlags_.default_open):
            return

        text = item.get_text()
        if "\n" in text:
            changed, new_text = imgui.input_text_multiline(
                "Content", text, imgui.ImVec2(0, 80)
            )
        else:
            changed, new_text = imgui.input_text("Content", text)
        if changed:
            item.set_text(new_text)

        changed, new_size = imgui.slider_int("Font size", item.get_font_size(), 6, 96)
        if changed:
            item.set_font_size(new_size)

    def _render_widget_section(self, item) -> None:
        """
        Draw the controls that only make sense for the selected widget type.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        ui_obj = item.ui
        renderers = (
            (Slider2D, self._render_slider_controls),
            (LineDoubleSlider2D, self._render_double_slider_controls),
            (ButtonGroup, self._render_button_group_controls),
            (Button2D, self._render_button_controls),
            (ComboBox2D, self._render_combo_controls),
            (ListBox2D, self._render_list_controls),
            (TextBox2D, self._render_textbox_controls),
            (TabUI, self._render_tab_controls),
            (PlaybackPanel, self._render_playback_controls),
        )
        renderer = None
        for widget_type, candidate in renderers:
            if isinstance(ui_obj, widget_type):
                renderer = candidate
                break
        if renderer is None:
            return

        title = f"{ui_obj.__class__.__name__} controls"
        if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
            return
        renderer(item)

    def _render_slider_controls(self, item) -> None:
        """
        Draw the value and range controls of a slider.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        slider = item.ui
        low, high = float(slider.min_value), float(slider.max_value)
        changed, value = imgui.slider_float("Value", float(slider.value), low, high)
        if changed:
            self._apply_value(item, "value", value, setattr)

        changed, low_edit = imgui.drag_float("Min", low, 1.0, -100000.0, 100000.0)
        if changed:
            self._apply_value(item, "min_value", low_edit, setattr)
        changed, high_edit = imgui.drag_float("Max", high, 1.0, -100000.0, 100000.0)
        if changed:
            self._apply_value(item, "max_value", high_edit, setattr)

        self._muted(f"ratio: {float(slider.ratio):.3f}")
        if hasattr(slider, "angle"):
            angle = float(slider.angle) * 180.0 / float(np.pi)
            self._muted(f"angle: {angle:.1f} deg")

    def _render_double_slider_controls(self, item) -> None:
        """
        Draw the two handle values of a double slider.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        slider = item.ui
        low, high = float(slider.min_value), float(slider.max_value)
        changed, left = imgui.slider_float(
            "Left value", float(slider.left_disk_value), low, high
        )
        if changed:
            self._apply_value(item, "left_disk_value", left, setattr)
        changed, right = imgui.slider_float(
            "Right value", float(slider.right_disk_value), low, high
        )
        if changed:
            self._apply_value(item, "right_disk_value", right, setattr)

    def _render_button_group_controls(self, item) -> None:
        """
        Draw one toggle per option of a checkbox or radio button group.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        group = item.ui
        checked = list(group.checked_labels)
        for label in list(getattr(group, "labels", [])):
            changed, is_checked = imgui.checkbox(f"{label}##opt", label in checked)
            if not changed:
                continue
            try:
                if is_checked:
                    group.select(label)
                else:
                    group.deselect(label)
                self.log_event("option", item.name, f"{label}={is_checked}")
            except Exception as err:
                self.status_message = f"{item.name}: {err}"
        self._muted(f"checked: {group.checked_labels}")

    def _render_button_controls(self, item) -> None:
        """
        Draw the state controls of a button.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        button = item.ui
        changed, enabled = imgui.checkbox("Enabled", bool(button.enabled))
        if changed:
            self._apply_value(item, "enabled", enabled, setattr)
        if getattr(button, "is_toggle", False):
            changed, toggled = imgui.checkbox("Toggled", bool(button.toggled))
            if changed:
                self._apply_value(item, "toggled", toggled, setattr)
        self._muted(
            f"hovered: {getattr(button, 'is_hovered', False)}"
            f"   pressed: {getattr(button, 'is_pressed', False)}"
        )
        if imgui.button("Trigger click"):
            try:
                button.do_click()
                self.log_event("click", item.name, "triggered from inspector")
            except Exception as err:
                self.status_message = f"{item.name}: {err}"

    def _render_combo_controls(self, item) -> None:
        """
        Draw the entries of a combo box.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        combo = item.ui
        self._label(f"selected: {combo.selected_text}")
        self._muted(f"index: {combo.selected_text_index}")
        for index, entry in enumerate(list(getattr(combo, "items", []))):
            self._muted(f"  [{index}] {entry}")

    def _render_list_controls(self, item) -> None:
        """
        Draw the entries and the selection of a list box.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        listbox = item.ui
        self._muted(f"multiselection: {getattr(listbox, 'multiselection', False)}")
        self._label(f"selected: {list(getattr(listbox, 'selected', []))}")
        if imgui.small_button("Clear selection"):
            try:
                listbox.clear_selection()
            except Exception as err:
                self.status_message = f"{item.name}: {err}"
        for index, entry in enumerate(list(getattr(listbox, "values", []))):
            self._muted(f"  [{index}] {entry}")

    def _render_textbox_controls(self, item) -> None:
        """
        Draw the editable content of a text box.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        textbox = item.ui
        changed, text = imgui.input_text("Message", str(textbox.text))
        if changed:
            try:
                textbox.set_message(text)
            except Exception as err:
                self.status_message = f"{item.name}: {err}"
        self._muted(f"caret: {getattr(textbox, 'caret_pos', 0)}")

    def _render_tab_controls(self, item) -> None:
        """
        Draw the tab selector of a tab container.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        tab_ui = item.ui
        self._muted(f"tabs: {tab_ui.nb_tabs}   active: {tab_ui.active_tab_idx}")
        for index, tab_title in enumerate(list(getattr(tab_ui, "tab_titles", []))):
            if imgui.small_button(f"{tab_title}##tab_{index}"):
                try:
                    tab_ui.select_tab(index)
                except Exception as err:
                    self.status_message = f"{item.name}: {err}"
            imgui.same_line()
        imgui.new_line()

    def _render_playback_controls(self, item) -> None:
        """
        Draw the timeline controls of a playback panel.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        panel = item.ui
        final_time = max(1.0, float(panel.final_time))
        changed, current = imgui.slider_float(
            "Current time", float(panel.current_time), 0.0, final_time
        )
        if changed:
            self._apply_value(item, "current_time", current, setattr)
        changed, final = imgui.drag_float("Final time", final_time, 1.0, 1.0, 100000.0)
        if changed:
            self._apply_value(item, "final_time", final, setattr)
        changed, speed = imgui.slider_float("Speed", float(panel.speed), 0.1, 8.0)
        if changed:
            self._apply_value(item, "speed", speed, setattr)

    def _render_all_properties_section(self, item) -> None:
        """
        Draw every property declared by the class of a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header(
            "All properties", imgui.TreeNodeFlags_.default_open
        ):
            return

        current_class = None
        for name, prop, class_name in iter_ui_properties(item.ui):
            if not self._property_applies(item.ui, name):
                continue
            if class_name != current_class:
                current_class = class_name
                imgui.separator_text(class_name)

            try:
                value = getattr(item.ui, name)
            except Exception as err:
                self._muted(f"{name} = <error: {err}>")
                continue

            editable = prop.fset is not None and name not in ASYMMETRIC_PROPERTIES
            changed, new_value = self._value_editor(
                f"{name}##prop_{item.id}", value, editable=editable
            )
            if changed:
                self._apply_value(item, name, new_value, setattr)

    @staticmethod
    def _property_applies(ui_obj, name):
        """
        Check whether reading a property makes sense for a component.

        Some getters warn or raise when the feature they describe is turned
        off, so the inspector skips them instead of reading them every frame.

        Parameters
        ----------
        ui_obj : UI
            The inspected component.
        name : str
            Name of the property.

        Returns
        -------
        bool
            True when the property should be read and shown.
        """
        if name in ("border_color", "border_width"):
            return bool(getattr(ui_obj, "has_border", False))
        return True

    def _render_attributes_section(self, item) -> None:
        """
        Draw the plain instance attributes of a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header("Instance attributes"):
            return

        _, self.show_private_attrs = imgui.checkbox(
            "Show private attributes", self.show_private_attrs
        )

        for name, value in sorted(vars(item.ui).items()):
            if name.startswith("_") and not self.show_private_attrs:
                continue
            if name.startswith("on_") or callable(value):
                continue

            if isinstance(value, UI):
                child_item = self.find_item_by_ui(value)
                self._muted(f"{name} = {value.__class__.__name__}")
                if child_item is not None:
                    imgui.same_line()
                    if imgui.small_button(f"select##attr_{name}"):
                        self.select(child_item)
                continue

            changed, new_value = self._value_editor(
                f"{name}##attr_{item.id}", value, editable=True
            )
            if changed:
                self._apply_value(item, name, new_value, setattr)

    def _apply_value(self, item, name, value, setter) -> None:
        """
        Write a value onto a component and report failures in the status bar.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        name : str
            Name of the attribute or property to write.
        value : object
            The value to write.
        setter : callable
            The function used to write, normally :func:`setattr`.
        """
        try:
            setter(item.ui, name, value)
            self.status_message = f"{item.name}.{name} = {self._short_repr(value)}"
            self.log_event("set", f"{item.name}.{name}", self._short_repr(value))
        except Exception as err:
            self.status_message = f"{item.name}.{name}: {err}"
            self.log_event("set failed", f"{item.name}.{name}", str(err))

    def _render_actors_section(self, item) -> None:
        """
        Draw the pygfx actors backing a component.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header("Actors"):
            return

        try:
            actors = list(item.ui.actors)
        except Exception as err:
            self._muted(f"<unavailable: {err}>")
            return

        if not actors:
            self._muted("This component draws through its sub components only.")
            return

        for index, actor in enumerate(actors):
            self._label(f"[{index}] {actor.__class__.__name__}")
            local = getattr(actor, "local", None)
            if local is not None:
                self._muted(f"    local: ({local.x:.1f}, {local.y:.1f}, {local.z:.3f})")
            self._muted(
                f"    render_order: {getattr(actor, 'render_order', 0)}"
                f"   visible: {getattr(actor, 'visible', True)}"
            )

    def _render_events_section(self, item) -> None:
        """
        Draw the callback slots of a component and how often they fired.

        Parameters
        ----------
        item : UIWorkbenchItem
            The inspected node.
        """
        if not imgui.collapsing_header("Events"):
            return

        if imgui.small_button("Reset counters"):
            item.event_counts.clear()
            item.last_event = ""
        imgui.same_line()
        self._muted(f"last: {item.last_event or '-'}")

        flags = (
            imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.sizing_stretch_prop
        )
        if not imgui.begin_table(f"events_{item.id}", 3, flags):
            return

        imgui.table_setup_column("callback")
        imgui.table_setup_column("bound")
        imgui.table_setup_column("count")
        imgui.table_headers_row()

        slots = list(HANDLER_NAMES)
        slots += sorted(
            name
            for name in vars(item.ui)
            if name.startswith("on_") and name not in HANDLER_NAMES
        )

        for name in slots:
            if name in item._original_handlers:
                original = item._original_handlers[name]
            else:
                original = getattr(item.ui, name, None)
            is_bound = original is not None and (
                getattr(original, "__name__", "<lambda>") != "<lambda>"
            )
            count = item.event_counts.get(name, 0)
            if not is_bound and count == 0:
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            self._label(name.replace("on_", ""))
            imgui.table_next_column()
            self._muted(getattr(original, "__name__", "-") if is_bound else "-")
            imgui.table_next_column()
            self._label(str(count))

        imgui.end_table()

    # ------------------------------------------------------------------
    # Create tab
    # ------------------------------------------------------------------

    def _render_builder_tab(self) -> None:
        """Draw the component creation form and the composite presets."""
        self._heading((0.3, 0.9, 0.7, 1.0), "Add a component")

        _, self.create_type_idx = imgui.combo(
            "Type", self.create_type_idx, self.create_types
        )
        element_type = self.create_types[self.create_type_idx]

        _, self.create_label = imgui.input_text("Label", self.create_label)
        _, self.create_pos = imgui.drag_float2(
            "Position", self.create_pos, 1.0, 0.0, 4000.0
        )
        _, self.create_size = imgui.drag_float2(
            "Size", self.create_size, 1.0, 10.0, 2000.0
        )
        _, self.create_color = imgui.color_edit3("Color", self.create_color)
        _, self.create_opacity = imgui.slider_float(
            "Opacity", self.create_opacity, 0.0, 1.0
        )
        _, self.create_border = imgui.checkbox("Panel border", self.create_border)
        if self.create_border:
            _, self.create_border_width = imgui.slider_float(
                "Border width", self.create_border_width, 1.0, 12.0
            )

        panels = self._panel_targets()
        options = ["Scene root"] + [f"Panel: {panel.name}" for panel in panels]
        self._create_dest_idx = min(self._create_dest_idx, len(options) - 1)
        _, self._create_dest_idx = imgui.combo(
            "Destination", self._create_dest_idx, options
        )
        destination = (
            panels[self._create_dest_idx - 1] if self._create_dest_idx else None
        )

        if imgui.button(f"Create {element_type}"):
            self.create_element(
                element_type,
                position=tuple(self.create_pos),
                size=tuple(self.create_size),
                color=tuple(self.create_color),
                opacity=self.create_opacity,
                label=self.create_label,
                parent=destination,
            )

        imgui.separator()
        self._heading((0.9, 0.8, 0.3, 1.0), "Composite presets")
        for preset in ("Settings Dialog", "Form Card", "Audio Mixer"):
            if imgui.button(preset):
                self.create_preset(preset, position=tuple(self.create_pos))

    # ------------------------------------------------------------------
    # Experiments tab
    # ------------------------------------------------------------------

    def _render_fuzzy_tab(self) -> None:
        """Draw the randomizers and the stress tests."""
        self._heading((1.0, 0.6, 0.2, 1.0), "Randomizers")
        target_mode = "selected" if self.selected_item is not None else "all"
        self._muted(f"Applies to: {target_mode.upper()} components")

        if imgui.button("Colors"):
            self.fuzzy_randomize_colors(target_mode)
        imgui.same_line()
        if imgui.button("Sizes"):
            self.fuzzy_randomize_sizes(
                target_mode, self.fuzzy_scale_min, self.fuzzy_scale_max
            )
        imgui.same_line()
        if imgui.button("Positions"):
            self.fuzzy_randomize_positions(target_mode, self.fuzzy_pos_delta)

        _, self.fuzzy_pos_delta = imgui.slider_float(
            "Position jitter", self.fuzzy_pos_delta, 5.0, 150.0
        )
        _, self.fuzzy_scale_min = imgui.slider_float(
            "Scale min", self.fuzzy_scale_min, 0.2, 1.0
        )
        _, self.fuzzy_scale_max = imgui.slider_float(
            "Scale max", self.fuzzy_scale_max, 1.0, 3.0
        )

        imgui.separator()
        self._heading((1.0, 0.35, 0.35, 1.0), "Boundary stress test")
        if self.selected_item is None:
            self._muted("Select a component to stress its geometry.")
        elif imgui.button(f"Stress {self.selected_item.name}"):
            self.fuzzy_extreme_stress_test(self.selected_item)

        imgui.separator()
        self._heading((0.5, 0.8, 1.0, 1.0), "Reparenting stress test")
        item = self.selected_item
        targets = self._panel_targets(exclude=item) if item is not None else []
        if item is None or not targets:
            self._muted("Select a component and keep one panel free.")
            return

        _, self.reparent_cycles = imgui.slider_int(
            "Cycles", self.reparent_cycles, 1, 40
        )
        if imgui.button(f"Run {self.reparent_cycles} cycles on {targets[0].name}"):
            self.run_reparent_stress_test(item, targets[0], self.reparent_cycles)

    # ------------------------------------------------------------------
    # Monitor tab
    # ------------------------------------------------------------------

    def _render_monitor_tab(self) -> None:
        """Draw the global UI context, scene counters and the event log."""
        self._heading((0.4, 0.8, 1.0, 1.0), "UI context")

        hot_ui = UIContext.hot_ui
        active_ui = UIContext.active_ui
        self._label(f"hovered:  {hot_ui.__class__.__name__ if hot_ui else '-'}")
        self._label(f"focused:  {active_ui.__class__.__name__ if active_ui else '-'}")
        self._label(f"canvas:   {tuple(int(v) for v in UIContext.canvas_size)}")
        self._label(f"z bounds: {tuple(int(v) for v in UIContext.z_order_bounds)}")

        if self.hit_item is not None:
            self._muted(f"last clicked: {self.hit_item.path}")

        imgui.separator()
        self._heading((0.4, 0.8, 1.0, 1.0), "Scene")
        if self.scene is not None:
            self._label(f"root components: {len(self.scene.ui_elements)}")
            self._label(f"actors in ui_scene: {len(self.scene.ui_scene.children)}")
        self._label(f"workbench nodes: {len(self.items)}")
        self._label(
            f"internal parts:  {sum(1 for i in self.items.values() if i.is_internal)}"
        )
        if self.show_manager is not None:
            self._muted(f"window: {getattr(self.show_manager, 'title', '-')}")

        imgui.separator()
        self._heading((0.85, 0.85, 0.35, 1.0), "Event log")
        imgui.same_line()
        if imgui.small_button("Clear"):
            self.event_log.clear()

        imgui.begin_child("EventLog", imgui.ImVec2(0, 0), imgui.ChildFlags_.borders)
        if not self.event_log:
            self._muted("No event recorded yet.")
        for entry in self.event_log:
            self._muted(f"[{entry['time']}]")
            imgui.same_line()
            self._heading((0.4, 1.0, 0.6, 1.0), entry["event"])
            imgui.same_line()
            self._label(f"{entry['target']} {entry['details']}".rstrip())
        imgui.end_child()


#: Backwards compatible alias of :class:`UIWorkbench`.
UIPlayground = UIWorkbench
