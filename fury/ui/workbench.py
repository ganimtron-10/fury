"""
FURY UI Workbench & Playground module.

Provides an interactive ImGui-powered side debug panel and workbench for
inspecting, adding, parenting, reparenting, merging, updating, and
stress-testing FURY UI components.
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
    Rectangle2D,
    RingSlider2D,
    RoundedRectangle2D,
    TabUI,
    TextBlock2D,
    TextBox2D,
    TextButton2D,
    UIContext,
)
from fury.window import add_ui_to_scene, remove_ui_from_scene

if imgui_bundle is not None:
    from imgui_bundle import imgui
else:
    imgui = None


@dataclass
class UIWorkbenchItem:
    """Represents a UI element inside the workbench hierarchy."""

    id: int
    name: str
    ui: UI
    parent: UIWorkbenchItem | None = None
    children: list[UIWorkbenchItem] = field(default_factory=list)
    is_internal: bool = False
    draggable: bool = False
    _drag_start_click: np.ndarray | None = None
    _drag_start_pos: np.ndarray | None = None
    _original_handlers: dict[str, Any] = field(default_factory=dict)

    @property
    def type_name(self) -> str:
        """Get class name of UI object."""
        return self.ui.__class__.__name__

    def get_position(self) -> np.ndarray:
        """Get current absolute position."""
        return np.array(self.ui.get_position(), dtype=float)

    def set_position(self, pos: tuple[float, float] | np.ndarray) -> None:
        """Set position and update parent panel offsets if parented."""
        pos_arr = np.array(pos, dtype=float)
        self.ui.set_position(pos_arr)
        if self.parent and isinstance(self.parent.ui, Panel2D):
            rel_offset = pos_arr - self.parent.get_position()
            if hasattr(self.parent.ui, "update_element_offset"):
                try:
                    self.parent.ui.update_element_offset(self.ui, rel_offset)
                except ValueError:
                    pass

    def get_size(self) -> tuple[float, float]:
        """Get size (width, height) of the UI component."""
        try:
            sz = self.ui.size
            if sz is not None:
                return float(sz[0]), float(sz[1])
        except Exception:
            pass
        return (100.0, 100.0)

    def set_size(self, size: tuple[float, float]) -> None:
        """Resize UI component if supported."""
        w, h = max(1.0, float(size[0])), max(1.0, float(size[1]))
        if hasattr(self.ui, "resize"):
            try:
                self.ui.resize((w, h))
                return
            except Exception:
                pass
        if hasattr(self.ui, "width") and hasattr(self.ui, "height"):
            try:
                self.ui.width = w
                self.ui.height = h
                return
            except Exception:
                pass
        if isinstance(self.ui, Disk2D):
            self.ui.outer_radius = w / 2.0
        elif isinstance(self.ui, LineSlider2D):
            self.ui.length = int(w)

    def get_color(self) -> tuple[float, float, float]:
        """Get RGB color in range [0, 1]."""
        if hasattr(self.ui, "color"):
            c = self.ui.color
            if c is not None and len(c) >= 3:
                return float(c[0]), float(c[1]), float(c[2])
        if hasattr(self.ui, "child") and hasattr(self.ui.child, "background"):
            c = self.ui.child.background.color
            if c is not None and len(c) >= 3:
                return float(c[0]), float(c[1]), float(c[2])
        return (1.0, 1.0, 1.0)

    def set_color(self, color: tuple[float, float, float] | list[float]) -> None:
        """Set color on UI component."""
        rgb = tuple(float(x) for x in color[:3])
        if hasattr(self.ui, "color"):
            try:
                self.ui.color = rgb
            except Exception:
                pass
        if hasattr(self.ui, "child") and hasattr(self.ui.child, "background"):
            try:
                self.ui.child.background.color = rgb
            except Exception:
                pass

    def get_opacity(self) -> float:
        """Get opacity [0, 1]."""
        if hasattr(self.ui, "opacity"):
            return float(self.ui.opacity)
        if hasattr(self.ui, "child") and hasattr(self.ui.child, "background"):
            return float(self.ui.child.background.opacity)
        return 1.0

    def set_opacity(self, opacity: float) -> None:
        """Set opacity [0, 1]."""
        val = max(0.0, min(1.0, float(opacity)))
        if hasattr(self.ui, "opacity"):
            try:
                self.ui.opacity = val
            except Exception:
                pass
        if hasattr(self.ui, "child") and hasattr(self.ui.child, "background"):
            try:
                self.ui.child.background.opacity = val
            except Exception:
                pass

    def get_z_order(self) -> int:
        """Get z_order."""
        return int(getattr(self.ui, "z_order", 0))

    def set_z_order(self, z: int) -> None:
        """Set z_order."""
        try:
            self.ui.z_order = int(z)
        except Exception:
            pass

    def get_text(self) -> str:
        """Get text or label message."""
        if hasattr(self.ui, "message"):
            return str(self.ui.message)
        if hasattr(self.ui, "default_label"):
            return str(self.ui.default_label)
        if hasattr(self.ui, "text"):
            if isinstance(self.ui.text, str):
                return self.ui.text
            if hasattr(self.ui.text, "message"):
                return str(self.ui.text.message)
        return ""

    def set_text(self, text: str) -> None:
        """Set text or label message."""
        if hasattr(self.ui, "message"):
            self.ui.message = text
        if hasattr(self.ui, "default_label"):
            self.ui.default_label = text
            if hasattr(self.ui, "child") and hasattr(self.ui.child, "message"):
                self.ui.child.message = text
        if hasattr(self.ui, "text"):
            if hasattr(self.ui.text, "message"):
                self.ui.text.message = text

    def get_font_size(self) -> int:
        """Get font size."""
        if hasattr(self.ui, "font_size"):
            return int(self.ui.font_size)
        return 18

    def set_font_size(self, size: int) -> None:
        """Set font size."""
        if hasattr(self.ui, "font_size"):
            try:
                self.ui.font_size = int(size)
            except Exception:
                pass


class UIWorkbench:
    """
    Interactive UI Workbench and Debugger for FURY.

    Provides a comprehensive ImGui side panel to inspect the active scene UI
    hierarchy, create new UI components, parent/reparent/merge them, live edit
    all properties, test fuzzy experiments, and diagnose dragging and layout
    bugs.
    """

    def __init__(
        self,
        scene=None,
        show_manager=None,
        *,
        title="UI Workbench & Debugger",
        panel_width=440,
        enable_universal_drag=True,
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
        self.multi_selected_ids: set[int] = set()

        # Event log ring buffer
        self.event_log: collections.deque = collections.deque(maxlen=40)
        self.status_message = "Ready"

        # Creation wizard defaults
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
        self.create_target_parent_id = 0  # 0 for root scene
        self.create_pos = [100.0, 100.0]
        self.create_size = [200.0, 80.0]
        self.create_label = "My Element"
        self.create_color = [0.2, 0.4, 0.8]
        self.create_opacity = 0.85
        self.create_border = False
        self.create_border_width = 2.0

        # Fuzzy controls state
        self.fuzzy_pos_delta = 30.0
        self.fuzzy_scale_min = 0.7
        self.fuzzy_scale_max = 1.4
        self.reparent_cycles = 5

        # Merge controls
        self.merge_padding = 15.0
        self.merge_panel_name = "merged_panel"
        self.merge_color = [0.12, 0.14, 0.18]

        # Scan existing items in scene if provided
        if self.scene is not None:
            self.scan_scene()

    def set_scene(self, scene) -> None:
        """Assign scene and scan for existing UI components."""
        self.scene = scene
        self.scan_scene()

    def set_show_manager(self, show_manager) -> None:
        """Assign ShowManager."""
        self.show_manager = show_manager

    def log_event(self, event_name: str, target_name: str, details: str = "") -> None:
        """Add entry to the workbench event monitor log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.event_log.appendleft(
            {
                "time": timestamp,
                "event": event_name,
                "target": target_name,
                "details": details,
            }
        )

    def scan_scene(self) -> None:
        """Scan scene.ui_elements and build the workbench item hierarchy."""
        if self.scene is None:
            return

        for ui_elem in list(self.scene.ui_elements):
            if not self.find_item_by_ui(ui_elem):
                self.register(ui_elem, parent=None, add_to_scene=False)

        self.status_message = f"Scanned scene ({len(self.items)} total items)"

    def find_item_by_ui(self, ui: UI) -> UIWorkbenchItem | None:
        """Lookup workbench item wrapping the given UI element."""
        for item in self.items.values():
            if item.ui is ui:
                return item
        return None

    def register(
        self,
        ui: UI,
        parent: UIWorkbenchItem | None = None,
        name: str | None = None,
        *,
        add_to_scene: bool = True,
    ) -> UIWorkbenchItem:
        """Register a UI component in the workbench hierarchy."""
        item_id = self._next_id
        self._next_id += 1

        if not name:
            name = f"{ui.__class__.__name__.lower()}_{item_id}"

        item = UIWorkbenchItem(id=item_id, name=name, ui=ui, parent=parent)
        self.items[item_id] = item

        if parent is None:
            self.root_items.append(item)
            if add_to_scene and self.scene is not None:
                self.scene.add(ui)
        else:
            parent.children.append(item)

        # Hook dragging support if enabled
        if self.enable_universal_drag:
            self._attach_drag_handlers(item)

        # Recursively register existing children (e.g. from containers)
        if hasattr(ui, "_children") and ui._children:
            for child_ui in ui._children:
                # Avoid registering internal background actors as separate root items
                if not self.find_item_by_ui(child_ui):
                    is_internal = (
                        hasattr(ui, "background") and child_ui is ui.background
                    )
                    child_item = self.register(
                        child_ui, parent=item, add_to_scene=False
                    )
                    child_item.is_internal = is_internal

        return item

    def unregister(
        self, item: UIWorkbenchItem, *, remove_from_scene: bool = True
    ) -> None:
        """Unregister item and its children from workbench and scene."""
        # Unregister children first
        for child in list(item.children):
            self.unregister(child, remove_from_scene=remove_from_scene)

        # Detach from parent
        if item.parent:
            if item in item.parent.children:
                item.parent.children.remove(item)
            if (
                isinstance(item.parent.ui, Panel2D)
                and item.ui in item.parent.ui._elements
            ):
                try:
                    item.parent.ui.remove_element(item.ui)
                except Exception:
                    pass
        elif item in self.root_items:
            self.root_items.remove(item)

        # Remove from scene
        if remove_from_scene and self.scene is not None:
            try:
                if item.ui in self.scene.ui_elements:
                    self.scene.ui_elements.remove(item.ui)
                remove_ui_from_scene(self.scene.ui_scene, item.ui)
            except Exception:
                pass

        if self.selected_item is item:
            self.selected_item = None
        self.multi_selected_ids.discard(item.id)

        if item.id in self.items:
            del self.items[item.id]

    def _attach_drag_handlers(self, item: UIWorkbenchItem) -> None:
        """Attach universal drag & drop handling to an item."""
        ui_obj = item.ui
        orig_pressed = getattr(ui_obj, "on_left_mouse_button_pressed", None)
        orig_dragged = getattr(ui_obj, "on_left_mouse_button_dragged", None)
        orig_released = getattr(ui_obj, "on_left_mouse_button_released", None)

        def _on_pressed(event):
            if orig_pressed:
                try:
                    orig_pressed(event)
                except Exception:
                    pass
            # Select this item in the workbench when clicked in viewport
            self.selected_item = item
            self.multi_selected_ids.clear()
            self.multi_selected_ids.add(item.id)
            self.log_event("Select", item.name, f"({event.x}, {event.y})")
            if item.draggable:
                item._drag_start_click = np.array([event.x, event.y], dtype=float)
                item._drag_start_pos = np.array(ui_obj.get_position(), dtype=float)
                self.log_event("Drag Start", item.name, f"({event.x}, {event.y})")

        def _on_dragged(event):
            if orig_dragged:
                try:
                    orig_dragged(event)
                except Exception:
                    pass
            if (
                item.draggable
                and item._drag_start_click is not None
                and item._drag_start_pos is not None
            ):
                current_click = np.array([event.x, event.y], dtype=float)
                delta = current_click - item._drag_start_click
                new_pos = item._drag_start_pos + delta
                item.set_position(new_pos)

        def _on_released(event):
            if orig_released:
                try:
                    orig_released(event)
                except Exception:
                    pass
            if item.draggable and item._drag_start_click is not None:
                item._drag_start_click = None
                item._drag_start_pos = None
                self.log_event("Drag End", item.name, f"pos={item.get_position()}")

        ui_obj.on_left_mouse_button_pressed = _on_pressed
        ui_obj.on_left_mouse_button_dragged = _on_dragged
        ui_obj.on_left_mouse_button_released = _on_released

    def parent_element(
        self,
        child_item: UIWorkbenchItem,
        parent_item: UIWorkbenchItem,
        offset: tuple[float, float] | None = None,
        anchor: str = "position",
    ) -> bool:
        """
        Parent a child UI component into a parent container.

        Maintains visual positioning and synchronizes actors in the graphics scene.
        """
        if child_item is parent_item or child_item in parent_item.children:
            return False
        if not isinstance(parent_item.ui, Panel2D):
            self.status_message = (
                f"Parent {parent_item.name} is not a Panel2D container"
            )
            return False

        # If already parented, unparent first
        if child_item.parent is not None:
            self.unparent_element(child_item)

        # Compute relative offset if not provided
        if offset is None:
            child_pos = child_item.get_position()
            parent_pos = parent_item.get_position()
            offset = (
                int(round(child_pos[0] - parent_pos[0])),
                int(round(child_pos[1] - parent_pos[1])),
            )
        else:
            if any(v > 1.0 or v < 0.0 for v in offset):
                offset = (int(round(offset[0])), int(round(offset[1])))

        # If child was a root item in scene, remove from root list
        if child_item in self.root_items:
            self.root_items.remove(child_item)
            if self.scene is not None and child_item.ui in self.scene.ui_elements:
                self.scene.ui_elements.remove(child_item.ui)

        # Add to parent Panel2D
        parent_item.ui.add_element(child_item.ui, offset, anchor=anchor)

        # Ensure actors are added to ui_scene if parent was already rendered
        if self.scene is not None:
            add_ui_to_scene(self.scene.ui_scene, child_item.ui)

        child_item.parent = parent_item
        if child_item not in parent_item.children:
            parent_item.children.append(child_item)

        self.log_event("Parented", child_item.name, f"to {parent_item.name}")
        self.status_message = f"Parented {child_item.name} -> {parent_item.name}"
        return True

    def unparent_element(self, child_item: UIWorkbenchItem) -> bool:
        """
        Unparent a child element, moving it to the root scene at its current
        world position.
        """
        old_parent = child_item.parent
        if old_parent is None:
            return False

        abs_pos = child_item.get_position()

        # Remove from parent container
        if isinstance(old_parent.ui, Panel2D):
            if child_item.ui in old_parent.ui._elements:
                old_parent.ui.remove_element(child_item.ui)
        if child_item in old_parent.children:
            old_parent.children.remove(child_item)

        child_item.parent = None

        # Add as root item
        if child_item not in self.root_items:
            self.root_items.append(child_item)

        if self.scene is not None and child_item.ui not in self.scene.ui_elements:
            self.scene.ui_elements.append(child_item.ui)

        child_item.set_position(abs_pos)

        self.log_event("Unparented", child_item.name, f"from {old_parent.name}")
        self.status_message = f"Unparented {child_item.name} to Root Scene"
        return True

    def reparent_element(
        self,
        child_item: UIWorkbenchItem,
        new_parent_item: UIWorkbenchItem,
        offset: tuple[float, float] | None = None,
        anchor: str = "position",
    ) -> bool:
        """Reparent an element from its current parent to a new parent."""
        abs_pos = child_item.get_position()
        self.unparent_element(child_item)
        if offset is None:
            offset = tuple(abs_pos - new_parent_item.get_position())
        return self.parent_element(
            child_item, new_parent_item, offset=offset, anchor=anchor
        )

    def merge_elements(
        self,
        items_to_merge: list[UIWorkbenchItem],
        padding: float = 15.0,
        panel_name: str | None = None,
        color: tuple[float, float, float] = (0.15, 0.15, 0.18),
        opacity: float = 0.85,
    ) -> UIWorkbenchItem | None:
        """
        Merge multiple UI components into a newly created parent Panel2D.

        Calculates the enclosing bounding box, creates a Panel2D covering it,
        and reparents each element with calculated relative coordinates.
        """
        if not items_to_merge:
            return None

        # Compute enclosing bounding box across all items
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        valid_items = [it for it in items_to_merge if not it.is_internal]
        if not valid_items:
            return None

        for it in valid_items:
            pos = it.get_position()
            sz = it.get_size()
            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0] + sz[0])
            max_y = max(max_y, pos[1] + sz[1])

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

        # Reparent all items into the new panel
        for it in valid_items:
            item_pos = it.get_position()
            rel_offset = item_pos - np.array([panel_x, panel_y])
            self.parent_element(it, new_panel_item, offset=tuple(rel_offset))

        self.selected_item = new_panel_item
        self.multi_selected_ids.clear()
        self.log_event(
            "Merged",
            new_panel_item.name,
            f"{len(valid_items)} items merged into panel",
        )
        self.status_message = (
            f"Merged {len(valid_items)} items into {new_panel_item.name}"
        )
        return new_panel_item

    def duplicate_element(
        self, item: UIWorkbenchItem, offset: tuple[float, float] = (25.0, 25.0)
    ) -> UIWorkbenchItem | None:
        """Create a duplicate clone of a UI component with an offset."""
        elem_type = item.type_name
        pos = item.get_position() + np.array(offset)
        sz = item.get_size()
        col = item.get_color()
        text = item.get_text()

        new_item = self.create_element(
            elem_type,
            position=tuple(pos),
            size=sz,
            color=col,
            label=text or "Clone",
            parent=item.parent,
        )
        if new_item:
            self.selected_item = new_item
            self.log_event("Duplicated", item.name, f"Clone {new_item.name} created")
        return new_item

    def create_element(
        self,
        element_type: str,
        *,
        position: tuple[float, float] = (100.0, 100.0),
        size: tuple[float, float] = (150.0, 50.0),
        color: tuple[float, float, float] = (0.2, 0.5, 0.8),
        opacity: float = 0.9,
        label: str = "New UI",
        parent: UIWorkbenchItem | None = None,
    ) -> UIWorkbenchItem | None:
        """Instantiate any supported FURY UI component and register it."""
        pos = (float(position[0]), float(position[1]))
        sz = (int(max(10, size[0])), int(max(10, size[1])))

        try:
            ui_instance: UI | None = None

            if element_type == "Panel2D":
                ui_instance = Panel2D(
                    size=sz,
                    position=pos,
                    color=color,
                    opacity=opacity,
                    has_border=self.create_border,
                    border_width=self.create_border_width,
                )
            elif element_type == "TextButton2D":
                ui_instance = TextButton2D(label=label, position=pos, size=sz)
            elif element_type == "LineSlider2D":
                ui_instance = LineSlider2D(
                    position=pos,
                    initial_value=50,
                    min_value=0,
                    max_value=100,
                    length=sz[0],
                )
            elif element_type == "LineDoubleSlider2D":
                ui_instance = LineDoubleSlider2D(
                    position=pos,
                    initial_values=(25, 75),
                    min_value=0,
                    max_value=100,
                    length=sz[0],
                )
            elif element_type == "RingSlider2D":
                radius = max(20, int(sz[0] // 2))
                ui_instance = RingSlider2D(
                    center=pos,
                    initial_value=45,
                    min_value=0,
                    max_value=100,
                    slider_inner_radius=max(10, radius - 15),
                    slider_outer_radius=radius,
                )
            elif element_type == "TextBlock2D":
                ui_instance = TextBlock2D(
                    text=label,
                    position=pos,
                    size=sz,
                    color=(1, 1, 1),
                    bg_color=color,
                    font_size=18,
                )
            elif element_type == "TextBox2D":
                chars = max(10, sz[0] // 12)
                lines = max(1, sz[1] // 25)
                ui_instance = TextBox2D(
                    width=chars, height=lines, text=label, position=pos
                )
            elif element_type == "Checkbox":
                ui_instance = Checkbox(
                    labels=["Option A", "Option B", "Option C"],
                    checked_labels=["Option A"],
                    position=pos,
                )
            elif element_type == "RadioButton":
                ui_instance = RadioButton(
                    labels=["Choice 1", "Choice 2", "Choice 3"],
                    checked_labels=["Choice 1"],
                    position=pos,
                )
            elif element_type == "ComboBox2D":
                ui_instance = ComboBox2D(
                    items=["Item 1", "Item 2", "Item 3", "Item 4"],
                    position=pos,
                    size=sz,
                )
            elif element_type == "ListBox2D":
                ui_instance = ListBox2D(
                    values=["Entry A", "Entry B", "Entry C", "Entry D"],
                    position=pos,
                    size=sz,
                )
            elif element_type == "Rectangle2D":
                ui_instance = Rectangle2D(
                    size=sz, position=pos, color=color, opacity=opacity
                )
            elif element_type == "RoundedRectangle2D":
                ui_instance = RoundedRectangle2D(
                    size=sz,
                    position=pos,
                    color=color,
                    opacity=opacity,
                    corner_radius=12.0,
                )
            elif element_type == "Disk2D":
                radius = max(10, int(sz[0] // 2))
                ui_instance = Disk2D(
                    outer_radius=radius,
                    inner_radius=0,
                    center=pos,
                    color=color,
                    opacity=opacity,
                )
            elif element_type == "TabUI":
                ui_instance = TabUI(
                    position=pos,
                    size=sz,
                    tab_titles=["Tab 1", "Tab 2"],
                    startup_tab_id=0,
                )
            elif element_type == "PlaybackPanel":
                ui_instance = PlaybackPanel(position=pos, size=sz)
            else:
                self.status_message = f"Unknown element type: {element_type}"
                return None

            if ui_instance is None:
                return None

            if parent is not None:
                item = self.register(ui_instance, parent=None, add_to_scene=False)
                self.parent_element(item, parent, offset=pos)
            else:
                item = self.register(ui_instance, parent=None, add_to_scene=True)

            self.selected_item = item
            self.log_event("Created", item.name, f"type={element_type}")
            self.status_message = f"Created {item.name} ({element_type})"
            return item

        except Exception as err:
            self.status_message = f"Error creating {element_type}: {err}"
            self.log_event("Create Error", element_type, str(err))
            return None

    def create_preset(
        self, preset_name: str, position: tuple[float, float] = (80.0, 80.0)
    ) -> UIWorkbenchItem | None:
        """Create composite pre-built UI components."""
        px, py = position
        if preset_name == "Settings Dialog":
            panel = Panel2D(
                size=(320, 280),
                position=(px, py),
                color=(0.14, 0.16, 0.22),
                opacity=0.92,
                has_border=True,
                border_width=2,
                border_color=(0.3, 0.5, 0.8),
            )
            p_item = self.register(panel, name="settings_dialog")

            title = TextBlock2D(
                text="Settings Panel",
                font_size=20,
                color=(1, 1, 1),
                bg_color=(0.2, 0.25, 0.35),
                size=(290, 35),
            )
            t_item = self.register(title, parent=None, add_to_scene=False)
            self.parent_element(t_item, p_item, offset=(15, 15))

            chk = Checkbox(
                labels=["Enable Shadows", "Anti-Aliasing", "VSync"],
                checked_labels=["Anti-Aliasing"],
            )
            c_item = self.register(chk, parent=None, add_to_scene=False)
            self.parent_element(c_item, p_item, offset=(15, 65))

            slider = LineSlider2D(
                initial_value=75,
                min_value=0,
                max_value=100,
                length=280,
                text_template="Volume: {value:.0f}%",
            )
            s_item = self.register(slider, parent=None, add_to_scene=False)
            self.parent_element(s_item, p_item, offset=(15, 175))

            btn = TextButton2D(label="Apply Settings", size=(140, 32))
            b_item = self.register(btn, parent=None, add_to_scene=False)
            self.parent_element(b_item, p_item, offset=(15, 230))

            self.status_message = "Created Settings Dialog Preset"
            return p_item

        elif preset_name == "Form Card":
            panel = Panel2D(
                size=(280, 220),
                position=(px, py),
                color=(0.18, 0.18, 0.2),
                opacity=0.9,
                has_border=True,
                border_width=1,
            )
            p_item = self.register(panel, name="form_card")

            header = TextBlock2D(
                text="User Profile",
                font_size=18,
                color=(1, 1, 1),
                size=(250, 30),
            )
            h_item = self.register(header, parent=None, add_to_scene=False)
            self.parent_element(h_item, p_item, offset=(15, 15))

            input_box = TextBox2D(width=20, height=1, text="Alice Doe")
            i_item = self.register(input_box, parent=None, add_to_scene=False)
            self.parent_element(i_item, p_item, offset=(15, 55))

            save_btn = TextButton2D(label="Save", size=(100, 32))
            s_item = self.register(save_btn, parent=None, add_to_scene=False)
            self.parent_element(s_item, p_item, offset=(15, 160))

            self.status_message = "Created Form Card Preset"
            return p_item

        elif preset_name == "Audio Mixer":
            panel = Panel2D(
                size=(360, 200),
                position=(px, py),
                color=(0.12, 0.12, 0.15),
                opacity=0.9,
                has_border=True,
            )
            p_item = self.register(panel, name="audio_mixer")

            master_slider = LineSlider2D(
                initial_value=80,
                min_value=0,
                max_value=100,
                length=200,
                text_template="Master: {value:.0f}dB",
            )
            ms_item = self.register(master_slider, parent=None, add_to_scene=False)
            self.parent_element(ms_item, p_item, offset=(20, 30))

            ring = RingSlider2D(
                center=(280, 70),
                initial_value=50,
                min_value=0,
                max_value=100,
                slider_inner_radius=25,
                slider_outer_radius=40,
            )
            r_item = self.register(ring, parent=None, add_to_scene=False)
            self.parent_element(r_item, p_item, offset=(220, 20))

            mute_btn = TextButton2D(label="Mute", size=(80, 30))
            m_item = self.register(mute_btn, parent=None, add_to_scene=False)
            self.parent_element(m_item, p_item, offset=(20, 140))

            self.status_message = "Created Audio Mixer Preset"
            return p_item

        return None

    # -------------------------------------------------------------------------
    # Fuzzy & Stress Experiments
    # -------------------------------------------------------------------------

    def fuzzy_randomize_colors(self, targets: str = "selected") -> None:
        """Randomize colors across selected items or all items."""
        items_to_modify = self._get_target_items(targets)
        for it in items_to_modify:
            r = random.uniform(0.1, 1.0)
            g = random.uniform(0.1, 1.0)
            b = random.uniform(0.1, 1.0)
            it.set_color((r, g, b))
        self.status_message = (
            f"Fuzzy: Randomized colors for {len(items_to_modify)} items"
        )
        self.log_event("Fuzzy Colors", f"{len(items_to_modify)} items")

    def fuzzy_randomize_positions(
        self, targets: str = "selected", max_delta: float = 35.0
    ) -> None:
        """Jitter / randomize positions to test bounds and dragging."""
        items_to_modify = self._get_target_items(targets)
        for it in items_to_modify:
            dx = random.uniform(-max_delta, max_delta)
            dy = random.uniform(-max_delta, max_delta)
            pos = it.get_position()
            new_pos = (max(0.0, pos[0] + dx), max(0.0, pos[1] + dy))
            it.set_position(new_pos)
        self.status_message = (
            f"Fuzzy: Jittered positions for {len(items_to_modify)} items"
        )
        self.log_event("Fuzzy Positions", f"jitter={max_delta}px")

    def fuzzy_randomize_sizes(
        self,
        targets: str = "selected",
        scale_min: float = 0.7,
        scale_max: float = 1.3,
    ) -> None:
        """Randomize sizes to test dynamic resizing and layouts."""
        items_to_modify = self._get_target_items(targets)
        for it in items_to_modify:
            scale_w = random.uniform(scale_min, scale_max)
            scale_h = random.uniform(scale_min, scale_max)
            w, h = it.get_size()
            new_w = max(20.0, w * scale_w)
            new_h = max(20.0, h * scale_h)
            it.set_size((new_w, new_h))
        self.status_message = f"Fuzzy: Scaled sizes for {len(items_to_modify)} items"
        self.log_event("Fuzzy Sizes", f"scale=[{scale_min:.1f}, {scale_max:.1f}]")

    def fuzzy_extreme_stress_test(self, target_item: UIWorkbenchItem) -> None:
        """Stress test an item with extreme boundary properties."""
        test_cases = [
            ("Zero size check", lambda: target_item.set_size((1.0, 1.0))),
            ("Oversize check", lambda: target_item.set_size((1600.0, 1200.0))),
            (
                "Negative coordinates check",
                lambda: target_item.set_position((-50.0, -50.0)),
            ),
            ("High Z-order check", lambda: target_item.set_z_order(999)),
            ("Restore normal size", lambda: target_item.set_size((200.0, 80.0))),
            ("Restore normal pos", lambda: target_item.set_position((100.0, 100.0))),
            ("Restore normal z_order", lambda: target_item.set_z_order(0)),
        ]
        succeeded = 0
        for desc, action in test_cases:
            try:
                action()
                succeeded += 1
            except Exception as e:
                self.log_event("Stress Failed", target_item.name, f"{desc}: {e}")
        self.status_message = (
            f"Extreme test on {target_item.name}: {succeeded}/{len(test_cases)} passed"
        )
        self.log_event("Extreme Test", target_item.name, f"{succeeded} passed")

    def run_reparent_stress_test(
        self,
        child_item: UIWorkbenchItem,
        target_panel: UIWorkbenchItem,
        cycles: int = 5,
    ) -> bool:
        """
        Stress test reparenting: rapidly parents and unparents an item in a
        loop to verify graphics scene synchronization and prevent actor leaks.
        """
        start_actors = (
            len(self.scene.ui_scene.children) if self.scene is not None else 0
        )
        for _i in range(cycles):
            self.parent_element(child_item, target_panel, offset=(20, 20))
            self.unparent_element(child_item)
        end_actors = len(self.scene.ui_scene.children) if self.scene is not None else 0
        ok = start_actors == end_actors
        delta = end_actors - start_actors
        msg = f"Reparent test ({cycles} cycles): Actor delta = {delta}"
        self.status_message = msg
        self.log_event("Reparent Stress", child_item.name, msg)
        return ok

    def _get_target_items(self, targets: str) -> list[UIWorkbenchItem]:
        """Get items matching target selector ('selected' or 'all')."""
        if targets == "selected":
            if self.multi_selected_ids:
                return [
                    self.items[mid]
                    for mid in self.multi_selected_ids
                    if mid in self.items and not self.items[mid].is_internal
                ]
            if self.selected_item and not self.selected_item.is_internal:
                return [self.selected_item]
        return [it for it in self.items.values() if not it.is_internal]

    # -------------------------------------------------------------------------
    # ImGui Side Debug Panel
    # -------------------------------------------------------------------------

    def render(self) -> None:
        """
        Render the UI Workbench debug panel inside the ImGui frame.

        Call this inside your ShowManager `imgui_draw_function` callback.
        """
        if imgui is None:
            return

        # Position and size the debug panel window on the right side
        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        vp_pos = viewport.pos

        panel_w = max(380.0, float(self.panel_width))
        imgui.set_next_window_pos(
            (vp_pos.x + vp_size.x - panel_w, vp_pos.y),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            (panel_w, vp_size.y),
            imgui.Cond_.always,
        )

        window_flags = (
            imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
        )

        expanded, _ = imgui.begin("FURY UI Workbench & Playground", True, window_flags)
        if not expanded:
            imgui.end()
            return

        # Header status bar
        imgui.text_colored((0.3, 0.8, 1.0, 1.0), "Workbench:")
        imgui.same_line()
        imgui.text(self.status_message)
        imgui.separator()

        # Tab Bar Navigation
        if imgui.begin_tab_bar("WorkbenchTabs"):
            # Tab 1: Hierarchy
            if imgui.begin_tab_item("Hierarchy")[0]:
                self._render_hierarchy_tab()
                imgui.end_tab_item()

            # Tab 2: Add UI & Presets
            if imgui.begin_tab_item("Add / Builder")[0]:
                self._render_builder_tab()
                imgui.end_tab_item()

            # Tab 3: Live Property Inspector
            if imgui.begin_tab_item("Inspector")[0]:
                self._render_inspector_tab()
                imgui.end_tab_item()

            # Tab 4: Fuzzy & Stress Experiments
            if imgui.begin_tab_item("Fuzzy Tests")[0]:
                self._render_fuzzy_tab()
                imgui.end_tab_item()

            # Tab 5: Event & Context Monitor
            if imgui.begin_tab_item("Monitor")[0]:
                self._render_monitor_tab()
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    def _render_hierarchy_tab(self) -> None:
        """Render the Hierarchy scene tree tab."""
        imgui.text_colored((0.8, 0.8, 0.2, 1.0), "Scene UI Graph")
        imgui.same_line()
        if imgui.button("Scan Scene"):
            self.scan_scene()
        imgui.same_line()
        if imgui.button("Clear Selection"):
            self.selected_item = None
            self.multi_selected_ids.clear()

        imgui.separator()

        # Tree view of root items
        imgui.begin_child("HierarchyTree", (0, 240), True)
        if not self.root_items:
            imgui.text_disabled("No UI elements in scene. Use 'Add / Builder' tab.")
        else:
            for root_it in list(self.root_items):
                self._render_tree_node(root_it)
        imgui.end_child()

        imgui.separator()

        # Selection Operations (Parenting, Reparenting, Unparenting, Merging)
        imgui.text_colored((0.4, 0.9, 0.4, 1.0), "Hierarchy Operations")

        if self.selected_item:
            sel = self.selected_item
            imgui.text(f"Selected: {sel.name} ({sel.type_name})")
            if sel.parent:
                imgui.text(f"Parent: {sel.parent.name}")
            else:
                imgui.text("Parent: Root Scene")

            # Potential parent containers (excluding itself and its children)
            available_panels = [
                it
                for it in self.items.values()
                if isinstance(it.ui, Panel2D)
                and it is not sel
                and it not in sel.children
                and not it.is_internal
            ]

            panel_names = [p.name for p in available_panels]

            if available_panels:
                # Parent / Reparent control
                if hasattr(self, "_parent_combo_idx"):
                    self._parent_combo_idx = min(
                        self._parent_combo_idx, len(panel_names) - 1
                    )
                else:
                    self._parent_combo_idx = 0

                _, self._parent_combo_idx = imgui.combo(
                    "Target Panel", self._parent_combo_idx, panel_names
                )
                target_panel = available_panels[self._parent_combo_idx]

                if imgui.button("Parent / Reparent to Panel"):
                    if sel.parent is None:
                        self.parent_element(sel, target_panel)
                    else:
                        self.reparent_element(sel, target_panel)

                imgui.same_line()

            if sel.parent is not None:
                if imgui.button("Unparent to Root"):
                    self.unparent_element(sel)
                imgui.same_line()

            if imgui.button("Duplicate"):
                self.duplicate_element(sel)

            imgui.same_line()
            if imgui.button("Delete"):
                self.unregister(sel)

        else:
            imgui.text_disabled(
                "Select an element above to parent, reparent, or delete."
            )

        imgui.separator()

        # Multi-Selection & Merging
        imgui.text_colored((0.9, 0.5, 0.2, 1.0), "Merge Elements into Panel")
        selected_count = len(self.multi_selected_ids)
        imgui.text(f"Multi-selected items: {selected_count}")

        _, self.merge_padding = imgui.slider_float(
            "Padding", self.merge_padding, 0.0, 50.0
        )
        _, self.merge_color = imgui.color_edit3("Panel Color", self.merge_color)

        can_merge = selected_count >= 2
        if not can_merge:
            imgui.begin_disabled()

        if imgui.button("Merge Selected Items into New Panel"):
            items_to_merge = [
                self.items[mid] for mid in self.multi_selected_ids if mid in self.items
            ]
            self.merge_elements(
                items_to_merge,
                padding=self.merge_padding,
                color=tuple(self.merge_color),
            )

        if not can_merge:
            imgui.end_disabled()
            imgui.text_disabled(
                "Check multiple items using boxes in the tree to merge."
            )

    def _render_tree_node(self, item: UIWorkbenchItem) -> None:
        """Recursively render an item node in the hierarchy tree."""
        is_selected = self.selected_item is item
        is_multi = item.id in self.multi_selected_ids

        # Multi-select checkbox
        changed, is_checked = imgui.checkbox(f"##chk_{item.id}", is_multi)
        if changed:
            if is_checked:
                self.multi_selected_ids.add(item.id)
            else:
                self.multi_selected_ids.discard(item.id)
        imgui.same_line()

        flags = (
            imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.span_avail_width
        )
        if is_selected:
            flags |= imgui.TreeNodeFlags_.selected
        if not item.children:
            flags |= imgui.TreeNodeFlags_.leaf

        icon = "📦" if isinstance(item.ui, Panel2D) else "🔹"
        label = f"{icon} {item.name} ({item.type_name})"

        opened = imgui.tree_node_ex(f"{label}##{item.id}", flags)
        if imgui.is_item_clicked():
            self.selected_item = item

        if opened:
            for child in list(item.children):
                self._render_tree_node(child)
            imgui.tree_pop()

    def _render_builder_tab(self) -> None:
        """Render the Add UI & Custom Builder tab."""
        imgui.text_colored((0.3, 0.9, 0.7, 1.0), "Add UI Component")

        _, self.create_type_idx = imgui.combo(
            "Element Type", self.create_type_idx, self.create_types
        )
        elem_type = self.create_types[self.create_type_idx]

        _, self.create_label = imgui.input_text("Label / Text", self.create_label)
        _, self.create_pos = imgui.drag_float2(
            "Position", self.create_pos, 1.0, 0.0, 2000.0
        )
        _, self.create_size = imgui.drag_float2(
            "Size (W, H)", self.create_size, 1.0, 10.0, 1000.0
        )
        _, self.create_color = imgui.color_edit3("Color", self.create_color)
        _, self.create_opacity = imgui.slider_float(
            "Opacity", self.create_opacity, 0.0, 1.0
        )

        # Container destination
        parent_options = ["Root Scene"]
        panels = [
            it
            for it in self.items.values()
            if isinstance(it.ui, Panel2D) and not it.is_internal
        ]
        parent_options.extend([f"Panel: {p.name}" for p in panels])

        if hasattr(self, "_create_dest_idx"):
            self._create_dest_idx = min(self._create_dest_idx, len(parent_options) - 1)
        else:
            self._create_dest_idx = 0

        _, self._create_dest_idx = imgui.combo(
            "Add Target", self._create_dest_idx, parent_options
        )

        target_parent = None
        if self._create_dest_idx > 0 and panels:
            target_parent = panels[self._create_dest_idx - 1]

        if imgui.button(f"Create {elem_type}"):
            self.create_element(
                elem_type,
                position=tuple(self.create_pos),
                size=tuple(self.create_size),
                color=tuple(self.create_color),
                opacity=self.create_opacity,
                label=self.create_label,
                parent=target_parent,
            )

        imgui.separator()

        # Composite Presets
        imgui.text_colored((0.9, 0.8, 0.3, 1.0), "Quick Composite UI Presets")
        if imgui.button("Add Settings Dialog"):
            self.create_preset("Settings Dialog", position=tuple(self.create_pos))
        imgui.same_line()
        if imgui.button("Add Form Card"):
            self.create_preset("Form Card", position=tuple(self.create_pos))
        imgui.same_line()
        if imgui.button("Add Audio Mixer"):
            self.create_preset("Audio Mixer", position=tuple(self.create_pos))

    def _render_inspector_tab(self) -> None:
        """Render the Property Inspector tab for live editing."""
        if not self.selected_item:
            imgui.text_disabled(
                "No UI element selected. Click an element in Hierarchy."
            )
            return

        item = self.selected_item
        imgui.text_colored((0.3, 0.8, 1.0, 1.0), f"Inspecting: {item.name}")
        imgui.text_disabled(f"Class: {item.type_name} | ID: {item.id}")

        # Live Name change
        changed, new_name = imgui.input_text("Name", item.name)
        if changed:
            item.name = new_name

        # Dragging toggle
        changed, is_drag = imgui.checkbox("Draggable (Mouse Drag)", item.draggable)
        if changed:
            item.draggable = is_drag
            self.log_event("Draggable", item.name, str(is_drag))

        # Visibility
        is_visible = getattr(item.ui, "visible", True)
        changed, new_vis = imgui.checkbox("Visible", is_visible)
        if changed:
            if hasattr(item.ui, "set_visibility"):
                item.ui.set_visibility(new_vis)
            elif hasattr(item.ui, "visible"):
                item.ui.visible = new_vis

        imgui.separator()

        # Transform & Geometry
        if imgui.collapsing_header(
            "Transform & Geometry", imgui.TreeNodeFlags_.default_open
        ):
            pos = item.get_position()
            changed, new_pos = imgui.drag_float2(
                "Position", [float(pos[0]), float(pos[1])], 1.0, -500.0, 3000.0
            )
            if changed:
                item.set_position(new_pos)

            sz = item.get_size()
            changed, new_sz = imgui.drag_float2(
                "Size (W, H)", [float(sz[0]), float(sz[1])], 1.0, 1.0, 2000.0
            )
            if changed:
                item.set_size(new_sz)

            z = item.get_z_order()
            changed, new_z = imgui.slider_int("Z-Order", z, -10, 50)
            if changed:
                item.set_z_order(new_z)

        # Visuals & Styling
        if imgui.collapsing_header(
            "Appearance & Styling", imgui.TreeNodeFlags_.default_open
        ):
            col = item.get_color()
            changed, new_col = imgui.color_edit3("Color", list(col))
            if changed:
                item.set_color(new_col)

            op = item.get_opacity()
            changed, new_op = imgui.slider_float("Opacity", op, 0.0, 1.0)
            if changed:
                item.set_opacity(new_op)

            # Border if Panel or RoundedRect
            if hasattr(item.ui, "has_border"):
                changed, has_b = imgui.checkbox("Has Border", item.ui.has_border)
                if changed:
                    item.ui.has_border = has_b

            if hasattr(item.ui, "corner_radius"):
                cr = float(getattr(item.ui, "corner_radius", 0.0))
                changed, new_cr = imgui.slider_float("Corner Radius", cr, 0.0, 60.0)
                if changed:
                    try:
                        item.ui.corner_radius = new_cr
                    except Exception:
                        pass

        # Text & Fonts
        text_val = item.get_text()
        if text_val or hasattr(item.ui, "message") or hasattr(item.ui, "default_label"):
            if imgui.collapsing_header("Typography", imgui.TreeNodeFlags_.default_open):
                changed, new_text = imgui.input_text("Text Content", text_val)
                if changed:
                    item.set_text(new_text)

                fs = item.get_font_size()
                changed, new_fs = imgui.slider_int("Font Size", fs, 8, 72)
                if changed:
                    item.set_font_size(new_fs)

        # Widget Specific (Sliders)
        if isinstance(item.ui, (LineSlider2D, RingSlider2D)):
            if imgui.collapsing_header(
                "Slider Controls", imgui.TreeNodeFlags_.default_open
            ):
                val = float(getattr(item.ui, "value", 50.0))
                min_v = float(getattr(item.ui, "min_value", 0.0))
                max_v = float(getattr(item.ui, "max_value", 100.0))
                changed, new_val = imgui.slider_float("Value", val, min_v, max_v)
                if changed:
                    try:
                        item.ui.value = new_val
                    except Exception:
                        pass

    def _render_fuzzy_tab(self) -> None:
        """Render the Fuzzy Experiments & Stress Testing tab."""
        imgui.text_colored((1.0, 0.6, 0.2, 1.0), "Fuzzy Property Randomizers")
        imgui.text_wrapped(
            "Use these tools to test arbitrary properties, positions, and "
            "extreme edge cases to debug layout and dragging behaviors."
        )

        target_mode = "selected" if self.selected_item else "all"
        imgui.text(f"Targeting: {target_mode.upper()} items")

        if imgui.button("Randomize Colors"):
            self.fuzzy_randomize_colors(target_mode)

        imgui.same_line()
        if imgui.button("Randomize Sizes"):
            self.fuzzy_randomize_sizes(
                target_mode, self.fuzzy_scale_min, self.fuzzy_scale_max
            )

        _, self.fuzzy_pos_delta = imgui.slider_float(
            "Pos Jitter Max", self.fuzzy_pos_delta, 5.0, 100.0
        )
        if imgui.button("Jitter Positions"):
            self.fuzzy_randomize_positions(target_mode, self.fuzzy_pos_delta)

        imgui.separator()

        # Extreme Boundary Torture
        imgui.text_colored((1.0, 0.3, 0.3, 1.0), "Extreme Boundary Torture")
        if self.selected_item:
            if imgui.button(f"Torture Test {self.selected_item.name}"):
                self.fuzzy_extreme_stress_test(self.selected_item)
        else:
            imgui.text_disabled("Select an element to run extreme boundary tests.")

        imgui.separator()

        # Reparenting Stress Test
        imgui.text_colored((0.5, 0.8, 1.0, 1.0), "Reparenting Stress Test")
        panels = [
            it
            for it in self.items.values()
            if isinstance(it.ui, Panel2D) and not it.is_internal
        ]
        if self.selected_item and panels:
            target_p = (
                panels[0]
                if self.selected_item not in panels
                else (panels[1] if len(panels) > 1 else None)
            )
            if target_p and target_p is not self.selected_item:
                _, self.reparent_cycles = imgui.slider_int(
                    "Cycles", self.reparent_cycles, 1, 20
                )
                if imgui.button(f"Run {self.reparent_cycles} Reparent Cycles"):
                    self.run_reparent_stress_test(
                        self.selected_item, target_p, self.reparent_cycles
                    )
            else:
                imgui.text_disabled(
                    "Need at least one separate Panel2D for reparent stress test."
                )
        else:
            imgui.text_disabled(
                "Select a child element and create a Panel2D to run stress test."
            )

    def _render_monitor_tab(self) -> None:
        """Render the Diagnostics & Event Monitor tab."""
        imgui.text_colored((0.4, 0.8, 1.0, 1.0), "Global UIContext Status")

        hot_ui = UIContext.hot_ui
        hot_name = hot_ui.__class__.__name__ if hot_ui else "None"
        active_ui = UIContext.active_ui
        act_name = active_ui.__class__.__name__ if active_ui else "None"
        canvas_sz = tuple(UIContext.canvas_size)

        imgui.text(f"Hot UI (Hover):   {hot_name}")
        imgui.text(f"Active UI (Focus): {act_name}")
        imgui.text(f"Canvas Size:       {canvas_sz}")

        if self.scene is not None:
            total_actors = len(self.scene.ui_scene.children)
            total_ui = len(self.scene.ui_elements)
            imgui.text(f"UI Elements:       {total_ui}")
            imgui.text(f"Scene Actors:      {total_actors}")

        imgui.separator()

        # Event Log
        imgui.text_colored((0.8, 0.8, 0.3, 1.0), "Live Event Logger")
        imgui.same_line()
        if imgui.button("Clear Log"):
            self.event_log.clear()

        imgui.begin_child("EventLogWindow", (0, 260), True)
        for entry in self.event_log:
            imgui.text_disabled(f"[{entry['time']}]")
            imgui.same_line()
            imgui.text_colored((0.3, 1.0, 0.5, 1.0), entry["event"])
            imgui.same_line()
            imgui.text(f"{entry['target']} {entry['details']}")
        imgui.end_child()


# Export aliases
UIPlayground = UIWorkbench
