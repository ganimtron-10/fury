"""
Flexbox-style layout engine for the FURY UI.

The solver is written from scratch for FURY and only uses the Python standard
library. It knows nothing about pygfx or about UI widgets: it works on a tree of
:class:`LayoutNode` objects, which keeps it easy to test without a window.

A layout is solved in six passes. Widths are resolved before heights because
the height of wrapped text depends on its final width.

1. Fit widths, bottom-up.
2. Grow and shrink widths, top-down.
3. Fit heights, bottom-up, measuring text at its final width.
4. Grow and shrink heights, top-down.
5. Place every node, top-down.
6. Place floating nodes, then notify every node of its final rectangle.
"""

from dataclasses import dataclass, field, fields
from enum import Enum
import math

_INF = math.inf
_EPSILON = 1e-6


class Direction(str, Enum):
    """Axis along which a container lays out its children."""

    ROW = "row"
    COLUMN = "column"


class Align(str, Enum):
    """Alignment of children inside a container."""

    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"
    SPACE_BETWEEN = "space_between"


class Position(str, Enum):
    """How a node takes part in the layout of its parent."""

    FLOW = "flow"
    ABSOLUTE = "absolute"
    FLOATING = "floating"


class _Kind(str, Enum):
    """Sizing strategies for one axis."""

    FIXED = "fixed"
    FIT = "fit"
    GROW = "grow"
    PERCENT = "percent"


@dataclass(frozen=True)
class Sizing:
    """
    Sizing rule for one axis of a node.

    Use :func:`fixed`, :func:`fit`, :func:`grow` or :func:`percent` to create
    instances rather than calling this class directly.

    Attributes
    ----------
    kind : _Kind
        The sizing strategy.
    value : float, optional
        Pixels for fixed, weight for grow and fraction for percent sizing.
    min : float, optional
        Smallest allowed size in pixels.
    max : float, optional
        Largest allowed size in pixels.
    """

    kind: _Kind
    value: float = 0.0
    min: float = 0.0
    max: float = _INF

    @property
    def is_fixed(self):
        """
        Check whether this rule is an exact size.

        Returns
        -------
        bool
            True for sizes created with :func:`fixed`.
        """
        return self.kind is _Kind.FIXED

    def clamp(self, size):
        """
        Limit a size to this rule's bounds.

        Parameters
        ----------
        size : float
            Size in pixels.

        Returns
        -------
        float
            The size, limited to ``[min, max]``.
        """
        return max(self.min, min(size, self.max))


def _check_bounds(min_size, max_size):
    """
    Validate a pair of size bounds.

    Parameters
    ----------
    min_size : float
        Smallest allowed size in pixels.
    max_size : float
        Largest allowed size in pixels.

    Raises
    ------
    ValueError
        If the bounds are negative or reversed.
    """
    if min_size < 0 or max_size < min_size:
        raise ValueError(f"Invalid size bounds: min={min_size}, max={max_size}.")


def fixed(size):
    """
    Size an axis to an exact number of pixels.

    Parameters
    ----------
    size : float
        Size in pixels.

    Returns
    -------
    Sizing
        The sizing rule.

    Raises
    ------
    ValueError
        If ``size`` is negative.
    """
    if size < 0:
        raise ValueError(f"Fixed size must not be negative, got {size}.")
    return Sizing(_Kind.FIXED, value=float(size), min=float(size), max=float(size))


def fit(min=0, max=_INF):  # noqa: A002
    """
    Size an axis to its content.

    Parameters
    ----------
    min : float, optional
        Smallest allowed size in pixels.
    max : float, optional
        Largest allowed size in pixels.

    Returns
    -------
    Sizing
        The sizing rule.
    """
    _check_bounds(min, max)
    return Sizing(_Kind.FIT, min=float(min), max=float(max))


def grow(weight=1, *, min=0, max=_INF):  # noqa: A002
    """
    Size an axis to its content, then take a share of the leftover space.

    Parameters
    ----------
    weight : float, optional
        Share of the leftover space relative to other growing siblings.
    min : float, optional
        Smallest allowed size in pixels.
    max : float, optional
        Largest allowed size in pixels.

    Returns
    -------
    Sizing
        The sizing rule.

    Raises
    ------
    ValueError
        If ``weight`` is not positive.
    """
    if weight <= 0:
        raise ValueError(f"Grow weight must be positive, got {weight}.")
    _check_bounds(min, max)
    return Sizing(_Kind.GROW, value=float(weight), min=float(min), max=float(max))


def percent(fraction, *, min=0, max=_INF):  # noqa: A002
    """
    Size an axis as a fraction of the parent's content box.

    On the parent's main axis the gaps are removed first, so two children at
    ``percent(0.5)`` always fit side by side.

    Parameters
    ----------
    fraction : float
        Fraction of the parent's content box, between 0 and 1.
    min : float, optional
        Smallest allowed size in pixels.
    max : float, optional
        Largest allowed size in pixels.

    Returns
    -------
    Sizing
        The sizing rule.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0, 1]``.
    """
    if not 0 <= fraction <= 1:
        raise ValueError(f"Percent must be between 0 and 1, got {fraction}.")
    _check_bounds(min, max)
    return Sizing(_Kind.PERCENT, value=float(fraction), min=float(min), max=float(max))


@dataclass(frozen=True)
class Edges:
    """
    Spacing on the four sides of a box, in pixels.

    Attributes
    ----------
    top : float, optional
        Space above the box.
    right : float, optional
        Space on the right of the box.
    bottom : float, optional
        Space below the box.
    left : float, optional
        Space on the left of the box.
    """

    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    @classmethod
    def of(cls, value):
        """
        Build edges from the shorthand forms accepted by the layout API.

        Parameters
        ----------
        value : Edges or float or (float, float) or (float, float, float, float)
            A single value for all sides, a ``(vertical, horizontal)`` pair or
            a ``(top, right, bottom, left)`` tuple.

        Returns
        -------
        Edges
            The spacing on each side.

        Raises
        ------
        ValueError
            If the value has the wrong shape or is negative.
        """
        if isinstance(value, Edges):
            return value
        if isinstance(value, (int, float)):
            sides = (value, value, value, value)
        elif len(value) == 2:
            sides = (value[0], value[1], value[0], value[1])
        elif len(value) == 4:
            sides = tuple(value)
        else:
            raise ValueError(f"Spacing needs 1, 2 or 4 values, got {value!r}.")
        if any(side < 0 for side in sides):
            raise ValueError(f"Spacing must not be negative, got {value!r}.")
        return cls(*(float(side) for side in sides))

    def total(self, axis):
        """
        Get the combined spacing along an axis.

        Parameters
        ----------
        axis : int
            0 for horizontal and 1 for vertical.

        Returns
        -------
        float
            The sum of both sides in pixels.
        """
        return self.left + self.right if axis == 0 else self.top + self.bottom

    def start(self, axis):
        """
        Get the spacing at the start of an axis.

        Parameters
        ----------
        axis : int
            0 for horizontal and 1 for vertical.

        Returns
        -------
        float
            The left or top spacing in pixels.
        """
        return self.left if axis == 0 else self.top


def _as_sizing(value):
    """
    Convert a sizing shorthand to a :class:`Sizing`.

    Parameters
    ----------
    value : Sizing or float
        A sizing rule, or a number of pixels.

    Returns
    -------
    Sizing
        The sizing rule.
    """
    return value if isinstance(value, Sizing) else fixed(value)


_STYLE_CONVERTERS = {
    "width": _as_sizing,
    "height": _as_sizing,
    "padding": Edges.of,
    "margin": Edges.of,
    "direction": Direction,
    "align_main": Align,
    "align_cross": Align,
    "position": Position,
}


@dataclass
class LayoutStyle:
    """
    Layout properties of one node.

    Attributes
    ----------
    width : Sizing, optional
        Horizontal sizing rule.
    height : Sizing, optional
        Vertical sizing rule.
    padding : Edges, optional
        Space between the border and the content.
    margin : Edges, optional
        Space outside the border, used by the parent when placing this node.
    border : float, optional
        Border thickness. It is part of the node's size, like padding.
    gap : float, optional
        Space between adjacent children, and between lines when wrapping.
    direction : Direction, optional
        Axis along which children are placed.
    align_main : Align, optional
        Placement of children along the main axis. One of ``START``,
        ``CENTER``, ``END`` or ``SPACE_BETWEEN``.
    align_cross : Align, optional
        Placement of children across the main axis. One of ``START``,
        ``CENTER``, ``END`` or ``STRETCH``.
    wrap : bool, optional
        If True, children that do not fit move to a new line. Only supported
        with ``Direction.ROW``.
    shrink : float, optional
        Share of the overflow this node gives back. 0 means it never shrinks.
    position : Position, optional
        Whether the node is in the flow, offset from its parent's content box
        (``ABSOLUTE``) or attached to another node (``FLOATING``).
    offset : (float, float), optional
        Offset used by absolute and floating nodes.
    float_to : LayoutNode, optional
        Node a floating node is attached to. Defaults to the parent.
    float_anchor : (float, float), optional
        Point on the target to attach to, as fractions of its size.
    """

    width: Sizing = field(default_factory=fit)
    height: Sizing = field(default_factory=fit)
    padding: Edges = field(default_factory=Edges)
    margin: Edges = field(default_factory=Edges)
    border: float = 0.0
    gap: float = 0.0
    direction: Direction = Direction.ROW
    align_main: Align = Align.START
    align_cross: Align = Align.START
    wrap: bool = False
    shrink: float = 1.0
    position: Position = Position.FLOW
    offset: tuple = (0.0, 0.0)
    float_to: object = None
    float_anchor: tuple = (0.0, 0.0)

    def update(self, **properties):
        """
        Change properties, accepting the same shorthands as the layout API.

        Parameters
        ----------
        **properties : dict
            The layout properties to change.

        Raises
        ------
        TypeError
            If a property does not exist.
        ValueError
            If the resulting combination of properties is invalid.
        """
        names = {item.name for item in fields(self)}
        for name, value in properties.items():
            if name not in names:
                raise TypeError(f"Unknown layout property {name!r}.")
            setattr(self, name, _STYLE_CONVERTERS.get(name, lambda v: v)(value))
        self.validate()

    def validate(self):
        """
        Check that the properties are compatible with each other.

        Raises
        ------
        ValueError
            If the properties are invalid.
        """
        if self.align_main is Align.STRETCH:
            raise ValueError("align_main does not support STRETCH.")
        if self.align_cross is Align.SPACE_BETWEEN:
            raise ValueError("align_cross does not support SPACE_BETWEEN.")
        if self.wrap and self.direction is not Direction.ROW:
            raise ValueError("wrap is only supported with Direction.ROW.")
        if self.gap < 0 or self.border < 0 or self.shrink < 0:
            raise ValueError("gap, border and shrink must not be negative.")

    def inset(self, axis):
        """
        Get the space taken by border and padding along an axis.

        Parameters
        ----------
        axis : int
            0 for horizontal and 1 for vertical.

        Returns
        -------
        float
            The total inset in pixels.
        """
        return self.padding.total(axis) + 2 * self.border

    def main_axis(self):
        """
        Get the index of the main axis.

        Returns
        -------
        int
            0 for rows and 1 for columns.
        """
        return 0 if self.direction is Direction.ROW else 1


@dataclass(frozen=True)
class LayoutRect:
    """
    Final position and size of a node, in whole pixels.

    Attributes
    ----------
    x : int, optional
        Left edge.
    y : int, optional
        Top edge.
    width : int, optional
        Width including border and padding.
    height : int, optional
        Height including border and padding.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def position(self):
        """
        Get the top-left corner.

        Returns
        -------
        (int, int)
            The ``(x, y)`` coordinates.
        """
        return (self.x, self.y)

    @property
    def size(self):
        """
        Get the size.

        Returns
        -------
        (int, int)
            The ``(width, height)`` in pixels.
        """
        return (self.width, self.height)


class LayoutNode:
    """
    One box in a layout tree.

    Parameters
    ----------
    style : LayoutStyle, optional
        Layout properties of this node.
    children : list of LayoutNode, optional
        Initial children.
    measure : callable, optional
        ``measure(max_width)`` returns the ``(width, height)`` of a leaf's
        content. ``max_width`` is None when measuring the unconstrained width.
    apply : callable, optional
        ``apply(rect)`` is called with the final :class:`LayoutRect`.
    resizable : bool, optional
        If False the node keeps its fit size: grow, shrink, percent and
        stretch are ignored.
    **properties : dict
        Layout properties applied on top of ``style``.
    """

    def __init__(
        self,
        style=None,
        *,
        children=(),
        measure=None,
        apply=None,
        resizable=True,
        **properties,
    ):
        """Initialize the node."""
        self.style = style if style is not None else LayoutStyle()
        self.style.update(**properties)
        self.children = []
        self.parent = None
        self.measure = measure
        self.apply = apply
        self.resizable = resizable
        self.rect = LayoutRect()
        self.dirty = True
        self._solving = False
        self._size = [0.0, 0.0]
        self._lines = []
        for child in children:
            self.add(child)

    @property
    def root(self):
        """
        Get the top of the tree containing this node.

        Returns
        -------
        LayoutNode
            The root node.
        """
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def add(self, child):
        """
        Append a child node.

        Parameters
        ----------
        child : LayoutNode
            The node to add.

        Returns
        -------
        LayoutNode
            The added child.

        Raises
        ------
        ValueError
            If the child already has a parent, or adding it creates a cycle.
        """
        if child.parent is not None:
            raise ValueError("Node already has a parent.")
        if child is self.root:
            raise ValueError("Adding this node would create a cycle.")
        child.parent = self
        self.children.append(child)
        self.mark_dirty()
        return child

    def remove(self, child):
        """
        Remove a child node.

        Parameters
        ----------
        child : LayoutNode
            The node to remove.
        """
        self.children.remove(child)
        child.parent = None
        self.mark_dirty()

    def mark_dirty(self):
        """Flag the tree containing this node as needing a new layout."""
        root = self.root
        if not root._solving:
            root.dirty = True


def _is_flexible(node, axis):
    """
    Check whether the solver may change a node's size along an axis.

    Parameters
    ----------
    node : LayoutNode
        The node to check.
    axis : int
        0 for horizontal and 1 for vertical.

    Returns
    -------
    bool
        True if the node's size is not fixed along this axis.
    """
    return node.resizable and _sizing(node, axis).kind is not _Kind.FIXED


def _sizing(node, axis):
    """
    Get a node's sizing rule along an axis.

    Parameters
    ----------
    node : LayoutNode
        The node.
    axis : int
        0 for horizontal and 1 for vertical.

    Returns
    -------
    Sizing
        The sizing rule.
    """
    return node.style.width if axis == 0 else node.style.height


def _flow_children(node):
    """
    Get the children that take part in the parent's flow.

    Parameters
    ----------
    node : LayoutNode
        The parent node.

    Returns
    -------
    list of LayoutNode
        The children positioned with ``Position.FLOW``.
    """
    return [c for c in node.children if c.style.position is Position.FLOW]


def _outer(node, axis):
    """
    Get a node's size along an axis including its margins.

    Parameters
    ----------
    node : LayoutNode
        The node.
    axis : int
        0 for horizontal and 1 for vertical.

    Returns
    -------
    float
        The outer size in pixels.
    """
    return node._size[axis] + node.style.margin.total(axis)


def _flow_content_size(node, flow, axis):
    """
    Compute the size of a container's flow children along an axis.

    Parameters
    ----------
    node : LayoutNode
        The container.
    flow : list of LayoutNode
        The container's flow children.
    axis : int
        0 for horizontal and 1 for vertical.

    Returns
    -------
    float
        The content size in pixels.
    """
    style = node.style
    if style.wrap and axis == 1:
        heights = [max(_outer(c, 1) for c in line) for line in node._lines]
        return sum(heights) + style.gap * max(len(heights) - 1, 0)
    sizes = [_outer(c, axis) for c in flow]
    if axis == style.main_axis():
        return sum(sizes) + style.gap * max(len(sizes) - 1, 0)
    return max(sizes)


def _leaf_content_size(node, axis):
    """
    Measure a leaf's content along an axis.

    Parameters
    ----------
    node : LayoutNode
        The leaf.
    axis : int
        0 for horizontal and 1 for vertical.

    Returns
    -------
    float
        The content size in pixels.
    """
    if node.measure is None:
        return 0.0
    max_width = None if axis == 0 else max(node._size[0] - node.style.inset(0), 0)
    return float(node.measure(max_width)[axis])


def _fit_pass(node, axis):
    """
    Compute the size each node wants from its content, bottom-up.

    Parameters
    ----------
    node : LayoutNode
        The subtree to process.
    axis : int
        0 for horizontal and 1 for vertical.
    """
    for child in node.children:
        _fit_pass(child, axis)

    flow = _flow_children(node)
    if flow:
        content = _flow_content_size(node, flow, axis)
    else:
        content = _leaf_content_size(node, axis)

    sizing = _sizing(node, axis)
    if not node.resizable and sizing.kind is not _Kind.FIXED:
        node._size[axis] = content + node.style.inset(axis)
    elif sizing.kind is _Kind.FIXED:
        node._size[axis] = sizing.value
    elif sizing.kind is _Kind.PERCENT:
        node._size[axis] = sizing.clamp(0.0)
    else:
        node._size[axis] = sizing.clamp(content + node.style.inset(axis))


def _share_space(nodes, amount, axis, weights, limits):
    """
    Share a positive amount of pixels between nodes, honouring their limits.

    Nodes that reach their limit are removed and the rest is shared again.

    Parameters
    ----------
    nodes : list of LayoutNode
        The nodes receiving the change.
    amount : float
        Signed number of pixels to add to the nodes.
    axis : int
        0 for horizontal and 1 for vertical.
    weights : callable
        Returns the share weight of a node.
    limits : callable
        Returns the size a node must not go past.
    """
    active = [n for n in nodes if weights(n) > 0]
    while active and abs(amount) > _EPSILON:
        total = sum(weights(n) for n in active)
        if total <= _EPSILON:
            return
        distributed = 0.0
        finished = []
        for node in active:
            change = amount * weights(node) / total
            target = node._size[axis] + change
            limit = limits(node)
            if (amount > 0 and target >= limit) or (amount < 0 and target <= limit):
                target = limit
                finished.append(node)
            distributed += target - node._size[axis]
            node._size[axis] = target
        amount -= distributed
        if not finished:
            return
        active = [n for n in active if n not in finished]


def _resolve_main_axis(line, content, gap, axis):
    """
    Resolve percent, grow and shrink along the main axis of one line.

    Parameters
    ----------
    line : list of LayoutNode
        The children on this line.
    content : float
        Size of the container's content box along the axis.
    gap : float
        Space between adjacent children.
    axis : int
        0 for horizontal and 1 for vertical.
    """
    gaps = gap * max(len(line) - 1, 0)
    for child in line:
        sizing = _sizing(child, axis)
        if child.resizable and sizing.kind is _Kind.PERCENT:
            child._size[axis] = sizing.clamp(sizing.value * max(content - gaps, 0))

    free = content - gaps - sum(_outer(c, axis) for c in line)
    if free > _EPSILON:
        growers = [
            c for c in line if c.resizable and _sizing(c, axis).kind is _Kind.GROW
        ]
        _share_space(
            growers,
            free,
            axis,
            lambda n: _sizing(n, axis).value,
            lambda n: _sizing(n, axis).max,
        )
    elif free < -_EPSILON:
        shrinkers = [c for c in line if _is_flexible(c, axis)]
        _share_space(
            shrinkers,
            free,
            axis,
            lambda n: n.style.shrink * n._size[axis],
            lambda n: _sizing(n, axis).min,
        )


def _resolve_cross_axis(line, available, axis, align):
    """
    Resolve percent, grow, stretch and shrink across the main axis of a line.

    Parameters
    ----------
    line : list of LayoutNode
        The children on this line.
    available : float
        Space available to the line across the main axis.
    axis : int
        0 for horizontal and 1 for vertical.
    align : Align
        The container's cross alignment.
    """
    for child in line:
        if not child.resizable:
            continue
        sizing = _sizing(child, axis)
        room = max(available - child.style.margin.total(axis), 0)
        if sizing.kind is _Kind.PERCENT:
            child._size[axis] = sizing.clamp(sizing.value * room)
        elif sizing.kind is _Kind.GROW or (
            sizing.kind is _Kind.FIT and align is Align.STRETCH
        ):
            child._size[axis] = sizing.clamp(room)
        elif (
            sizing.kind is _Kind.FIT
            and child.style.shrink > 0
            and child._size[axis] > room
        ):
            child._size[axis] = sizing.clamp(room)


def _build_lines(node, flow, content):
    """
    Split a container's flow children into lines.

    Parameters
    ----------
    node : LayoutNode
        The container.
    flow : list of LayoutNode
        The container's flow children.
    content : float
        Width of the content box.

    Returns
    -------
    list of list of LayoutNode
        The children of each line.
    """
    style = node.style
    if not style.wrap or not flow:
        return [flow] if flow else []

    lines = [[]]
    used = 0.0
    for child in flow:
        size = _outer(child, 0)
        extra = size if not lines[-1] else size + style.gap
        if lines[-1] and used + extra > content + _EPSILON:
            lines.append([])
            used = 0.0
            extra = size
        lines[-1].append(child)
        used += extra
    return lines


def _resolve_out_of_flow(node, content, axis):
    """
    Resolve percent and grow for children outside the flow.

    Parameters
    ----------
    node : LayoutNode
        The parent node.
    content : float
        Size of the parent's content box along the axis.
    axis : int
        0 for horizontal and 1 for vertical.
    """
    for child in node.children:
        if child.style.position is Position.FLOW or not child.resizable:
            continue
        sizing = _sizing(child, axis)
        if sizing.kind is _Kind.PERCENT:
            child._size[axis] = sizing.clamp(sizing.value * content)
        elif sizing.kind is _Kind.GROW:
            child._size[axis] = sizing.clamp(content)


def _distribute_pass(node, axis):
    """
    Share each container's space between its children, top-down.

    Parameters
    ----------
    node : LayoutNode
        The subtree to process.
    axis : int
        0 for horizontal and 1 for vertical.
    """
    style = node.style
    content = max(node._size[axis] - style.inset(axis), 0)
    flow = _flow_children(node)

    if axis == 0:
        node._lines = _build_lines(node, flow, content)

    if flow:
        if axis == style.main_axis():
            for line in node._lines:
                _resolve_main_axis(line, content, style.gap, axis)
        elif style.wrap:
            for line in node._lines:
                height = max(_outer(c, axis) for c in line)
                _resolve_cross_axis(line, height, axis, style.align_cross)
        else:
            _resolve_cross_axis(flow, content, axis, style.align_cross)

    _resolve_out_of_flow(node, content, axis)

    for child in node.children:
        _distribute_pass(child, axis)


def _resolve_root(root, available, axis):
    """
    Size the root node from the space available to the tree.

    Parameters
    ----------
    root : LayoutNode
        The root node.
    available : (float, float) or None
        Space available to the tree.
    axis : int
        0 for horizontal and 1 for vertical.
    """
    if available is None or available[axis] is None or not root.resizable:
        return
    sizing = _sizing(root, axis)
    if sizing.kind is _Kind.GROW:
        root._size[axis] = sizing.clamp(available[axis])
    elif sizing.kind is _Kind.PERCENT:
        root._size[axis] = sizing.clamp(sizing.value * available[axis])


def _main_offsets(style, free, count):
    """
    Compute the start offset and spacing of a line along the main axis.

    Parameters
    ----------
    style : LayoutStyle
        The container's style.
    free : float
        Unused space along the main axis.
    count : int
        Number of children on the line.

    Returns
    -------
    (float, float)
        The offset of the first child and the space between children.
    """
    free = max(free, 0)
    if style.align_main is Align.CENTER:
        return free / 2, style.gap
    if style.align_main is Align.END:
        return free, style.gap
    if style.align_main is Align.SPACE_BETWEEN and count > 1:
        return 0.0, style.gap + free / (count - 1)
    return 0.0, style.gap


def _cross_offset(align, free):
    """
    Compute a child's offset across the main axis.

    Parameters
    ----------
    align : Align
        The container's cross alignment.
    free : float
        Unused space across the line.

    Returns
    -------
    float
        The child's offset in pixels.
    """
    if align is Align.CENTER:
        return free / 2
    if align is Align.END:
        return free
    return 0.0


def _to_rect(x, y, size):
    """
    Round a box to whole pixels, keeping adjacent boxes gap-free.

    Parameters
    ----------
    x : float
        Left edge.
    y : float
        Top edge.
    size : list of float
        Width and height.

    Returns
    -------
    LayoutRect
        The rounded rectangle.
    """
    left, top = round(x), round(y)
    return LayoutRect(left, top, round(x + size[0]) - left, round(y + size[1]) - top)


def _place_line(node, line, origin, cross_start, cross_size):
    """
    Place the children of one line.

    Parameters
    ----------
    node : LayoutNode
        The container.
    line : list of LayoutNode
        The children on this line.
    origin : (float, float)
        Top-left corner of the container's content box.
    cross_start : float
        Offset of this line across the main axis.
    cross_size : float
        Size of this line across the main axis.
    """
    style = node.style
    main = style.main_axis()
    cross = 1 - main
    content = max(node._size[main] - style.inset(main), 0)
    used = sum(_outer(c, main) for c in line)
    free = content - used - style.gap * max(len(line) - 1, 0)
    cursor, spacing = _main_offsets(style, free, len(line))

    for child in line:
        margin = child.style.margin
        position = [0.0, 0.0]
        position[main] = origin[main] + cursor + margin.start(main)
        leftover = cross_size - _outer(child, cross)
        position[cross] = (
            origin[cross]
            + cross_start
            + _cross_offset(style.align_cross, leftover)
            + margin.start(cross)
        )
        _place_pass(child, position[0], position[1])
        cursor += _outer(child, main) + spacing


def _place_pass(node, x, y):
    """
    Give every node in a subtree its final rectangle, top-down.

    Parameters
    ----------
    node : LayoutNode
        The subtree to process.
    x : float
        Left edge of this node.
    y : float
        Top edge of this node.
    """
    node.rect = _to_rect(x, y, node._size)
    style = node.style
    origin = (
        x + style.padding.left + style.border,
        y + style.padding.top + style.border,
    )

    cross = 1 - style.main_axis()
    if style.wrap:
        cursor = 0.0
        for line in node._lines:
            height = max(_outer(c, 1) for c in line)
            _place_line(node, line, origin, cursor, height)
            cursor += height + style.gap
    elif node._lines:
        content = max(node._size[cross] - style.inset(cross), 0)
        _place_line(node, node._lines[0], origin, 0.0, content)

    for child in node.children:
        if child.style.position is Position.ABSOLUTE:
            margin = child.style.margin
            _place_pass(
                child,
                origin[0] + child.style.offset[0] + margin.left,
                origin[1] + child.style.offset[1] + margin.top,
            )


def _place_floating(node, root):
    """
    Place floating nodes once the nodes they attach to have their rectangles.

    Parameters
    ----------
    node : LayoutNode
        The subtree to process.
    root : LayoutNode
        The root of the tree.
    """
    for child in node.children:
        style = child.style
        if style.position is Position.FLOATING:
            target = style.float_to if style.float_to is not None else node
            if target.root is not root:
                raise ValueError("float_to must be a node in the same tree.")
            _place_pass(
                child,
                target.rect.x
                + target.rect.width * style.float_anchor[0]
                + style.offset[0],
                target.rect.y
                + target.rect.height * style.float_anchor[1]
                + style.offset[1],
            )
        _place_floating(child, root)


def _notify(node):
    """
    Call every node's apply callback, parents first.

    Parameters
    ----------
    node : LayoutNode
        The subtree to process.
    """
    if node.apply is not None:
        node.apply(node.rect)
    for child in node.children:
        _notify(child)


def compute_sizes(root, *, available=None):
    """
    Compute the size of every node in a tree.

    Parameters
    ----------
    root : LayoutNode
        The root of the tree.
    available : (float, float), optional
        Space available to the tree. Used when the root grows or is a percent.
    """
    for axis in (0, 1):
        _fit_pass(root, axis)
        _resolve_root(root, available, axis)
        _distribute_pass(root, axis)
    root.rect = LayoutRect(root.rect.x, root.rect.y, *_to_rect(0, 0, root._size).size)


def place(root, origin=(0, 0)):
    """
    Position an already sized tree and notify its nodes.

    Parameters
    ----------
    root : LayoutNode
        The root of the tree.
    origin : (float, float), optional
        Position of the root's top-left corner.
    """
    root._solving = True
    try:
        _place_pass(root, float(origin[0]), float(origin[1]))
        _place_floating(root, root)
        root.dirty = False
        _notify(root)
    finally:
        root._solving = False


def solve(root, *, available=None, origin=(0, 0)):
    """
    Compute the layout of a tree.

    Parameters
    ----------
    root : LayoutNode
        The root of the tree.
    available : (float, float), optional
        Space available to the tree. Used when the root grows or is a percent.
    origin : (float, float), optional
        Position of the root's top-left corner.

    Returns
    -------
    LayoutRect
        The final rectangle of the root.
    """
    root.style.validate()
    compute_sizes(root, available=available)
    place(root, origin)
    return root.rect
