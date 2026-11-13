"""Tests for the pure layout solver."""

import math
import time

import numpy.testing as npt
import pytest

from fury.ui.layout import (
    Align,
    Direction,
    Edges,
    LayoutNode,
    LayoutStyle,
    Position,
    fit,
    fixed,
    grow,
    percent,
    place,
    solve,
)


def box(width=10, height=10, **properties):
    """Create a leaf node with a fixed size by default."""
    return LayoutNode(width=width, height=height, **properties)


def text_leaf(natural_width, line_height, **properties):
    """Create a leaf that wraps like text, keeping its area constant."""

    def measure(max_width):
        if max_width is None or max_width >= natural_width:
            return (natural_width, line_height)
        lines = -(-natural_width // max(max_width, 1))
        return (max_width, lines * line_height)

    return LayoutNode(measure=measure, **properties)


def test_fit_container_sums_children_gap_and_padding():
    a, b = box(30, 10), box(20, 40)
    root = LayoutNode(children=[a, b], gap=5, padding=(1, 2, 3, 4))
    rect = solve(root)
    assert rect.size == (61, 44)
    assert a.rect.position == (4, 1)
    assert b.rect.position == (39, 1)


def test_column_stacks_vertically():
    a, b = box(30, 10), box(20, 40)
    root = LayoutNode(children=[a, b], gap=5, direction=Direction.COLUMN)
    assert solve(root).size == (30, 55)
    assert b.rect.position == (0, 15)


def test_fixed_ignores_content():
    root = LayoutNode(children=[box(500, 500, shrink=0)], width=fixed(100), height=50)
    assert solve(root).size == (100, 50)


def test_fit_is_clamped_by_min_and_max():
    assert solve(LayoutNode(children=[box(5, 5)], width=fit(min=50))).width == 50
    assert solve(LayoutNode(children=[box(90, 5)], width=fit(max=50))).width == 50


def test_grow_shares_leftover_by_weight():
    a, b = box(width=grow(weight=1)), box(width=grow(weight=3))
    root = LayoutNode(children=[a, b], width=fixed(100), height=10)
    solve(root)
    assert a.rect.width + b.rect.width == 100
    assert (a.rect.width, b.rect.width) == (25, 75)


def test_grow_starts_from_fit_size():
    a = text_leaf(40, 10, width=grow())
    b = text_leaf(0, 10, width=grow())
    solve(LayoutNode(children=[a, b], width=fixed(100)))
    assert (a.rect.width, b.rect.width) == (70, 30)


def test_grow_respects_max_and_redistributes():
    a, b = box(width=grow(max=10)), box(width=grow())
    solve(LayoutNode(children=[a, b], width=fixed(100), height=10))
    assert (a.rect.width, b.rect.width) == (10, 90)


def test_percent_removes_gaps_on_main_axis():
    a, b = box(width=percent(0.5)), box(width=percent(0.5))
    solve(LayoutNode(children=[a, b], width=fixed(110), height=10, gap=10))
    assert (a.rect.width, b.rect.width) == (50, 50)
    assert b.rect.x == 60


def test_percent_uses_content_box_on_cross_axis():
    child = box(width=10, height=percent(0.5))
    solve(LayoutNode(children=[child], width=10, height=120, padding=(10, 0)))
    assert child.rect.height == 50


def test_shrink_is_proportional_and_keeps_fixed_siblings():
    a = text_leaf(100, 10, width=fit())
    b = text_leaf(300, 10, width=fit())
    c = box(100, 10)
    solve(LayoutNode(children=[a, b, c], width=fixed(300)))
    assert c.rect.width == 100
    assert (a.rect.width, b.rect.width) == (50, 150)


def test_shrink_stops_at_min_and_redistributes():
    a = text_leaf(100, 10, width=fit(min=90))
    b = text_leaf(100, 10, width=fit())
    solve(LayoutNode(children=[a, b], width=fixed(100)))
    assert (a.rect.width, b.rect.width) == (90, 10)


def test_shrink_zero_never_shrinks():
    a = text_leaf(100, 10, shrink=0)
    b = text_leaf(100, 10)
    solve(LayoutNode(children=[a, b], width=fixed(150)))
    assert (a.rect.width, b.rect.width) == (100, 50)


def test_text_height_follows_final_width():
    label = text_leaf(100, 10)
    root = LayoutNode(children=[label], width=fixed(30))
    assert solve(root).height == 40
    assert label.rect.size == (30, 40)


def test_nested_fit_height_uses_wrapped_text():
    label = text_leaf(100, 10)
    inner = LayoutNode(children=[label], direction=Direction.COLUMN)
    outer = LayoutNode(
        children=[inner], width=fixed(50), direction=Direction.COLUMN, padding=2
    )
    assert solve(outer).height == 34
    assert inner.rect.height == 30


def test_margin_is_outside_the_box():
    a = box(10, 10, margin=(1, 2, 3, 4))
    b = box(10, 10)
    root = LayoutNode(children=[a, b])
    assert solve(root).size == (26, 14)
    assert a.rect.position == (4, 1)
    assert b.rect.x == 16


def test_border_counts_as_inset():
    child = box(10, 10)
    root = LayoutNode(children=[child], border=2, padding=1)
    assert solve(root).size == (16, 16)
    assert child.rect.position == (3, 3)


@pytest.mark.parametrize(
    "align,expected", [(Align.START, 0), (Align.CENTER, 40), (Align.END, 80)]
)
def test_align_main(align, expected):
    child = box(20, 10)
    solve(LayoutNode(children=[child], width=100, height=10, align_main=align))
    assert child.rect.x == expected


def test_space_between_pins_first_and_last():
    a, b, c = box(10, 10), box(10, 10), box(10, 10)
    solve(
        LayoutNode(
            children=[a, b, c], width=100, height=10, align_main=Align.SPACE_BETWEEN
        )
    )
    assert [n.rect.x for n in (a, b, c)] == [0, 45, 90]


@pytest.mark.parametrize(
    "align,expected", [(Align.START, 0), (Align.CENTER, 15), (Align.END, 30)]
)
def test_align_cross(align, expected):
    child = box(10, 20)
    solve(LayoutNode(children=[child], width=10, height=50, align_cross=align))
    assert child.rect.y == expected


def test_stretch_fills_cross_axis_but_not_fixed():
    fill = text_leaf(10, 5)
    keep = box(10, 5)
    solve(
        LayoutNode(
            children=[fill, keep], width=20, height=50, align_cross=Align.STRETCH
        )
    )
    assert fill.rect.height == 50
    assert keep.rect.height == 5


def test_grow_on_cross_axis_respects_margin():
    child = box(width=10, height=grow(), margin=(5, 0))
    solve(LayoutNode(children=[child], width=10, height=50))
    assert child.rect.height == 40


def test_wrap_breaks_lines_and_sums_height():
    a, b, c = box(40, 10), box(40, 20), box(40, 5)
    root = LayoutNode(children=[a, b, c], width=fixed(90), wrap=True, gap=4)
    rect = solve(root)
    assert b.rect.position == (44, 0)
    assert c.rect.position == (0, 24)
    assert rect.height == 29


def test_wrap_lines_grow_independently():
    a, b = box(width=grow()), box(40, 10)
    c = box(width=grow())
    a.style.update(width=grow(min=40))
    c.style.update(width=grow(min=80))
    solve(LayoutNode(children=[a, b, c], width=fixed(100), wrap=True))
    assert (a.rect.width, b.rect.x) == (60, 60)
    assert (c.rect.width, c.rect.y) == (100, 10)


def test_wrap_rejected_for_columns():
    with pytest.raises(ValueError):
        LayoutNode(wrap=True, direction=Direction.COLUMN)


def test_absolute_is_out_of_flow():
    floating = box(500, 500, position=Position.ABSOLUTE, offset=(3, 4))
    root = LayoutNode(children=[box(10, 10), floating], padding=2)
    assert solve(root).size == (14, 14)
    assert floating.rect.position == (5, 6)


def test_absolute_grow_fills_parent_content():
    overlay = box(width=grow(), height=grow(), position=Position.ABSOLUTE)
    solve(LayoutNode(children=[overlay], width=50, height=60, padding=5))
    assert overlay.rect.size == (40, 50)


def test_floating_attaches_to_target_and_stays_out_of_flow():
    button = box(40, 20, margin=(0, 0, 0, 10))
    menu = box(100, 300, position=Position.FLOATING)
    menu.style.update(float_to=button, float_anchor=(0, 1), offset=(0, 2))
    root = LayoutNode(children=[button, menu])
    assert solve(root).size == (50, 20)
    assert menu.rect.position == (10, 22)


def test_floating_target_from_another_tree_is_rejected():
    stray = box()
    menu = box(position=Position.FLOATING, float_to=stray)
    with pytest.raises(ValueError):
        solve(LayoutNode(children=[menu]))


def test_root_grow_uses_available():
    root = LayoutNode(width=grow(), height=percent(0.5))
    assert solve(root, available=(200, 100)).size == (200, 50)


def test_origin_offsets_the_tree():
    child = box()
    solve(LayoutNode(children=[child], padding=1), origin=(100, 200))
    assert child.rect.position == (101, 201)


def test_rounding_leaves_no_gaps():
    children = [box(width=grow()) for _ in range(3)]
    solve(LayoutNode(children=children, width=100, height=10))
    assert children[0].rect.x + children[0].rect.width == children[1].rect.x
    assert children[1].rect.x + children[1].rect.width == children[2].rect.x
    assert children[2].rect.x + children[2].rect.width == 100


def test_non_resizable_keeps_fit_size():
    leaf = LayoutNode(measure=lambda _: (30, 12), width=grow(), resizable=False)
    solve(LayoutNode(children=[leaf], width=100, height=50, align_cross=Align.STRETCH))
    assert leaf.rect.size == (30, 12)


def test_empty_container_is_only_its_inset():
    assert solve(LayoutNode(padding=3)).size == (6, 6)


def test_oversized_inset_does_not_go_negative():
    child = box(width=grow(), height=grow())
    solve(LayoutNode(children=[child], width=4, height=4, padding=10))
    assert child.rect.size == (0, 0)


def test_apply_runs_parent_first_and_clears_dirty():
    calls = []
    child = LayoutNode(apply=lambda r: calls.append("child"))
    root = LayoutNode(children=[child], apply=lambda r: calls.append("root"))
    assert root.dirty
    solve(root)
    assert calls == ["root", "child"]
    assert not root.dirty


def test_mark_dirty_reaches_root_but_not_during_apply():
    leaf = LayoutNode()
    leaf.apply = lambda rect: leaf.mark_dirty()
    root = LayoutNode(children=[LayoutNode(children=[leaf])])
    solve(root)
    assert not root.dirty
    leaf.mark_dirty()
    assert root.dirty


def test_add_rejects_reparenting_and_cycles():
    child = LayoutNode()
    root = LayoutNode(children=[child])
    with pytest.raises(ValueError):
        LayoutNode(children=[child])
    with pytest.raises(ValueError):
        child.add(root)


def test_remove_detaches_and_marks_dirty():
    child = box(10, 10)
    root = LayoutNode(children=[child])
    solve(root)
    root.remove(child)
    assert root.dirty and child.parent is None
    assert solve(root).size == (0, 0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (3, Edges(3, 3, 3, 3)),
        ((1, 2), Edges(1, 2, 1, 2)),
        ((1, 2, 3, 4), Edges(1, 2, 3, 4)),
    ],
)
def test_edges_shorthands(value, expected):
    assert Edges.of(value) == expected


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Edges.of((1, 2, 3)),
        lambda: Edges.of(-1),
        lambda: fixed(-1),
        lambda: fit(min=5, max=1),
        lambda: grow(weight=0),
        lambda: percent(1.5),
        lambda: LayoutNode(align_main=Align.STRETCH),
        lambda: LayoutNode(align_cross=Align.SPACE_BETWEEN),
        lambda: LayoutNode(gap=-1),
    ],
)
def test_invalid_values_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_unknown_property_raises():
    with pytest.raises(TypeError):
        LayoutNode(colour="red")


def test_string_enum_values_are_accepted():
    node = LayoutNode(direction="column", align_cross="center")
    assert node.style.direction is Direction.COLUMN
    assert node.style.align_cross is Align.CENTER


def test_large_tree_solves_quickly():
    rows = [
        LayoutNode(children=[box(width=grow()) for _ in range(9)], width=grow())
        for _ in range(100)
    ]
    root = LayoutNode(
        children=rows, width=1000, height=1000, direction=Direction.COLUMN
    )
    start = time.perf_counter()
    solve(root)
    elapsed = time.perf_counter() - start
    npt.assert_array_less(elapsed, 0.2)
    assert rows[-1].children[-1].rect.x + rows[-1].children[-1].rect.width == 1000


@pytest.mark.parametrize("size,space", [(1, 4), (3, 6), (11, 12)])
def test_centering_keeps_integer_size(size, space):
    child = box(size, 1)
    solve(LayoutNode(children=[child], width=space, height=1, align_main=Align.CENTER))
    assert child.rect.width == size
    assert child.rect.x == math.floor((space - size) / 2 + 0.5)


def test_wrap_resolves_percent_before_breaking_lines():
    children = [box(width=percent(0.5)) for _ in range(4)]
    solve(LayoutNode(children=children, width=fixed(100), wrap=True))
    assert [c.rect.width for c in children] == [50, 50, 50, 50]
    assert [c.rect.position for c in children] == [(0, 0), (50, 0), (0, 10), (50, 10)]


@pytest.mark.parametrize(
    "fraction,positions",
    [
        (0.45, [(0, 0), (55, 0), (0, 20)]),
        (0.5, [(0, 0), (0, 20), (0, 40)]),
    ],
)
def test_wrap_percent_ignores_gaps_and_matches_across_lines(fraction, positions):
    children = [box(width=percent(fraction)) for _ in range(3)]
    solve(LayoutNode(children=children, width=fixed(100), wrap=True, gap=10))
    assert len({c.rect.width for c in children}) == 1
    assert children[0].rect.width == fraction * 100
    assert [c.rect.position for c in children] == positions


def test_wrap_percent_height_in_fit_row_does_not_overflow():
    tall = box(fixed(60), 40)
    half = box(50, percent(1.0))
    root = LayoutNode(children=[tall, half], width=100, wrap=True)
    rect = solve(root)
    assert half.rect.y + half.rect.height <= rect.height


def test_wrap_percent_height_in_fit_row_takes_content_height():
    head = text_leaf(10, 30, height=percent(0.5))
    base = text_leaf(10, 80)
    tail = text_leaf(95, 25, height=percent(0.5))
    root = LayoutNode(children=[head, base, tail], width=fixed(100), wrap=True)
    assert solve(root).height == 105
    assert head.rect.height == 30
    assert tail.rect.position == (0, 80)
    assert tail.rect.height == 25


def _stretched_wrap_row(content, row_height):
    child = LayoutNode(measure=lambda _: (10, content), height=percent(0.5))
    row = LayoutNode(children=[child], width=fixed(100), wrap=True)
    solve(LayoutNode(children=[row], height=row_height, align_cross=Align.STRETCH))
    return row, child


def _grow_wrap_row(content, column_height):
    child = LayoutNode(measure=lambda _: (10, content), height=percent(0.5))
    row = LayoutNode(children=[child], width=fixed(100), height=grow(), wrap=True)
    solve(LayoutNode(children=[row], height=column_height, direction=Direction.COLUMN))
    return row, child


def _bounded_wrap_row(content, row_height):
    child = LayoutNode(measure=lambda _: (10, content), height=percent(0.5))
    row = LayoutNode(children=[child], width=fixed(100), height=row_height, wrap=True)
    solve(row)
    return row, child


@pytest.mark.parametrize(
    "build,content,row_size,expected",
    [
        (_stretched_wrap_row, 30, 29, (29, 15)),
        (_stretched_wrap_row, 30, 30, (30, 15)),
        (_stretched_wrap_row, 30, 31, (31, 16)),
        (_grow_wrap_row, 40, 40, (40, 20)),
        (_grow_wrap_row, 40, 41, (41, 21)),
        (_bounded_wrap_row, 20, fit(min=100), (100, 50)),
        (_bounded_wrap_row, 100, fit(max=30), (30, 15)),
    ],
)
def test_wrap_percent_height_follows_how_the_row_was_sized(
    build, content, row_size, expected
):
    row, child = build(content, row_size)
    assert (row.rect.height, child.rect.height) == expected


def test_wrap_percent_height_in_grow_row_uses_final_height():
    head = text_leaf(10, 30, height=percent(0.5))
    row = LayoutNode(children=[head], width=fixed(100), height=grow(), wrap=True)
    solve(LayoutNode(children=[row], height=100, direction=Direction.COLUMN))
    assert row.rect.height == 100
    assert head.rect.height == 50


def test_wrap_percent_height_in_stretched_row_uses_final_height():
    head = text_leaf(10, 30, height=percent(0.5))
    row = LayoutNode(children=[head], width=fixed(100), wrap=True)
    solve(LayoutNode(children=[row], height=100, align_cross=Align.STRETCH))
    assert row.rect.height == 100
    assert head.rect.height == 50


def _solve_wrap_row_time(count):
    """Return the best of three times to solve a wrap row of ``count`` boxes."""
    times = []
    for _ in range(3):
        row = LayoutNode(children=[box() for _ in range(count)], wrap=True)
        start = time.perf_counter()
        solve(row)
        times.append(time.perf_counter() - start)
    return min(times)


def test_wrap_line_breaking_scales_linearly():
    small = _solve_wrap_row_time(1000)
    large = _solve_wrap_row_time(4000)
    npt.assert_array_less(large / small, 8)


def test_wrap_cross_percent_uses_content_box():
    child = box(width=10, height=percent(0.5))
    solve(LayoutNode(children=[child], width=100, height=100, wrap=True))
    assert child.rect.height == 50


def test_floating_is_placed_after_its_floating_target():
    button = box(50, 20)
    menu = box(40, 30, position=Position.FLOATING)
    menu.style.update(float_to=button, float_anchor=(1, 0))
    submenu = box(20, 20, position=Position.FLOATING)
    submenu.style.update(float_to=menu, float_anchor=(0, 1))
    root = LayoutNode(children=[submenu, button, menu])
    solve(root)
    assert submenu.rect.position == (50, 30)
    button.style.update(margin=(0, 0, 0, 100))
    solve(root)
    assert submenu.rect.position == (150, 30)


def test_floating_cycle_is_placed_once_with_a_warning(caplog):
    first = box(10, 10, position=Position.FLOATING)
    second = box(10, 10, position=Position.FLOATING, offset=(5, 0))
    first.style.update(float_to=second)
    second.style.update(float_to=first)
    root = LayoutNode(children=[first, second])
    solve(root)
    assert "2 floating nodes form a cycle" in caplog.text
    assert second.rect.x == first.rect.x + 5


def test_floating_cycle_warns_once_per_cycle(caplog):
    first, second = box(position=Position.FLOATING), box(position=Position.FLOATING)
    first.style.update(float_to=second)
    second.style.update(float_to=first)
    root = LayoutNode(children=[first, second])
    for _ in range(3):
        solve(root)
    assert caplog.text.count("cycle") == 1


def test_floating_attached_to_own_child_warns_in_the_singular(caplog):
    menu = LayoutNode(position=Position.FLOATING, offset=(1, 0))
    item = menu.add(box())
    menu.style.update(float_to=item)
    solve(LayoutNode(children=[menu]))
    assert "itself or to its own descendant" in caplog.text
    assert "1 floating nodes" not in caplog.text


def test_floating_chain_into_cycle_places_followers_from_fresh_rects():
    lead = box(position=Position.FLOATING, offset=(1, 0))
    first = box(position=Position.FLOATING, offset=(10, 0))
    second = box(position=Position.FLOATING, offset=(100, 0))
    lead.style.update(float_to=first)
    first.style.update(float_to=second)
    second.style.update(float_to=first)
    solve(LayoutNode(children=[lead, first, second]))
    assert second.rect.x == first.rect.x + 100
    assert lead.rect.x == first.rect.x + 1


def test_long_floating_chain_in_reverse_order():
    nodes = [box(position=Position.FLOATING, offset=(1, 0)) for _ in range(1500)]
    for node, target in zip(nodes, nodes[1:], strict=False):
        node.style.update(float_to=target)
    solve(LayoutNode(children=nodes))
    assert nodes[0].rect.x == 1500


def test_floating_margin_moves_and_shrinks_alike():
    menu = box(width=percent(0.5), position=Position.FLOATING, margin=5)
    solve(LayoutNode(children=[menu], width=100, height=100))
    assert menu.rect.position == (5, 5)
    assert menu.rect.width == 45


@pytest.mark.parametrize(
    "height,align", [(grow(), Align.START), (fit(), Align.STRETCH)]
)
def test_shrink_zero_never_shrinks_on_cross_axis(height, align):
    child = text_leaf(10, 50, height=height, shrink=0)
    solve(LayoutNode(children=[child], width=10, height=10, align_cross=align))
    assert child.rect.height == 50


@pytest.mark.parametrize(
    "width,content",
    [(fit(max=50), 100), (grow(min=50), 5), (percent(0.1, max=50), 100)],
)
def test_non_resizable_fit_size_is_clamped(width, content):
    leaf = LayoutNode(measure=lambda _: (content, 10), width=width, resizable=False)
    solve(LayoutNode(children=[leaf], width=200))
    assert leaf.rect.width == 50


def test_absolute_grow_and_percent_respect_margin():
    overlay = box(width=grow(), height=grow(), position=Position.ABSOLUTE, margin=5)
    half = box(
        width=percent(0.5),
        height=percent(0.5),
        position=Position.ABSOLUTE,
        margin=5,
    )
    solve(LayoutNode(children=[overlay, half], width=50, height=60))
    assert overlay.rect.position == (5, 5)
    assert overlay.rect.size == (40, 50)
    assert half.rect.size == (20, 25)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit(1, 2),
        lambda: grow(2),
        lambda: place(LayoutNode(), (0, 0)),
        lambda: LayoutNode(LayoutStyle()),
        lambda: LayoutStyle(fixed(10)),
    ],
)
def test_optional_parameters_are_keyword_only(call):
    with pytest.raises(TypeError):
        call()
