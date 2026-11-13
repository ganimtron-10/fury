"""Test containers module."""

from os.path import join as pjoin

import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.data import DATA_DIR
from fury.testing import VisualTest


def test_panel2d_initialization_default():
    """Test Panel2D initialization and default property values."""
    panel = ui.Panel2D(size=(100, 100))

    npt.assert_equal(panel.size, (100, 100))
    npt.assert_equal(panel.has_border, False)
    npt.assert_array_almost_equal(panel.color, [0.1, 0.1, 0.1])
    npt.assert_almost_equal(panel.opacity, 0.7)
    assert panel.background in panel._elements


def test_panel2d_initialization_with_border():
    """
    Test Panel2D initialization ensures border elements are created and
    sized.
    """
    panel = ui.Panel2D(size=(300, 150), has_border=True, border_width=10)

    assert panel.has_border is True
    assert isinstance(panel.borders["left"], ui.Rectangle2D)
    assert len(panel.borders) == 4

    assert len(panel._elements) == 5

    npt.assert_equal(panel.borders["left"].size, (10, 160))
    npt.assert_equal(panel.borders["right"].size, (10, 160))
    npt.assert_equal(panel.borders["top"].size, (310, 10))
    npt.assert_equal(panel.borders["bottom"].size, (310, 10))


def test_panel2d_color_opacity_properties():
    """Test Panel2D color and opacity setters/getters proxy to background."""
    color = (0.9, 0.5, 0.1)
    opacity = 0.8

    panel = ui.Panel2D(size=(10, 10), color=color, opacity=opacity)

    npt.assert_array_almost_equal(panel.color, color)
    npt.assert_array_almost_equal(panel.background.color, color)

    npt.assert_almost_equal(panel.opacity, opacity)
    npt.assert_almost_equal(panel.background.opacity, opacity)

    new_color = (0.2, 0.4, 0.6)
    new_opacity = 0.2

    panel.color = new_color
    npt.assert_array_almost_equal(panel.color, new_color)
    npt.assert_array_almost_equal(panel.background.color, new_color)

    panel.opacity = new_opacity
    npt.assert_almost_equal(panel.opacity, new_opacity)
    npt.assert_almost_equal(panel.background.opacity, new_opacity)


def test_panel2d_color_normalization():
    """Test Panel2D normalizes hex / [0, 255] colors for body and border."""
    panel = ui.Panel2D(
        size=(10, 10), color="#FF0000", has_border=True, border_color=(0, 255, 0)
    )
    npt.assert_array_almost_equal(panel.color, [1, 0, 0])
    npt.assert_array_almost_equal(panel.background.color, [1, 0, 0])
    npt.assert_array_almost_equal(panel.borders["top"].color, [0, 1, 0])

    panel.color = (0, 0, 255)
    npt.assert_array_almost_equal(panel.color, [0, 0, 1])

    panel.border_color = ("left", "#FFFFFF")
    npt.assert_array_almost_equal(panel.borders["left"].color, [1, 1, 1])


def test_panel2d_resize_and_border_sync():
    """Test resizing updates background and correctly sizes border elements."""
    size = (100, 50)
    panel = ui.Panel2D(size=size, has_border=True, border_width=5)

    npt.assert_equal(panel.size, size)
    npt.assert_equal(panel.background.size, size)

    npt.assert_equal(panel.borders["left"].size, (5, 55))
    npt.assert_equal(panel.borders["right"].size, (5, 55))
    npt.assert_equal(panel.borders["top"].size, (105, 5))
    npt.assert_equal(panel.borders["bottom"].size, (105, 5))

    new_size = (300, 150)
    panel.resize(new_size)

    npt.assert_equal(panel.size, new_size)
    npt.assert_equal(panel.background.size, new_size)

    npt.assert_equal(panel.borders["left"].size, (5, 155))
    npt.assert_equal(panel.borders["right"].size, (5, 155))
    npt.assert_equal(panel.borders["top"].size, (305, 5))
    npt.assert_equal(panel.borders["bottom"].size, (305, 5))


def test_panel2d_add_remove_element():
    """Test add_element handles normalization and removes correctly."""
    panel = ui.Panel2D(size=(100, 100))
    mock_child = ui.Rectangle2D(size=(10, 10))

    initial_elements_count = len(panel._elements)

    panel.add_element(mock_child, coords=(0.5, 0.75))

    assert mock_child in panel._elements
    assert mock_child in panel._children
    assert len(panel._elements) == initial_elements_count + 1

    mock_child_2 = ui.Rectangle2D(size=(5, 5))
    panel.add_element(mock_child_2, coords=(10, 20))
    assert mock_child_2 in panel._elements
    assert mock_child_2 in panel._children

    panel.remove_element(mock_child)
    assert mock_child not in panel._elements
    assert mock_child not in panel._children

    with npt.assert_raises(ValueError):
        panel.remove_element(mock_child)


def test_panel2d_no_border_actors_and_scene():
    """
    Test Panel2D without borders can return actors and be added to a
    scene.
    """
    panel = ui.Panel2D(size=(200, 200))

    # _get_actors should work without borders
    actors = panel.actors
    assert len(actors) > 0

    # _update_actors_position should work without borders
    panel.set_position((10, 10))

    # Adding to a scene should not raise
    scene = window.Scene()
    scene.add(panel)

    # Adding child elements should also work
    child = ui.Rectangle2D(size=(20, 20))
    panel.add_element(child, (0.5, 0.5))
    panel._update_actors_position()

    # Border properties should return empty when has_border=False
    assert panel.border_color == []
    assert panel.border_width == []
    panel.update_border_coords()
    panel.resize((300, 300))


def test_panel2d_z_ordering():
    """
    Test that internal Z-ordering is set correctly by
    _update_actors_position.
    """
    panel = ui.Panel2D(size=(100, 100), has_border=True, border_width=1)

    custom_element = ui.Rectangle2D(size=(5, 5))
    panel.add_element(custom_element, coords=(0, 0))

    panel._update_actors_position()

    assert panel.background.z_order == 0
    assert all(border.z_order == 1 for border in panel.borders.values())
    assert custom_element.z_order == 2


def test_panel2d_visual_snapshot():
    """
    Visual test for Panel2D: Verifies composite rendering and color
    checks.
    """
    panel_size = (300, 150)
    panel_pos = (100, 100)

    panel = ui.Panel2D(
        size=panel_size,
        position=panel_pos,
        color=(0.1, 0.1, 0.1),
        has_border=True,
        border_width=5,
        border_color=(1.0, 0.0, 1.0),
    )

    disk = ui.Disk2D(outer_radius=25, color=(1, 1, 0))
    panel.add_element(disk, coords=(0.5, 0.5), anchor="center")

    scene = window.Scene()
    scene.add(panel)

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1
    assert np.sum(arr) > 0

    mean_r, mean_g, mean_b, _ = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert mean_r > mean_g
    assert mean_b > mean_g

    scene = window.Scene()
    scene.add(panel)

    panel.set_visibility(False)
    arr_hidden = window.snapshot(scene=scene, fname=None, return_array=True)
    report_hidden = window.analyze_snapshot(arr_hidden, find_objects=True)
    assert report_hidden.objects == 0
    npt.assert_almost_equal(np.mean(arr_hidden[..., :3]), 0, decimal=0)


def test_tab_panel2d_properties_and_content():
    """Test TabPanel2D title proxies and content element management."""
    tab_panel = ui.TabPanel2D(size=(120, 30), title="Tab 1")

    npt.assert_equal(tab_panel.size, (120, 30))
    npt.assert_equal(tab_panel.title, "Tab 1")
    npt.assert_equal(tab_panel.title_bold, False)
    npt.assert_equal(tab_panel.title_italic, False)
    npt.assert_equal(tab_panel.title_font_size, 18)
    npt.assert_equal(tab_panel.actors, [])
    assert not tab_panel.content_panel.actors[0].visible

    tab_panel_with_content = ui.TabPanel2D(
        size=(120, 30), content_size=(200, 180), content_visible=True
    )
    npt.assert_equal(tab_panel_with_content.content_panel.size, (200, 180))
    assert tab_panel_with_content.content_panel.actors[0].visible

    tab_panel.title = "Updated"
    tab_panel.title_bold = True
    tab_panel.title_italic = True
    tab_panel.title_color = (1, 0, 0)
    tab_panel.title_font_size = 12

    npt.assert_equal(tab_panel.title, "Updated")
    npt.assert_equal(tab_panel.title_bold, True)
    npt.assert_equal(tab_panel.title_italic, True)
    npt.assert_array_almost_equal(tab_panel.title_color, (1, 0, 0))
    npt.assert_equal(tab_panel.title_font_size, 12)

    element = ui.Rectangle2D(size=(10, 10))
    tab_panel.add_element(element, (0.5, 0.5))
    assert element in tab_panel.content_panel._elements
    assert not element.actors[0].visible

    tab_panel.update_element(element, (10, 10))
    tab_panel.remove_element(element)
    assert element not in tab_panel.content_panel._elements


def test_tab_panel2d_color_normalization():
    """Test TabPanel2D normalizes hex / [0, 255] colors."""
    tab_panel = ui.TabPanel2D(size=(120, 30), color="#FF0000")
    npt.assert_array_almost_equal(tab_panel.color, [1, 0, 0])
    npt.assert_array_almost_equal(tab_panel.panel.color, [1, 0, 0])

    tab_panel.color = (0, 0, 255)
    npt.assert_array_almost_equal(tab_panel.color, [0, 0, 1])

    tab_panel.title_color = "#00FF00"
    npt.assert_array_almost_equal(tab_panel.title_color, [0, 1, 0])


def test_tab_ui_initialization_and_selection():
    """Test TabUI initialization, element visibility, and tab selection."""
    tab_ui = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        font_size=14,
        startup_tab_id=1,
    )

    npt.assert_equal(tab_ui.size, (300, 300))
    npt.assert_equal(len(tab_ui.tabs), 3)
    npt.assert_equal(tab_ui.actors, [])
    npt.assert_equal(tab_ui.tabs[0].title, "Tab 1")
    npt.assert_equal(tab_ui.tabs[1].title, "Tab 2")
    npt.assert_equal(tab_ui.tabs[2].title, "Tab 3")
    npt.assert_equal(tab_ui.tabs[0].title_font_size, 14)
    npt.assert_equal(tab_ui.tabs[1].title_font_size, 14)
    npt.assert_equal(tab_ui.tabs[2].title_font_size, 14)
    npt.assert_equal(tab_ui.active_tab_idx, 1)
    npt.assert_equal(tab_ui.collapsed, False)

    assert not tab_ui.tabs[0].content_panel.actors[0].visible
    assert tab_ui.tabs[1].content_panel.actors[0].visible
    assert not tab_ui.tabs[2].content_panel.actors[0].visible

    element = ui.Rectangle2D(size=(20, 20))
    tab_ui.add_element(1, element, (0.5, 0.5))
    assert element.actors[0].visible

    changes = []
    collapses = []
    tab_ui.on_change = lambda ui_obj: changes.append(ui_obj.active_tab_idx)
    tab_ui.on_collapse = lambda ui_obj: collapses.append(ui_obj.collapsed)

    tab_ui.select_tab(2)
    npt.assert_equal(tab_ui.active_tab_idx, 2)
    npt.assert_equal(changes, [2])
    assert not tab_ui.tabs[1].content_panel.actors[0].visible
    assert tab_ui.tabs[2].content_panel.actors[0].visible

    tab_ui.select_tab(2)
    npt.assert_equal(tab_ui.active_tab_idx, 2)
    npt.assert_equal(tab_ui.collapsed, False)
    npt.assert_equal(changes, [2, 2])
    npt.assert_equal(collapses, [])
    assert not tab_ui.tabs[2].content_panel.actors[0].visible

    tab_ui.collapse_tab_ui()
    npt.assert_equal(tab_ui.active_tab_idx, None)
    npt.assert_equal(tab_ui.collapsed, True)
    npt.assert_equal(collapses, [True])


def test_tab_ui_layout_and_invalid_index():
    """Test TabUI layout and invalid index checks."""
    tab_ui_top = ui.TabUI(
        position=(50, 50), size=(300, 300), tab_titles=["Tab 1", "Tab 2", "Tab 3"]
    )
    tab_ui_bottom = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="bottom",
    )
    tab_ui_left = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="left",
    )
    tab_ui_right = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="right",
    )
    tab_ui_accordion = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        startup_tab_id=1,
        tab_bar_pos="accordion",
    )
    tab_ui_invalid = ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        tab_bar_pos="side",
    )

    npt.assert_equal(tab_ui_top.tabs[0].size, (100, 30))
    npt.assert_equal(tab_ui_top.tabs[0].content_panel.size, (300, 270))
    npt.assert_array_equal(tab_ui_top.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_top.tabs[0].content_panel.get_position(), (50, 80))

    npt.assert_array_equal(tab_ui_bottom.tabs[0].content_panel.get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_bottom.tabs[0].get_position(), (50, 320))

    npt.assert_equal(tab_ui_left.tabs[0].size, (30, 100))
    npt.assert_equal(tab_ui_left.tabs[0].content_panel.size, (270, 300))
    npt.assert_array_equal(tab_ui_left.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_left.tabs[1].get_position(), (50, 150))
    npt.assert_array_equal(tab_ui_left.tabs[0].content_panel.get_position(), (80, 50))

    npt.assert_equal(tab_ui_right.tabs[0].size, (30, 100))
    npt.assert_equal(tab_ui_right.tabs[0].content_panel.size, (270, 300))
    npt.assert_array_equal(tab_ui_right.tabs[0].content_panel.get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_right.tabs[0].get_position(), (320, 50))
    npt.assert_array_equal(tab_ui_right.tabs[1].get_position(), (320, 150))

    npt.assert_equal(tab_ui_accordion.tabs[0].size, (300, 30))
    npt.assert_equal(tab_ui_accordion.tabs[0].content_panel.size, (300, 210))
    npt.assert_array_equal(tab_ui_accordion.tabs[0].get_position(), (50, 50))
    npt.assert_array_equal(tab_ui_accordion.tabs[1].get_position(), (50, 80))
    npt.assert_array_equal(
        tab_ui_accordion.tabs[1].content_panel.get_position(), (50, 110)
    )
    npt.assert_array_equal(tab_ui_accordion.tabs[2].get_position(), (50, 320))

    npt.assert_equal(tab_ui_invalid.tab_bar_pos, "top")
    npt.assert_equal(tab_ui_invalid.tabs[0].size, (100, 30))
    npt.assert_array_equal(
        tab_ui_invalid.tabs[0].content_panel.get_position(), (50, 80)
    )

    with npt.assert_raises(IndexError):
        tab_ui_top.add_element(3, ui.Rectangle2D(size=(10, 10)), (0.5, 0.5))

    with npt.assert_raises(IndexError):
        tab_ui_top.remove_element(3, ui.Rectangle2D(size=(10, 10)))

    with npt.assert_raises(IndexError):
        tab_ui_top.update_element(3, ui.Rectangle2D(size=(10, 10)), (0.5, 0.5))

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles=[])

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles="Tab")

    with npt.assert_raises(ValueError):
        ui.TabUI(position=(50, 50), size=(300, 300), tab_titles=["Tab", 2])


# def setup_module():
#     fetch_viz_icons()


# def test_wrong_interactor_style():
#     panel = ui.Panel2D(size=(300, 150))
#     dummy_scene = window.Scene()
#     _ = window.ShowManager(scene=dummy_scene, interactor_style="trackball")
#     npt.assert_raises(TypeError, panel.add_to_scene, dummy_scene)


# @pytest.mark.skipif(
#     skip_linux or skip_win,
#     reason="This test does not work on Windows. Need to be introspected",
# )
# def test_grid_ui1(interactive=False):
#     vol1 = np.zeros((100, 100, 100))
#     vol1[25:75, 25:75, 25:75] = 100

#     colors = distinguishable_colormap(nb_colors=3)
#     contour_actor1 = actor.contour_from_roi(
#         vol1,
#         affine=np.eye(4),
#         color=colors[0],
#         opacity=1.0,
#     )

#     vol2 = np.zeros((100, 100, 100))
#     vol2[25:75, 25:75, 25:75] = 100

#     contour_actor2 = actor.contour_from_roi(
#         vol2,
#         affine=np.eye(4),
#         color=colors[1],
#         opacity=1.0,
#     )

#     vol3 = np.zeros((100, 100, 100))
#     vol3[25:75, 25:75, 25:75] = 100

#     contour_actor3 = actor.contour_from_roi(
#         vol3,
#         affine=np.eye(4),
#         color=colors[2],
#         opacity=1.0,
#     )

#     scene = window.Scene()
#     actors = []
#     texts = []

#     actors.append(contour_actor1)
#     text_actor1 = actor.text_3d("cube 1", justification="center")
#     texts.append(text_actor1)

#     actors.append(contour_actor2)
#     text_actor2 = actor.text_3d("cube 2", justification="center")
#     texts.append(text_actor2)

#     actors.append(contour_actor3)
#     text_actor3 = actor.text_3d("cube 3", justification="center")
#     texts.append(text_actor3)

#     actors.append(shallow_copy(contour_actor1))
#     text_actor1 = actor.text_3d("cube 4", justification="center")
#     texts.append(text_actor1)

#     actors.append(shallow_copy(contour_actor2))
#     text_actor2 = actor.text_3d("cube 5", justification="center")
#     texts.append(text_actor2)

#     actors.append(shallow_copy(contour_actor3))
#     text_actor3 = actor.text_3d("cube 6", justification="center")
#     texts.append(text_actor3)

#     actors.append(shallow_copy(contour_actor1))
#     text_actor1 = actor.text_3d("cube 7", justification="center")
#     texts.append(text_actor1)

#     actors.append(shallow_copy(contour_actor2))
#     text_actor2 = actor.text_3d("cube 8", justification="center")
#     texts.append(text_actor2)

#     actors.append(shallow_copy(contour_actor3))
#     text_actor3 = actor.text_3d("cube 9", justification="center")
#     texts.append(text_actor3)

#     counter = itertools.count()
#     show_m = window.ShowManager(scene=scene)

#     def timer_callback(_obj, _event):
#         nonlocal show_m, counter
#         cnt = next(counter)
#         show_m.scene.zoom(1)
#         show_m.render()
#         if cnt == 10:
#             show_m.exit()

#     # show the grid with the captions
#     grid_ui = ui.GridUI(
#         actors=actors,
#         captions=texts,
#         caption_offset=(0, -50, 0),
#         cell_padding=(60, 60),
#         dim=(3, 3),
#         rotation_axis=(1, 0, 0),
#     )

#     scene.add(grid_ui)

#     show_m.add_timer_callback(True, 200, timer_callback)
#     show_m.start()

#     arr = window.snapshot(scene)
#     report = window.analyze_snapshot(arr)
#     npt.assert_equal(report.objects > 9, True)
#     # print(report.objects)

#     # testing grid without captions
#     new_sm = window.ShowManager()
#     t = 0
#     try:
#         grid_ui_2 = ui.GridUI(actors=actors)
#         new_sm.scene.add(grid_ui_2)
#         t = 1
#     except:  # noqa: E722
#         pass

#     npt.assert_equal(t, 1)


# def test_grid_ui2(interactive=False):
#     vol1 = np.zeros((100, 100, 100))
#     vol1[25:75, 25:75, 25:75] = 100

#     colors = distinguishable_colormap(nb_colors=3)
#     contour_actor1 = actor.contour_from_roi(
#         vol1,
#         affine=np.eye(4),
#         color=colors[0],
#         opacity=1.0,
#     )

#     vol2 = np.zeros((100, 100, 100))
#     vol2[25:75, 25:75, 25:75] = 100

#     contour_actor2 = actor.contour_from_roi(
#         vol2,
#         affine=np.eye(4),
#         color=colors[1],
#         opacity=1.0,
#     )

#     vol3 = np.zeros((100, 100, 100))
#     vol3[25:75, 25:75, 25:75] = 100

#     contour_actor3 = actor.contour_from_roi(
#         vol3,
#         affine=np.eye(4),
#         color=colors[2],
#         opacity=1.0,
#     )

#     scene = window.Scene()
#     actors = []
#     texts = []

#     actors.append(contour_actor1)
#     text_actor1 = actor.text_3d("cube 1", justification="center")
#     texts.append(text_actor1)

#     actors.append(contour_actor2)
#     text_actor2 = actor.text_3d("cube 2", justification="center")
#     texts.append(text_actor2)

#     actors.append(contour_actor3)
#     text_actor3 = actor.text_3d("cube 3", justification="center")
#     texts.append(text_actor3)

#     actors.append(shallow_copy(contour_actor1))
#     text_actor1 = actor.text_3d("cube 4", justification="center")
#     texts.append(text_actor1)

#     actors.append(shallow_copy(contour_actor2))
#     text_actor2 = actor.text_3d("cube 5", justification="center")
#     texts.append(text_actor2)

#     actors.append(shallow_copy(contour_actor3))
#     text_actor3 = actor.text_3d("cube 6", justification="center")
#     texts.append(text_actor3)

#     actors.append(shallow_copy(contour_actor1))
#     text_actor1 = actor.text_3d("cube 7", justification="center")
#     texts.append(text_actor1)

#     actors.append(shallow_copy(contour_actor2))
#     text_actor2 = actor.text_3d("cube 8", justification="center")
#     texts.append(text_actor2)

#     actors.append(shallow_copy(contour_actor3))
#     text_actor3 = actor.text_3d("cube 9", justification="center")
#     texts.append(text_actor3)

#     # this needs to happen automatically when start() ends.
#     # for act in actors:
#     #     act.RemoveAllObservers()

#     filename = "test_grid_ui"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     current_size = (900, 600)
#     scene = window.Scene()
#     show_manager = window.ShowManager(
#         scene=scene, size=current_size, title="FURY GridUI"
#     )

#     grid_ui2 = ui.GridUI(
#         actors=actors,
#         captions=texts,
#         caption_offset=(0, -50, 0),
#         cell_padding=(60, 60),
#         dim=(3, 3),
#         rotation_axis=None,
#     )

#     scene.add(grid_ui2)

#     event_counter = EventCounter()
#     event_counter.monitor(grid_ui2)

#     if interactive:
#         show_manager.start()

#     recording = False

#     if recording:
#         # Record the following events
#         # 1. Left click on top left box (will rotate the box)
#         show_manager.record_events_to_file(recording_filename)
#         # print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_image_container_2d(interactive=False):
#     image_test = ui.ImageContainer2D(img_path=read_viz_icons(fname="home3.png"))

#     image_test.center = (300, 300)
#     npt.assert_equal(image_test.size, (100, 100))

#     image_test.scale((2, 2))
#     npt.assert_equal(image_test.size, (200, 200))

#     current_size = (600, 600)
#     show_manager = window.ShowManager(size=current_size, title="FURY Button")
#     show_manager.scene.add(image_test)
#     if interactive:
#         show_manager.start()


# def test_ui_tab_ui(interactive=False):
#     filename = "test_ui_tab_ui"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     tab_ui = ui.TabUI(position=(50, 50), size=(300, 300), nb_tabs=3, draggable=True)

#     tab_ui.tabs[0].title = "Tab 1"
#     tab_ui.tabs[1].title = "Tab 2"
#     tab_ui.tabs[2].title = "Tab 3"

#     npt.assert_equal(tab_ui.tabs[0].title_bold, False)
#     npt.assert_equal(tab_ui.tabs[1].title_bold, False)
#     npt.assert_equal(tab_ui.tabs[2].title_bold, False)

#     tab_ui.tabs[0].title_bold = True
#     tab_ui.tabs[1].title_bold = False
#     tab_ui.tabs[2].title_bold = True

#     npt.assert_equal(tab_ui.tabs[0].title_bold, True)
#     npt.assert_equal(tab_ui.tabs[1].title_bold, False)
#     npt.assert_equal(tab_ui.tabs[2].title_bold, True)

#     npt.assert_equal(tab_ui.tabs[0].title_color, (0.0, 0.0, 0.0))
#     npt.assert_equal(tab_ui.tabs[1].title_color, (0.0, 0.0, 0.0))
#     npt.assert_equal(tab_ui.tabs[2].title_color, (0.0, 0.0, 0.0))

#     tab_ui.tabs[0].title_color = (1, 0, 0)
#     tab_ui.tabs[1].title_color = (0, 1, 0)
#     tab_ui.tabs[2].title_color = (0, 0, 1)

#     npt.assert_equal(tab_ui.tabs[0].title_color, (1.0, 0.0, 0.0))
#     npt.assert_equal(tab_ui.tabs[1].title_color, (0.0, 1.0, 0.0))
#     npt.assert_equal(tab_ui.tabs[2].title_color, (0.0, 0.0, 1.0))

#     npt.assert_equal(tab_ui.tabs[0].title_font_size, 18)
#     npt.assert_equal(tab_ui.tabs[1].title_font_size, 18)
#     npt.assert_equal(tab_ui.tabs[2].title_font_size, 18)

#     tab_ui.tabs[0].title_font_size = 10
#     tab_ui.tabs[1].title_font_size = 20
#     tab_ui.tabs[2].title_font_size = 30

#     npt.assert_equal(tab_ui.tabs[0].title_font_size, 10)
#     npt.assert_equal(tab_ui.tabs[1].title_font_size, 20)
#     npt.assert_equal(tab_ui.tabs[2].title_font_size, 30)

#     npt.assert_equal(tab_ui.tabs[0].title_italic, False)
#     npt.assert_equal(tab_ui.tabs[1].title_italic, False)
#     npt.assert_equal(tab_ui.tabs[2].title_italic, False)

#     tab_ui.tabs[0].title_italic = False
#     tab_ui.tabs[1].title_italic = True
#     tab_ui.tabs[2].title_italic = False

#     npt.assert_equal(tab_ui.tabs[0].title_italic, False)
#     npt.assert_equal(tab_ui.tabs[1].title_italic, True)
#     npt.assert_equal(tab_ui.tabs[2].title_italic, False)

#     tab_ui.add_element(0, ui.Checkbox(["Option 1", "Option 2"]), (0.5, 0.5))
#     tab_ui.add_element(1, ui.LineSlider2D(), (0.0, 0.5))
#     tab_ui.add_element(2, ui.TextBlock2D(), (0.5, 0.5))

#     with npt.assert_raises(IndexError):
#         tab_ui.add_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     with npt.assert_raises(IndexError):
#         tab_ui.remove_element(3, ui.TextBlock2D())

#     with npt.assert_raises(IndexError):
#         tab_ui.update_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     npt.assert_equal("Tab 1", tab_ui.tabs[0].title)
#     npt.assert_equal("Tab 2", tab_ui.tabs[1].title)
#     npt.assert_equal("Tab 3", tab_ui.tabs[2].title)

#     npt.assert_equal(3, tab_ui.nb_tabs)

#     collapses = itertools.count()
#     changes = itertools.count()

#     def collapse(tab_ui):
#         if tab_ui.collapsed:
#             next(collapses)

#     def tab_change(tab_ui):
#         next(changes)

#     tab_ui.on_change = tab_change
#     tab_ui.on_collapse = collapse

#     event_counter = EventCounter()
#     event_counter.monitor(tab_ui)

#     current_size = (800, 800)
#     show_manager = window.ShowManager(size=current_size, title="Tab UI Test")
#     show_manager.scene.add(tab_ui)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     npt.assert_equal(0, tab_ui.active_tab_idx)
#     npt.assert_equal(11, next(changes))
#     npt.assert_equal(5, next(collapses))


# def test_ui_tab_ui_position(interactive=False):
#     filename = "test_ui_tab_ui_top_position"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     tab_ui_top = ui.TabUI(
#         position=(50, 50), size=(300, 300), nb_tabs=3,
#         draggable=True, tab_bar_pos="top"
#     )

#     tab_ui_top.tabs[0].title = "Tab 1"
#     tab_ui_top.tabs[1].title = "Tab 2"
#     tab_ui_top.tabs[2].title = "Tab 3"

#     tab_ui_top.add_element(0, ui.Checkbox(["Option 1", "Option 2"]), (0.5, 0.5))
#     tab_ui_top.add_element(1, ui.LineSlider2D(), (0.0, 0.5))
#     tab_ui_top.add_element(2, ui.TextBlock2D(), (0.5, 0.5))

#     npt.assert_equal("Tab 1", tab_ui_top.tabs[0].title)
#     npt.assert_equal("Tab 2", tab_ui_top.tabs[1].title)
#     npt.assert_equal("Tab 3", tab_ui_top.tabs[2].title)

#     npt.assert_equal(3, tab_ui_top.nb_tabs)

#     npt.assert_equal((50, 50), tab_ui_top.position)
#     npt.assert_equal((300, 300), tab_ui_top.size)

#     with npt.assert_raises(IndexError):
#         tab_ui_top.add_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     with npt.assert_raises(IndexError):
#         tab_ui_top.remove_element(3, ui.TextBlock2D())

#     with npt.assert_raises(IndexError):
#         tab_ui_top.update_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     tab_ui_bottom = ui.TabUI(
#         position=(350, 50),
#         size=(300, 300),
#         nb_tabs=3,
#         draggable=True,
#         tab_bar_pos="bottom",
#     )

#     tab_ui_bottom.tabs[0].title = "Tab 1"
#     tab_ui_bottom.tabs[1].title = "Tab 2"
#     tab_ui_bottom.tabs[2].title = "Tab 3"

#     tab_ui_bottom.add_element(0, ui.Checkbox(["Option 1", "Option 2"]), (0.5, 0.5))
#     tab_ui_bottom.add_element(1, ui.LineSlider2D(), (0.0, 0.5))
#     tab_ui_bottom.add_element(2, ui.TextBlock2D(), (0.5, 0.5))

#     npt.assert_equal("Tab 1", tab_ui_bottom.tabs[0].title)
#     npt.assert_equal("Tab 2", tab_ui_bottom.tabs[1].title)
#     npt.assert_equal("Tab 3", tab_ui_bottom.tabs[2].title)

#     npt.assert_equal(3, tab_ui_bottom.nb_tabs)

#     npt.assert_equal((350, 50), tab_ui_bottom.position)
#     npt.assert_equal((300, 300), tab_ui_bottom.size)

#     with npt.assert_raises(IndexError):
#         tab_ui_bottom.add_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     with npt.assert_raises(IndexError):
#         tab_ui_bottom.remove_element(3, ui.TextBlock2D())

#     with npt.assert_raises(IndexError):
#         tab_ui_bottom.update_element(3, ui.TextBlock2D(), (0.5, 0.5, 0.5))

#     collapses = itertools.count()
#     changes = itertools.count()

#     def collapse(tab_ui_top):
#         if tab_ui_top.collapsed or tab_ui_bottom.collapsed:
#             next(collapses)

#     def tab_change(tab_ui_top):
#         next(changes)

#     tab_ui_top.on_change = tab_change
#     tab_ui_top.on_collapse = collapse

#     tab_ui_bottom.on_change = tab_change
#     tab_ui_bottom.on_collapse = collapse

#     event_counter = EventCounter()
#     event_counter.monitor(tab_ui_top)
#     event_counter.monitor(tab_ui_bottom)

#     current_size = (800, 800)
#     show_manager = window.ShowManager(size=current_size, title="Tab UI Test")
#     show_manager.scene.add(tab_ui_top)
#     show_manager.scene.add(tab_ui_bottom)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     npt.assert_equal(0, tab_ui_top.active_tab_idx)
#     npt.assert_equal(0, tab_ui_bottom.active_tab_idx)
#     npt.assert_equal(14, next(changes))
#     npt.assert_equal(5, next(collapses))


def test_ui_panel2d_visual(recording=False):
    panel = ui.Panel2D(
        size=(200, 200), color=(0.2, 0.3, 0.4), opacity=0.8, position=(300, 300)
    )

    event_counter = VisualTest()
    event_counter.monitor(panel)
    event_counter.record_or_test(
        pjoin(DATA_DIR, "test_ui_panel2d_container.log.gz"),
        pjoin(DATA_DIR, "test_ui_panel2d_container.json"),
        size=(600, 600),
        title="Panel2D Visual Test",
        recording=recording,
    )


def test_ui_tabui_visual(recording=False):
    tab_ui = ui.TabUI(
        size=(400, 300), tab_titles=["Home", "Settings"], position=(100, 100)
    )

    event_counter = VisualTest()
    event_counter.monitor(tab_ui)
    event_counter.record_or_test(
        pjoin(DATA_DIR, "test_ui_tabui_container.log.gz"),
        pjoin(DATA_DIR, "test_ui_tabui_container.json"),
        size=(600, 600),
        title="TabUI Visual Test",
        recording=recording,
    )


def _block(size=(40, 20)):
    """Create a plain rectangle to place in a layout."""
    return ui.Rectangle2D(size=size)


def test_column_places_children_and_fits():
    a, b = _block(), _block((60, 30))
    column = ui.Column(a, b, position=(10, 20), padding=5, gap=4)
    column.update_layout()
    npt.assert_array_equal(column.size, (70, 64))
    npt.assert_array_equal(a.get_position(), (15, 25))
    npt.assert_array_equal(b.get_position(), (15, 49))


def test_row_moves_children_without_a_new_layout():
    a, b = _block(), _block()
    row = ui.Row(a, b, gap=10)
    row.update_layout()
    row.set_position((100, 50))
    npt.assert_array_equal(a.get_position(), (100, 50))
    npt.assert_array_equal(b.get_position(), (150, 50))


def test_nested_boxes_keep_relative_offsets_when_moved():
    leaf = _block()
    inner = ui.Column(leaf, padding=3)
    outer = ui.Row(_block(), inner, gap=2)
    outer.update_layout()
    npt.assert_array_equal(leaf.get_position(), (45, 3))
    outer.set_position((10, 10))
    npt.assert_array_equal(leaf.get_position(), (55, 13))


def test_root_can_be_positioned_by_center_anchor():
    row = ui.Row(_block((100, 20)))
    row.set_position((400, 400), ui.Anchor.CENTER, ui.Anchor.CENTER)
    row.update_layout()
    npt.assert_array_equal(row.layout_rect.position, (350, 390))


def test_growing_children_resize_and_shrink_back():
    a, b = _block(), _block()
    a.set_layout(width=ui.grow())
    row = ui.Row(a, b, width=200)
    row.update_layout()
    npt.assert_array_equal(a.size, (160, 20))
    row.set_layout(width=120)
    row.update_layout()
    npt.assert_array_equal(a.size, (80, 20))


def test_panel_add_uses_layout_and_keeps_fixed_size():
    panel = ui.Panel2D((200, 100), position=(10, 10))
    panel.set_layout(direction=ui.Direction.COLUMN, padding=8, gap=2)
    a = panel.add(_block())
    b = panel.add(_block(), width=ui.grow(), height=ui.grow())
    panel.update_layout()
    npt.assert_array_equal(panel.size, (200, 100))
    npt.assert_array_equal(a.get_position(), (18, 18))
    npt.assert_array_equal(b.size, (184, 62))


def test_panel_resize_reflows_children():
    panel = ui.Panel2D((200, 100))
    child = panel.add(_block(), width=ui.grow())
    panel.update_layout()
    npt.assert_array_equal(child.size[0], 200)
    panel.resize((300, 100))
    panel.update_layout()
    npt.assert_array_equal(child.size[0], 300)


def test_panel_drag_keeps_flow_children_attached():
    panel = ui.Panel2D((200, 100), position=(0, 0))
    panel.set_layout(padding=10)
    child = panel.add(_block())
    panel.update_layout()
    panel.set_position((50, 60))
    npt.assert_array_equal(child.get_position(), (60, 70))


def test_panel_fit_height_resizes_background():
    panel = ui.Panel2D((200, 10))
    panel.set_layout(height=ui.fit(), direction=ui.Direction.COLUMN, padding=5)
    panel.add(_block((10, 40)))
    panel.update_layout()
    npt.assert_array_equal(panel.size, (200, 50))
    npt.assert_array_equal(panel.background.size, (200, 50))


def test_panel_add_element_stays_manual():
    panel = ui.Panel2D((200, 100))
    panel.set_layout(padding=10)
    manual = _block()
    panel.add_element(manual, (5, 5))
    flow = panel.add(_block())
    panel.update_layout()
    npt.assert_array_equal(manual.get_position(), (5, 5))
    npt.assert_array_equal(flow.get_position(), (10, 10))


def test_panel_remove_element_leaves_the_layout():
    panel = ui.Panel2D((200, 100))
    a, b = panel.add(_block()), panel.add(_block())
    panel.update_layout()
    panel.remove_element(a)
    panel.update_layout()
    npt.assert_array_equal(b.get_position(), (0, 0))


def test_panel_border_insets_content():
    panel = ui.Panel2D((100, 100), has_border=True, border_width=4)
    child = panel.add(_block())
    panel.update_layout()
    npt.assert_array_equal(child.get_position(), (4, 4))


def test_non_resizable_widget_keeps_its_size_when_asked_to_grow():
    slider = ui.LineSlider2D(length=100)
    natural = tuple(slider.size)
    row = ui.Row(slider, width=500)
    slider.set_layout(width=ui.grow())
    row.update_layout()
    npt.assert_array_equal(slider.size, natural)


def test_layout_is_solved_by_the_render_loop():
    child = _block()
    row = ui.Row(child, position=(30, 40), padding=5)
    scene = window.Scene()
    scene.add(row)
    show_m = window.ShowManager(scene=scene, size=(200, 200), window_type="offscreen")
    show_m._draw_function()
    assert not row._layout_node.dirty
    npt.assert_array_equal(child.get_position(), (35, 45))

    child.resize((50, 25))
    child.mark_layout_dirty()
    assert row._layout_node.dirty
    show_m._draw_function()
    npt.assert_array_equal(row.size, (60, 35))
    show_m.close()


def test_window_resize_refills_growing_root():
    child = _block()
    row = ui.Row(child, width=ui.grow())
    child.set_layout(width=ui.grow())
    scene = window.Scene()
    scene.add(row)
    show_m = window.ShowManager(scene=scene, size=(300, 200), window_type="offscreen")
    show_m._draw_function()
    npt.assert_array_equal(child.size[0], 300)
    show_m._resize((500, 200))
    show_m._draw_function()
    npt.assert_array_equal(child.size[0], 500)
    show_m.close()


def test_layout_inside_a_widget_is_solved_by_the_render_loop():
    """Layouts not directly added to the scene are solved too."""
    child = _block()
    inner = ui.Row(child, padding=3)
    holder = ui.Panel2D((100, 100))
    holder.add_element(inner, (0, 0))
    scene = window.Scene()
    scene.add(holder)
    show_m = window.ShowManager(scene=scene, size=(200, 200), window_type="offscreen")
    show_m._draw_function()
    child.resize((70, 20))
    child.mark_layout_dirty()
    assert inner._layout_node.dirty
    show_m._draw_function()
    assert not inner._layout_node.dirty
    npt.assert_array_equal(inner.size, (76, 26))
    show_m.close()


def test_clean_frames_do_no_layout_work():
    """After a layout has been solved a frame must not solve it again."""
    row = ui.Row(_block(), padding=3)
    scene = window.Scene()
    scene.add(row)
    show_m = window.ShowManager(scene=scene, size=(200, 200), window_type="offscreen")
    show_m._draw_function()
    solved = []
    row._layout_node.apply = lambda rect: solved.append(rect)
    show_m._draw_function()
    assert solved == []
    assert len(ui.UIContext._dirty_layouts) == 0
    show_m.close()


def test_childless_box_reports_its_own_size():
    """A box without children is solved like any other root."""
    npt.assert_array_equal(ui.Row(width=200, height=50).size, (200, 50))
    npt.assert_array_equal(ui.Column(padding=5).size, (10, 10))


def test_box_shrinks_when_its_last_child_is_removed():
    child = _block((40, 20))
    row = ui.Row(child, padding=5)
    npt.assert_array_equal(row.size, (50, 30))
    row.remove(child)
    npt.assert_array_equal(row.size, (10, 10))
    assert not row._layout_node.dirty


def test_fit_panel_shrinks_when_its_last_child_is_removed():
    panel = ui.Panel2D((100, 100))
    panel.set_layout(height=ui.fit(), padding=(5, 0))
    child = panel.add(_block((40, 30)))
    panel.update_layout()
    npt.assert_array_equal(panel.size, (100, 40))
    panel.remove_element(child)
    panel.update_layout()
    npt.assert_array_equal(panel.size, (100, 10))


def test_panel_border_side_insets_only_that_side():
    panel = ui.Panel2D((100, 100), has_border=True, border_width=4)
    panel.set_layout(padding=1)
    child = panel.add(_block())
    panel.border_width = ("left", 20)
    panel.update_layout()
    npt.assert_array_equal(child.get_position(), (21, 5))
    assert panel.border_width == [20, 4, 4, 4]

    panel.border_width = ("Top", 10)
    panel.update_layout()
    npt.assert_array_equal(child.get_position(), (21, 11))


def test_panel_border_side_keeps_its_width_after_resize():
    panel = ui.Panel2D((100, 100), has_border=True, border_width=4)
    panel.border_width = ("right", 9)
    panel.resize((150, 120))
    assert panel.border_width == [4, 9, 4, 4]


def test_invalid_panel_border_side_changes_nothing():
    panel = ui.Panel2D((100, 100), has_border=True, border_width=4)
    child = panel.add(_block())
    with npt.assert_raises(ValueError):
        panel.border_width = ("middle", 20)
    panel.update_layout()
    npt.assert_array_equal(child.get_position(), (4, 4))
    assert panel.border_width == [4, 4, 4, 4]


def test_panel_children_are_drawn_above_the_background():
    panel = ui.Panel2D((200, 100), has_border=True, border_width=2)
    panel.z_order = 3
    laid_out = panel.add(_block())
    placed = ui.Rectangle2D(size=(10, 10))
    panel.add_element(placed, (50, 50))
    panel.update_layout()
    assert laid_out.z_order == 5
    assert placed.z_order == 5
    assert laid_out.actor.render_order > panel.background.actor.render_order


def test_panel_ignores_an_explicit_layout_border(caplog):
    panel = ui.Panel2D((100, 100), has_border=True, border_width=4)
    child = panel.add(_block())
    with caplog.at_level("WARNING"):
        panel.set_layout(border=0)
    assert "border" in caplog.text
    panel.border_width = ("left", 10)
    panel.update_layout()
    npt.assert_array_equal(child.get_position(), (10, 4))
