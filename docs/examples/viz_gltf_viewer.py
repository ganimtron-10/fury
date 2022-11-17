"""
===========
gLTF Viewer
===========

"""
import os

from fury import ui, window
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

# Fury files path
if 'FURY_HOME' in os.environ:
    fury_home = os.environ['FURY_HOME']
else:
    fury_home = os.path.join(os.path.expanduser('~'), '.fury')

current_path = fury_home
current_actors = []

f = ui.FileMenu2D(directory_path=fury_home, size=(250, 720), multiselection=False)


def open_gltf_file():
    global current_path
    global current_actors

    current_selection = f.listbox.selected

    if current_selection[0] == '../':
        current_path = os.path.dirname(current_path)
        return

    if current_selection[0].endswith(".gltf"):
        gltf_obj_path = os.path.join(current_path, current_selection[0])
        gltf_obj = glTF(gltf_obj_path, apply_normals=False)
        actors = gltf_obj.actors()

        if current_actors:
            sm.scene.rm(*current_actors)
        current_actors = actors

        sm.scene.add(*actors)

        cameras = gltf_obj.cameras
        if cameras:
            sm.scene.SetActiveCamera(cameras[0])

        return

    current_path = os.path.join(current_path, current_selection[0])


f.listbox.on_change = open_gltf_file


# Show Manager
sm = window.ShowManager(size=(1280, 720))
sm.scene.SetBackground(0.1, 0.1, 0.4)

sm.scene.add(f)

interactive = 1

if interactive:
    sm.start()

window.record(sm.scene, out_path='viz_gltf_viewer.png')
