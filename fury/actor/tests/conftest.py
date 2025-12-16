import numpy as np
import pytest
import rendercanvas.glfw

from fury import actor


def _do_nothing_patch(self):
    pass


rendercanvas.glfw.RenderCanvas._rc_close = _do_nothing_patch


@pytest.fixture
def sphere_actor():
    centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    return actor.sphere(centers=centers, colors=colors)
