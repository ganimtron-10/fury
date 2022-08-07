"""
============
UI animation
============
Bouncing Ball Animation using UI element in FURY.
"""

import numpy as np
from fury import actor, window, ui
from fury.animation.timeline import Timeline
from fury.animation.interpolator import cubic_spline_interpolator, linear_interpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

fury = ui.TextBlock2D(text="FURY", font_size=50)

###############################################################################
# Creating a timeline to animate the actor
timeline = Timeline(playback_panel=Timeline)

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
timeline.add_actor(fury.actor)

###############################################################################
# Adding some position keyframes

# Below Bouncing Ball Program:
# https://physics.stackexchange.com/questions/256468/model-formula-for-bouncing-ball

h0 = 5         # m/s
v = 0          # m/s, current velocity
g = 10         # m/s/s
t = 0          # starting time
dt = 0.1       # time step
rho = 0.75     # coefficient of restitution
tau = 0.10     # contact time for bounce
hmax = h0      # keep track of the maximum height
h = h0
hstop = 0.1   # stop when bounce is less than 1 cm
freefall = True  # state: freefall or in contact
t_last = -np.sqrt(2*h0/g)  # time we would have launched to get to h0 at t=0
vmax = np.sqrt(2 * hmax * g)
H = []
T = []
while(hmax > hstop):
    if(freefall):
        hnew = h + v*dt - 0.5*g*dt*dt
        if(hnew < 0):
            t = t_last + 2*np.sqrt(2*hmax/g)
            freefall = False
            t_last = t + tau
            h = 0
        else:
            t = t + dt
            v = v - g*dt
            h = hnew
    else:
        t = t + tau
        vmax = vmax * rho
        v = vmax
        freefall = True
        h = 0
    hmax = 0.5*vmax*vmax/g
    H.append(h)
    T.append(t)

    timeline.set_position(t, np.array([100 + 100*t, 100 + 100*h]))


# ###############################################################################
# # change the position interpolator to Cubic spline interpolator.
# timeline.set_position_interpolator(linear_interpolator)

###############################################################################
# Main timeline to control all the timelines.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding timelines to the main Timeline.
scene.add(timeline)


###############################################################################
# making a function to update the animation and render the scene.
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation.
showm.add_timer_callback(True, 10, timer_callback)

interactive = 1

if interactive:
    showm.start()

window.record(scene, out_path='viz_ui_animation.png',
              size=(900, 768))
