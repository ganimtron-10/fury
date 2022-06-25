from fury import ui, window

scroll_1 = ui.ScrollBar((20, 200), 3, 10, position=(100, 100), orientation="vertical")
scroll_2 = ui.ScrollBar((200, 20), 4, 10, position=(100, 70), orientation="horizontal")
sli = ui.LineSlider2D(300, 400)

sm = window.ShowManager(size=(500, 500))
sm.scene.add(scroll_1, scroll_2, sli)
sm.start()