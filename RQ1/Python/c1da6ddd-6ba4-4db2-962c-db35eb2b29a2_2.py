def on_button_press(self, widget, event):
    if event.button == 1:  # Left mouse button
        image_x, image_y = self.component_to_image_space(event.x, event.y)
        # Rest of the function...

def on_button_release(self, widget, event):
    image_x, image_y = self.component_to_image_space(event.x, event.y)
    # Rest of the function...

def on_motion_notify(self, widget, event):
    image_x, image_y = self.component_to_image_space(event.x, event.y)
    # Rest of the function...
