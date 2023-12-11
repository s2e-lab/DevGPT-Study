def on_button_press(self, widget, event):
    if event.button == 1:  # Left mouse button
        x, y = self.component_to_image_space(event.x, event.y)
        closest_index, is_close = self.get_closest_point(x, y)

        if is_close:
            self.dragging_point = closest_index
        else:
            self.point_positions.append((x, y))

