# Example usage:
bbox = Rectangle(100, 200, 300, 400)  # Replace these values with your actual bounding box coordinates
angle = 30  # Replace this with your desired rotation angle in degrees
pivot = bbox.center()

rotated_bbox = rotate_bounding_box(bbox, angle, pivot)

print(f"Original Bounding Box: (x_min={bbox.minX}, y_min={bbox.minY}), (x_max={bbox.maxX}, y_max={bbox.maxY})")
print(f"Rotated Bounding Box: (x_min={rotated_bbox.minX}, y_min={rotated_bbox.minY}), (x_max={rotated_bbox.maxX}, y_max={rotated_bbox.maxY})")
