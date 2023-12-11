# Calculate the inverse rotation angle
inverse_rotation_angle = -angle

# Create a matrix for inverse rotation
inverse_rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), inverse_rotation_angle, 1.0)

# Transform the bounding box coordinates using the inverse rotation matrix
# Assuming you have the original bounding box as (x_min, y_min, x_max, y_max)
original_bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
transformed_bbox = cv2.transform(np.array([original_bbox]), inverse_rotation_matrix)[0]

# Get the new coordinates of the rotated bounding box
new_x_min = int(transformed_bbox[:, 0].min())
new_y_min = int(transformed_bbox[:, 1].min())
new_x_max = int(transformed_bbox[:, 0].max())
new_y_max = int(transformed_bbox[:, 1].max())
