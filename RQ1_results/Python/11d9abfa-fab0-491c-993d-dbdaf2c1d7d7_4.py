# Assuming you have the rotation angle in degrees as 'angle'
angle = 30  # Replace this with your desired rotation angle

# Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

# Perform the rotation on the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
