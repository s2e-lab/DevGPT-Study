quaternion = [w, x, y, z]  # Replace w, x, y, z with actual values
translation_vector = [tx, ty, tz]  # Replace tx, ty, tz with actual values

transformation_matrix = opencv_quaternion_to_opengl_transform(quaternion, translation_vector)
print(transformation_matrix)
