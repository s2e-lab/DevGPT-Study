import numpy as np
import pyrr

def opencv_quaternion_to_opengl_transform(quaternion, translation_vector):
    # Step 1: Convert the quaternion to a rotation matrix
    rotation_matrix = pyrr.matrix33.create_from_quaternion(quaternion)

    # Step 2: Rotate the rotation matrix by 180 degrees around the x-axis
    x_rotation_matrix = pyrr.matrix33.create_from_x_rotation(np.radians(180))
    rotated_rotation_matrix = pyrr.matrix33.multiply(rotation_matrix, x_rotation_matrix)

    # Step 3: Create the translation matrix from the translation vector
    translation_matrix = pyrr.matrix44.create_from_translation(translation_vector)

    # Step 4: Append the translation matrix to the rotated rotation matrix
    transformation_matrix = np.dot(translation_matrix, np.hstack((rotated_rotation_matrix, np.array([[0], [0], [0]]))))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

    return transformation_matrix
