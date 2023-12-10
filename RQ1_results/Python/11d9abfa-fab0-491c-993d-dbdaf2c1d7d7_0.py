import math

class Rectangle:
    def __init__(self, minX, minY, maxX, maxY):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

    def width(self):
        return self.maxX - self.minX

    def height(self):
        return self.maxY - self.minY

    def center(self):
        return [(self.minX + self.maxX) // 2, (self.minY + self.maxY) // 2]

def rotate_bounding_box(bbox, angle, pivot):
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Calculate the sine and cosine of the angle
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Calculate the coordinates of the rotated corners
    corners = [(bbox.minX, bbox.minY), (bbox.maxX, bbox.minY),
               (bbox.maxX, bbox.maxY), (bbox.minX, bbox.maxY)]

    rotated_corners = []
    for x, y in corners:
        # Translate the corners to be relative to the pivot point
        translated_x = x - pivot[0]
        translated_y = y - pivot[1]

        # Perform the rotation
        new_x = translated_x * cos_angle - translated_y * sin_angle
        new_y = translated_x * sin_angle + translated_y * cos_angle

        # Translate the corners back to their original position
        rotated_x = new_x + pivot[0]
        rotated_y = new_y + pivot[1]

        rotated_corners.append((rotated_x, rotated_y))

    # Find the new bounding box coordinates from the rotated corners
    new_minX = min(x for x, _ in rotated_corners)
    new_maxX = max(x for x, _ in rotated_corners)
    new_minY = min(y for _, y in rotated_corners)
    new_maxY = max(y for _, y in rotated_corners)

    # Create a new Rectangle object with the rotated bounding box coordinates
    rotated_bbox = Rectangle(new_minX, new_minY, new_maxX, new_maxY)

    return rotated_bbox
