def is_inside_polygon(x, y, polygon):
    n = len(polygon)
    odd_nodes = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes

        j = i

    return odd_nodes

def points_inside_polygon(vertices, width, height):
    # Encuentra los puntos dentro del polígono
    points_inside = []
    for x in range(width):
        for y in range(height):
            if is_inside_polygon(x, y, vertices):
                points_inside.append((x, y))

    return points_inside

# Ejemplo de uso:
vertices = [(0, 0), (4, 0), (4, 4), (2, 6), (0, 4)]
width, height = 7, 7

points_inside = points_inside_polygon(vertices, width, height)
print(points_inside)
