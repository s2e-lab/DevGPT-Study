import numpy as np

def find_elbow(wcss):
    # Calculate the coordinates of the line from the first point to the last point
    n_points = len(wcss)
    coords = np.column_stack((range(n_points), wcss))
    
    # Initialize variables to hold the distances
    max_dist = 0
    elbow_point = 0
    
    # Calculate line from first to last point (y = mx + b)
    first_point = coords[0]
    last_point = coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Check distance of each point from the line
    for i in range(n_points):
        point = coords[i]
        vec_from_first = point - first_point
        scalar_proj = np.dot(vec_from_first, line_vec_norm)
        vec_on_line = line_vec_norm * scalar_proj + first_point
        dist_from_line = np.linalg.norm(point - vec_on_line)
        
        if dist_from_line > max_dist:
            max_dist = dist_from_line
            elbow_point = i
    
    return elbow_point

# Example usage
wcss = [1000, 900, 800, 400, 200, 100, 50, 40, 30]  # replace with your WCSS values
elbow_point = find_elbow(wcss)
print(f"The optimal number of clusters is at index {elbow_point}, which corresponds to k = {elbow_point + 1}")
