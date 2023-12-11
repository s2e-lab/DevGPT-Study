def main():
    # ... rest of the code before this stays the same ...
    
    obstacles = generate_obstacles(50, exclude_positions=[(start_x, start_y), (end_x, end_y)])
    
    # Create a set with obstacle positions for easy lookup
    obstacle_positions = set((obstacle.x, obstacle.y) for obstacle in obstacles)
    
    # Calculate the path using A* algorithm
    path = astar(None, (rover.x, rover.y), (end.x, end.y), obstacle_positions)
    
    # Iterator for path
    path_iter = iter(path)
    
    # Skip the first position as it is the rover's starting position
    next(path_iter)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Make the rover follow the path
        try:
            next_position = next(path_iter)
            rover.x, rover.y = next_position
        except StopIteration:
            # Path ended
            print("End goal reached!")
            pygame.quit()
            return
        
        # ... rest of the code stays the same, except for key handling logic ...
