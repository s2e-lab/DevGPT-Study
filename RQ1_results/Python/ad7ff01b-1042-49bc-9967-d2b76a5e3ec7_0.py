def create_bricks():
    bricks = []  # Create a new list for the bricks
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    for i in range(4):
        row = [Brick(x, 50 + i * 25, 60, 20, colors[i]) for x in range(0, WINDOW_WIDTH, 75)]
        bricks.extend(row)  # Add the new row of bricks to the list
    return bricks  # Return the list of bricks
