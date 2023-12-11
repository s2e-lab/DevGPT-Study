def get_agent_position(grid: Grid) -> Tuple[int, int]:
    """
    Return the position of the agent
    """
    for row in range(grid.rows):
        for col in range(grid.cols):
            cell = grid.cells[row][col]
            # Check for any of the characters representing the agent
            if cell[0] in ('>', 'V', '<', '^'):
                return (row, col)
    return (-1, -1)
