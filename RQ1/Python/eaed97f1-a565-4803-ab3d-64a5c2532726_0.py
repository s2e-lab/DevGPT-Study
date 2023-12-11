def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, end, obstacles):
    # The set of discovered nodes that may need to be (re-)expanded
    open_set = set([start])
    
    # For node n, came_from[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    came_from = {}

    # For node n, gscore[n] is the cost of the cheapest path from start to end.
    gscore = {start: 0}

    # For node n, fscore[n] := gscore[n] + h(n).
    fscore = {start: heuristic(start, end)}

    while open_set:
        # the node in openSet having the lowest fScore[] value
        current = min(open_set, key=lambda x: fscore.get(x, float('inf')))

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        open_set.remove(current)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = current[0] + dx, current[1] + dy
            tentative_gscore = gscore[current] + 1

            if 0 <= neighbor[0] < NUM_COLS and 0 <= neighbor[1] < NUM_ROWS:
                if neighbor in obstacles or tentative_gscore >= gscore.get(neighbor, float('inf')):
                    continue

                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, end)
                open_set.add(neighbor)

    return []  # Return an empty list if there is no path to the end
